/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONTROL_ELEMENT_CACHE_H
#define EQ_MIVT_CONTROL_ELEMENT_CACHE_H

#include <controlCache.h>
#include <linkedList.h>

#include <iostream> 
#include <algorithm>
#include <queue>

#include <boost/unordered_map.hpp>

#ifdef TIMING
#include <lunchbox/clock.h>
#endif

#define PROCESSING -1
#define PROCESSED -2
#define WAITING 200

#define MAX_QUEUE 250

namespace eqMivt
{

template< class TYPE >
class  ControlElementCache : public ControlCache 
{
	private:
		LinkedList<TYPE>										_lruElement;
		boost::unordered_map<TYPE, NodeLinkedList<TYPE> * >		_currentElement;
		std::queue<TYPE>										_pendingElement;
		std::queue<NodeLinkedList<TYPE> *>						_readingElement;

		lunchbox::Condition			_emptyPending;
		lunchbox::Condition			_fullSlots;

		#ifdef TIMING
		double _searchN;
		double _insertN;
		double _readingN;
		double _reading;
		double _search;
		double _insert;
		#endif

		bool readElement(NodeLinkedList<TYPE> * element)
		{
		#ifdef TIMING
			lunchbox::Clock clock;
			clock.reset();

			bool r = _readElement(element);
			_reading += clock.getTimed()/1000.0;
			_readingN += 1.0;

			return r; 
		#else
			return _readElement(element);
		#endif
		}

		void _threadWork()
		{
			if (_readingElement.size() > 0)
			{
				NodeLinkedList<TYPE> * element = _readingElement.front();
				if (readElement(element))
				{
					element->refs = PROCESSED;
					_readingElement.pop();
				}
			}
			else
			{
				_emptyPending.lock();
				if (_pendingElement.size() == 0)
				{
					if (!_emptyPending.timedWait(WAITING))
					{
						_emptyPending.unlock();
						return;
					}
				}

				TYPE element = _pendingElement.front();
				_emptyPending.unlock();

				_fullSlots.lock();

				typename boost::unordered_map<TYPE, NodeLinkedList<TYPE> *>::iterator it;
				it = _currentElement.find(element);
				if (it == _currentElement.end())
				{
					if (_freeSlots == 0 && !_fullSlots.timedWait(WAITING))
					{
						_fullSlots.unlock();
						return;
					}


					NodeLinkedList<TYPE> * c = _lruElement.getFirstFreePosition();

					if (c == 0)
					{
						std::cerr<<"Error control element cache, cache is full"<<std::endl;
						throw;
					}

					it = _currentElement.find(c->id);
					if (it != _currentElement.end())
						_currentElement.erase(it);	
				
					_freeSlots--;
					_fullSlots.unlock();

					#ifndef NDEBUG
					if (c->refs != 0)
					{
						std::cerr<<"Control Element Cache, unistable state, free plane slot with references "<<c->id<<" refs "<<c->refs<<std::endl;
						throw;
					}
					#endif

					c->id = element;
					c->refs = PROCESSING;
					if (!readElement(c))
						_readingElement.push(c);
					else
						c->refs = PROCESSED;

					_fullSlots.lock();

					_currentElement.insert(std::make_pair<index_node_t, NodeLinkedList<TYPE> *>(c->id, c));
					_lruElement.moveToLastPosition(c);

					_fullSlots.unlock();

					_emptyPending.lock();

					_pendingElement.pop();

					_emptyPending.unlock();

				}
				else
				{
					_lruElement.moveToLastPosition(it->second);
					_fullSlots.unlock();

					_emptyPending.lock();

					_pendingElement.pop();

					_emptyPending.unlock();
				}
			}
		}

	protected:
		virtual bool _readElement(NodeLinkedList<TYPE> * element) = 0;

		virtual bool _threadInit()
		{
			_currentElement.clear();
			std::queue<TYPE> emptyQ;
			std::queue<NodeLinkedList<TYPE>*> emptyQN;
			std::swap(_pendingElement, emptyQ);
			std::swap(_readingElement, emptyQN);
			_freeSlots = 0;
			_memory = 0;
			_sizeElement = 0;

			return true;
		}

		virtual void _threadStop()
		{
			#ifndef NDEBUG
			std::cout<<"Control Element Cache stopped"<<std::endl;
			#endif
			_memory= 0;
		}

		virtual void _freeCache()
		{
			#ifdef TIMING
			if (_searchN != 0.0 && _insertN != 0.0 && _readingN != 0.0)
			{
				std::cout<<"Time searching "<<_search<<" seconds"<<std::endl;
				std::cout<<"Time inserting "<<_insert<<" seconds"<<std::endl;
				std::cout<<"Time reading "<<_reading<<" seconds"<<std::endl;
				std::cout<<"Average searching "<<_search/_searchN<<" seconds, operations "<<_searchN<<std::endl;
				std::cout<<"Average inserting "<<_insert/_insertN<<" seconds, operations "<<_insertN<<std::endl;
				std::cout<<"Average reading "<<_reading/_readingN<<" seconds, operations "<<_readingN<<std::endl;
				_searchN = 0.0;
				_insertN = 0.0;
				_readingN = 0.0;
				_reading = 0.0;
				_search = 0.0;
				_insert = 0.0;
			}
			#endif
		}

		virtual void _reSizeCache()
		{
			if (!_lruElement.reSize(_freeSlots))
			{
				std::cerr<<"Error resizing control element cache "<<_freeSlots <<" elements"<<std::endl;
			}
			_currentElement.clear();
			std::queue<TYPE> emptyQ;
			std::queue<NodeLinkedList<TYPE>*> emptyQN;
			std::swap(_pendingElement, emptyQ);
			std::swap(_readingElement, emptyQN);
		}

		bool _init()
		{
			#ifdef TIMING
			_searchN = 0.0;
			_insertN = 0.0;
			_readingN = 0.0;
			_reading = 0.0;
			_search = 0.0;
			_insert = 0.0;
			#endif

			return  ControlCache::_initControlCache();
		}

		bool _freeCacheAndPause()
		{
			return _pauseWorkAndFreeCache();
		}

		bool _reSizeCacheAndContinue()
		{
			if (_checkRunning())
				return false;

			return _continueWorkAndReSize();
		}

		unsigned int	_freeSlots;
		unsigned int	_sizeElement;
		float	*		_memory;
		TYPE			_minValue;
		TYPE			_maxValue;

	public:

		float * getAndBlockElement(TYPE element)
		{
			if (element < _minValue || element > _maxValue)
			{
				std::cerr<<"Try to get element out of range"<<std::endl;
				throw;
			}

			typename boost::unordered_map<TYPE, NodeLinkedList<TYPE> * >::iterator it;
			float * data = 0;

			#ifdef TIMING
			lunchbox::Clock clock;
			clock.reset();
			#endif
			_fullSlots.lock();
			it = _currentElement.find(element);
			#ifdef TIMING
			_search += clock.getTimed()/1000.0;
			_searchN += 1.0;
			#endif
			if (it != _currentElement.end())
			{
				if (it->second->refs != PROCESSING)
				{
					if (it->second->refs == 0)
						_freeSlots--;
					else if (it->second->refs == PROCESSED)
						it->second->refs = 0;

					it->second->refs += 1;
					data = _memory + it->second->element*_sizeElement;
				}

				_fullSlots.unlock();
			}
			else
			{
				_fullSlots.unlock();
			#ifdef TIMING
			clock.reset();
			#endif
				_emptyPending.lock();
				if (_pendingElement.size() <=  MAX_QUEUE)
				{	
					if (_pendingElement.size() == 0)
					{
						_pendingElement.push(element);
						_emptyPending.signal();
					}
					else if (_pendingElement.front() != element)
					{
						_pendingElement.push(element);
					}
			#ifdef TIMING
			_insert += clock.getTimed()/1000.0;
			_insertN += 1.0;
			#endif
				}
				_emptyPending.unlock();
			}

			return data;
		}

		void	unlockElement(TYPE element)
		{
			if (element < _minValue || element > _maxValue)
			{
				std::cerr<<"Try to get element out of range"<<std::endl;
				throw;
			}

			typename boost::unordered_map<TYPE, NodeLinkedList<TYPE> * >::iterator it;
			_fullSlots.lock();
			it = _currentElement.find(element);
			if (it != _currentElement.end())
			{
				it->second->refs -= 1;
				if (it->second->refs == 0)
				{
					_freeSlots++;
					_fullSlots.signal();
				}

				#ifndef NDEBUG
				if (it->second->refs < 0)
				{
					std::cerr<<"Control Element Cache, error unlocking cube"<<std::endl;
					throw;
				}
				#endif
			}
			#ifndef NDEBUG
			else
			{
				std::cerr<<"Control Element Cache, error unlocking cube that not exists"<<std::endl;
				throw;
			}
			#endif

			_fullSlots.unlock();
		}

};

}
#endif
