/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlCache.h>

#include <iostream>

#define RUNNING 4
#define STOPPED 3
#define PAUSED  2
#define STARTED 1

namespace eqMivt
{

bool ControlCache::_initControlCache()
{
	_operationCond.lock();

	bool r = start();
	if (r)
	{
		_operationCond.wait();
	}
	_operationCond.unlock();
	return r;
}

bool ControlCache::stopWork()
{
	bool r = false;

	if (Thread::isStopped())
		return true;

	_operationCond.lock();
	if (_state != STOPPED)
	{
		if (_state == PAUSED || _state == STARTED)
		{
			_stateCond.lock();
			_stateCond.signal();
			_stateCond.unlock();
		}
		_notEnd = false;
		_operationCond.wait();
		r = true;
	}
	_operationCond.unlock();
	if (r)
		return join();
	else
		return false;
}

bool ControlCache::pauseWork()
{
	bool r = false;

	_operationCond.lock();
	if (_state == RUNNING)
	{
		_pause = true;
		_operationCond.wait();
		r = true;
	}
	_operationCond.unlock();

	return r;
}

bool ControlCache::_pauseWorkAndFreeCache()
{
	bool r = false;

	_operationCond.lock();
	if (_state == STARTED)
		r = true;
	else if (_state == RUNNING)
	{
		r = true;
		_free = true;
		_operationCond.wait();
	}
	_operationCond.unlock();

	return r;
}

bool ControlCache::continueWork()
{
	bool r = false;

	_operationCond.lock();
	if (_state == STARTED)
	{
		std::cerr<<"Try to continue first time without resize"<<std::endl;
		throw;
	}
	else if (_free) 
	{
		std::cerr<<"Try to continue, after free"<<std::endl;
		throw;
	}
	if (_state == PAUSED)
	{
		r = true;
		_stateCond.lock();
		_stateCond.signal();
		_stateCond.unlock();
		_operationCond.wait();
	}
	_operationCond.unlock();

	return r;
}

bool ControlCache::_continueWorkAndReSize()
{
	bool r = false;

	_operationCond.lock();
	if (_state == STARTED)
	{
		_stateCond.lock();
		_stateCond.signal();
		_resize = true;
		_stateCond.unlock();
		_operationCond.wait();
		r = true;
	}
	else if (_free)
	{
		std::cerr<<"Try to continue and resize without pause and free"<<std::endl;
		throw;
	}
	else if (_state == PAUSED)
	{
		_stateCond.lock();
		_stateCond.signal();
		_stateCond.unlock();
		_resize = true;
		_operationCond.wait();
		r = true;
	}
	_operationCond.unlock();

	return r;
}

bool ControlCache::_checkRunning()
{
	bool r = false;

	_operationCond.lock();
	if (_state == RUNNING)
		r = true;
	_operationCond.unlock();

	return r;
}

bool ControlCache::_checkStarted()
{
	bool r = false;

	_operationCond.lock();
	if (_state == STARTED)
		r = true;
	_operationCond.unlock();

	return r;
}

void ControlCache::run()
{
	_operationCond.lock();

		_resize = false;
		_free = false;
		_notEnd = true;
		_pause = false;
		_state = STARTED;
		_stateCond.lock();

	_operationCond.signal();
	_operationCond.unlock();


	if (_threadInit())
	{

		while(_notEnd)
		{
			if (_state == RUNNING)
			{
				_operationCond.lock();
				if (_pause)
				{
					_pause = false;
					_state = PAUSED;
					_operationCond.signal();
					_operationCond.unlock();
				}
				else if (_free)
				{
					_freeCache();
					_free = false;
					_state = PAUSED;
					_operationCond.signal();
					_operationCond.unlock();
				}
				else if (!_notEnd)
				{
					_state = STOPPED; 
				}
				else
				{
					_operationCond.unlock();
					_threadWork();
				}
			}
			else if (_state == STARTED)
			{
				_stateCond.wait();

				_operationCond.lock();
				if (!_notEnd)
				{
					_state = STOPPED;
				}
				else if (_resize)
				{
					_reSizeCache();
					_resize = false;
					_state = RUNNING;
					_operationCond.signal();
				}
				else
				{
					std::cerr<<"Control Cache, try to continue wihtout resizing the first time"<<std::endl;
					throw;
				}
				_operationCond.unlock();
			}
			else if (_state == PAUSED)
			{
				_stateCond.wait();

				_operationCond.lock();
				if (!_notEnd)
				{
					_state = STOPPED;
				}
				else if (_resize)
				{
					_reSizeCache();
					_resize = false;
					_state = RUNNING;
					_operationCond.signal();
				}
				else 
				{
					_state = RUNNING;
					_operationCond.signal();
				}
				_operationCond.unlock();
			}
			else if (_state == STOPPED)
			{
				break;
			}
			else
			{
				std::cerr<<"Contal Cache, worng state"<<std::endl;
				throw;
			}
		}

		_threadStop();

		_operationCond.signal();
		_operationCond.unlock(); 
		_stateCond.unlock();

	}
	else
		_state = STOPPED;

	lunchbox::Thread::exit();

	return;
}

}
