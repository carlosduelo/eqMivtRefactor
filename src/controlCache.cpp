/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlCache.h>

#include <iostream>

namespace eqMivt
{

bool ControlCache::_initControlCache()
{
	_state = STARTED;
	_resize = false;
	_free = false;
	_safeArea = false;
	_notEnd = true;
	_pause = false;

	return start();
}

bool ControlCache::stopWork()
{
	bool r = false;

	_stateCond.lock();
	if (_state != STOPPED)
	{
		_notEnd = false;
		if (_state == PAUSED || _state == STARTED)
		{
			_pauseCond.signal();
		}

		r = true;
		_stateCond.wait();
	}
	_stateCond.unlock();
	if (r)
		join();

	return r;
}

bool ControlCache::_startProceted()
{
	_stateCond.lock();
	if (_state == RUNNING)
	{
		_safeArea = true;
		return true;
	}

	_stateCond.unlock();
	return false;
}

bool ControlCache::_endProtected()
{
	if (!_safeArea)
	{
		std::cerr<<"Control Cache, try to end safe aread not protected before"<<std::endl;
		return false;
	}
}

bool ControlCache::pauseWork()
{
	bool r = false;

	_stateCond.lock();
	if (_state == RUNNING)
	{
		_pause = true;
		_stateCond.wait();
		r = true;
	}
	_stateCond.unlock();

	return r;
}

bool ControlCache::_pauseWorkAndFreeCache()
{
	bool r = false;

	_stateCond.lock();
	if (_state == RUNNING)
	{
		r = true;
		_free = true;
		_stateCond.wait();
	}
	_stateCond.unlock();

	return r;
}

bool ControlCache::continueWork()
{
	bool r = false;

	_stateCond.lock();
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
		_pauseCond.signal();
		_stateCond.wait();
	}
	_stateCond.unlock();

	return r;
}

bool ControlCache::_continueWorkAndReSize()
{
	bool r = false;

	_stateCond.lock();
	if (_state == STARTED)
	{
		_resize = true;
		_pauseCond.signal();
		_stateCond.wait();
	}
	else if (!_free)
	{
		std::cerr<<"Try to continue and resize without pause and free"<<std::endl;
		throw;
	}
	else if (_state == PAUSED)
	{
		r = true;
		_resize = true;
		_pauseCond.signal();
		_stateCond.wait();
	}
	_stateCond.unlock();

	return r;
}

bool ControlCache::_checkRunning()
{
	bool r = false;

	_stateCond.lock();
	if (_state == RUNNING)
		r = true;
	_stateCond.unlock();

	return r;
}

bool ControlCache::_checkStarted()
{
	bool r = false;

	_stateCond.lock();
	if (_state == STARTED)
		r = true;
	_stateCond.unlock();

	return r;
}

void ControlCache::run()
{
	_stateCond.lock();

	if (_threadInit())
	{
		_stateCond.unlock();

		while(_notEnd)
		{
			_stateCond.lock();

			if (_state == STARTED)
			{
				_stateCond.unlock();
				_pauseCond.wait();
				_pauseCond.unlock();

				if (!_notEnd)
				{
					_stateCond.lock();
					_state = STOPPED;
				}
				else if (_resize)
				{
					_stateCond.lock();
					_reSizeCache();
					_resize = false;
					_state = RUNNING;
					_stateCond.signal();
				}
				else
				{
					std::cerr<<"Control Cache, try to continue wihtout resizing the first time"<<std::endl;
					throw;
				}
			}
			else if (_state == PAUSED)
			{
				_stateCond.unlock();
				_pauseCond.wait();
				_pauseCond.unlock();

				if (!_notEnd)
				{
					_stateCond.lock();
					_state = STOPPED;
				}
				else if (_resize)
				{
					_stateCond.lock();
					_reSizeCache();
					_resize = false;
					_state = RUNNING;
					_stateCond.signal();
				}
				else 
				{
					_stateCond.lock();
					_state = RUNNING;
					_stateCond.signal();
				}
			}
			else if (_state == RUNNING)
			{
				if (_pause)
				{
					_pause = false;
					_state == PAUSED;
					_stateCond.signal();
				}
				else if (_free)
				{
					_freeCache();
					_free = false;
					_state == PAUSED;
					_stateCond.signal();
				}
				else if (!_notEnd)
				{
					_state = STOPPED; 
				}
				else
				{
					_threadWork();
				}
			}
			else if (_state == STOPPED)
			{
				_stateCond.signal();
				break;
			}
			else
			{
				std::cerr<<"Contal Cache, worng state"<<std::endl;
				throw;
			}

			_stateCond.unlock();
		}

		_threadStop();

		_stateCond.signal();
	}
	else
		_state = STOPPED;

	_stateCond.unlock();

	lunchbox::Thread::exit();
}

}
