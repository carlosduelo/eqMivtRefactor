/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONTROL_CACHE_H
#define EQ_MIVT_CONTROL_CACHE_H

#include <lunchbox/thread.h>
#include <lunchbox/condition.h>

#include <typedef.h>

namespace eqMivt
{

class ControlCache : public lunchbox::Thread
{
private:
	lunchbox::Condition		_pauseCond;
	bool					_state;
	lunchbox::Condition		_stateCond;

	bool	_notEnd;
	bool	_resize;
	bool	_free;
	bool	_pause;

	bool	_safeArea;

protected:
	bool _initControlCache();

	bool _checkRunning();

	bool _checkStarted();

	bool _startProceted();

	bool _endProtected();

	virtual void _threadWork() = 0;

	virtual bool _threadInit() = 0;

	virtual void _threadStop() = 0;

	virtual void _freeCache() = 0;

	virtual void _reSizeCache() = 0;

	virtual void run();

	bool _pauseWorkAndFreeCache();

	bool _continueWorkAndReSize();
public:

	virtual ~ControlCache(){};

	bool pauseWork();

	bool continueWork();

	bool stopWork();


};

}

#endif /*EQ_MIVT_CONTROL_CACHE_H*/
