/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "config.h"

#include "configEvent.h"

#include <math.h>
#include <boost/filesystem.hpp>

namespace eqMivt
{

Config::Config( eq::ServerPtr parent )
        : eq::Config( parent )
        , _redraw( true )
		, _numFramesAA( 0 )
{
}

Config::~Config()
{
}

bool Config::init()
{
    registerObject( &_frameData );
   _frameData.setAutoObsolete( getLatency( ));

    _initData.setFrameDataID( _frameData.getID( ));
    registerObject( &_initData );

    if( !eq::Config::init( _initData.getID( )))
    {
        _deregisterData();
        return false;
    }

    if (_surfaceInfo.init(_initData.getOctreeFilename()))
	{
		_frameData.setRealDim(_surfaceInfo.getRealDimVolume());
		_frameData.setStartCoord(_surfaceInfo.getStartCoord());
		_frameData.setEndCoord(_surfaceInfo.getEndCoord());
		_frameData.reset();
		return true;
	}
	else
	{
		return false;
	}
}

void Config::_reset()
{
	_frameData.reset();
}

bool Config::exit()
{
    const bool ret = eq::Config::exit();
    _deregisterData();

    return ret;
}

void Config::_deregisterData()
{
    deregisterObject( &_initData );
    deregisterObject( &_frameData );

    _initData.setFrameDataID( eq::UUID( ));
}

bool Config::loadData( const eq::uint128_t& initDataID )
{
    if( !_initData.isAttached( ))
    {
        const uint32_t request = mapObjectNB( &_initData, initDataID,
                                              co::VERSION_OLDEST,
                                              getApplicationNode( ));
        if( !mapObjectSync( request ))
            return false;
        unmapObject( &_initData ); // data was retrieved, unmap immediately
    }
    else // appNode, _initData is registered already
    {
        LBASSERT( _initData.getID() == initDataID );
    }

    // Check needed files exist
    if (!boost::filesystem::exists(_initData.getOctreeFilename()))
    {
	    LBERROR << "Cannot open "<<_initData.getOctreeFilename()<<" file."<< std::endl;
	    return false;
    }
    if (!boost::filesystem::exists(_initData.getDataFilename()[0]))
    {
	    LBERROR << "Cannot open "<<_initData.getDataFilename()[0]<<" file."<< std::endl;
	    return false;
    }

    return true;
}

uint32_t Config::startFrame()
{
	// idle mode
	if( isIdleAA( ))
	{
		LBASSERT( _numFramesAA > 0 );
		_frameData.setIdle( true );
	}
	else
		_frameData.setIdle( false );

	_numFramesAA = 0;

    const eq::uint128_t& version = _frameData.commit();

    _redraw = false;
    return eq::Config::startFrame( version );
}

uint32_t Config::finishFrame()
{
	_frameData.setNone();
	return eq::Config::finishFrame();
}
	
bool Config::handleEvent( const eq::ConfigEvent* event )
{
    switch( event->data.type)
    {
        case eq::Event::KEY_PRESS:
        {
            if( _handleKeyEvent( event->data.keyPress ))
            {
                _redraw = true;
                return true;
            }
            break;
        }

        case eq::Event::CHANNEL_POINTER_BUTTON_PRESS:
        {
			const eq::uint128_t& viewID = event->data.context.view.identifier;
			_frameData.setCurrentViewID( viewID );
			if (viewID == 0)
			{
				return false;
			}
			break;
        }

        case eq::Event::CHANNEL_POINTER_BUTTON_RELEASE:
        {
            break;
        }
        case eq::Event::CHANNEL_POINTER_MOTION:
        {
            switch( event->data.pointerMotion.buttons )
            {
              case eq::PTR_BUTTON1:

                      _frameData.spinCamera(
                          -0.005f * event->data.pointerMotion.dy,
                          -0.005f * event->data.pointerMotion.dx );
                  _redraw = true;
                  return true;

              case eq::PTR_BUTTON2:
                  _frameData.zoom( event->data.pointerMotion.dy );
                  _redraw = true;
                  return true;

              case eq::PTR_BUTTON3:
                  _frameData.moveCamera(  event->data.pointerMotion.dx,
                                         -event->data.pointerMotion.dy,
                                          0.f );
                  _redraw = true;
                  return true;
            }
            break;
        }

        case eq::Event::CHANNEL_POINTER_WHEEL:
        {
            _frameData.moveCamera( -event->data.pointerWheel.yAxis,
                                   0.f,
                                   event->data.pointerWheel.xAxis );
            _redraw = true;
            return true;
        }

        case eq::Event::MAGELLAN_AXIS:
        {
	    break;
        }

        case eq::Event::MAGELLAN_BUTTON:
        {
	    break;
        }

        case eq::Event::WINDOW_EXPOSE:
        case eq::Event::WINDOW_RESIZE:
        case eq::Event::WINDOW_CLOSE:
        case eq::Event::VIEW_RESIZE:
            _redraw = true;
            break;
    default:
        break;
    }

    _redraw |= eq::Config::handleEvent( event );
    return _redraw;
}

bool Config::handleEvent( eq::EventICommand command )
{
    switch( command.getEventType( ))
	{
		case IDLE_AA_LEFT:
		{
			const int32_t steps = command.get< int32_t >();
			_numFramesAA = LB_MAX( _numFramesAA, steps );
			return false;
		}
        case eq::Event::KEY_PRESS:
        {
			const eq::Event& event = command.get< eq::Event >();
            if( _handleKeyEvent( event.keyPress ))
            {
                _redraw = true;
                return true;
            }
            break;
        }

        case eq::Event::CHANNEL_POINTER_BUTTON_PRESS:
        {
			const eq::Event& event = command.get< eq::Event >();
			const eq::uint128_t& viewID = event.context.view.identifier;
			_frameData.setCurrentViewID( viewID );
			if (viewID == 0)
			{
				return false;
			}
			break;
        }

        case eq::Event::CHANNEL_POINTER_BUTTON_RELEASE:
        {
            break;
        }
        case eq::Event::CHANNEL_POINTER_MOTION:
        {
	    const eq::Event& event = command.get< eq::Event >();
            switch( event.pointerMotion.buttons )
            {
              case eq::PTR_BUTTON1:

                      _frameData.spinCamera(
                          -0.005f * event.pointerMotion.dy,
                          -0.005f * event.pointerMotion.dx );
                  _redraw = true;
                  return true;

              case eq::PTR_BUTTON2:
                  _frameData.zoom(event.pointerMotion.dy );
                  _redraw = true;
                  return true;

              case eq::PTR_BUTTON3:
                  _frameData.moveCamera(  event.pointerMotion.dx,
                                         -event.pointerMotion.dy,
                                          0.f );
                  _redraw = true;
                  return true;
            }
            break;
        }

        case eq::Event::CHANNEL_POINTER_WHEEL:
        {
			const eq::Event& event = command.get< eq::Event >();
            _frameData.moveCamera( -event.pointerWheel.yAxis,
                                   0.f,
                                   event.pointerWheel.xAxis );
            _redraw = true;
            return true;
        }

        case eq::Event::MAGELLAN_AXIS:
        {
	    break;
        }

        case eq::Event::MAGELLAN_BUTTON:
        {
	    break;
        }

        case eq::Event::WINDOW_EXPOSE:
        case eq::Event::WINDOW_RESIZE:
        case eq::Event::WINDOW_CLOSE:
        case eq::Event::VIEW_RESIZE:
            _redraw = true;
            break;
    default:
        break;
    }

    _redraw |= eq::Config::handleEvent( command );
    return _redraw;
}

bool Config::_handleKeyEvent( const eq::KeyEvent& event )
{
    switch( event.key )
    {
		case 'r':
		{
			_reset();
			return true;
		}
		case 'R':
		{
			_reset();
			return true;
		}
		case 'c':
		{
			_frameData.setRenderCubes();
			return true;
		}
		case 'C':
		{
			_frameData.setRenderCubes();
			return true;
		}
		
		case 'b':
		{
			_frameData.setDrawBox();
			return true;
		}
		case 'B':
		{
			_frameData.setDrawBox();
			return true;
		}
		case 's':
		{
			_frameData.setStatistics();
			return true;
		}
		case 'S':
		{
			_frameData.setStatistics();
			return true;
		}
		case 'n':
		{
			if (_surfaceInfo.checkLoadNextPosition())
			{
				_frameData.setStartCoord(_surfaceInfo.getStartCoord());
				_frameData.setEndCoord(_surfaceInfo.getEndCoord());
				_frameData.setNextPosition();
				_reset();
			}
			return true;
		}
		case 'N':
		{
			if (_surfaceInfo.checkLoadNextPosition())
			{
				_frameData.setStartCoord(_surfaceInfo.getStartCoord());
				_frameData.setEndCoord(_surfaceInfo.getEndCoord());
				_frameData.setNextPosition();
				_reset();
			}
			return true;
		}
		case 'p':
		{
			if (_surfaceInfo.checkLoadPreviusPosition())
			{
				_frameData.setStartCoord(_surfaceInfo.getStartCoord());
				_frameData.setEndCoord(_surfaceInfo.getEndCoord());
				_frameData.setPreviusPosition();
				_reset();
			}
			return true;
		}
		case 'P':
		{
			if (_surfaceInfo.checkLoadPreviusPosition())
			{
				_frameData.setStartCoord(_surfaceInfo.getStartCoord());
				_frameData.setEndCoord(_surfaceInfo.getEndCoord());
				_frameData.setPreviusPosition();
				_reset();
			}
			return true;
		}
		case 'o':
		{
			if (_surfaceInfo.checkLoadNextIsosurface())
			{
				_frameData.setNextIsosurface();
				_reset();
			}
			return true;
		}
		case 'O':
		{
			if (_surfaceInfo.checkLoadNextIsosurface())
			{
				_frameData.setNextIsosurface();
				_reset();
			}
			return true;
		}
		case 'l':
		{
			if (_surfaceInfo.checkLoadPreviusIsosurface())
			{
				_frameData.setPreviusIsosurface();
				_reset();
			}
			return true;
		}
		case 'L':
		{
			if (_surfaceInfo.checkLoadPreviusIsosurface())
			{
				_frameData.setPreviusIsosurface();
				_reset();
			}
			return true;
		}
        default:
            return false;
    }
}

bool Config::needRedraw()
{
    return _redraw || _numFramesAA > 0;
}

bool Config::isIdleAA()
{
    return ( !_redraw && _numFramesAA > 0 );//&& !_frameData.isDrawBox());
}

co::uint128_t Config::sync( const co::uint128_t& version )
{
    return eq::Config::sync( version );
}
}

