/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "frameData.h"
#include <typedef.h>

namespace eqMivt
{

FrameData::FrameData() 
        : _center( eq::Vector4f::ZERO )
		, _radio( 0.0f )
		, _angle( 0.0f )
        , _position( eq::Vector4f::ZERO )
        , _up( eq::Vector4f::ZERO )
        , _viewM( eq::Matrix4f::IDENTITY )
        , _invViewM( eq::Matrix4f::IDENTITY )
		, _idle( false )
		, _drawBox( true )
		, _renderCubes( false )
		, _statistics( false )
		, _nextPosition( false )
		, _previusPosition( false )
		, _nextIsosurface( false )
		, _previusIsosurface( false )
        , _startCoord( eq::Vector4f::ZERO )
        , _endCoord( eq::Vector4f::ZERO )
        , _realDim( eq::Vector4f::ZERO )
{
    reset();
}

void FrameData::serialize( co::DataOStream& os, const uint64_t dirtyBits )
{
    co::Serializable::serialize( os, dirtyBits );
    if( dirtyBits & DIRTY_CAMERA )
        os << _position << _center << _radio << _angle << _up << _viewM << _invViewM;
	if( dirtyBits & DIRTY_FLAGS )
		os << _idle << _statistics << _drawBox <<  _renderCubes;
	if( dirtyBits & DIRTY_VIEW )
		os << _currentViewID;
	if( dirtyBits & DIRTY_MODEL )
		os << _nextPosition << _previusPosition << _nextIsosurface << _previusIsosurface << _startCoord << _endCoord << _realDim; 
}

void FrameData::deserialize( co::DataIStream& is, const uint64_t dirtyBits )
{
    co::Serializable::deserialize( is, dirtyBits );
    if( dirtyBits & DIRTY_CAMERA )
        is >> _position >> _center >> _radio >> _angle >> _up >> _viewM >> _invViewM;
	if( dirtyBits & DIRTY_FLAGS )
		is >> _idle >> _statistics >> _drawBox >> _renderCubes;
	if( dirtyBits & DIRTY_VIEW )
		is >> _currentViewID;
	if( dirtyBits & DIRTY_MODEL )
		is >> _nextPosition >> _previusPosition >> _nextIsosurface >> _previusIsosurface >> _startCoord >> _endCoord >> _realDim; 
}

void FrameData::setRealDim(eq::Vector3f coord)
{
	_realDim = coord;
	setDirty( DIRTY_MODEL);
}

void FrameData::setStartCoord(eq::Vector3f coord)
{
	_startCoord = coord;
	setDirty( DIRTY_MODEL);
}

void FrameData::setEndCoord(eq::Vector3f coord)
{
	_endCoord = coord;
	setDirty( DIRTY_MODEL);
}

void FrameData::setNone()
{
	bool c = _nextPosition || _previusPosition || _nextIsosurface || _previusIsosurface;

	_nextPosition = false;
	_previusPosition = false;
	_nextIsosurface = false;
	_previusIsosurface = false;

	if (c)
	    setDirty( DIRTY_MODEL);
}

void FrameData::setNextPosition()
{
	if (_previusPosition || _nextIsosurface || _previusIsosurface)
		return;

	_nextPosition = true;
    setDirty( DIRTY_MODEL);
}
void FrameData::setPreviusPosition()
{
	if (_nextPosition || _nextIsosurface || _previusIsosurface)
		return;

	_previusPosition = true;
    setDirty( DIRTY_MODEL);
}
void FrameData::setNextIsosurface()
{
	if (_nextPosition || _previusPosition || _previusIsosurface)
		return;

	_nextIsosurface = true;
    setDirty( DIRTY_MODEL);
}
void FrameData::setPreviusIsosurface()
{
	if (_nextPosition || _previusPosition || _nextIsosurface)
		return;

	_previusIsosurface = true;
    setDirty( DIRTY_MODEL);
}

void FrameData::spinCamera( const float x, const float y )
{
    if( x == 0.f && y == 0.f )
        return;
	
	_viewM.rotate_y(y);
	_viewM.rotate_x(x);

	compute_inverse(_viewM, _invViewM);
    setDirty( DIRTY_CAMERA );
}

void FrameData::zoom( const float x)
{
	vmml::vector<4, int> l = _center - _position;

	float d = fminf(l.length()*0.1f*x, 500.0f);
	d = d == 0 ? x : d;

	eq::Matrix4f t =  eq::Matrix4f::IDENTITY;
	t.set_translation(0.0f, 0.0f, d);
	_viewM = _viewM * t;
	compute_inverse(_viewM, _invViewM);

    setDirty( DIRTY_CAMERA );
}

void FrameData::moveCamera( const float x, const float y, const float z )
{
	vmml::vector<4, int> l = _center - _position;
	float d = fminf(l.length(), 20.0f);

	if ( x != 0)
	{
		float angleX = d*x*0.0005f;
		eq::Matrix4f rotX = eq::Matrix4f::IDENTITY;
		eq::Matrix4f t1 = eq::Matrix4f::IDENTITY;
		eq::Matrix4f t2 = eq::Matrix4f::IDENTITY;
		t1.set_translation(TO3V(-_center));
		t2.set_translation(TO3V(_center));
		rotX.rotate(angleX, TO3V(_up));
		_viewM = t2 * (rotX * (t1 * _viewM));
	}

	if (y != 0)
	{
		eq::Matrix4f t = eq::Matrix4f::IDENTITY;
		t.set_translation(vmml::vector<3,float>(0.0f, y*d*0.05f, 0.0f));
		_viewM = t * _viewM;
	}

	compute_inverse(_viewM, _invViewM);

    setDirty( DIRTY_CAMERA );
}

void FrameData::setIdle( const bool idle )
{
	if( _idle == idle )
		return;

	_idle = idle;
	setDirty( DIRTY_FLAGS );
}

void FrameData::setStatistics()
{ 
	_statistics = !_statistics;
	setDirty( DIRTY_FLAGS );
}

void FrameData::setDrawBox()
{
	_drawBox = !_drawBox;
	setDirty( DIRTY_FLAGS );
}

void FrameData::setRenderCubes()
{
	_renderCubes = !_renderCubes;
	setDirty( DIRTY_FLAGS );
}

void FrameData::setCurrentViewID( const eq::uint128_t& id )
{
	_currentViewID = id;
	setDirty( DIRTY_VIEW );
}

void FrameData::reset()
{
	_position   = eq::Vector4f::ZERO;
	_center		= eq::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
	_up			= eq::Vector4f(0.0f, 1.0f, 0.0f, 0.0f);

	_center.x() = _startCoord.x() + ((_endCoord.x()-_startCoord.x())/2.0f);
	_center.y() = _startCoord.y() + ((_endCoord.y()-_startCoord.y())/2.0f);
	_center.z() = _startCoord.z() + ((_endCoord.z()-_startCoord.z())/2.0f);

std::cout<<_center<<_startCoord<<_endCoord<<std::endl;
	_radio =  _center.length();
	_angle = 0;
	_position.x() = _center.x() + _radio * sin(_angle);
	_position.y() = _center.y() + _radio * sin(_angle);
	_position.z() = _center.z() + _radio * cos(_angle);

	_createViewMatrix();

    setDirty( DIRTY_CAMERA );
}

void FrameData::_createViewMatrix()
{
	#if 0
	vmml::vector<3,float> wAux = TO3V((_position - _center)); wAux.normalize();
	vmml::vector<3,float> uAux = TO3V(_up); uAux = uAux.cross(wAux); uAux.normalize(); 
	vmml::vector<3,float> vAux = wAux.cross(uAux); 
	vmml::vector<4,float> u = uAux; u[3] = 0.0f;
	vmml::vector<4,float> v = vAux; v[3] = 0.0f;
	vmml::vector<4,float> w = wAux; w[3] = 0.0f; 

	_viewM[0][0] = u.x(); _viewM[1][0] = v.x(); _viewM[2][0] = w.x(); _viewM[3][0] = 0.0f;
	_viewM[0][1] = u.y(); _viewM[1][1] = v.y(); _viewM[2][1] = w.y(); _viewM[3][1] = 0.0f;
	_viewM[0][2] = u.z(); _viewM[1][2] = v.z(); _viewM[2][2] = w.z(); _viewM[3][2] = 0.0f;
	_viewM[0][3] = u.w(); _viewM[1][3] = v.w(); _viewM[2][3] = w.w(); _viewM[3][3] = 1.0f;
	#endif
	_viewM = eq::Matrix4f::IDENTITY;
	_viewM.set_translation(TO3V(_position));
	compute_inverse(_viewM, _invViewM);

    setDirty( DIRTY_CAMERA );
}
}

