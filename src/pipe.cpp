/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "pipe.h"

#include "config.h"
#include "node.h"
#include <eq/eq.h>

namespace eqMivt
{
bool Pipe::configInit( const eq::uint128_t& initID )
{
    if( !eq::Pipe::configInit( initID ))
        return false;

    Config*         config      = static_cast<Config*>( getConfig( ));
    const InitData& initData    = config->getInitData();
    const eq::uint128_t&  frameDataID = initData.getFrameDataID();

	int ds = -1;
	if (cudaSuccess != cudaGetDevice(&ds))
	{
		LBERROR<<"Pipe: Error checking cuda device capable"<<std::endl;
		return false;
	}

	if (getDevice() < 32 && ds != (int)getDevice())
		if (cudaSuccess != cudaSetDevice(getDevice()))
		{
			LBERROR<<"Pipe: Error setting cuda device capable"<<std::endl;
			return false;
		}
		
	_lastState = true;
	// _lastState = _render.init(getDevice());
	

    return _lastState && config->mapObject( &_frameData, frameDataID );
}

bool Pipe::configExit()
{
    eq::Config* config = getConfig();
    config->unmapObject( &_frameData );

	//_render.destroy();

    return eq::Pipe::configExit();
}

void Pipe::frameStart( const eq::uint128_t& frameID, const uint32_t frameNumber)
{
    eq::Pipe::frameStart( frameID, frameNumber );
    _frameData.sync( frameID );
}

#if 0
Render * Pipe::getRender()
{
    Node*       node = static_cast<Node*>( getNode( ));

	//_lastState = node->updateRender(&_render); 	

	if (_lastState)
		return &_render;	
	else
		return 0;
}
#endif

}

