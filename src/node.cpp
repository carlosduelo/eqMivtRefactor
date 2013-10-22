/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "node.h"

#include "config.h"
#include "error.h"

namespace eqMivt
{
	bool Node::configInit( const eq::uint128_t& initID )
	{
		// All render data is static or multi-buffered, we can run asynchronously
		if( getIAttribute( IATTR_THREAD_MODEL ) == eq::UNDEFINED )
			setIAttribute( IATTR_THREAD_MODEL, eq::DRAW_SYNC );

		if( !eq::Node::configInit( initID ))
			return false;

		Config* config = static_cast< Config* >( getConfig( ));
		if( !config->loadData( initID ))
		{
			setError( ERROR_EQ_MIVT_FAILED );
			return false;
		}

		// Init cpu Cache
		const InitData& initData = config->getInitData();
		_status = true;

		if (!_resourcesManager.isInit())
		{

			_status =	_resourcesManager.init(initData.getDataFilename(), initData.getOctreeFilename(), initData.getTransferFunctionFile(), initData.getMemoryOccupancy());
		}

		if (!_status)
		{
			LBERROR<<"Node: error init resources"<<std::endl;
		}

		config->mapObject(&_frameData, initData.getFrameDataID());

		return _status;
	}

	void Node::frameStart( const eq::uint128_t& frameID, const uint32_t frameNumber )
	{
		_frameData.sync( frameID );

		if (_frameData.checkNextPosition())
			_resourcesManager.loadNextPosition();
		else if (_frameData.checkPreviusPosition())
			_resourcesManager.loadPreviusPosition();
		else if (_frameData.checkNextIsosurface())
			_resourcesManager.loadNextIsosurface();
		else if (_frameData.checkPreviusIsosurface())
			_resourcesManager.loadPreviusIsosurface();

		eq::Node::frameStart(frameID, frameNumber);
	}

	
	bool Node::updateRender(Render * render)
	{
		return _resourcesManager.updateRender(render);
	}

	bool Node::configExit()
	{
		Config* config = static_cast< Config* >( getConfig( ));
		config->unmapObject( &_frameData );

		return eq::Node::configExit();
	}
	
	vmml::vector<3, float>    Node::getVolumeCoord()
	{ 
		return _resourcesManager.getRealDimVolume(); 
	}

	vmml::vector<3, float>    Node::getStartCoord()
	{
		return _resourcesManager.getStartCoord();
	}

	vmml::vector<3, float>    Node::getFinishCoord()
	{
		return _resourcesManager.getEndCoord();
	}
}

