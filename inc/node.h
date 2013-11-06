/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_NODE_H
#define EQ_MIVT_NODE_H

#include "eqMivt.h"
#include "initData.h"
#include "resourcesManager.h"
#include "frameData.h"

#include <eq/eq.h>

namespace eqMivt
{
	/**
	 * Representation of a node in the cluster
	 * 
	 * Manages node-specific data, namely requesting the mapping of the
	 * initialization data by the local Config instance.
	 */
	class Node : public eq::Node
	{
		public:
			Node( eq::Config* parent ) : eq::Node( parent ) {}

			bool	checkStatus() { return _status; }
			vmml::vector<3, float>    getStartCoord();
			vmml::vector<3, float>    getFinishCoord();
			vmml::vector<3, float>    getVolumeCoord();
			vmml::vector<3, float> getGridStartCoord();
			vmml::vector<3, float> getGridEndCoord();
			vmml::vector<3, float> getGridRealDimVolume();

			bool updateRender(Render * render);

		protected:
			virtual ~Node(){}

			virtual bool configInit( const eq::uint128_t& initID );

			virtual void frameStart( const eq::uint128_t& frameID, const uint32_t frameNumber );

			virtual bool configExit();

		private:

			bool				_status;
			ResourcesManager	_resourcesManager;
			FrameData			_frameData;
	};
}

#endif // EQ_MIVT_NODE_H

