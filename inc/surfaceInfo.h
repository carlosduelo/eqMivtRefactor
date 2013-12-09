/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_SURFACE_INFO_H
#define EQ_MIVT_SURFACE_INFO_H

#include <typedef.h>

#include <fstream>

namespace eqMivt
{

class SurfaceInfo
{
	private:
		int		_numOctrees;
		int		_currentPosition;
		int		_currentIsosurface;

		float		*		_xGrid;
		float		*		_yGrid;
		float		*		_zGrid;

		std::vector<octreePosition_t> _octrees;

		vmml::vector<3, int> _realDimVolume;
	public:
		bool init(std::string file_name);

		bool checkLoadNextPosition();
		bool checkLoadPreviusPosition();

		bool checkLoadNextIsosurface();
		bool checkLoadPreviusIsosurface();

		int getNumOctrees(){ return _numOctrees; }
		vmml::vector<3, float> getRealDimVolume();
		vmml::vector<3, float> getStartCoord();
		vmml::vector<3, float> getEndCoord();
		int getnLevels();
		int getmaxLevel();
		int	getCubeLevel();
		int getRayCastingLevel();
		int	getMaxHeight();
		float getIsosurface();


};

}
#endif /* EQ_MIVT_SURFACE_INFO_H */
