/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <surfaceInfo.h>

namespace eqMivt
{

bool SurfaceInfo::init(std::string file_name)
{
	std::ifstream file;
	try
	{
		file.open(file_name.c_str(), std::ifstream::binary);
	}
	catch(...)
	{
		std::cerr<<"SurfaceInfo, error opening octree file"<<std::endl;
		return false;
	}

	int magicWord;

	file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<magicWord<<std::endl;
		return false;
	}

	file.read((char*)&_numOctrees,sizeof(_numOctrees));
	file.read((char*)&_realDimVolume.array[0],3*sizeof(int));
	_xGrid = new float[2*CUBE_INC + _realDimVolume.x()];
	_yGrid = new float[2*CUBE_INC + _realDimVolume.y()];
	_zGrid = new float[2*CUBE_INC + _realDimVolume.z()];
	file.read((char*)(_xGrid+CUBE_INC),_realDimVolume.x()*sizeof(float));
	file.read((char*)(_yGrid+CUBE_INC),_realDimVolume.y()*sizeof(float));
	file.read((char*)(_zGrid+CUBE_INC),_realDimVolume.z()*sizeof(float));
	for(int i=CUBE_INC-1; i>=0 ;i--)
	{
		_xGrid[i] = _xGrid[i+1] - 1.0f;
		_yGrid[i] = _yGrid[i+1] - 1.0f;
		_zGrid[i] = _zGrid[i+1] - 1.0f;
	}
	for(int i=0; i<CUBE_INC; i++)
	{
		_xGrid[CUBE_INC + _realDimVolume.x() + i] = _xGrid[CUBE_INC + _realDimVolume.x() + i - 1] + 1.0f;
		_yGrid[CUBE_INC + _realDimVolume.y() + i] = _yGrid[CUBE_INC + _realDimVolume.y() + i - 1] + 1.0f;
		_zGrid[CUBE_INC + _realDimVolume.z() + i] = _zGrid[CUBE_INC + _realDimVolume.z() + i - 1] + 1.0f;
		
	}
	_currentPosition = 0;
	_currentIsosurface = 0;

	int curO[_numOctrees];
	int rest = 0;
	int d = 0;
	while(rest < _numOctrees)
	{
		octreePosition_t o;
		int n = 0;
		file.read((char*)&n,sizeof(int));
		file.read((char*)&o.start.array,3*sizeof(int));
		file.read((char*)&o.end.array,3*sizeof(int));
		file.read((char*)&o.nLevels,sizeof(int));
		file.read((char*)&o.maxLevel,sizeof(int));

		for(int j=0; j<n; j++)
		{
			curO[rest + j] = d;
			o.index.push_back(rest + j);
		}

		_octrees.push_back(o);
		rest += n;
		d++;
	}

	for(int i=0; i<_numOctrees; i++)
	{
		float iso = 0.0;
		file.read((char*)&iso, sizeof(float));
		_octrees[curO[i]].isos.push_back(iso);
	}

	int desp[_numOctrees];
	file.read((char*)desp, _numOctrees*sizeof(int));

	for(int i=0; i<_numOctrees; i++)
	{
		int mH = 0;
		file.seekg(desp[0], std::ios_base::beg);
		for(int d=1; d<=i; d++)
			file.seekg(desp[d], std::ios_base::cur);
		file.read((char*)&mH, sizeof(int));
		_octrees[curO[i]].maxHeight.push_back(mH);
	}

	file.close();

	return  true;
}

vmml::vector<3, float> SurfaceInfo::getRealDimVolume()
{
	return vmml::vector<3, float>(
				_xGrid[CUBE_INC + _realDimVolume.x()],
				_yGrid[CUBE_INC + _realDimVolume.y()],
				_zGrid[CUBE_INC + _realDimVolume.z()]);
}

vmml::vector<3, float> SurfaceInfo::getStartCoord()
{
	return vmml::vector<3, float>(
				_xGrid[CUBE_INC + _octrees[_currentPosition].start.x()],
				_yGrid[CUBE_INC + _octrees[_currentPosition].start.y()],
				_zGrid[CUBE_INC + _octrees[_currentPosition].start.z()]);
}

vmml::vector<3, float> SurfaceInfo::getEndCoord()
{
	return vmml::vector<3, float>(
				_xGrid[CUBE_INC + _octrees[_currentPosition].end.x()],
				_yGrid[CUBE_INC + _octrees[_currentPosition].end.y()],
				_zGrid[CUBE_INC + _octrees[_currentPosition].end.z()]);
}

int SurfaceInfo::getnLevels()
{
	return _octrees[_currentPosition].nLevels;
}

int SurfaceInfo::getmaxLevel()
{
	return _octrees[_currentPosition].maxLevel;
}

int	SurfaceInfo::getCubeLevel()
{
	return _octrees[_currentPosition].cubeLevel;
}

int SurfaceInfo::getRayCastingLevel()
{
	return _octrees[_currentPosition].rayCastingLevel;
}

int	SurfaceInfo::getMaxHeight()
{	
	return _octrees[_currentPosition].maxHeight[_currentIsosurface];
}

float SurfaceInfo::getIsosurface()
{	
	return _octrees[_currentPosition].isos[_currentIsosurface];
}
bool SurfaceInfo::checkLoadNextPosition()
{
	return _currentPosition != (int)_octrees.size() - 1 && (_currentPosition += 1) >= 0; 
}

bool SurfaceInfo::checkLoadPreviusPosition()
{
	return _currentPosition != 0 && (_currentPosition -= 1)>= 0;
}

bool SurfaceInfo::checkLoadNextIsosurface()
{
	return _currentIsosurface != (int)_octrees[_currentPosition].isos.size() - 1 && (_currentIsosurface +=1 ) >=0; 
}

bool SurfaceInfo::checkLoadPreviusIsosurface()
{
	return _currentIsosurface != 0 && (_currentIsosurface -= 1) >= 0; 
}

}
