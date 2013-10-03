/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <octreeContainer.h>

#include <algorithm>

namespace eqMivt
{

OctreeContainer::OctreeContainer()
{
	_numOctrees = 0;
	_currentPosition = 0;
	_currentIsosurface = 0;
	_octrees.clear();

	_desp = 0;

	_xGrid = 0;
	_yGrid = 0;
	_zGrid = 0;

	_sizes = 0;
	_octree = 0;
}

void OctreeContainer::stop()
{
	if (_xGrid != 0)
		delete[] _xGrid;
	if (_yGrid != 0)
		delete[] _yGrid;
	if (_zGrid != 0)
		delete[] _zGrid;

	if (_desp != 0)
		delete[] _desp;

	if(_octree != 0)
		delete[] _octree;

	if(_sizes!= 0)
		delete[] _sizes;

}

bool OctreeContainer::init(std::string file_name)
{
	try
	{
		_file.open(file_name.c_str(), std::ifstream::binary);
	}
	catch(...)
	{
		std::cerr<<"Octree Container, error opening octree file"<<std::endl;
		return false;
	}
	int magicWord;

	_file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<magicWord<<std::endl;
		return false;
	}

	_file.read((char*)&_numOctrees,sizeof(_numOctrees));
	_file.read((char*)_realDimVolume.array,3*sizeof(int));
	_xGrid = new float[4 + _realDimVolume.x()];
	_yGrid = new float[4 + _realDimVolume.y()];
	_zGrid = new float[4 + _realDimVolume.z()];
	_file.read((char*)(_xGrid+2),_realDimVolume.x()*sizeof(float));
	_file.read((char*)(_yGrid+2),_realDimVolume.y()*sizeof(float));
	_file.read((char*)(_zGrid+2),_realDimVolume.z()*sizeof(float));
	for(int i=1; i>=0 ;i--)
	{
		_xGrid[i] = _xGrid[i+1] - 1.0f;
		_yGrid[i] = _yGrid[i+1] - 1.0f;
		_zGrid[i] = _zGrid[i+1] - 1.0f;
	}
	for(int i=0; i<2; i++)
	{
		_xGrid[2 + _realDimVolume.x() + i] = _xGrid[2 + _realDimVolume.x() + i - 1] + 1.0f;
		_yGrid[2 + _realDimVolume.y() + i] = _yGrid[2 + _realDimVolume.y() + i - 1] + 1.0f;
		_zGrid[2 + _realDimVolume.z() + i] = _zGrid[2 + _realDimVolume.z() + i - 1] + 1.0f;
		
	}

	int curO[_numOctrees];
	int rest = 0;
	int d = 0;
	while(rest < _numOctrees)
	{
		octreePosition_t o;
		int n = 0;
		_file.read((char*)&n,sizeof(int));
		_file.read((char*)&o.start.array,3*sizeof(int));
		_file.read((char*)&o.end.array,3*sizeof(int));
		_file.read((char*)&o.nLevels,sizeof(int));
		_file.read((char*)&o.maxLevel,sizeof(int));

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
		_file.read((char*)&iso, sizeof(float));
		_octrees[curO[i]].isos.push_back(iso);
	}

	_desp = new int[_numOctrees];
	_file.read((char*)_desp, _numOctrees*sizeof(int));

	for(int i=0; i<_numOctrees; i++)
	{
		int mH = 0;
		_file.seekg(_desp[0], std::ios_base::beg);
		for(int d=1; d<=i; d++)
			_file.seekg(_desp[d], std::ios_base::cur);
		_file.read((char*)&mH, sizeof(int));
		_octrees[curO[i]].maxHeight.push_back(mH);
	}

	_setBestCubeLevel();
	_readCurrentOctree();

	return true;

}

void OctreeContainer::_setBestCubeLevel()
{

	for(std::vector<octreePosition_t>::iterator it=_octrees.begin(); it!=_octrees.end(); it++)
	{
		int mH = * std::max_element(it->maxHeight.begin(), it->maxHeight.end());
 		float aux = logf(mH)/logf(2.0);
		float aux2 = aux - floorf(aux);
		int nL = aux2>0.0 ? aux+1 : aux;

		if (it->nLevels - 6 > 0)
			nL = it->nLevels - 6;
		if (nL <= 0)
			nL = 0;
		if (it->maxLevel < nL)
			nL = it->maxLevel;

		it->cubeLevel = nL;

		int oL = it->nLevels - 5;
		if (oL <= 0)
			oL = 0;
		if (it->maxLevel < oL)
			oL = it->maxLevel;
		it->rayCastingLevel = oL; 

		octreePosition_t  o=*it;
	}

}

void OctreeContainer::_readCurrentOctree()
{
	if(_octree != 0)
		delete[] _octree;

	if(_sizes!= 0)
		delete[] _sizes;

	std::cout<<_octrees[_currentPosition];
	std::cout<<"Selected isosurface : "<<_octrees[_currentPosition].isos[_currentIsosurface]<<std::endl;


	_file.seekg(_desp[0], std::ios_base::beg);
	for(int d=1; d<=_octrees[_currentPosition].index[_currentIsosurface]; d++)
		_file.seekg(_desp[d], std::ios_base::cur);

	_file.seekg((_octrees[_currentPosition].maxLevel+2)*sizeof(int), std::ios_base::cur);
	_sizes = new int[(_octrees[_currentPosition].maxLevel+1)];
	_file.read((char*)_sizes, (_octrees[_currentPosition].maxLevel+1)*sizeof(int));
	
	int  s = 0;
	for(int i=0; i<=_octrees[_currentPosition].maxLevel; i++)
		s += _sizes[i];

	_octree = new index_node_t[s];
	s = 0;
	for(int i=0; i<=_octrees[_currentPosition].maxLevel; i++)
	{
		_file.read((char*)_octree + s, _sizes[i]*sizeof(index_node_t));
		s+=_sizes[i];
	}
}

bool OctreeContainer::loadNextPosition()
{
	if (_currentPosition == _octrees.size() - 1)
		return true;
	
	_currentPosition++;
	_currentIsosurface = 0;
	_readCurrentOctree();
	return true;
}

bool OctreeContainer::loadPreviusPosition()
{
	if (_currentPosition == 0)
		return true;

	_currentPosition--;
	_currentIsosurface = 0;
	_readCurrentOctree();
	return true;
}

bool OctreeContainer::loadNextIsosurface()
{
	if (_currentIsosurface == _octrees[_currentPosition].isos.size() - 1)
		return true;

	_currentIsosurface++;
	_readCurrentOctree();
	return true;
}

bool OctreeContainer::loadPreviusIsosurface()
{
	if (_currentIsosurface == 0)
		return true;

	_currentIsosurface--;
	_readCurrentOctree();
	return true;
}

int OctreeContainer::getnLevels()
{
	return _octrees[_currentPosition].nLevels;
}

int OctreeContainer::getmaxLevel()
{
	return _octrees[_currentPosition].maxLevel;
}

int	OctreeContainer::getCubeLevel()
{
	return _octrees[_currentPosition].cubeLevel;
}

int OctreeContainer::getRayCastingLevel()
{
	return _octrees[_currentPosition].rayCastingLevel;
}

int	OctreeContainer::getCurrentOctree()
{
	return _octrees[_currentPosition].index[_currentIsosurface];
}

}