/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <octreeConstructor.h>

#include <mortonCodeUtil_CPU.h>

#include <algorithm>

namespace eqMivt
{
bool octreeConstructor::_addElement(index_node_t id, int level)
{
	int size = _octree[level].size();

	try
	{
		// Firts
		if (size == 0)
		{
			_numCubes[level] = 1;
			_octree[level].push_back(id);
			_octree[level].push_back(id);
		}
		else if (_octree[level].back() == (id - (index_node_t)1))
		{
			_numCubes[level] += 1;
			_octree[level][size-1] = id;
		}
		else if(_octree[level].back() == id)
		{
			//std::cout<<"repetido in level "<<level<<" "<< id <<std::endl;
			return true;
		}
		else if(_octree[level].back() > id)
		{
			std::cout<<"=======>   ERROR: insert index in order "<< id <<" (in level "<<level<<") last inserted "<<_octree[level].back()<<std::endl;
			throw;
		}
		else
		{
			_numCubes[level] += 1;
			_octree[level].push_back(id);
			_octree[level].push_back(id);
		}
	}
	catch (...)
	{
		std::cerr<<"No enough memory aviable"<<std::endl;
		throw;
	}

	return false;
}


octreeConstructor::octreeConstructor(int nLevels, int maxLevel, float iso, vmml::vector<3, int> start, vmml::vector<3, int> finish)
{

	_octree		= new std::vector<index_node_t>[maxLevel + 1];
	_numCubes	= new int[maxLevel + 1];	
	bzero(_numCubes, (maxLevel + 1)*sizeof(int));
	_numElements = 0;

	_iso		= iso;
	_maxLevel	= maxLevel;
	_nLevels	= nLevels;
	_maxHeight	= 0;
	_dim = exp2(_nLevels - _maxLevel);

	_start = start;
	_finish = finish;

	std::ostringstream convert;
	convert <<rand() <<nLevels << maxLevel << iso << ".tmp";
	_nameFile = convert.str();

	_tempFile.open(_nameFile.c_str(), std::ofstream::binary | std::ofstream::trunc);
}

octreeConstructor::~octreeConstructor()
{
	remove(_nameFile.c_str());
	if (_octree != 0)
		delete[] _octree;
	if (_numCubes != 0)
		delete[] _numCubes;
}

void octreeConstructor::completeOctree()
{
	try
	{
		_tempFile.close();
		std::vector<index_node_t> lastLevel;
		std::ifstream File(_nameFile.c_str(), std::ifstream::binary);

		for(int i=0; i< _numElements; i++)
		{
			index_node_t a = 0;
			File.read((char*) &a, sizeof(index_node_t));
			lastLevel.push_back(a);
		}

		std::sort(lastLevel.begin(), lastLevel.end());
		lastLevel.erase( std::unique( lastLevel.begin(), lastLevel.end() ), lastLevel.end() );
		for (std::vector<index_node_t>::iterator it=lastLevel.begin(); it!=lastLevel.end(); ++it)
		{
			index_node_t id = *it;

			vmml::vector<3, int> coorFinishStart = getMinBoxIndex2(id, _maxLevel, _nLevels) + (vmml::vector<3, int>(1,1,1)*_dim);
			if (coorFinishStart.y() > _maxHeight)
				_maxHeight = coorFinishStart.y();

			for(int i=_maxLevel; i>=0; i--)
			{
				if (_addElement(id, i))
					break;
				id >>= 3;
			}
		}
		lastLevel.clear();

	}
	catch (...)
	{
		std::cerr<<"Not enough memory aviable"<<std::endl;
		throw;
	}
}

void octreeConstructor::addVoxel(index_node_t id)
{
	_numElements++;
	_tempFile.write((char*)&id, sizeof(index_node_t));
}

float octreeConstructor::getIso()
{
	return _iso;
}

int octreeConstructor::getSize()
{
	int size = (2*(_maxLevel+1) + 1)*sizeof(int);

	for(int i=0; i<=_maxLevel; i++)
		size +=_octree[i].size()*sizeof(index_node_t); 

	return size;
}

void octreeConstructor::writeToFile(std::ofstream * file)
{

	file->write((char*)&_maxHeight, sizeof(int));
	file->write((char*)_numCubes, (_maxLevel+1)*sizeof(int));
	for(int i=0; i<=_maxLevel; i++)
	{
		int s = _octree[i].size();
		file->write((char*)&s, sizeof(int));
	}

	for(int i=0; i<=_maxLevel; i++)
	{
		file->write((char*)_octree[i].data(), _octree[i].size()*sizeof(index_node_t));
	}
	remove(_nameFile.c_str());
	delete[] _octree; _octree = 0;
	delete[] _numCubes; _numCubes = 0;

}

void octreeConstructor::printTree()
{
	std::cout<<"Isosurface "<<_iso<<std::endl;
	std::cout<<"Maximum height "<<_maxHeight<<std::endl;
	for(int i=_maxLevel; i>=0; i--)
	{
		std::vector<index_node_t>::iterator it;

		std::cout	<<"Level: "<<i<<" num cubes "<<_numCubes[i]<<" size "<<_octree[i].size()<<" porcentaje "<<100.0f - ((_octree[i].size()*100.0f)/(float)_numCubes[i])
					<<" cube dimension "<<pow(2,_nLevels-i)<<"^3 "<<" memory needed for level "<< (_numCubes[i]*pow(pow(2,_nLevels-i),3)*sizeof(float))/1024.0f/1024.0f<<" MB"<<std::endl;
		#if 0
		for ( it=_octree[i].begin() ; it != _octree[i].end(); it++ )
		{
			std::cout << "From " << *it;
			it++;
			std::cout << " to " << *it<<std::endl;
		}
		#endif
	}
}
}
