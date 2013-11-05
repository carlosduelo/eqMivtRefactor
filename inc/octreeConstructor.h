/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_OCTREE_CONSTRUCTOR 
#define EQ_MIVT_OCTREE_CONSTRUCTOR 

#include <typedef.h>

#include <vmmlib/vector.hpp>

#include <fstream>

namespace eqMivt
{
	class octreeConstructor
	{
		private: 
			std::vector<index_node_t> 	_lastLevel;
			std::vector<index_node_t> *	_octree;
			int		*					_numCubes;
			int							_dim;
			int							_maxHeight;
			int							_maxLevel;
			int							_nLevels;
			float						_iso;
			vmml::vector<3, int>		_start;
			vmml::vector<3, int>		_finish;
			int							_numElements;
			std::string					_nameFile;
			std::ofstream				_tempFile;
			bool						_disk;

			bool _addElement(index_node_t id, int level);

		public:
			octreeConstructor(int nLevels, int maxLevel, float iso, vmml::vector<3, int> start, vmml::vector<3, int> finish, bool disk);

			~octreeConstructor();

			void completeOctree();

			void addVoxel(index_node_t id);
			
			float getIso();

			int getMaxLevel() { return _maxLevel; }

			int getnLevels() { return _nLevels; }

			int getSize();

			void writeToFile(std::ofstream * file);

			void printTree();
	};
}
#endif /*EQ_MIVT_OCTREE_CONSTRUCTOR */
