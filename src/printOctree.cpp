/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <typedef.h>

#include <iostream>
#include <fstream>
#include <cmath>

int main( const int argc, char ** argv)
{
	if (argc < 2)
	{
		std::cerr<<"Please provaid octree file"<<std::endl;
		return 0;
	}

	/* Read octree from file */
	std::ifstream file;

	try
	{
		file.open(argv[1], std::ifstream::binary);
	}
	catch(...)
	{
		std::cerr<<"Octree: error opening octree file"<<std::endl;
		return 0;
	}

	int magicWord;

	file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format "<<magicWord<<std::endl;
		return false;
	}
	
	int realDimensionV[3] = {0, 0, 0};
	int numOctrees = 0;

	file.read((char*)&numOctrees,sizeof(numOctrees));
	file.read((char*)&realDimensionV[0],sizeof(realDimensionV[0]));
	file.read((char*)&realDimensionV[1],sizeof(realDimensionV[1]));
	file.read((char*)&realDimensionV[2],sizeof(realDimensionV[2]));

	std::cout<<"Num octrees: "<<numOctrees<<std::endl;
	std::cout<<"Real dimension: "<<realDimensionV[0]<<"x"<<realDimensionV[1]<<"x"<<realDimensionV[2]<<std::endl;

	float * xgrid = new float[realDimensionV[0]];
	float * ygrid = new float[realDimensionV[1]];
	float * zgrid = new float[realDimensionV[2]];

	file.read((char*)xgrid, realDimensionV[0]*sizeof(float));
	file.read((char*)ygrid, realDimensionV[1]*sizeof(float));
	file.read((char*)zgrid, realDimensionV[2]*sizeof(float));

	int nLevels[numOctrees];
	int maxLevel[numOctrees]; 
	int dimension[numOctrees]; 
	int realDim[3*numOctrees];
	int startC[3*numOctrees];
	int finishC[3*numOctrees]; 
	int rest = 0;
	while(rest < numOctrees)
	{
		int n = 0;
		int nL = 0;
		int mL = 0;
		int s[3];
		int f[3];
		int d[3];
		file.read((char*)&n,sizeof(int));
		file.read((char*)s,3*sizeof(int));
		file.read((char*)f,3*sizeof(int));
		file.read((char*)&nL,sizeof(int));
		file.read((char*)&mL,sizeof(int));
		d[0] = f[0] - s[0];
		d[1] = f[1] - s[1];
		d[2] = f[2] - s[2];
		for(int j=0; j<n; j++)
		{
			nLevels[rest+j] = nL;
			maxLevel[rest+j] = mL;
			dimension[rest+j] = exp2(nL);
			startC[3*(rest+j)] = s[0];
			startC[3*(rest+j)+1] = s[1];
			startC[3*(rest+j)+2] = s[2];
			finishC[3*(rest+j)] = f[0];
			finishC[3*(rest+j)+1] = f[1];
			finishC[3*(rest+j)+2] = f[2];
			realDim[3*(rest+j)] = d[0];
			realDim[3*(rest+j)+1] = d[1];
			realDim[3*(rest+j)+2] = d[2];
		}
		rest += n;
	}

	float isos[numOctrees];
	int	desp[numOctrees];
	file.read((char*)isos, numOctrees*sizeof(float));
	file.read((char*)desp, numOctrees*sizeof(int));

	bool cont = true;
	while(cont) 
	{
		int selection = -1;
		std::cout<<"Select octree to print [0,"<<numOctrees-1<<"]  to exit -1: ";
		std::cin>>selection;
		if (selection == -1)
			cont = false;
		else if (selection < 0 || selection >= numOctrees)
			std::cout<<"Octree not aviable, repeat"<<std::endl;
		else
		{
			int mH = 0;
			int numCubes[maxLevel[selection]+1];
			int sizes[maxLevel[selection]+1];
			file.seekg(desp[0], std::ios_base::beg);
			for(int d=1; d<=selection; d++)
				file.seekg(desp[d], std::ios_base::cur);
			file.read((char*)&mH, sizeof(int));
			file.read((char*)numCubes, (maxLevel[selection]+1)*sizeof(int));
			file.read((char*)sizes, (maxLevel[selection]+1)*sizeof(int));
			std::cout<<std::endl;
			std::cout<<"Octree "<<selection<<std::endl;
			std::cout<<"nLevels "<<nLevels[selection]<<std::endl;
			std::cout<<"maxLevel "<<maxLevel[selection]<<std::endl;
			std::cout<<"Isosurface "<<isos[selection]<<std::endl;
			std::cout<<"Start coordinates "<<startC[3*selection]<<" "<<startC[3*selection+1]<<" "<<startC[3*selection+2]<<std::endl;
			std::cout<<"Finish coordinates "<<finishC[3*selection]<<" "<<finishC[3*selection+1]<<" "<<finishC[3*selection+2]<<std::endl;
			std::cout<<"Real dimension "<<realDim[3*selection]<<" "<<realDim[3*selection+1]<<" "<<realDim[3*selection+2]<<std::endl;

			std::cout<<"Levels dimension and concentration:"<<std::endl;
			float s = 0;
			for(int k=0; k<=maxLevel[selection]; k++)
			{
				s += sizes[k]*sizeof(eqMivt::index_node_t)/1024.f/1024.f;
				std::cout<<"Level "<<k<<" dimension "<<sizes[k]<<" num cubes "<<numCubes[k]<<" concentration "<<(numCubes[k]*100.0f)/(float)sizes[k]<<" %"<<" size: "<<s<<" MB"<<std::endl;
			}

		}
	}

	delete[] xgrid;
	delete[] ygrid;
	delete[] zgrid;

	file.close();
	return 0;

}
