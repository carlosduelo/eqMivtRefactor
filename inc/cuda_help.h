/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef _EQ_MIVT_CUDA_HELP_H_
#define _EQ_MIVT_CUDA_HELP_H_

#define BLOCK_SIZE 128

#define EPS 0.0001f

namespace eqMivt
{

inline dim3 getBlocks(int dim)
{
	if (dim <= BLOCK_SIZE)
	{
		dim3 blocks(1);//,0,0);
		return blocks;
	}
	else// if (dim<=(BLOCK_SIZE*BLOCK_SIZE))
	{
		int numBlocks = dim / BLOCK_SIZE;
		if (dim % BLOCK_SIZE !=0) numBlocks++;
		int bpA = sqrt(numBlocks);
		int bp  = floorf(bpA) + 1;
		dim3 blocks(bp,bp);//,0); 
		return blocks;
	}
}

inline dim3 getThreads(int dim)
{
	int t = 32;
	while(dim>t && t<BLOCK_SIZE)
	{
		t+=32;
	}

	dim3 threads(t);//,0,0);
	return threads;
}
}
#endif
