/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_TEXTURE_FUNCTIONS_H
#define EQ_MIVT_TEXTURE_FUNCTIONS_H

namespace eqMivt
{
	bool initTextures();

	bool bindTextures(float * xGrid, float * yGrid, float * zGrid, int3 realDim);

	bool unBindTextures();
}

#endif /* EQ_MIVT_TEXTURE_FUNCTIONS_H */
