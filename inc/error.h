/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_ERROR_H
#define EQ_MIVT_ERROR_H

#include <eq/eq.h>

namespace eqMivt
{
	enum Error
	{
		ERROR_EQ_MIVT_FAILED = eq::ERROR_CUSTOM,
	};

	void initErrors();

	void exitErrors();
}
#endif /* EQ_MIVT_ERROR_H */
