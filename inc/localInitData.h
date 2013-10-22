/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_LOCALINITDATA_H
#define EQ_MIVT_LOCALINITDATA_H

#include "initData.h"

namespace eqMivt
{
    class LocalInitData : public InitData
    {
    public:
        LocalInitData();

        bool parseArguments( const int argc, char** argv );

        uint32_t           getMaxFrames()   const { return _maxFrames; }
		bool               isResident()     const { return _isResident; }

	const LocalInitData& operator = ( const LocalInitData& from );

    private:
        uint32_t    _maxFrames;
		bool        _isResident;
    };
}
#endif // EQ_MIVT_LOCALINITDATA_H

