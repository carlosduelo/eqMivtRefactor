/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "initData.h"

namespace eqMivt
{

InitData::InitData()
        : _frameDataID()
{}

InitData::~InitData()
{
    setFrameDataID( eq::uint128_t( ) );
}

void InitData::getInstanceData( co::DataOStream& os )
{
    os << _frameDataID << _octreeFilename << _dataFilename << _memoryOccupancy << _transferFunctionFile;
}

void InitData::applyInstanceData( co::DataIStream& is )
{
    is >> _frameDataID >> _octreeFilename >> _dataFilename >> _memoryOccupancy >> _transferFunctionFile;
 
    LBASSERT( _frameDataID != 0 );
}

}

