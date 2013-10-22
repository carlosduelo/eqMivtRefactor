/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_INITDATA_H
#define EQ_MIVT_INITDATA_H

#include "eqMivt.h"

#include <string>
#include <vector>

namespace eqMivt
{
    class InitData : public co::Object
    {
    public:
        InitData();
        virtual ~InitData();

        void setFrameDataID( const eq::uint128_t& id ) { _frameDataID = id; }
        eq::uint128_t getFrameDataID() const  { return _frameDataID; }

		std::string	getOctreeFilename() { return _octreeFilename; }
		std::string	getOctreeFilename() const { return _octreeFilename; }
		void		setOctreeFilename(std::string octFilename) {  _octreeFilename = octFilename; }

		std::vector<std::string>	getDataFilename() { return _dataFilename; }
		std::vector<std::string>	getDataFilename() const { return _dataFilename; }
		void	setDataFilename(std::vector<std::string> dataFilename) { _dataFilename = dataFilename; }

		float		getMemoryOccupancy() { return _memoryOccupancy; }
		float		getMemoryOccupancy() const { return _memoryOccupancy; }
		void		setMemoryOccupancy(float memoryOccupancy){ _memoryOccupancy = memoryOccupancy; }

		std::string	getTransferFunctionFile() { return _transferFunctionFile; }
		std::string	getTransferFunctionFile() const { return _transferFunctionFile; }
		void		setTransferFunctionFile(std::string transferFunctionFile) { _transferFunctionFile = transferFunctionFile; }
    protected:
        virtual void getInstanceData( co::DataOStream& os );
        virtual void applyInstanceData( co::DataIStream& is );

    private:
        eq::uint128_t	_frameDataID;

		std::string		_octreeFilename;

		std::vector<std::string>		_dataFilename;

		std::string						_transferFunctionFile;
		
		float							_memoryOccupancy;
    };
}


#endif /* EQ_MIVT_INITDATA_H */

