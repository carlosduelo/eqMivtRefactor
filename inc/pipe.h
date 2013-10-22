/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_PIPE_H
#define EQ_MIVT_PIPE_H

#include <eq/eq.h>

#include "frameData.h"
//#include "render.h"

namespace eqMivt
{
    class Pipe : public eq::Pipe
    {
    public:
        Pipe( eq::Node* parent ) : eq::Pipe( parent ) {}

        const FrameData& getFrameData() const { return _frameData; }
		
		//Render * getRender();

    protected:
        virtual ~Pipe() {}

        virtual bool configInit( const eq::uint128_t& initID );
        virtual bool configExit();
        virtual void frameStart( const eq::uint128_t& frameID, 
                                 const uint32_t frameNumber );

    private:
        FrameData	_frameData;

		bool		_lastState;
		//Render		_render;
    };
}

#endif // EQ_MIVT_PIPE_H

