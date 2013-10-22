/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONFIG_H
#define EQ_MIVT_CONFIG_H

#include "localInitData.h"
#include "frameData.h"
#include "surfaceInfo.h"

namespace eqMivt
{
    class Config : public eq::Config
    {
    public:
        Config( eq::ServerPtr parent );

        virtual bool init();

        virtual bool exit();

        virtual uint32_t startFrame();

        void setInitData( const LocalInitData& data ) { _initData = data; }
        const InitData& getInitData() const { return _initData; }

        bool loadData( const eq::uint128_t& initDataID );

        virtual bool handleEvent( eq::EventICommand command );
		virtual bool handleEvent( const eq::ConfigEvent* event );

        bool needRedraw();
		bool isIdleAA();

    protected:
        virtual ~Config();

        virtual co::uint128_t sync(const co::uint128_t& version = co::VERSION_HEAD);

    private:
        LocalInitData	_initData;
        FrameData		_frameData;
        bool			_redraw;
		int32_t			_numFramesAA;
		SurfaceInfo		_surfaceInfo;

        bool _handleKeyEvent( const eq::KeyEvent& event );
		void _deregisterData();
		void _reset();
    };
}

#endif /* EQ_MIVT_CONFIG_H */

