/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_FRAMEDATA_H
#define EQ_MIVT_FRAMEDATA_H

#include "eqMivt.h"

namespace eqMivt
{
class FrameData : public co::Serializable
{
    public:
		FrameData();
		virtual ~FrameData() {};

		void reset();

		void spinCamera( const float x, const float y );
		void moveCamera( const float x, const float y, const float z );
		void zoom( const float x);

		const eq::Matrix4f& getViewMatrix() const { return _viewM; }
		const eq::Matrix4f& getInvViewMatrix() const {return _invViewM; }

		void setIdle( const bool idleMode );
		bool isIdle() const { return _idle; }

		void setDrawBox();
		bool isDrawBox() const { return _drawBox; }
		void setRenderCubes();
		bool isRenderCubes() const { return _renderCubes; }

		void setCurrentViewID( const eq::uint128_t& id );
		eq::uint128_t getCurrentViewID() const { return _currentViewID; }

		void setStatistics();
		bool getStatistics() const { return _statistics; }

		void setNone();
		void setNextPosition();
		void setPreviusPosition();
		void setNextIsosurface();
		void setPreviusIsosurface();

		void setRealDim(eq::Vector3f coord);
		void setStartCoord(eq::Vector3f coord);
		void setEndCoord(eq::Vector3f coord);
		eq::Vector3f getRealDim() const { return _realDim; }
		eq::Vector3f getStartCoord() const { return _startCoord; }
		eq::Vector3f getEndCoord()const { return _endCoord;}

		bool checkNextPosition() const { return _nextPosition;}
		bool checkPreviusPosition() const { return _previusPosition;}
		bool checkNextIsosurface() const { return _nextIsosurface;}
		bool checkPreviusIsosurface() const { return _previusIsosurface;}


    protected:
		virtual void serialize( co::DataOStream& os, const uint64_t dirtyBits );
		virtual void deserialize( co::DataIStream& is, const uint64_t dirtyBits );

		enum DirtyBits
		{
			DIRTY_CAMERA = co::Serializable::DIRTY_CUSTOM << 0,
			DIRTY_FLAGS   = co::Serializable::DIRTY_CUSTOM << 1,
			DIRTY_VIEW    = co::Serializable::DIRTY_CUSTOM << 3,
			DIRTY_MODEL	= co::Serializable::DIRTY_CUSTOM << 4,
		};
    private:
		eq::Vector4f	_center;
		float			_radio;
		float			_angle;
		eq::Vector4f	_position;
		eq::Vector4f	_up;
		eq::Matrix4f	_rotation;
		eq::Matrix4f	_viewM;
		eq::Matrix4f	_invViewM;
		bool            _idle;
		bool			_drawBox;
		bool			_renderCubes;
		eq::uint128_t	_currentViewID;

		bool			_statistics;

		bool			_nextPosition;
		bool			_previusPosition;
		bool			_nextIsosurface;
		bool			_previusIsosurface;
		eq::Vector3f	_startCoord;
		eq::Vector3f	_endCoord;
		eq::Vector3f	_realDim;
		void			_createViewMatrix();
};
}
#endif // EQ_MIVT_FRAMEDATA_H

