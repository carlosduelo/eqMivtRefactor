/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/
#include "view.h"
#include "config.h"

namespace eqMivt
{

#pragma warning( push )
#pragma warning( disable : 4355 )

	View::View( eq::Layout* parent )
		: eq::View( parent )
		  , _proxy( this )
		  , _idleSteps( 0 )
	{
		setUserData( &_proxy );
	}

#pragma warning( pop )

	View::~View()
	{
		setUserData( 0 );
		_idleSteps = 0;
	}

	void View::Proxy::serialize( co::DataOStream& os, const uint64_t dirtyBits )
	{
		if( dirtyBits & DIRTY_IDLE )
			os << _view->_idleSteps;
	}

	void View::Proxy::deserialize( co::DataIStream& is, const uint64_t dirtyBits )
	{
		if( dirtyBits & DIRTY_IDLE )
		{
			is >> _view->_idleSteps;
			if( isMaster( ))
				setDirty( DIRTY_IDLE ); // redistribute slave settings
		}
	}

	void View::setIdleSteps( const int32_t steps )
	{
		if( _idleSteps == steps )
			return;

		_idleSteps = steps;
		_proxy.setDirty( Proxy::DIRTY_IDLE );
	}

	void View::toggleEqualizer()
	{
		if( getEqualizers() & eq::fabric::LOAD_EQUALIZER )
			useEqualizer( eq::fabric::EQUALIZER_ALL & ~eq::fabric::LOAD_EQUALIZER );
		else
			useEqualizer( eq::fabric::EQUALIZER_ALL & ~eq::fabric::TILE_EQUALIZER );
	}

}

