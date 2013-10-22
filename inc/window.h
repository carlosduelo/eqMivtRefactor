/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_WINDOW_H
#define EQ_MIVT_WINDOW_H

#include <eq/eq.h>

namespace eqMivt
{
	class VertexBufferState;

	/**
	 * A window represent an OpenGL drawable and context
	 */
	class Window : public eq::Window
	{
		public:
			Window( eq::Pipe* parent ) 
				: eq::Window( parent ) {}

		protected:
			virtual ~Window() {}
			virtual bool configInitSystemWindow( const eq::uint128_t& initID );
			virtual bool configInitGL( const eq::uint128_t& initID );
			virtual bool configExitGL();
			virtual void frameStart( const eq::uint128_t& frameID,
					const uint32_t frameNumber );
	};
}

#endif /* EQ_MIVT_WINDOW_H */

