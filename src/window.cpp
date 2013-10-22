/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "window.h"

#include "config.h"
#include "pipe.h"

//#include <GL/glew.h>

#include <fstream>
#include <sstream>

namespace eqMivt
{

	bool Window::configInitSystemWindow( const eq::uint128_t& initID )
	{
#ifndef Darwin
		if( !eq::Window::configInitSystemWindow( initID ))
			return false;

		// OpenGL version is less than 2.0.
		if( !GLEW_EXT_framebuffer_object )
		{
			if( getDrawableConfig().accumBits )
				return true;

			configExitSystemWindow();
#endif
			// try with 64 bit accum buffer
			setIAttribute( IATTR_PLANES_ACCUM, 16 );
			if( eq::Window::configInitSystemWindow( initID ))
				return true;

			// no anti-aliasing possible
			setIAttribute( IATTR_PLANES_ACCUM, eq::AUTO );

			return eq::Window::configInitSystemWindow( initID );

#ifndef Darwin
		}

		return true;
#endif
	}

	bool Window::configInitGL( const eq::uint128_t& initID )
	{
		if( !eq::Window::configInitGL( initID ))
			return false;

		glLightModeli( GL_LIGHT_MODEL_LOCAL_VIEWER, 1 );
		glEnable( GL_CULL_FACE ); // OPT - produces sparser images in DB mode
		glCullFace( GL_BACK );

		return true;
	}

	bool Window::configExitGL()
	{
		return eq::Window::configExitGL();
	}



	void Window::frameStart( const eq::uint128_t& frameID, const uint32_t frameNumber )
	{
		//const Pipe*      pipe      = static_cast<Pipe*>( getPipe( ));
		//const FrameData& frameData = pipe->getFrameData();

		eq::Window::frameStart( frameID, frameNumber );
	}

}

