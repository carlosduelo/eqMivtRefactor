/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "eqMivt.h"

#include "error.h"
#include "config.h"
#include "node.h"
#include "pipe.h"
#include "channel.h"
#include "view.h"
#include "window.h"

#include <stdlib.h>

class NodeFactory : public eq::NodeFactory
{
public:
    virtual eq::Config*  createConfig( eq::ServerPtr parent )
        { return new eqMivt::Config( parent ); }
    virtual eq::Pipe*    createPipe( eq::Node* parent )
        { return new eqMivt::Pipe( parent ); }
    virtual eq::Channel* createChannel( eq::Window* parent )
        { return new eqMivt::Channel( parent ); }
    virtual eq::Node*    createNode( eq::Config* parent ) 
        { return new eqMivt::Node( parent ); }
    virtual eq::View* createView( eq::Layout* parent )
        { return new eqMivt::View( parent ); }
    virtual eq::Window*  createWindow( eq::Pipe* parent )
        { return new eqMivt::Window( parent ); }
};

int main( const int argc, char** argv )
{
    eqMivt::initErrors();

    // parse arguments
    eqMivt::LocalInitData initData;
    if (!initData.parseArguments( argc, argv ))
	{
		return EXIT_FAILURE;
	}

    NodeFactory nodeFactory;
    //1. Equalizer initialization
    if( !eq::init( argc, argv, &nodeFactory ))
    {
        LBERROR << "Equalizer init failed" << std::endl;
        return EXIT_FAILURE;
    }

    // 3. initialization of local client node
    lunchbox::RefPtr< eqMivt::EqMivt > client = new eqMivt::EqMivt( initData );
    if( !client->initLocal( argc, argv ))
    {
        LBERROR << "Can't init client" << std::endl;
        eq::exit();
        return EXIT_FAILURE;
    }

    // run client
    const int ret = client->run();

    // cleanup and exit
    client->exitLocal();

    LBASSERTINFO( client->getRefCount() == 1, client );
    client = 0;

    eq::exit();
    eqMivt::exitErrors();
    
    return ret;
}
