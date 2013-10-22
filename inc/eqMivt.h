/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_H
#define EQ_MIVT_H

#define VERSION_EQ_MIVT 0.3

#include <eq/eq.h>

namespace eqMivt
{
    class LocalInitData;

    class EqMivt : public eq::Client
    {
    public:
        EqMivt( const LocalInitData& initData );
        virtual ~EqMivt() {}

        int run();

        static const std::string& getHelp();

    protected:
        virtual void clientLoop();

    private:
	const LocalInitData& _initData;
    };

    enum LogTopics
    {
	    LOG_STATS = eq::LOG_CUSTOM << 0, // 65536
	    LOG_CULL  = eq::LOG_CUSTOM << 1  // 131072
    };
}
#endif /* EQ_MIVT_H */

