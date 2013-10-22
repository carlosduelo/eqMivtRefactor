/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_VIEW_H
#define EQ_MIVT_VIEW_H

#include <eq/eq.h>

#include <string>

namespace eqMivt
{
	class View : public eq::View
	{
		public:
			View(eq::Layout* parent);
			virtual ~View();
			void setIdleSteps(const int32_t steps);
			int32_t getIdleSteps() const { return _idleSteps; }
			void toggleEqualizer();
		private:
			class Proxy : public co::Serializable
		{
			public:
				Proxy( View* view ) : _view( view ) {}
			protected:
				/** The changed parts of the view. */
				enum DirtyBits
				{
					DIRTY_IDLE  = co::Serializable::DIRTY_CUSTOM << 0
				};
				virtual void serialize( co::DataOStream&, const uint64_t );
				virtual void deserialize( co::DataIStream&, const uint64_t );
				virtual void notifyNewVersion() { sync(); }
			private:
				View* const _view;
				friend class eqMivt::View;
		};
			Proxy _proxy;
			friend class Proxy;
			int32_t _idleSteps;
	};
}

#endif /* EQ_MIVT_VIEW_H */
