/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <octree.h>
#include <cache.h>
#include <queue>
#include <lunchbox/condition.h>
#include <lunchbox/thread.h>

#include <cuda_help.h>

#ifndef EQ_MIVT_RENDER_WORKERS_H
#define EQ_MIVT_RENDER_WORKERS_H

#define STOP 0
#define START_FRAME 1
#define FINISH_FRAME 2
#define FRAME 3

namespace eqMivt
{
struct workpackage_t{ int work[3]; };

struct queue_t
{
	std::queue<workpackage_t>	queue;
	lunchbox::Condition			cond;
};
struct queuePOP_t
{
	std::queue<index_node_t>	queue;
	lunchbox::Condition			cond;
};

struct sharedParameters_t
{
	int size;
	visibleCube_t * visibleCubes;
	visibleCubeGPU_t visibleCubesGPU;
	int pvpW;
	int pvpH;
	float w;
	float h;
	float3 origin;
	float3 LB;
	float3 up;
	float3 right;
	float * pixelBuffer;
};

class worker : public lunchbox::Thread
{
	protected:
		Octree * _octree;

		sharedParameters_t *	_parameters;
		device_t				_device;
		cudaStream_t			_stream;
		queue_t * _IN;
		queue_t * _OUT;

		bool _init(Octree * octree, device_t device, queue_t * IN, queue_t * OUT, sharedParameters_t *	parameters);
		
		bool _updateCPU(int offset, int size);
		bool _sync();
		bool _updateGPU(int offset, int size);

		virtual bool _frame(workpackage_t work) = 0;
		virtual bool _startFrame() = 0;
		virtual bool _finishFrame() = 0;
		virtual bool _stopWorking() = 0;

	public:
		bool destroy();

		virtual void run();
};

class workerOctree : public worker
{
	private:
		virtual bool _frame(workpackage_t work);
		virtual bool _startFrame();
		virtual bool _finishFrame();
		virtual bool _stopWorking();
	
	public:
		bool init(Octree * octree, device_t device, queue_t * IN, queue_t * OUT, sharedParameters_t *	parameters);
};

class workerCache : public worker
{
	private:
		Cache			_cache;

		virtual bool _frame(workpackage_t work);
		virtual bool _startFrame();
		virtual bool _finishFrame();
		virtual bool _stopWorking();

	public:
		bool init(Octree * octree, ControlCubeCache * ccc, device_t device, queue_t * IN, queue_t * OUT, sharedParameters_t *	parameters);
		Cache * getCache(){ return &_cache; }
};

class workerRayCaster : public worker
{
	private:
		queuePOP_t	*	_POP;
		queue_t	*	_END;
		color_t *	_colors;

		virtual bool _frame(workpackage_t work);
		virtual bool _startFrame();
		virtual bool _finishFrame();
		virtual bool _stopWorking();
	public:
		bool init(Octree * octree, color_t * colors, device_t device, queue_t * IN, queue_t * OUT, sharedParameters_t *	parameters, queue_t * END, queuePOP_t * POP);
};

class workerPoper : public lunchbox::Thread
{
	private:
		queuePOP_t	*	_POP;
		Cache * _cache;

		virtual void run();
	public:
	bool init(queuePOP_t * POP, Cache * cache);

};

}
#endif /* EQ_MIVT_RENDER_WORKERS_H */
