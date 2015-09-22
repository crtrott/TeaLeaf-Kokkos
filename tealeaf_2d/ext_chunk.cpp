#include "ext_chunk.hpp"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

/* 
 * 		CHUNK CLASS
 */

using namespace Kokkos;
using std::cout;
using std::endl;

// Globally shared data structure.
std::vector<TeaLeafChunk<DEVICE>*> Chunks;

// Entry point for extension initialisation.
extern "C"
void ext_init_(
		int* xMax, 
		int* yMax, 
		int* rank)
{
	Chunks.push_back(new TeaLeafChunk<DEVICE>(*xMax, *yMax, *rank));
}

// Entry point for finalising the process.
extern "C"
void ext_finalise_()
{
	for(TeaLeafChunk<DEVICE>* c : Chunks)
		delete c;
}

template <class Device>
TeaLeafChunk<Device>::TeaLeafChunk(
		int xMax,
		int yMax, 
		int rank) 
:	rank(rank)
{
	if(rank == 0)
	{
		cout << "Problem Size: " << xMax << "x" << yMax << endl;
	}

    // Initialise Kokkos
	Kokkos::initialize();

	dims.x = xMax+HALO_PAD*2;
	dims.y = yMax+HALO_PAD*2;
	innerDomain = xMax*yMax;
	fullDomain = dims.x*dims.y;
	inOneDomain = (dims.x-2)*(dims.y-2);

	// Creates Kokkos views
	density = View<double*,Device>("density", fullDomain);
	energy0 = View<double*,Device>("energy0", fullDomain);
	energy = View<double*,Device>("energy", fullDomain);
	u = View<double*,Device>("u", fullDomain);
	u0 = View<double*,Device>("u0", fullDomain);
	p = View<double*,Device>("p", fullDomain);
	r = View<double*,Device>("r", fullDomain);
	mi = View<double*,Device>("mi", fullDomain);
	w = View<double*,Device>("w",fullDomain);
	z = View<double*,Device>("z", fullDomain);
	kx = View<double*,Device>("kx", fullDomain);
	ky = View<double*,Device>("ky", fullDomain);
	sd = View<double*,Device>("sd", fullDomain);
	volume = View<double*,Device>("volume", fullDomain);
	xArea = View<double*,Device>("xArea", (dims.x+1)*dims.y);
	yArea = View<double*,Device>("yArea", dims.x*(dims.y+1));
	cellX = View<double*,Device>("cellX", dims.x);
	cellY = View<double*,Device>("cellY", dims.y);
	vertexX = View<double*,Device>("vertexX", dims.x+1);
	vertexY = View<double*,Device>("vertexY", dims.y+1);

	lrLineLen = yMax;
	tbLineLen = xMax;
	lrBuffer = View<double*,Device>("lrBuffer", lrLineLen*HALO_PAD);
	tbBuffer = View<double*,Device>("tbBuffer", tbLineLen*HALO_PAD);
}

template <class Device>
TeaLeafChunk<Device>::~TeaLeafChunk()
{
#ifdef ENABLE_PROFILING
	if(rank == 0)
	{
		PRINT_PROFILING_RESULTS;

		std::cout << "Profiling may affect the wallclock and can be disabled"
			<< " by removing '-DENABLE_PROFILING' from the Makefile." << std::endl;
	}
#endif

	// Finalise the Kokkos runtime
	Kokkos::finalize();
}

