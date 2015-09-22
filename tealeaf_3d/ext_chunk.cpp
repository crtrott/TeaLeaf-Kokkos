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
using std::cerr;

// Globally shared data structure.
std::vector<TeaLeafChunk<DEVICE>*> Chunks;

// Entry point for extension initialisation.
extern "C"
void ext_init_(
		int* xMax, 
		int* yMax, 
		int* zMax,
		int* rank)
{
	Chunks.push_back(new TeaLeafChunk<DEVICE>(*xMax, *yMax, *zMax, *rank));
}

// Entry point for finalising the process.
extern "C"
void ext_finalise_()
{
	for(TeaLeafChunk<DEVICE>* c : Chunks)
		delete c;
}

// Constructor for a chunk.
template <class Device>
TeaLeafChunk<Device>::TeaLeafChunk(
		int xMax,
		int yMax, 
		int zMax,
		int rank) 
:	rank(rank)
{
	cout << "Problem Size: " << xMax << "x" << yMax << "x" << zMax << endl;

#ifdef CUDA
        HostSpace::execution_space::initialize(1);

        const unsigned int deviceCount = Cuda::detect_device_count();
        if(!deviceCount)
        {
            cerr << "No CUDA-enabled devices discovered." << endl;
            exit(1);
        }

        Cuda::initialize(Kokkos::Cuda::SelectDevice(0));
#else
        Kokkos::initialize();
#endif

    dims.x = xMax+HALO_PAD*2;
    dims.y = yMax+HALO_PAD*2;
    dims.z = zMax+HALO_PAD*2;

    page = dims.x*dims.y;
    innerDomain = xMax*yMax*zMax;
    fullDomain = dims.x*dims.y*dims.z;
    inOneDomain = (dims.x-2)*(dims.y-2)*(dims.z-2);

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
    kz = View<double*,Device>("kz", fullDomain);
    sd = View<double*,Device>("sd", fullDomain);
    volume = View<double*,Device>("volume", fullDomain);
    xArea = View<double*,Device>("xArea", (dims.x+1)*dims.y*dims.z);
    yArea = View<double*,Device>("yArea", dims.x*(dims.y+1)*dims.z);
    zArea = View<double*,Device>("zArea", dims.x*dims.y*(dims.z+1));
    cellX = View<double*,Device>("cellX", dims.x);
    cellY = View<double*,Device>("cellY", dims.y);
    cellZ = View<double*,Device>("cellZ", dims.z);
    vertexX = View<double*,Device>("vertexX", dims.x+1);
    vertexY = View<double*,Device>("vertexY", dims.y+1);
    vertexZ = View<double*,Device>("vertexZ", dims.z+1);

    lrLineLen = yMax*zMax;
    tbLineLen = xMax*zMax;
    fbLineLen = xMax*yMax;
    lrBuffer = View<double*,Device>("lrBuffer", lrLineLen*HALO_PAD);
    tbBuffer = View<double*,Device>("tbBuffer", tbLineLen*HALO_PAD);
    fbBuffer = View<double*,Device>("fbBuffer", fbLineLen*HALO_PAD);
}

// Destructor for a chunk.
    template <class Device>
TeaLeafChunk<Device>::~TeaLeafChunk()
{
    if(rank == 0)
    {
        PRINT_PROFILING_RESULTS;
    }

    Kokkos::finalize();
}

