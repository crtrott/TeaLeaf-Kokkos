#ifndef __CHUNK
#define __CHUNK

#include <cmath>
#include <vector>
#include <map>
#include <string>
#include "ext_shared.hpp"

/*
 * 		CHUNK CLASS
 */

// The core Tealeaf interface class.
template <class Device>
struct TeaLeafChunk
{
	TeaLeafChunk(
			const int xMax, 
			const int yMax, 
			const int zMax,
			const int rank);

	~TeaLeafChunk();

	void Plot3d(
			Kokkos::View<double*,Device> buffer, 
			std::string name);

	// Kokkos
	Kokkos::View<double*,Device> density;
	Kokkos::View<double*,Device> energy0;
	Kokkos::View<double*,Device> energy;
	Kokkos::View<double*,Device> kx;
	Kokkos::View<double*,Device> ky;
	Kokkos::View<double*,Device> kz;
	Kokkos::View<double*,Device> mi;
	Kokkos::View<double*,Device> p;
	Kokkos::View<double*,Device> r;
	Kokkos::View<double*,Device> sd;
	Kokkos::View<double*,Device> u;
	Kokkos::View<double*,Device> u0;
	Kokkos::View<double*,Device> w;
	Kokkos::View<double*,Device> z;
	Kokkos::View<double*,Device> vertexX;
	Kokkos::View<double*,Device> vertexY;
	Kokkos::View<double*,Device> vertexZ;
	Kokkos::View<double*,Device> volume;
	Kokkos::View<double*,Device> cellX;
	Kokkos::View<double*,Device> cellY;
	Kokkos::View<double*,Device> cellZ;
	Kokkos::View<double*,Device> xArea;
	Kokkos::View<double*,Device> yArea;
	Kokkos::View<double*,Device> zArea;
	Kokkos::View<double*,Device> alphas;
	Kokkos::View<double*,Device> betas;

	// MPI Comms
	Kokkos::View<double*,Device> lrBuffer;
	Kokkos::View<double*,Device> tbBuffer;
	Kokkos::View<double*,Device> fbBuffer;

	int lrLineLen;
	int tbLineLen;
	int fbLineLen;

	TLDims dims;
	int fullDomain;
	int innerDomain;
	int inOneDomain;
	int rank;
	int page;
	double dx;
	double dy;
	double dz;
	double theta;
	bool preconditioner;
};

// Globally stored list of chunks.
extern std::vector<TeaLeafChunk<DEVICE>*> Chunks;

#endif
