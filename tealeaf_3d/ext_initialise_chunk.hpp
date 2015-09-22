#ifndef __INITCHUNK
#define __INITCHUNK

#include "ext_chunk.hpp"

/*
 * 		INITIALISE CHUNK KERNEL
 * 		Initialises the chunk's mesh data.
 */

// Initialises the vertices for a chunk.
template <class Device>
struct InitialiseChunkVertices
{
	typedef Device device_type;
	typedef Kokkos::View<double*, Device> KView;

	InitialiseChunkVertices(
			TLDims dims, KView vertexX, KView vertexY,
			KView vertexZ, KView cellX, KView cellY, KView cellZ, 
			const double xMin, const double yMin, const double zMin,
			const double dx, const double dy, const double dz) 
		: dims(dims), vertexX(vertexX), vertexY(vertexY), vertexZ(vertexZ),
		cellX(cellX), cellY(cellY), cellZ(cellZ),xMin(xMin), yMin(yMin), 
		zMin(zMin), dx(dx), dy(dy), dz(dz) {} 

	KOKKOS_INLINE_FUNCTION
	void operator() (const int index) const
    {
		if(index < dims.x+1)
        {
			vertexX(index)= xMin+dx*(index-HALO_PAD);
        }

		if(index < dims.y+1)
        {
			vertexY(index) = yMin+dy*(index-HALO_PAD);
        }

		if(index < dims.z+1)
        {
			vertexZ(index) = zMin+dz*(index-HALO_PAD);
        }

		if(index < dims.x)
        {
			cellX(index) = 0.5*(vertexX(index)+vertexX(index+1));
        }

		if(index < dims.y)
        {
			cellY(index) = 0.5*(vertexY(index)+vertexY(index+1));
        }

		if(index < dims.z)
        {
			cellZ(index) = 0.5*(vertexZ(index)+vertexZ(index+1));
        }
	}

	TLDims dims;
	KView vertexX;
	KView vertexY;
	KView vertexZ;
	KView cellX;
	KView cellY;
	KView cellZ;
	const double xMin;
	const double yMin;
	const double zMin;
	const double dx;
	const double dy;
	const double dz;
};

// Initialises the areas of a chunk.
template <class Device>
struct InitialiseChunkAreas
{
	typedef Device device_type;
	typedef Kokkos::View<double*, Device> KView;

	InitialiseChunkAreas(
			KView volume, KView xArea, KView yArea,
			KView zArea, const double xMin, const double yMin, const double zMin,
			const double dx, const double dy, const double dz) 
		: volume(volume), xArea(xArea), yArea(yArea), zArea(zArea),
		xMin(xMin), yMin(yMin), zMin(zMin), dx(dx), dy(dy), dz(dz) {}

	KOKKOS_INLINE_FUNCTION
	void operator() (const int index) const
    {
		volume(index) = dx*dy*dz;
		xArea(index) = dy*dz;
		yArea(index) = dx*dz;
		zArea(index) = dx*dy;
	}

	KView volume;
	KView xArea;
	KView yArea;
	KView zArea;
	const double xMin;
	const double yMin;
	const double zMin;
	const double dx;
	const double dy;
	const double dz;
};

#endif
