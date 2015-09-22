#ifndef __INITCHUNK
#define __INITCHUNK

#include "ext_chunk.hpp"

/*
 * 		INITIALISE CHUNK KERNEL
 * 		Initialises the chunk's mesh data.
 */

// Initialises the vertices
template <class Device>
struct InitialiseChunkVertices
{
	typedef Device device_type;
	typedef Kokkos::View<double*, Device> KView;

	InitialiseChunkVertices(
			TLDims dims, KView vertexX, KView vertexY, KView cellX, 
			KView cellY, const double xMin, const double yMin,
			const double dx, const double dy) 
		: dims(dims), vertexX(vertexX), vertexY(vertexY),
		cellX(cellX), cellY(cellY), xMin(xMin), yMin(yMin), 
		dx(dx), dy(dy) {} 

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

		if(index < dims.x)
        {
			cellX(index) = 0.5*(vertexX(index)+vertexX(index+1));
        }

		if(index < dims.y)
        {
			cellY(index) = 0.5*(vertexY(index)+vertexY(index+1));
        }
	}

	TLDims dims;
	KView vertexX;
	KView vertexY;
	KView cellX;
	KView cellY;
	const double xMin;
	const double yMin;
	const double dx;
	const double dy;
};

// Initialises the volume and areas
template <class Device>
struct InitialiseChunkAreas
{
	typedef Device device_type;
	typedef Kokkos::View<double*, Device> KView;

	InitialiseChunkAreas(
			KView volume, KView xArea, KView yArea,
			const double xMin, const double yMin,
			const double dx, const double dy) 
		: volume(volume), xArea(xArea), yArea(yArea),
		xMin(xMin), yMin(yMin), dx(dx), dy(dy) {}

	KOKKOS_INLINE_FUNCTION
	void operator() (const int index) const 
    {
		volume(index) = dx*dy;
		xArea(index) = dy;
		yArea(index) = dx;
	}

	KView volume;
	KView xArea;
	KView yArea;
	const double xMin;
	const double yMin;
	const double dx;
	const double dy;
};

#endif
