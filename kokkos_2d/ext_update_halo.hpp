#ifndef __UPDATEHALO
#define __UPDATEHALO

#include "ext_chunk.hpp"

/*
 * 		UPDATE HALO KERNEL
 */	

// Updates an individual buffer halo.
template <class Device>
struct UpdateHalo
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	UpdateHalo(TLDims dims, KView buffer, int face, int depth) 
		: dims(dims), buffer(buffer), face(face), depth(depth) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const 
    {
		switch(face)
		{
			case CHUNK_LEFT:
				{
					int flip = index % depth;
					int lines = index/depth;
					int offset = HALO_PAD + lines*(dims.x-depth);

					int fromIndex = offset+index;
					int toIndex = fromIndex-(1+flip*2);
					buffer(toIndex) = buffer(fromIndex);
					break;
				}
			case CHUNK_RIGHT:
				{
					int flip = index % depth;
					int lines = index/depth;
					int offset = dims.x-HALO_PAD + lines*(dims.x-depth);

					int toIndex = offset+index;
					int fromIndex = toIndex-(1+flip*2);
					buffer(toIndex) = buffer(fromIndex);
					break;
				}
			case CHUNK_TOP:
				{
					int lines = index/dims.x;
					int offset = dims.x*(dims.y-HALO_PAD);

					int toIndex = offset+index;
					int fromIndex = toIndex-(1+lines*2)*dims.x;
					buffer(toIndex) = buffer(fromIndex);
					break;
				}
			case CHUNK_BOTTOM:
				{
					int lines = index/dims.x;
					int offset = dims.x*HALO_PAD;

					int fromIndex = offset+index;
					int toIndex = fromIndex-(1+lines*2)*dims.x;
					buffer(toIndex) = buffer(fromIndex);
					break;
				}
			default:
				break;
		}
	}

	int face;
	int depth;
	TLDims dims;
	KView buffer;
};

#endif

