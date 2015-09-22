#include <numeric>
#include <cstdlib>
#include <cstdio>
#include "ext_chunk.hpp"

// Packs/unpacks the top/bottom buffer
template <class Device>
struct TopBottomPacker
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	TopBottomPacker(TLDims dims, KView buffer, KView field, const int depth, int face, bool pack) 
		: dims(dims), buffer(buffer), field(field), depth(depth), face(face), pack(pack){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const 
    {
		int offset = 0;
		int width = dims.x-2*HALO_PAD;
		int lines = index/width;

		if(face == CHUNK_TOP && pack)
		{
			offset = dims.x*(dims.y-HALO_PAD-depth) + lines*2*HALO_PAD;
			buffer(index) = field(offset+index);
		}
		else if(face == CHUNK_BOTTOM && pack)
		{
			offset = dims.x*HALO_PAD + lines*2*HALO_PAD;
			buffer(index) = field(offset+index);
		}
		else if (face == CHUNK_TOP)
		{
			offset = dims.x*(dims.y-HALO_PAD) + lines*2*HALO_PAD;
			field(offset+index)=buffer(index);
		}
		else
		{
			offset = dims.x*(HALO_PAD-depth) + lines*2*HALO_PAD;
			field(offset+index)=buffer(index);
		}
	}

	TLDims dims;
	KView buffer;
	KView field;
	const int depth;
	int face;
	bool pack;
};

// Packs/unpacks the left/right buffer
template <class Device>
struct LeftRightPacker
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	LeftRightPacker(TLDims dims, KView buffer, KView field, const int depth, int face, bool pack) 
		: dims(dims), buffer(buffer), field(field), depth(depth), face(face), pack(pack){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const 
    {
		int offset = 0;
		int lines = index/depth;

		if(face == CHUNK_LEFT && pack)
		{
			offset = HALO_PAD + lines*(dims.x-depth);
			buffer(index) = field(offset+index);
		}
		else if(face == CHUNK_RIGHT && pack)
		{
			offset = dims.x-HALO_PAD-depth + lines*(dims.x-depth);
			buffer(index) = field(offset+index);
		}
		else if(face == CHUNK_LEFT)
		{
			offset = HALO_PAD-depth + lines*(dims.x-depth);
			field(offset+index)=buffer(index);
		}
		else
		{
			offset = dims.x-HALO_PAD + lines*(dims.x-depth);
			field(offset+index)=buffer(index);
		}
	}

	TLDims dims;
	KView buffer;
	KView field;
	const int depth;
	int face;
	bool pack;
};

