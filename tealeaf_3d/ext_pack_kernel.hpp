#include <numeric>
#include <cstdlib>
#include <cstdio>
#include "ext_chunk.hpp"

using std::ceil;
using std::accumulate;

// Packs or unpacks the top and bottom buffers.
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
		int width = dims.x-HP2;
		int lines = index/width;

		if(face == CHUNK_TOP && pack)
		{
			offset = dims.x*dims.y*HALO_PAD
				+ dims.x*(dims.y-HALO_PAD-depth)
				+ HALO_PAD
				+ lines*HP2
				+ dims.x*(dims.y-depth)*(lines/depth);
			buffer(index) = field(offset+index);
		}
		else if(face == CHUNK_BOTTOM && pack)
		{
			offset = dims.x*dims.y*HALO_PAD
				+ dims.x*HALO_PAD
				+ HALO_PAD
				+ lines*HP2
				+ dims.x*(dims.y-depth)*(lines/depth);
			buffer(index) = field(offset+index);
		}
		else if (face == CHUNK_TOP)
		{
			offset = dims.x*dims.y*HALO_PAD
				+ dims.x*(dims.y-HALO_PAD)
				+ HALO_PAD
				+ lines*HP2
				+ dims.x*(dims.y-depth)*(lines/depth);
			field(offset+index)=buffer(index);
		}
		else
		{
			offset = dims.x*dims.y*HALO_PAD
				+ dims.x*(HALO_PAD-depth)
				+ HALO_PAD
				+ lines*HP2
				+ dims.x*(dims.y-depth)*(lines/depth);
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

// Packs or unpacks the left and right halos.
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
		int height = dims.y-HP2;
		int lines = index/depth;

		if(face == CHUNK_LEFT && pack)
		{
			offset = dims.x*dims.y*HALO_PAD
				+ dims.x*HALO_PAD
				+ HALO_PAD
				+ lines *(dims.x-depth)
				+ (lines/height)*dims.x*HP2;
			buffer(index) = field(offset+index);
		}
		else if(face == CHUNK_RIGHT && pack)
		{
			offset = dims.x*dims.y*HALO_PAD
				+ dims.x*HALO_PAD
				+ (dims.x-HALO_PAD-depth)
				+ lines *(dims.x-depth)
				+ (lines/height)*dims.x*HP2;
			buffer(index) = field(offset+index);
		}
		else if(face == CHUNK_LEFT)
		{
			offset = dims.x*dims.y*HALO_PAD
				+ dims.x*HALO_PAD
				+ (HALO_PAD-depth)
				+ lines *(dims.x-depth)
				+ (lines/height)*dims.x*HP2;
			field(offset+index)=buffer(index);
		}
		else
		{
			offset = dims.x*dims.y*HALO_PAD
				+ dims.x*HALO_PAD
				+ dims.x-HALO_PAD
				+ lines *(dims.x-depth)
				+ (lines/height)*dims.x*HP2;
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

// Packs or unpacks the front and back buffers.
template <class Device>
struct FrontBackPacker
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	FrontBackPacker(TLDims dims, KView buffer, KView field, const int depth, int face, bool pack)
		: dims(dims), buffer(buffer), field(field), depth(depth), face(face), pack(pack){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const
    {
		int offset = 0;
		int height = dims.y-HP2;
		int lines = index/(dims.x-HP2);

		if(face == CHUNK_FRONT && pack)
		{				
			offset = dims.x*dims.y*(dims.z-HALO_PAD-depth)
				+ dims.x*HALO_PAD
				+ HALO_PAD
				+ lines*HP2
				+ (lines/height)*dims.x*HP2;
			buffer(index) = field(offset+index);
		}
		else if (face == CHUNK_BACK && pack)
		{	
			offset = dims.x*dims.y*HALO_PAD
				+ dims.x*HALO_PAD
				+ HALO_PAD
				+ lines*HP2
				+ (lines/height)*dims.x*HP2;
			buffer(index) = field(offset+index);
		}
		else if(face == CHUNK_FRONT)
		{
			offset = dims.x*dims.y*(dims.z-HALO_PAD)
				+ dims.x*HALO_PAD
				+ HALO_PAD
				+ lines*HP2
				+ (lines/height)*dims.x*HP2;
			field(offset+index)=buffer(index);
		}
		else
		{
			offset = dims.x*dims.y*(HALO_PAD-depth)
				+ dims.x*HALO_PAD
				+ HALO_PAD
				+ lines*HP2
				+ (lines/height)*dims.x*HP2;
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
