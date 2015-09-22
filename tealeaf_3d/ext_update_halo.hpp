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
        : dims(dims), buffer(buffer), face(face), depth(depth){}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int index) const
    {
        switch(face)
        {
            case CHUNK_LEFT:
                {
                    int height = dims.y-HP2;
                    int lines = index/depth;
                    int flip = index % depth;
                    int offset = dims.x*dims.y*HALO_PAD
                        + HALO_PAD
                        + dims.x*HALO_PAD
                        + lines *(dims.x-depth)
                        + (lines/height)*dims.x*HP2;

                    int fromIndex = offset+index;
                    int toIndex = fromIndex-(1+flip*2);
                    buffer(toIndex) = buffer(fromIndex);
                    break;
                }
            case CHUNK_RIGHT:
                {
                    int height = dims.y-HP2;
                    int lines = index/depth;
                    int flip = index % depth;
                    int offset = dims.x*dims.y*HALO_PAD
                        + (dims.x-HALO_PAD)
                        + dims.x*HALO_PAD
                        + lines *(dims.x-depth)
                        + (lines/height)*dims.x*HP2;

                    int toIndex = offset+index;
                    int fromIndex = toIndex-(1+flip*2);
                    buffer(toIndex) = buffer(fromIndex);

                    break;
                }
            case CHUNK_TOP:
                {
                    int width = dims.x-HP2;
                    int lines = index/width;
                    int flip = lines%depth;
                    int offset = dims.x*dims.y*HALO_PAD
                        + dims.x*(dims.y-HALO_PAD)
                        + HALO_PAD
                        + lines*HP2
                        + dims.x*(dims.y-depth)*(lines/depth);

                    int toIndex = offset+index;
                    int fromIndex = toIndex-(1+flip*2)*dims.x;
                    buffer(toIndex) = buffer(fromIndex);
                    break;
                }
            case CHUNK_BOTTOM:
                {
                    int width = dims.x-HP2;
                    int lines = index/width;
                    int flip = lines%depth;
                    int offset = dims.x*dims.y*HALO_PAD
                        + dims.x*HALO_PAD
                        + HALO_PAD
                        + lines*HP2
                        + dims.x*(dims.y-depth)*(lines/depth);

                    int fromIndex = offset+index;
                    int toIndex = fromIndex-(1+flip*2)*dims.x;
                    buffer(toIndex) = buffer(fromIndex);
                    break;
                }
            case CHUNK_FRONT:
                {
                    int width = dims.x-HP2;
                    int height = dims.y-HP2;
                    int lines = index/width;
                    int flip = index / (width*height);
                    int page = dims.x*dims.y;
                    int offset = page*(dims.z-HALO_PAD)
                        + dims.x*HALO_PAD
                        + HALO_PAD
                        + lines*HP2
                        + (lines/height)*dims.x*HP2;

                    int toIndex = offset+index;
                    int fromIndex = toIndex-(1+flip*2)*page;
                    buffer(toIndex) = buffer(fromIndex);
                    break;
                }
            case CHUNK_BACK:
                {
                    int width = dims.x-HP2;
                    int height = dims.y-HP2;
                    int lines = index/width;
                    int flip = index / (width*height);
                    int page = dims.x*dims.y;
                    int offset = page*HALO_PAD
                        + dims.x*HALO_PAD
                        + HALO_PAD
                        + lines*HP2
                        + (lines/height)*dims.x*HP2;

                    int fromIndex = offset+index;
                    int toIndex = fromIndex-(1+flip*2)*page;
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

