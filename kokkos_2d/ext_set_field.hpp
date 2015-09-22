#ifndef __SETFIELD
#define __SETFIELD

#include "ext_chunk.hpp"

/*
 * 		SET FIELD KERNEL
 * 		Sets energy1 to energy0.
 */	

// Copies energy0 into energy1.
template <class Device>
struct SetField
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	SetField(KView energy, KView energy0) 
		: energy(energy), energy0(energy0){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const 
    {
		energy(index) = energy0(index);
	}

	KView energy;
	KView energy0;
};

#endif
