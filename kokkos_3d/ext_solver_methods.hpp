#ifndef __SOLVERMETHODS
#define __SOLVERMETHODS

#include "ext_chunk.hpp"

/*
 *		SHARED SOLVER METHODS
 */

// Copies the inner u into u0.
template <class Device>
struct CopyU
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	CopyU(TLDims dims, KView u, KView u0) 
		: dims(dims), u(u), u0(u0){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const
    {
		KOKKOS_INDICES

		if(INDEX_IN_INNER_DOMAIN)
		{
			u0(index) = u(index);	
		}
	}

	TLDims dims;
	KView u;
	KView u0;
};

// Calculates the residual r.
template <class Device>
struct CalculateResidual
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	CalculateResidual(TLDims dims, KView u, KView u0, KView r, 
			KView kx, KView ky, KView kz) 
		: dims(dims), u(u), u0(u0), r(r), 
		kx(kx), ky(ky), kz(kz){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const
    {
		KOKKOS_INDICES

		if(INDEX_IN_INNER_DOMAIN)
		{
			const double smvp = SMVP(u);
			r(index) = u0(index) - smvp;
		}
	}

	TLDims dims;
	KView u;
	KView u0;
	KView r;
	KView kx;
	KView ky;
	KView kz;
};

// Calculates the 2 norm of the provided buffer.
template <class Device>
struct Calculate2Norm
{
	typedef Device device_type;
	typedef double value_type;
	typedef Kokkos::View<double*,Device> KView;

	Calculate2Norm(TLDims dims, KView buffer) 
		: dims(dims), buffer(buffer){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index, double& norm) const
    {
		KOKKOS_INDICES

		if(INDEX_IN_INNER_DOMAIN)
		{
			norm += buffer(index)*buffer(index);			
		}
	}

	TLDims dims;
	KView buffer;
};

// Finalises the energy field.
template <class Device>
struct Finalise
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	Finalise(TLDims dims, KView u, KView density, KView energy) 
		: dims(dims), u(u), density(density), energy(energy) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const
    {
		KOKKOS_INDICES

		if(INDEX_IN_INNER_DOMAIN)
		{
			energy(index) = u(index)/density(index);
		}
	}

	TLDims dims;
	KView u;
	KView density;
	KView energy;
};

#endif
