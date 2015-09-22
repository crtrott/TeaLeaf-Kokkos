#ifndef __JACOBISOLVER
#define __JACOBISOLVER

#include "ext_chunk.hpp"

/*
 *		JACOBI SOLVER KERNEL
 */

using std::ceil;

// Copies the inner u into u0.
template <class Device>
struct JacobiInit
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	JacobiInit(TLDims dims, KView u, KView u0, KView density, KView energy,
			KView kx, KView ky, KView kz, const double dt, const int coefficient,
			double rx, double ry, double rz) 
		: dims(dims), u(u), u0(u0), density(density), energy(energy), kx(kx), 
		ky(ky), kz(kz), dt(dt), coefficient(coefficient), rx(rx), ry(ry), rz(rz){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const
    {
		KOKKOS_INDICES

		if(INDEX_IN_ONE_DOMAIN)
		{
			u0(index) = energy(index)*density(index);
			u(index) = u0(index);
		}

		if(INDEX_SKEW_DOMAIN)
		{
			double densityCentre = (coefficient == CONDUCTIVITY) 
				? density(index) : 1.0/density(index);
			double densityLeft = (coefficient == CONDUCTIVITY) 
				? density(index-1) : 1.0/density(index-1);
			double densityDown = (coefficient == CONDUCTIVITY) 
				? density(index-dims.x) : 1.0/density(index-dims.x);
			double densityBack = (coefficient == CONDUCTIVITY) 
				? density(index-dims.x*dims.y) : 1.0/density(index-dims.x*dims.y);

			kx(index) = rx*(densityLeft+densityCentre)/(2.0*densityLeft*densityCentre);
			ky(index) = ry*(densityDown+densityCentre)/(2.0*densityDown*densityCentre);
			kz(index) = rz*(densityBack+densityCentre)/(2.0*densityBack*densityCentre);
		}
	}

	TLDims dims;
	KView u;
	KView u0;
	KView density;
	KView energy;
	KView kx;
	KView ky;
	KView kz;
	const double dt;
	const int coefficient;
	double rx;
	double ry;
	double rz;
};

// Copies the value of u.
template <class Device>
struct JacobiCopyU
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	JacobiCopyU(KView r, KView u) 
		: r(r), u(u){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const
    {
		r(index) = u(index);	
	}

	KView r;
	KView u;
};

// Main Jacobi solver method.
template <class Device>
struct JacobiSolve
{
	typedef double value_type;
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	JacobiSolve(TLDims dims, KView u, KView u0, KView r, KView kx, KView ky, KView kz) 
		: dims(dims), u(u), u0(u0), r(r), kx(kx), ky(ky), kz(kz){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index, double& error) const {
		KOKKOS_INDICES

		if(INDEX_IN_INNER_DOMAIN)
		{
			int page = dims.x*dims.y;
			u(index) = (u0(index) 
					+ (kx(index+1)*r(index+1) + kx(index)*r(index-1))
					+ (ky(index+dims.x)*r(index+dims.x) + ky(index)*r(index-dims.x))
					+ (kz(index+page)*r(index+page) + kz(index)*r(index-page)))
				/ (1.0 + (kx(index)+kx(index+1))
						+ (ky(index)+ky(index+dims.x))
						+ (kz(index)+kz(index+page)));

			error += fabs(u(index)-r(index));
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

#endif
