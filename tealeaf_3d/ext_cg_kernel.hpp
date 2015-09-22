#include "ext_chunk.hpp"
#include <cstdlib>

/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */

// Initialises u.
template <class Device>
struct CGInitU
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	CGInitU(KView p, KView r, KView u, KView density, KView energy) 
		: p(p), r(r), u(u), density(density), energy(energy){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const 
    {
		p[index] = 0.0;
		r[index] = 0.0;
		u[index] = energy[index]*density[index];
	}

	KView r;
	KView u;
	KView p;
	KView density;
	KView energy;
};

// Initialises w.
template <class Device>
struct CGInitW
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	CGInitW(TLDims dims, KView w, KView density, const int coefficient) 
		: dims(dims), w(w), density(density), coefficient(coefficient) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const 
    {
		KOKKOS_INDICES;

		if(INDEX_IN_ONE_DOMAIN)
		{
			w[index] = (coefficient == CONDUCTIVITY) ? density[index] : 1.0/density[index];
		}
	}

	TLDims dims;
	KView w;
	KView density;
	const int coefficient;
};

// Initialises directions kx, ky and kz.
template <class Device>
struct CGInitK
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	CGInitK(TLDims dims, KView w, KView kx, KView ky, KView kz, double rx, double ry, double rz) 
		: dims(dims), w(w), kx(kx), ky(ky), kz(kz), rx(rx), ry(ry), rz(rz) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const 
    {
		KOKKOS_INDICES;

		if(INDEX_SKEW_DOMAIN)
		{
			kx[index] = rx*(w[index-1]+w[index])/(2.0*w[index-1]*w[index]);
			ky[index] = ry*(w[index-dims.x]+w[index])/(2.0*w[index-dims.x]*w[index]);
			kz[index] = rz*(w[index-dims.x*dims.y]+w[index])/(2.0*w[index-dims.x*dims.y]*w[index]);
		}
	}

	TLDims dims;
	KView w;
	KView kx;
	KView ky;
	KView kz;
	double rx;
	double ry;
	double rz;
};

// Calculates a value for rro.
template <class Device>
struct CGCalcRRO
{
	typedef double value_type;
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	CGCalcRRO(TLDims dims, KView mi, KView kx, KView ky, KView kz, 
			KView z, KView p, KView r, KView u, KView w, bool preconditioner) 
		: dims(dims), mi(mi), kx(kx), ky(ky), kz(kz), z(z), p(p), 
		r(r), u(u), w(w), preconditioner(preconditioner) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index, value_type& rro) const
    {
		KOKKOS_INDICES;

		if(INDEX_IN_INNER_DOMAIN)
		{
			const double smvp = SMVP(u);
			w[index] = smvp;
			r[index] = u[index]-w[index];

			if(preconditioner)
			{
				mi[index] = (1.0
						+ (kx[index+1]+kx[index])
						+ (ky[index+dims.x]+ky[index])
						+ (kz[index+dims.x*dims.y]+kz[index]));
				mi[index] = 1.0/mi[index];
				z[index] = mi[index]*r[index];
				p[index] = z[index];
			}
			else
			{
				p[index] = r[index];
			}

			rro += r[index]*p[index];
		}
	}

	TLDims dims;
	KView mi;
	KView w;
	KView p;
	KView r;
	KView z;
	KView u;
	KView kx;
	KView ky;
	KView kz;
	bool preconditioner;
};

// Calculates a value for w.
template <class Device>
struct CGCalcW
{
	typedef double value_type;
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	CGCalcW(TLDims dims, KView w, KView p, KView kx, KView ky, KView kz) 
		: dims(dims), w(w), p(p), kx(kx), ky(ky), kz(kz) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index, value_type& pw) const 
    {
		KOKKOS_INDICES;

		if(INDEX_IN_INNER_DOMAIN)
		{
			const double smvp = SMVP(p);
			w[index] = smvp;
			pw += w[index]*p[index];
		}
	}

	TLDims dims;
	KView w;
	KView p;
	KView kx;
	KView ky;
	KView kz;
};

// Calculates a value for u and r.
template <class Device>
struct CGCalcUr
{
	typedef double value_type;
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	CGCalcUr(TLDims dims, KView u, KView r, KView mi, KView z, KView p, 
			KView w, double alpha, bool preconditioner) 
		: dims(dims), u(u), r(r), mi(mi), z(z), p(p), w(w),
		alpha(alpha), preconditioner(preconditioner)	{}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index, value_type& rrn) const 
    {
		KOKKOS_INDICES;

		if(INDEX_IN_INNER_DOMAIN)
		{
			u[index] += alpha*p[index];
			r[index] -= alpha*w[index];

			if(preconditioner)
			{
				z[index] = mi[index]*r[index];
				rrn += r[index]*z[index];
			}
			else
			{
				rrn += r[index]*r[index];
			}
		}
	}

	TLDims dims;
	KView u;
	KView r;
	KView mi;
	KView z;
	KView p;
	KView w;
	double alpha;
	bool preconditioner;
};

// Calculates a value for p.
template <class Device>
struct CGCalcP
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	CGCalcP(TLDims dims, KView p, KView z, KView r, double beta, bool preconditioner) 
		: dims(dims), p(p), z(z), r(r), beta(beta), preconditioner(preconditioner) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const 
    {
		KOKKOS_INDICES;

		if(INDEX_IN_INNER_DOMAIN)
		{
			p[index] = beta*p[index] + ((preconditioner) ? z[index] : r[index]);
		}
	}

	TLDims dims;
	KView p;
	KView r;
	KView z;
	double beta;
	bool preconditioner;
};

