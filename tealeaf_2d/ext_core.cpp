#include "ext_initialise_chunk.hpp"
#include "ext_generate_chunk.hpp"
#include "ext_field_summary.hpp"
#include "ext_solver_methods.hpp"
#include "ext_set_field.hpp"
#include "ext_jacobi_kernel.hpp"
#include "ext_update_halo.hpp"
#include "ext_pack_kernel.hpp"
#include "ext_cg_kernel.hpp"
#include "ext_cheby_kernel.hpp"
#include "ext_ppcg_kernel.hpp"
#include <cstdarg>
#include <algorithm>

using namespace Kokkos;
using std::accumulate;

void abort(int lineNum, const char* file, const char* format, ...);

// Straps profiling around dispatch
#define PROFILED_PARALLEL_FOR(n,kernel,name)\
	START_PROFILING;\
	Kokkos::parallel_for(n,kernel);\
	STOP_PROFILING(name);

#define PROFILED_PARALLEL_REDUCE(n,kernel,reductee,name)\
	START_PROFILING;\
	Kokkos::parallel_reduce(n,kernel,reductee);\
	STOP_PROFILING(name);

// Extended kernel for the chunk initialisation
extern "C"
void ext_initialise_chunk_( 
		const int* chunk,
		const double* xMin,
		const double* yMin,
		const double* dx,
		const double* dy)
{
	auto c = Chunks[*chunk-1];
	c->dx = *dx;
	c->dy = *dy;

	int n = std::max(c->dims.x,c->dims.y);

	InitialiseChunkVertices<DEVICE> vertexKernel(c->dims, c->vertexX, c->vertexY, 
			c->cellX, c->cellY, *xMin, *yMin, *dx, *dy);
	PROFILED_PARALLEL_FOR(n, vertexKernel, "Initialise Chunk Vertices");

	InitialiseChunkAreas<DEVICE> areaKernel(c->volume, c->xArea, c->yArea, *xMin, *yMin, *dx, *dy);
	PROFILED_PARALLEL_FOR(c->fullDomain, areaKernel, "Initialise Chunk Areas");
}


// Entry point for the the chunk generation method.
	extern "C"
void ext_generate_chunk_(
		const int* chunk,
		const int* numStates,
		const double* stateDensity,
		const double* stateEnergy,
		const double* stateXMin,
		const double* stateXMax,
		const double* stateYMin,
		const double* stateYMax,
		const double* stateRadius,
		const int* stateGeometry,
		const int* rectParam,
		const int* circParam,
		const int* pointParam)
{
	auto c = Chunks[*chunk-1];

	int nStates = *numStates;

	// Device
	View<double*,DEVICE> dStateDensity("stateDensity",nStates);
	View<double*,DEVICE> dStateEnergy("stateEnergy",nStates);
	View<double*,DEVICE> dStateXMin("stateXMin",nStates);
	View<double*,DEVICE> dStateXMax("stateXMax",nStates);
	View<double*,DEVICE> dStateYMin("stateYMin",nStates);
	View<double*,DEVICE> dStateYMax("stateYMax",nStates);
	View<double*,DEVICE> dStateRadius("stateRadius",nStates);
	View<int*,DEVICE> dStateGeometry("stateGeometry",nStates);

	// Host mirrors
	typename View<double*>::HostMirror hStateDensity = create_mirror_view(dStateDensity);
	KokkosHelper::InitMirror<double>(hStateDensity,stateDensity,nStates);

	typename View<double*>::HostMirror hStateEnergy = create_mirror_view(dStateEnergy);
	KokkosHelper::InitMirror<double>(hStateEnergy,stateEnergy,nStates);

	typename View<double*>::HostMirror hStateXMin = create_mirror_view(dStateXMin);
	KokkosHelper::InitMirror<double>(hStateXMin,stateXMin,nStates);

	typename View<double*>::HostMirror hStateXMax = create_mirror_view(dStateXMax);
	KokkosHelper::InitMirror<double>(hStateXMax,stateXMax,nStates);

	typename View<double*>::HostMirror hStateYMin = create_mirror_view(dStateYMin);
	KokkosHelper::InitMirror<double>(hStateYMin,stateYMin,nStates);

	typename View<double*>::HostMirror hStateYMax = create_mirror_view(dStateYMax);
	KokkosHelper::InitMirror<double>(hStateYMax,stateYMax,nStates);

	typename View<double*>::HostMirror hStateRadius = create_mirror_view(dStateRadius);
	KokkosHelper::InitMirror<double>(hStateRadius,stateRadius,nStates);

	typename View<int*>::HostMirror hStateGeometry = create_mirror_view(dStateGeometry);
	KokkosHelper::InitMirror<int>(hStateGeometry,stateGeometry,nStates);

	// Copy onto device
	deep_copy(dStateDensity, hStateDensity);
	deep_copy(dStateEnergy, hStateEnergy);
	deep_copy(dStateXMin,hStateXMin);
	deep_copy(dStateXMax,hStateXMax);
	deep_copy(dStateYMin,hStateYMin);
	deep_copy(dStateYMax,hStateYMax);
	deep_copy(dStateRadius,hStateRadius);
	deep_copy(dStateGeometry,hStateGeometry);

	GenerateChunk<DEVICE> kernel(c->dims, nStates, *rectParam, *pointParam, *circParam, 
			c->energy0, c->density, c->u, c->cellX, c->cellY, c->vertexX, c->vertexY,
			dStateXMin, dStateXMax, dStateYMin, dStateYMax,
			dStateEnergy, dStateDensity, dStateGeometry, dStateRadius);
	PROFILED_PARALLEL_FOR(c->fullDomain, kernel, "Generate Chunk");
}

// Entry point for field summary method.
	extern "C"
void ext_field_summary_kernel_(
		const int* chunk,
		double* vol,
		double* mass,
		double* ie,
		double* temp)
{
	auto c = Chunks[*chunk-1];

	FieldSummary<DEVICE>::value_type result;
	FieldSummary<DEVICE> kernel(c->dims, c->u, c->density, c->energy0, c ->volume);
	PROFILED_PARALLEL_REDUCE(c->fullDomain, kernel, result, "Field Summary");

	*vol = result.vol;
	*mass = result.mass;
	*ie = result.ie;
	*temp = result.temp;
}

// Entry point for packing messages.
	extern "C"
void ext_pack_message_(
		const int* chunk,
		const int* fields,
		const int* offsets,
		const int* depth,
		const int* face,
		double* buffer,
		bool* pack)
{
	auto c = Chunks[*chunk-1];
	const int exchanges = accumulate(fields, fields+NUM_FIELDS, 0);

	if(exchanges < 1) return;

	for(int ii = 0; ii < NUM_FIELDS; ++ii)
	{
		if(fields[ii])
		{
			View<double*,DEVICE> field;
			switch(ii+1)
			{
				case FIELD_DENSITY:
					field = c->density;
					break;
				case FIELD_ENERGY0:
					field = c->energy0;
					break;
				case FIELD_ENERGY1:
					field = c->energy;
					break;
				case FIELD_U:
					field = c->u;
					break;
				case FIELD_P:
					field = c->p;
					break;
				case FIELD_SD:
					field = c->sd;
					break;
				default:
					abort(__LINE__,__FILE__,
							"Incorrect field provided: %d.\n", ii+1);
			}

			int bLen;
			double* extBuf = buffer+offsets[ii];
			View<double*,DEVICE> dBuf;

#define PRE_KERNEL(len, buf)\
			bLen = len; \
			dBuf = buf; \
			if(!*pack)\
			{\
				typename View<double*>::HostMirror hBuf = create_mirror_view(dBuf);\
				KokkosHelper::InitMirror<double>(hBuf,extBuf,len);\
				Kokkos::deep_copy(dBuf, hBuf);\
			}

			switch(*face)
			{
				case CHUNK_LEFT:
				case CHUNK_RIGHT:
					{
						PRE_KERNEL(c->lrLineLen**depth,	c->lrBuffer);

						LeftRightPacker<DEVICE> kernel(c->dims, dBuf, field, *depth, *face, *pack);
						PROFILED_PARALLEL_FOR(bLen, kernel, "Packing R/L");
						break;
					}
				case CHUNK_TOP:
				case CHUNK_BOTTOM:
					{
						PRE_KERNEL(c->tbLineLen**depth, c->tbBuffer);

						TopBottomPacker<DEVICE> kernel(c->dims, dBuf, field, *depth, *face, *pack);
						PROFILED_PARALLEL_FOR(bLen, kernel, "Packing T/B");
						break;
					}
				default:
					abort(__LINE__,__FILE__, "Incorrect face provided: %d.\n", *face);
					break;
			}

			if(*pack)
			{
				typename View<double*>::HostMirror hBuf = create_mirror_view(dBuf);
				Kokkos::deep_copy(hBuf, dBuf);
				for(int jj = 0; jj < bLen; ++jj)
				{
					extBuf[jj] = hBuf(jj);
				}
			}
		}
	}
}

// Entry point for the the set field method.
	extern "C"
void ext_set_field_kernel_(
		const int* chunk)
{
	auto c = Chunks[*chunk-1];

	SetField<DEVICE> kernel(c->energy, c->energy0);
	PROFILED_PARALLEL_FOR(c->fullDomain, kernel, "Set Field");
}

// Entry point to copy U.
	extern "C"
void ext_solver_copy_u_(
		const int* chunk)
{
	auto c = Chunks[*chunk-1];

	CopyU<DEVICE> kernel(c->dims, c->u, c->u0);
	PROFILED_PARALLEL_FOR(c->fullDomain, kernel, "Copy U");
}

// Entry point for calculating residual.
	extern "C"
void ext_calculate_residual_(
		const int* chunk)
{
	auto c = Chunks[*chunk-1];

	CalculateResidual<DEVICE> kernel(c->dims, c->u, c->u0, c->r, c->kx, c->ky);
	PROFILED_PARALLEL_FOR(c->fullDomain, kernel, "Calc Residual");
}

// Entry point for calculating 2norm.
	extern "C"
void ext_calculate_2norm_(
		const int* chunk,
		double* norm,
		bool* normArray)
{
	auto c = Chunks[*chunk-1];

	Calculate2Norm<DEVICE> kernel(c->dims, *normArray ? c->r : c->u0);
	PROFILED_PARALLEL_REDUCE(c->fullDomain, kernel, *norm, "Calc 2 Norm");
}

// Entry point for finalising solution.
	extern "C"
void ext_solver_finalise_(
		const int* chunk)
{
	auto c = Chunks[*chunk-1];

	Finalise<DEVICE> kernel(c->dims, c->u, c->density, c->energy);
	PROFILED_PARALLEL_FOR(c->fullDomain, kernel, "Finalise");
}

	extern "C"
void ext_update_halo_kernel_(
		const int* chunk,
		const int* chunkNeighbours,
		const int* fields,
		const int* depth)
{
	auto c = Chunks[*chunk-1];
	TLDims dims = c->dims;

#define UPDATE_FACE(face,buffer,n) \
	if(chunkNeighbours[face-1] == EXTERNAL_FACE)\
	{\
		UpdateHalo<DEVICE> kernel(dims, buffer, face, *depth);\
		PROFILED_PARALLEL_FOR(n, kernel, "Update Halo " + std::to_string((long long)face));\
	}

#define LAUNCH_UPDATE(index, buffer)\
	if(fields[index-1])\
	{\
		UPDATE_FACE(CHUNK_LEFT, buffer, dims.y**depth);\
		UPDATE_FACE(CHUNK_RIGHT, buffer, dims.y**depth);\
		UPDATE_FACE(CHUNK_TOP, buffer, dims.x**depth);\
		UPDATE_FACE(CHUNK_BOTTOM, buffer, dims.x**depth);\
	}

	LAUNCH_UPDATE(FIELD_DENSITY, c->density);
	LAUNCH_UPDATE(FIELD_P, c->p);
	LAUNCH_UPDATE(FIELD_ENERGY0, c->energy0);
	LAUNCH_UPDATE(FIELD_ENERGY1, c->energy);
	LAUNCH_UPDATE(FIELD_U, c->u);
	LAUNCH_UPDATE(FIELD_SD, c->sd);
}

// Entry point for Jacobi initialisation.
	extern "C"
void ext_jacobi_kernel_init_(
		const int* chunk,
		const int* coefficient,
		const double* dt)
{
	auto c = Chunks[*chunk-1];

	if(*coefficient < CONDUCTIVITY && *coefficient < RECIP_CONDUCTIVITY)
	{
		abort(__LINE__, __FILE__, "Coefficient %d is not valid.\n", *coefficient);
	}

	double rx = *dt/(c->dx*c->dx);
	double ry = *dt/(c->dy*c->dy);

	JacobiInit<DEVICE> kernel(c->dims, c->u, c->u0, c->density, c->energy,
			c->kx, c->ky, *dt, *coefficient, rx, ry);
	PROFILED_PARALLEL_FOR(c->fullDomain, kernel, "Jacobi Init");
}

// Entry point for Jacobi solver main method.
	extern "C"
void ext_jacobi_kernel_solve_(
		const int* chunk,
		double* error)
{
	auto c = Chunks[*chunk-1];

	JacobiCopyU<DEVICE> uKernel(c->r, c->u);
	PROFILED_PARALLEL_FOR(c->fullDomain, uKernel, "Jacobi Copy U");

	JacobiSolve<DEVICE> solveKernel(c->dims, c->u, c->u0, c->r, c->kx, c->ky);
	PROFILED_PARALLEL_REDUCE(c->fullDomain, solveKernel, *error, "Jacobi Solve");
}

// Entry point for CG initialisation.
	extern "C"
void ext_cg_solver_init_(
		const int* chunk,
		const int* coefficient,
		const int* preconditioner,
		double* dt,
		double* rro)
{
	auto c = Chunks[*chunk-1];

	if(*coefficient < CONDUCTIVITY && *coefficient < RECIP_CONDUCTIVITY)
	{
		abort(__LINE__, __FILE__, "Coefficient %d is not valid.\n", *coefficient);
	}

	c->preconditioner = *preconditioner;

	if(c->rank == 0)
	{
		std::cout << "Setting preconditioner to " << *preconditioner << std::endl;
	}

	CGInitU<DEVICE> uKernel(c->p, c->r, c->u, c->density, c->energy);
	PROFILED_PARALLEL_FOR(c->fullDomain, uKernel, "CG Init U");

	double rx = *dt/(c->dx*c->dx);
	double ry = *dt/(c->dy*c->dy);

	CGInitW<DEVICE> wKernel(c->dims, c->w, c->density, *coefficient);
	PROFILED_PARALLEL_FOR(c->fullDomain, wKernel, "CG Init W");

	CGInitK<DEVICE> kKernel(c->dims, c->w, c->kx, c->ky, rx, ry);
	PROFILED_PARALLEL_FOR(c->fullDomain, kKernel, "CG Init K");

	CGCalcRRO<DEVICE> rroKernel(c->dims, c->mi, c->kx, c->ky, c->z,
			c->p, c->r, c->u, c->w, c->preconditioner);
	PROFILED_PARALLEL_REDUCE(c->fullDomain, rroKernel, *rro, "CG Calc RRO");
}

// Entry point for calculating w
	extern "C"
void ext_cg_calc_w_(
		const int* chunk,
		double* pw)
{
	auto c = Chunks[*chunk-1];

	CGCalcW<DEVICE> kernel(c->dims, c->w, c->p, c->kx, c->ky);
#ifdef USE_TEAMS
  PROFILED_PARALLEL_REDUCE(Kokkos::TeamPolicy<DEVICE>(c->dims.x-4,Kokkos::AUTO), kernel, *pw, "CG Calc W");
#else
  PROFILED_PARALLEL_REDUCE(c->fullDomain, kernel, *pw, "CG Calc W");
#endif
}

// Entry point for calculating ur
	extern "C"
void ext_cg_calc_ur_(
		const int* chunk,
		const double* alpha,
		double* rrn)
{
	auto c = Chunks[*chunk-1];

	CGCalcUr<DEVICE> kernel(c->dims, c->u, c->r, c->mi, c->z, c->p, c->w,
			*alpha, c->preconditioner);
#ifdef USE_TEAMS
  PROFILED_PARALLEL_REDUCE(Kokkos::TeamPolicy<DEVICE>(c->dims.x-4,Kokkos::AUTO), kernel, *rrn, "CG Calc UR");
#else
	PROFILED_PARALLEL_REDUCE(c->fullDomain, kernel, *rrn, "CG Calc UR");
#endif
}

// Entry point for calculating p
	extern "C"
void ext_cg_calc_p_(
		const int* chunk,
		const double* beta)
{
	auto c = Chunks[*chunk-1];

	CGCalcP<DEVICE> kernel(c->dims, c->p, c->z, c->r, *beta, c->preconditioner);

#ifdef USE_TEAMS
	PROFILED_PARALLEL_FOR(Kokkos::TeamPolicy<DEVICE>(c->dims.x-4,Kokkos::AUTO), kernel, "CG Calc P");
#else
	PROFILED_PARALLEL_FOR(c->fullDomain, kernel, "CG Calc P");
#endif
}

// Entry point for Chebyshev initialisation.
	extern "C"
void ext_cheby_solver_init_(
		const int* chunk,
		double* alphas, 
		double* betas,
		const double* theta,
		const int* preconditioner,
		const int* maxChebyIters)
{
	auto c = Chunks[*chunk-1];

	c->preconditioner = *preconditioner;
	c->alphas = View<double*,DEVICE>("alphas", *maxChebyIters);
	c->betas = View<double*,DEVICE>("betas", *maxChebyIters);

	// Create the host mirrored objects, initialise them, and copy them to the device
	View<double*>::HostMirror hAlphas = create_mirror_view(c->alphas);
	KokkosHelper::InitMirror<double>(hAlphas,alphas,*maxChebyIters);
	deep_copy(c->alphas, hAlphas);

	View<double*>::HostMirror hBetas = create_mirror_view(c->betas);
	KokkosHelper::InitMirror<double>(hBetas,betas,*maxChebyIters);
	deep_copy(c->betas, hBetas);

	ChebyInit<DEVICE> kernel(c->dims, c->p, c->r, c->u, c->mi, c->u0,
			c->w, c->kx, c->ky, *theta, *preconditioner);
	PROFILED_PARALLEL_FOR(c->fullDomain, kernel, "Cheby Init");

	ChebyCalcU<DEVICE> uKernel(c->dims, c->p, c->u);
	PROFILED_PARALLEL_FOR(c->fullDomain, uKernel, "Cheby Calc U");
}

// Entry point for the Chebyshev iterations.
	extern "C"
void ext_cheby_solver_iterate_(
		const int* chunk,
		const int* chebyCalcStep)
{
	auto c = Chunks[*chunk-1];
	int step = *chebyCalcStep-1;

	ChebyIterate<DEVICE> kernel(c->dims, c->p, c->r, c->u, c->mi,
			c->u0, c->w, c->kx, c->ky, c->alphas, c->betas, 
			step, c->preconditioner);
	PROFILED_PARALLEL_FOR(c->fullDomain, kernel, "Cheby Iterate");

	ChebyCalcU<DEVICE> uKernel(c->dims, c->p, c->u);
	PROFILED_PARALLEL_FOR(c->fullDomain, uKernel, "Cheby Calc U");
}

// Entry point for CG initialisation.
	extern "C"
void ext_ppcg_init_(
		const int* chunk,
		double* alphas,
		double* betas,
		const double* theta,
		const int* maxChebyIters)
{
	auto c = Chunks[*chunk-1];
	c->theta = *theta;
    c->alphas = View<double*,DEVICE>("alphas", *maxChebyIters);
    c->betas = View<double*,DEVICE>("betas", *maxChebyIters);

	// Create the host mirrors, initialise them, and copy data to the device
    View<double*>::HostMirror hAlphas = create_mirror_view(c->alphas);
    KokkosHelper::InitMirror<double>(hAlphas,alphas,*maxChebyIters);
    deep_copy(c->alphas, hAlphas);

    View<double*>::HostMirror hBetas = create_mirror_view(c->betas);
    KokkosHelper::InitMirror<double>(hBetas,betas,*maxChebyIters);
    deep_copy(c->betas, hBetas);
}

// Entry point for initialising sd.
	extern "C"
void ext_ppcg_init_sd_(
		const int* chunk)
{
	auto c = Chunks[*chunk-1];

	PPCGInitSd<DEVICE> kernel(c->dims, c->sd, c->r, c->mi, c->theta, c->preconditioner);
	PROFILED_PARALLEL_FOR(c->fullDomain, kernel, "PPCG Init SD");
}

// Entry point for the main PPCG step.
	extern "C"
void ext_ppcg_inner_(
		const int* chunk,
		int* currentStep)
{
	auto c = Chunks[*chunk-1];
	int step = *currentStep-1;

	PPCGCalcU<DEVICE> uKernel(c->dims, c->sd, c->r, c->u, c->kx, c->ky);
	PROFILED_PARALLEL_FOR(c->fullDomain, uKernel, "PPCG Calc U");

	PPCGCalcSd<DEVICE> sdKernel(c->dims, c->sd, c->r, c->mi, c->alphas, 
			c->betas, c->theta, c->preconditioner, step);
	PROFILED_PARALLEL_FOR(c->fullDomain, sdKernel, "PPCG Calc SD");
}


// Aborts the application.
void abort(int lineNum, const char* file, const char* format, ...)
{
	fprintf(stderr, "\x1b[31m");
	fprintf(stderr, "\nError at line %d in %s:", lineNum, file);
	fprintf(stderr, "\x1b[0m \n");

	va_list arglist;
	va_start(arglist, format);
	vfprintf(stderr, format, arglist);
	va_end(arglist);

	exit(1);
}

// Plots a three-dimensional dat file for debugging.
template <class Device>
void TeaLeafChunk<Device>::Plot3d(View<double*,Device> buffer, std::string name)
{
	// Open the plot file
	FILE* fp = fopen("plot2d.dat", "wb");
	if(!fp) { fprintf(stderr, "Could not open plot file.\n"); }

	double bSum = 0.0;

	// Plot the data structure
	for(int jj = 0; jj < dims.y; ++jj)
	{
		for(int kk = 0; kk < dims.x; ++kk)
		{
			double val = buffer(kk+jj*dims.x);
			fprintf(fp, "%d %d %.12E\n", kk, jj, val);
			bSum += val;
		}
	}

	printf("%s: %.12E\n", name.c_str(), bSum);
	fclose(fp);
}

