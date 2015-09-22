#ifndef __GENERATECHUNK
#define __GENERATECHUNK

#include "ext_chunk.hpp"

/*
 *		GENERATE CHUNK KERNEL
 *		Sets up the chunk geometry.
 */

// Sets up a chunk, setting state provided.
template <class Device>
struct GenerateChunk
{
	typedef Device device_type;
	typedef Kokkos::View<int*,Device> KIView;
	typedef Kokkos::View<double*,Device> KView;

	GenerateChunk(
			TLDims dims, int numStates, int rectParam, int pointParam, int circParam, KView energy0, 
			KView density, KView u, KView cellX, KView cellY, KView cellZ, KView vertexX, KView vertexY, 
			KView vertexZ, KView stateXMin, KView stateXMax, KView stateYMin, KView stateYMax, 
			KView stateZMin, KView stateZMax, KView stateEnergy, KView stateDensity, 
			KIView stateGeometry, KView stateRadius) : 
		dims(dims), numStates(numStates), rectParam(rectParam), pointParam(pointParam), 
		circParam(circParam), energy0(energy0), density(density), u(u), cellX(cellX), cellY(cellY), 
		cellZ(cellZ), vertexX(vertexX), vertexY(vertexY), vertexZ(vertexZ), stateXMin(stateXMin), 
		stateXMax(stateXMax), stateYMin(stateYMin), stateYMax(stateYMax), stateZMin(stateZMin), 
		stateZMax(stateZMax), stateEnergy(stateEnergy), stateDensity(stateDensity), 
		stateGeometry(stateGeometry), stateRadius(stateRadius){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const
    {
		energy0(index)=stateEnergy(0);
		density(index)=stateDensity(0);

		KOKKOS_INDICES

		for(int ss = 1; ss < numStates; ++ss)
		{
			int applyState = 0;

			if(stateGeometry(ss) == rectParam) // Rectangular state
			{
				applyState = (
						vertexX(kk+1) >= stateXMin(ss) && 
						vertexX(kk) < stateXMax(ss)    &&
						vertexY(jj+1) >= stateYMin(ss) &&
						vertexY(jj) < stateYMax(ss) 	 &&
						vertexZ(ii+1) >= stateZMin(ss) &&
						vertexZ(ii) < stateZMax(ss));
			}
			else if(stateGeometry(ss) == circParam) // Circular state
			{
				double radius = sqrt(
						(cellX(kk)-stateXMin(ss))*(cellX(kk)-stateXMin(ss))+
						(cellY(jj)-stateYMin(ss))*(cellY(jj)-stateYMin(ss))+
						(cellZ(ii)-stateZMin(ss))*(cellZ(ii)-stateZMin(ss)));

				applyState = (radius <= stateRadius(ss));
			}
			else if(stateGeometry(ss) == pointParam) // Point state
			{
				applyState = (
						vertexX(kk) == stateXMin(ss) &&
						vertexY(jj) == stateYMin(ss) &&
						vertexZ(ii) == stateZMin(ss));
			}

			// Check if state applies at this vertex, and apply
			if(applyState)
			{
				energy0(index) = stateEnergy(ss);
				density(index) = stateDensity(ss);
			}
		}

		if(kk > 0 && kk < dims.x-1 && jj > 0 && jj < dims.y-1 && ii > 0 && ii < dims.z-1) 
		{
			u(index)=energy0(index)*density(index);
		}
	}

	TLDims dims;
	int numStates;
	int rectParam;
	int pointParam;
	int circParam;
	KView energy0; 
	KView density; 
	KView u; 
	KView cellX; 
	KView cellY; 
	KView cellZ; 
	KView vertexX; 
	KView vertexY; 
	KView vertexZ; 
	KView stateXMin; 
	KView stateXMax; 
	KView stateYMin; 
	KView stateYMax; 
	KView stateZMin; 
	KView stateZMax; 
	KView stateEnergy; 
	KView stateDensity; 
	KView stateRadius; 
	KIView stateGeometry; 
};

#endif

