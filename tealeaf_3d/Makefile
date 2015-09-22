#Crown Copyright 2014 AWE.
#
# This file is part of TeaLeaf.
#
# TeaLeaf is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
#
# TeaLeaf is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# TeaLeaf. If not, see http://www.gnu.org/licenses/.

#  @brief Makefile for TeaLeaf
#  @author David Beckingsale, Wayne Gaudin, Matthew Martineau
#  @details Makefile for the Kokkos implementation of TeaLeaf.

default: build

# User defined parameters, suggests tested implementations
# TARGET   = <OMP|CUDA|MIC>   : OpenMP, CUDA, MIC native compilation
# COMPILER = <INTEL|GNU> 	  : Currently tested with Intel, GNU compilers
# KOKKOS_PATH is the root of Kokkos source
TARGET   	 = OMP
COMPILER 	 = INTEL
MPI_F90	 	 = mpiifort
MPI_CPP      = mpiicpc
KOKKOS_PATH ?= ../kokkos_src
OPTIONS     += #-DENABLE_TIMELOGGING -DENABLE_PROFILING
EXE 		 = tea_leaf

# Compiler-specific flags
OMP_INTEL     = -openmp
OMP_SUN       = -xopenmp=parallel -vpara
OMP_GNU       = -fopenmp -cpp
OMP_CRAY      = -e Z
OMP_PGI       = -mp=nonuma
OMP_PATHSCALE = -mp
OMP_XL        = -qsmp=omp -qthreaded
OMP=$(OMP_$(COMPILER))

FLAGS_          = -O3
FLAGS_INTEL     = -O3 -no-prec-div -fpp
FLAGS_SUN       = -fast -xipo=2 -Xlistv4
FLAGS_GNU       = -O3 -march=native -funroll-loops
FLAGS_CRAY      = -em -ra -h acc_model=fast_addr:no_deep_copy:auto_async_all
FLAGS_PGI       = -fastsse -gopt -Mipa=fast -Mlist
FLAGS_PATHSCALE = -O3
FLAGS_XL       = -O5 -qipa=partition=large -g -qfullpath -Q -qsigtrap -qextname=flush:ideal_gas_kernel_c:viscosity_kernel_c:pdv_kernel_c:revert_kernel_c:accelerate_kernel_c:flux_calc_kernel_c:advec_cell_kernel_c:advec_mom_kernel_c:reset_field_kernel_c:timer_c:unpack_top_bottom_buffers_c:pack_top_bottom_buffers_c:unpack_left_right_buffers_c:pack_left_right_buffers_c:field_summary_kernel_c:update_halo_kernel_c:generate_chunk_kernel_c:initialise_chunk_kernel_c:calc_dt_kernel_c -qlistopt -qattr=full -qlist -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036
	
CFLAGS_          = -O3
CFLAGS_INTEL     = -O3 -march=native -no-prec-div -restrict -fno-alias 
CFLAGS_SUN       = -fast -xipo=2
CFLAGS_GNU       = -O3 -march=native -funroll-loops
CFLAGS_CRAY      = -em -h list=a
CFLAGS_PGI       = -fastsse -gopt -Mipa=fast -Mlist
CFLAGS_PATHSCALE = -O3
CFLAGS_XL        = -O5 -qipa=partition=large -g -qfullpath -Q -qlistopt -qattr=full -qlist -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036 -qsrcmsg

ifdef DEBUG
  FLAGS_INTEL     = -O0 -g -debug all -check all -traceback -check noarg_temp_created
  FLAGS_SUN       = -g -xopenmp=noopt -stackvar -u -fpover=yes -C -ftrap=common
  FLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check
  FLAGS_CRAY      = -O0 -g -em -eD
  FLAGS_PGI       = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk -Mchkptr
  FLAGS_PATHSCALE = -O0 -g
  FLAGS_XL       = -O0 -g -qfullpath -qcheck -qflttrap=ov:zero:invalid:en -qsource -qinitauto=FF -qmaxmem=-1 -qinit=f90ptr -qsigtrap -qextname=flush:ideal_gas_kernel_c:viscosity_kernel_c:pdv_kernel_c:revert_kernel_c:accelerate_kernel_c:flux_calc_kernel_c:advec_cell_kernel_c:advec_mom_kernel_c:reset_field_kernel_c:timer_c:unpack_top_bottom_buffers_c:pack_top_bottom_buffers_c:unpack_left_right_buffers_c:pack_left_right_buffers_c:field_summary_kernel_c:update_halo_kernel_c:generate_chunk_kernel_c:initialise_chunk_kernel_c:calc_dt_kernel_c
  FLAGS_          = -O0 -g
  CFLAGS_          = -O0 -g
  CFLAGS_INTEL    = -O0 -g -debug all -traceback
  CFLAGS_SUN      = -g -O0 -xopenmp=noopt -stackvar -u -fpover=yes -C -ftrap=common
  CFLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check
  CFLAGS_CRAY     = -O0 -g -em -eD
  CFLAGS_PGI      = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk
  CFLAGS_PATHSCALE= -O0 -g
  CFLAGS_XL      = -O0 -g -qfullpath -qcheck -qflttrap=ov:zero:invalid:en -qsource -qinitauto=FF -qmaxmem=-1 -qsrcmsg
endif

ifdef IEEE
  I3E_INTEL     = -fp-model strict -fp-model source -prec-div -prec-sqrt
  I3E_SUN       = -fsimple=0 -fns=no
  I3E_GNU       = -ffloat-store
  I3E_CRAY      = -hflex_mp=intolerant
  I3E_PGI       = -Kieee
  I3E_PATHSCALE = -mieee-fp
  I3E_XL       = -qfloat=nomaf
  I3E=$(I3E_$(COMPILER))
endif

FLAGS=$(FLAGS_$(COMPILER)) $(OMP) $(I3E) $(OPTIONS)
CXXFLAGS=$(CFLAGS_$(COMPILER)) $(OMP) $(I3E) $(OPTIONS)
LFLAGS=-lstdc++

# Kokkos settings
ifeq ($(TARGET), CUDA)
  CXX = $(KOKKOS_PATH)/config/nvcc_wrapper
  KOKKOS_DEVICES = "Cuda"
  KOKKOS_ARCH = "Kepler35"
  CXXFLAGS += -DCUDA
else
  CXX = $(MPI_CPP)
  KOKKOS_DEVICES = "OpenMP"
  KOKKOS_ARCH = "SNB"
endif

ifeq ($(TARGET), MIC)
  KOKKOS_ARCH = "KNC"
  FLAGS += -mmic
endif

FOBJ=\
			  data.o						\
			  definitions.o					\
			  pack_kernel.o					\
			  tea.o							\
			  report.o						\
			  timer.o						\
			  parse.o						\
			  read_input.o					\
			  initialise_chunk_kernel.o		\
			  initialise_chunk.o			\
			  build_field.o					\
			  update_halo_kernel.o			\
			  update_halo.o					\
			  start.o						\
			  generate_chunk_kernel.o		\
			  generate_chunk.o				\
			  initialise.o					\
			  field_summary_kernel.o		\
			  field_summary.o				\
			  calc_dt.o						\
			  timestep.o					\
			  set_field_kernel.o            \
			  set_field.o                   \
			  tea_leaf_jacobi.o             \
			  tea_leaf_cg.o             	\
			  tea_leaf_cheby.o             	\
			  tea_leaf_ppcg.o             	\
			  tea_solve.o                   \
			  visit.o						\
			  tea_leaf.o					\
			  diffuse.o

OBJ=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
OBJ+=$(FOBJ)

include $(KOKKOS_PATH)/Makefile.kokkos
include make.deps

build: $(EXE)

$(EXE): Makefile $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(MPI_F90) $(KOKKOS_LDFLAGS) $(FLAGS) $(OBJ) $(KOKKOS_LIBS) $(LFLAGS) -o $(EXE)
	@echo $(MESSAGE)

.PHONY: clean 

clean: kokkos-clean
	rm -f *.o *.cuda *.host *.mod tea_leaf

%_module.mod: %.f90 %.o
	@true

%.o: %.f90 Makefile make.deps
	$(MPI_F90) $(FLAGS) -c $<

%.o: %.cpp $(KOKKOS_CPP_DEPENDS) Makefile make.deps
	$(CXX) $(KOKKOS_CXXFLAGS) $(KOKKOS_CPPFLAGS) $(CXXFLAGS) -c $<

