KOKKOS_OPTIONS=KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest

lib:
	cd core; \
	make -j -j -f /core/src/Makefile KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest

install: lib
	cd core; \
	make -j -f /core/src/Makefile KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest install

build-test:
	cd core/unit_test; \
	make -j -f /core/unit_test/Makefile KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest
	cd core/perf_test; \
	make -j -f /core/perf_test/Makefile KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest
	cd containers/unit_tests; \
	make -j -f /containers/unit_tests/Makefile KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest
	cd containers/performance_tests; \
	make -j -f /containers/performance_tests/Makefile KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest
	cd algorithms/unit_tests; \
	make -j -f /algorithms/unit_tests/Makefile KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest

test: build-test
	cd core/unit_test; \
	make -f /core/unit_test/Makefile KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest test
	cd core/perf_test; \
	make -f /core/perf_test/Makefile KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest test
	cd containers/unit_tests; \
	make -f /containers/unit_tests/Makefile KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest test
	cd containers/performance_tests; \
	make -f /containers/performance_tests/Makefile KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest test
	cd algorithms/unit_tests; \
	make -f /algorithms/unit_tests/Makefile KOKKOS_PATH= KOKKOS_DEVICES=,Cuda,OpenMP CUDA_PATH=/cm/shared/apps/cuda60/toolkit/6.0.37 GTEST_PATH=/tpls/gtest test
