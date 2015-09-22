!Crown Copyright 2014 AWE.
!
! This file is part of TeaLeaf.
!
! TeaLeaf is free software: you can redistribute it and/or modify it under 
! the terms of the GNU General Public License as published by the 
! Free Software Foundation, either version 3 of the License, or (at your option) 
! any later version.
!
! TeaLeaf is distributed in the hope that it will be useful, but 
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
! details.
!
! You should have received a copy of the GNU General Public License along with 
! TeaLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief Driver for the field summary kernels
!>  @author David Beckingsale, Wayne Gaudin
!>  @details The user specified field summary kernel is invoked here. A summation
!>  across all mesh chunks is then performed and the information outputed.
!>  If the run is a test problem, the final result is compared with the expected
!>  result and the difference output.
!>  Note the reference solution is the value returned from an Intel compiler with
!>  ieee options set on a single core run.

SUBROUTINE field_summary()

    USE tea_module
    USE field_summary_kernel_module

    IMPLICIT NONE

    INTEGER      :: c
    REAL(KIND=8) :: vol,mass,ie,temp
    REAL(KIND=8) :: kernel_time,timer
    REAL(KIND=8) :: qa_diff

    !$ INTEGER :: OMP_GET_THREAD_NUM

    IF(parallel%boss)THEN
        WRITE(g_out,*)
        WRITE(g_out,*) 'Time ',time
        WRITE(g_out,'(a13,5a16)')'           ','Volume','Mass','Density','Energy','U'
    ENDIF

    IF(profiler_on) kernel_time=timer()

    IF(use_fortran_kernels)THEN
      DO c=1,chunks_per_task
        IF(chunks(c)%task.EQ.parallel%task) THEN
            CALL field_summary_kernel(chunks(c)%field%x_min,                   &
                chunks(c)%field%x_max,                   &
                chunks(c)%field%y_min,                   &
                chunks(c)%field%y_max,                   &
                chunks(c)%field%z_min,                   &
                chunks(c)%field%z_max,                   &
                chunks(c)%field%volume,                  &
                chunks(c)%field%density,                 &
                chunks(c)%field%energy0,                 &
                chunks(c)%field%u,                       &
                vol,mass,ie,temp                         )
        ENDIF
      ENDDO
    ELSEIF(use_ext_kernels) THEN
      DO c=1,chunks_per_task
        IF(chunks(c)%task.EQ.parallel%task) THEN
            CALL ext_field_summary_kernel(c, vol, mass, ie, temp)
        ENDIF
      ENDDO
    ENDIF

    ! For mpi I need a reduction here
    CALL tea_sum(vol)
    CALL tea_sum(mass)
    CALL tea_sum(ie)
    CALL tea_sum(temp)
    IF(profiler_on) profiler%summary=profiler%summary+(timer()-kernel_time)

    IF(parallel%boss) THEN
        !$ IF(OMP_GET_THREAD_NUM().EQ.0) THEN
        WRITE(g_out,'(a6,i7,5e16.7)')' step:',step,vol,mass,mass/vol,ie,temp
        WRITE(g_out,*)
        !$ ENDIF
    ENDIF

    !Check if this is the final call and if it is a test problem, check the result.
    IF(complete) THEN
        IF(parallel%boss) THEN
            !$ IF(OMP_GET_THREAD_NUM().EQ.0) THEN
            IF(test_problem.GE.1) THEN
                ! Note that the "correct" solution is with IEEE switched on, 1 task, 1 thread, Intel compiler on Ivy Bridge
                IF(test_problem.EQ.1) qa_diff=ABS((100.0_8*(temp/157.550841832793_8))-100.0_8)
                IF(test_problem.EQ.2) qa_diff=ABS((100.0_8*(temp/116.067951160930_8))-100.0_8)
                IF(test_problem.EQ.3) qa_diff=ABS((100.0_8*(temp/95.4865103390698_8))-100.0_8)
                IF(test_problem.EQ.4) qa_diff=ABS((100.0_8*(temp/166.838315378708_8))-100.0_8)
                IF(test_problem.EQ.5) qa_diff=ABS((100.0_8*(temp/116.482111627676_8))-100.0_8)

                ! Three dimensional problem (GNU compiler, IEEE, 1 core, Sandy Bridge) 20 20 20
                IF(test_problem.EQ.6) qa_diff=ABS((100.0_8*(temp/57.275950005439007))-100.0_8)

                ! 10x10x10
                IF(test_problem.EQ.7) qa_diff=ABS((100.0_8*(temp/56.905915016570454_08))-100.0_8)

                ! 16x16x16
                IF(test_problem.EQ.8) qa_diff=ABS((100.0_8*(temp/70.914754278021732_08))-100.0_8)

                ! 25x25x25
                IF(test_problem.EQ.9) qa_diff=ABS((100.0_8*(temp/58.156372502592568_08))-100.0_8)

                ! 40x40x40
                IF(test_problem.EQ.10) qa_diff=ABS((100.0_8*(temp/57.571559142436392_08))-100.0_8)

                ! 64x64x64
                IF(test_problem.EQ.11) qa_diff=ABS((100.0_8*(temp/58.442896702081427_08))-100.0_8)

                ! 100x100x100
                IF(test_problem.EQ.12) qa_diff=ABS((100.0_8*(temp/57.788169431988784_08))-100.0_8)

                ! 160x160x160
                IF(test_problem.EQ.13) qa_diff=ABS((100.0_8*(temp/57.847149250777079_08))-100.0_8)

                ! 256x256x256
                IF(test_problem.EQ.14) qa_diff=ABS((100.0_8*(temp/58.633563819463340_08))-100.0_8)

                ! 58x58x58
                IF(test_problem.EQ.15) qa_diff=ABS((100.0_8*(temp/59.336123231095804_08))-100.0_8)

                ! 67x67x67
                IF(test_problem.EQ.16) qa_diff=ABS((100.0_8*(temp/60.219852602894704_08))-100.0_8)

                ! 74x74x74
                IF(test_problem.EQ.17) qa_diff=ABS((100.0_8*(temp/58.374475944856059_08))-100.0_8)

                ! 79x79x79
                IF(test_problem.EQ.18) qa_diff=ABS((100.0_8*(temp/58.605123548564777_08))-100.0_8)

                ! 84x84x84
                IF(test_problem.EQ.19) qa_diff=ABS((100.0_8*(temp/58.322724309579144_08))-100.0_8)

                ! 89x89x89
                IF(test_problem.EQ.20) qa_diff=ABS((100.0_8*(temp/58.529888354664266_08))-100.0_8)

                ! 93x93x93
                IF(test_problem.EQ.21) qa_diff=ABS((100.0_8*(temp/59.029784947717950_08))-100.0_8)

                ! 97x97x97
                IF(test_problem.EQ.22) qa_diff=ABS((100.0_8*(temp/59.496231543737323_08))-100.0_8)
 
                ! 100x100x100
                IF(test_problem.EQ.23) qa_diff=ABS((100.0_8*(temp/57.788169432006349_08))-100.0_8)

                ! 103x103x103
                IF(test_problem.EQ.24) qa_diff=ABS((100.0_8*(temp/58.922579985226299_08))-100.0_8)

                ! 106x106x106
                IF(test_problem.EQ.25) qa_diff=ABS((100.0_8*(temp/59.629007203000995_08))-100.0_8)

                ! 109x109x109
                IF(test_problem.EQ.26) qa_diff=ABS((100.0_8*(temp/58.421562127057406_08))-100.0_8)

                ! 112x112x112
                IF(test_problem.EQ.27) qa_diff=ABS((100.0_8*(temp/59.093728049719466_08))-100.0_8)

                ! 114x114x114
                IF(test_problem.EQ.28) qa_diff=ABS((100.0_8*(temp/58.222880800927285_08))-100.0_8)
                
                ! 117x117x117
                IF(test_problem.EQ.29) qa_diff=ABS((100.0_8*(temp/59.225127737540063_08))-100.0_8)


                ! 50 50 50
                !IF(test_problem.EQ.7) qa_diff=ABS((100.0_8*(temp/57.640367914233721))-100.0_8)

                ! 250 250 250
                !IF(test_problem.EQ.8) qa_diff=ABS((100.0_8*(temp/57.883485034282600))-100.0_8)

                ! 64 64 64
                !IF(test_problem.EQ.9) qa_diff=ABS((100.0_8*(temp/58.442896702081427))-100.0_8)

                WRITE(*,'(a,i4,a,f16.7,a)')"Test problem", Test_problem," is within",qa_diff,"% of the expected solution"
                WRITE(g_out,'(a,i4,a,e16.7,a)')"Test problem", Test_problem," is within",qa_diff,"% of the expected solution"
                IF(qa_diff.LT.0.001) THEN
                    WRITE(*,*)"This test is considered "//achar(27)//"[32m PASSED"//achar(27)//"[0m."
                    WRITE(g_out,*)"This test is considered PASSED"
                ELSE
                    WRITE(*,*)"This test is considered "//achar(27)//"[31m NOT PASSED"//achar(27)//"[0m."
                    WRITE(g_out,*)"This test is considered NOT PASSED"
                ENDIF
            ENDIF
            !$ ENDIF
        ENDIF
    ENDIF

END SUBROUTINE field_summary
