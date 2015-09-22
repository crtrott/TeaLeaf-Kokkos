SUBROUTINE plot3d(x_min,x_max,y_min,y_max,z_min,z_max,buffer)

    USE tea_module

    IMPLICIT NONE

    INTEGER :: j,k,l
    INTEGER :: x_min,x_max,y_min,y_max,z_min,z_max
    REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2,z_min-2:z_max+2) :: buffer
    INTEGER, PARAMETER :: out_unit=20

    WRITE(0,*) "RANK", parallel%task, "writing out the plot file..."
    OPEN(UNIT=out_unit,FILE="plot3d.dat",ACTION="write",STATUS="replace")

    DO l=z_min-2,z_max+2
        DO k=y_min-2,y_max+2
            DO j=x_min-2,x_max+2
                WRITE(out_unit,*) j+1, k+1, l+1, buffer(j,k,l)
            ENDDO
        ENDDO
    ENDDO

    CLOSE(out_unit)
    WRITE(0,*) "finished writing out the plot file..."

    STOP 0

END SUBROUTINE plot3d
