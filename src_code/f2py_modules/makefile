#
# Source files
#
SRCS =  get_omega.F

#
# libraries
#
#LIBS = -L/opt/intel/mkl/8.0.1/lib/em64t -lmkl_lapack -lmkl_em64t 
#LIBS = -llapack
#
# compiler flags
FFLG = --f90flags="-extend_source -O1 "
#
#
# Make fortran extension
#
pysw:	$(SRCS)
	f2py $(FFLG) -c -m omega_fort get_omega.F
#f2py $(FFLG) -c -m part particles_3d.f90

clean:
	rm *.so
