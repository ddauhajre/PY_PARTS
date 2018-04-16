#include "cppdefs.h"
#ifdef SOLVE3D
 
      subroutine omega (tile)
      implicit none
      integer tile
# include "param.h"
# include "private_scratch.h"
# include "compute_tile_bounds.h"
      call omega_tile (istr,iend,jstr,jend, A2d(1,1))
      return
      end
 
      subroutine omega_tile (istr,iend,jstr,jend, wrk)

! Compute S-coordinate vertical velocity, w=[Hz/(m*n)]*omega [m^3/s],
! which has meaning of FINITE_VOLUME WATER FLUX across MOVING grid-
! box interfaces of RHO-boxes. To do so, we integrate divergence of
! horizontal mass fluxes from bottom up, starting with the no-normal
! flow boundary condition at the bottom (k=0); After this operation
! W(:,:,N) contains vertical velocity flux at the free surface, which
! is the time-tendency of the free surface, d_zeta/d_t multiplied by
! grid-box area as seen from above;  To convert W(:,:,:) into
! S-coordinate vertical velocity flux, one needs to subtract the
! vertical velocities of moving S-coordinate surfaces, which are
! proportional the product of d_zeta/d_t and the fraction of the
! distance from the point to the bottom divided by the total
! depth of water column, i.e. the whole S-coordinate system is
! "breathes" by linear in Z-space expansion and contraction set by
! variation in free surface.

      implicit none
      integer istr,iend,jstr,jend, i,j,k
# include "param.h"
      real wrk(PRIVATE_1D_SCRATCH_ARRAY)
# ifdef Q_PSOURCE
      integer is,ii,jj
#  include "sources.h"
# endif
# include "grid.h"
# include "ocean3d.h"
# include "scalars.h"

# include "compute_auxiliary_bounds.h"

      do j=jstr,jend
        do i=istr,iend
          W(i,j,0)=0.
        enddo
        do k=1,N,+1        !--> recursive
          do i=istr,iend
            W(i,j,k)=W(i,j,k-1) -FlxU(i+1,j,k)+FlxU(i,j,k)
     &                          -FlxV(i,j+1,k)+FlxV(i,j,k)
          enddo
        enddo

# ifdef Q_PSOURCE
        do is=1,Nsrc
          if (Dsrc(is).eq.2) then
            ii=Isrc(is)
            jj=Jsrc(is)
            if (istrR.le.ii.and.ii.le.iendR.and.
     &          jstrR.le.jj.and.jj.le.jendR.and.j.eq.jj) then
              do k=1,N-1                          ! rho- to w-point
                wrk(k)=0.5*(Qsrc(is,k)+Qsrc(is,k+1))
              enddo
              wrk(0)=1.5*wrk(1)-0.5*wrk(2)        ! linear profile
              wrk(N)=1.5*wrk(N-1)-0.5*wrk(N-2)
              do k=0,N
                W(ii,jj,k)=W(ii,jj,k)+wrk(k)
              enddo
            endif
          endif
        enddo
# endif

        do i=istr,iend
          wrk(i)=W(i,j,N)/(z_w(i,j,N)-z_w(i,j,0))
          W(i,j,N)=0.
        enddo
        do k=N-1,1,-1
          do i=istr,iend
            W(i,j,k)=W(i,j,k)-wrk(i)*(z_w(i,j,k)-z_w(i,j,0))
          enddo
        enddo
      enddo

# ifndef EW_PERIODIC
      if (WESTERN_EDGE) then                       ! Set lateral
        do k=0,N                                   ! boundary 
          do j=jstr,jend                           ! conditions
            W(istr-1,j,k)=W(istr,j,k)
          enddo
        enddo
      endif
      if (EASTERN_EDGE) then
        do k=0,N
          do j=jstr,jend
            W(iend+1,j,k)=W(iend,j,k)
          enddo
        enddo
      endif
# endif
# ifndef NS_PERIODIC
      if (SOUTHERN_EDGE) then
        do k=0,N
          do i=istr,iend
            W(i,jstr-1,k)=W(i,jstr,k)
          enddo
        enddo
      endif
      if (NORTHERN_EDGE) then
        do k=0,N
          do i=istr,iend
            W(i,jend+1,k)=W(i,jend,k)
          enddo
        enddo
      endif
#  ifndef EW_PERIODIC
      if (WESTERN_EDGE .and. SOUTHERN_EDGE) then
        do k=0,N
          W(istr-1,jstr-1,k)=W(istr,jstr,k)
        enddo
      endif
      if (WESTERN_EDGE .and. NORTHERN_EDGE) then
        do k=0,N
          W(istr-1,jend+1,k)=W(istr,jend,k)
        enddo
      endif
      if (EASTERN_EDGE .and. SOUTHERN_EDGE) then
        do k=0,N
          W(iend+1, jstr-1,k)=W(iend,jstr,k)
        enddo
      endif
      if (EASTERN_EDGE .and. NORTHERN_EDGE) then
        do k=0,N
          W(iend+1,jend+1,k)=W(iend,jend,k)
        enddo
      endif
#  endif
# endif
# ifdef EXCHANGE
      call exchange_tile (istr,iend,jstr,jend, W,N+1)
# endif
      return
      end
#else
      subroutine omega_empty
      end
#endif /* SOLVE3D */
 
