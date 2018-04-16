!----------------------------------------------------------------------------------------------
       subroutine seed(px,py,pz,npmx,mask,nx,ny,np)
       implicit none 
!      import/export
       integer(kind=4)                 ,intent(in)   :: nx,ny
       integer(kind=4)                 ,intent(in)   :: np
       integer(kind=4),dimension(ny,nx),intent(in)   :: mask
       real(kind=8)   ,dimension(np)   ,intent(inout):: px,py,pz
       integer(kind=4)                 ,intent(inout):: npmx
!      local
       integer(kind=4) :: ip,i,j,id,zlev
       integer(kind=4),allocatable,dimension(:,:) :: grid

!f2py intent(inout) px,py,pz
!f2py intent(in)    mask
!f2py intent(inout) npmx
!f2py intent(in)    nx,ny,nz
!f2py intent(in)    np

!      print *,'fo: ',npmx
       allocate(grid(nx,ny))
       grid = 0

       ip = npmx + 1
       px(ip) = 0.6*nx;
       py(ip) = 0.6*ny;
       pz(ip) = 0.0
       npmx = npmx + 1
*
       deallocate(grid)
       end
!----------------------------------------------------------------------------------------------
       subroutine part_rhs(px,py,pz,dpx,dpy,dpz,u,v,w,dxi,dyi,dz,dt,i0,
     &                     j0,nx,ny,nz,np)
       implicit none 
!      import/export
       integer(kind=4)                    ,intent(in)  :: nx,ny,nz
       integer(kind=4)                    ,intent(in)  :: np
       integer(kind=4)                    ,intent(in)  :: i0,j0
       real(kind=8)   ,dimension(0:nz+1,0:ny+1,0:nx)  ,intent(in) :: u
       real(kind=8)   ,dimension(0:nz+1,0:ny,0:nx+1)  ,intent(in) :: v
       real(kind=8)   ,dimension(0:nz,0:ny+1,0:nx+1)  ,intent(in) :: w
       real(kind=8)   ,dimension(0:nz+1,0:ny+1,0:nx+1),intent(in) :: dz
       real(kind=8)   ,dimension(np)                  ,intent(in) :: px,
     &                                                             py,pz
       real(kind=8)   ,dimension(np)                  ,intent(out):: dpx
     &                                                          ,dpy,dpz
       real(kind=8)   ,dimension(0:ny+1,0:nx+1)       ,intent(in) :: dxi
     &                                                              ,dyi
       real(kind=8)                                   ,intent(in) :: dt
!      local
       integer(kind=4):: ip,chunk,id
       real(kind=8)   :: pu,pv,pw
       real(kind=8)   :: pdxi,pdyi,pdz
       real(kind=8)   :: x,y,z,xu,yv,zw
       integer(kind=4):: i,j,k,iu,jv,kw

!f2py intent(in)  u,v,w
!f2py intent(in)  px,py,pz
!f2py intent(out) dpx,dpy,dpz
!f2py intent(in)  dz
!f2py intent(in)  dx,dy,dt
!f2py intent(in)  nx,ny,nz
!f2py intent(in)  i0,j0,k0
!f2py intent(in)  np

       chunk = np/4

!!!$OMP  PARALLEL DO DEFAULT(shared)
!!!$OMP& PRIVATE(i,j,k,ip,fcx,fcy,fcz,wt,pu,pv,pw)
!!!$OMP& SCHEDULE(static,chunk)
       do ip = 1,np
         !!! Check 1 offset for i0 etc
           i = floor(px(ip))-i0
           j = floor(py(ip))-j0
           k = floor(pz(ip))
           iu= floor(px(ip)-0.5)-i0
           jv= floor(py(ip)-0.5)-j0
           kw= floor(pz(ip)-0.5)

           x =  px(ip)-i0-i
           y =  py(ip)-j0-j
           z =  pz(ip)   -k

           xu=  px(ip)+0.5-i0-iu
           yv=  py(ip)+0.5-j0-jv
           zw=  pz(ip)+0.5   -kw

           call int_lin_3d(pu,u(k:k+1,j:j+1,iu:iu+1),xu,y,z)
           call int_lin_3d(pv,v(k:k+1,jv:jv+1,i:i+1),x,yv,z)
           call int_lin_3d(pw,w(kw:kw+1,j:j+1,i:i+1),x,y,zw)

           call int_lin_3d(pdz,dz(k:k+1,j:j+1,i:i+1),x,y,z)
           call int_lin_2d(pdxi,dxi(j:j+1,i:i+1),x,y)
           call int_lin_2d(pdyi,dyi(j:j+1,i:i+1),x,y)

!          call int_cubic_3d(pu,u(iu-1:iu+2,j-1:j+2,k-1:k+2),xu,y,z)
!          call int_cubic_3d(pv,v(i-1:i+2,jv-1:jv+2,k-1:k+2),x,yv,z)
!          call int_cubic_3d(pw,w(i-1:i+2,j-1:j+2,kw-1:kw+2),x,y,zw)

!          call int_weno_3d(pu,u(iu-1:iu+2,j-1:j+2,k-1:k+2),xu,y,z)
!          call int_weno_3d(pv,v(i-1:i+2,jv-1:jv+2,k-1:k+2),x,yv,z)
!          call int_weno_3d(pw,w(i-1:i+2,j-1:j+2,kw-1:kw+2),x,y,zw)

           dpx(ip) = dt*pu*pdxi
           dpy(ip) = dt*pv*pdyi
           dpz(ip) = dt*pw/pdz
       enddo
!!!$OMP  END PARALLEL DO 
!
       end
!----------------------------------------------------------------------------------------------
       subroutine advance_3d(px,py,pz,u,v,w,dxi,dyi,dz,dt,i0,j0,nx,ny,nz
     &                                                          ,np)
       implicit none 
       save ab,first
!      import/export
       integer(kind=4)                    ,intent(in)  :: nx,ny,nz
       integer(kind=4)                    ,intent(in)  :: np
       integer(kind=4)                    ,intent(in)  :: i0,j0
       real(kind=8)   ,dimension(0:nz+1,0:ny+1,0:nx)  ,intent(in) :: u
       real(kind=8)   ,dimension(0:nz+1,0:ny,0:nx+1)  ,intent(in) :: v
       real(kind=8)   ,dimension(0:nz,0:ny+1,0:nx+1)  ,intent(in) :: w
       real(kind=8)   ,dimension(0:nz+1,0:ny+1,0:nx+1),intent(in) :: dz
       real(kind=8)   ,dimension(0:ny+1,0:nx+1)       ,intent(in) :: dxi
     &                                                              ,dyi
       real(kind=8)   ,dimension(np)                  ,intent(inout):: 
     &                                                   px,py,pz
       real(kind=8)                                   ,intent(in) :: dt
!      local
       integer(kind=4),dimension(2)   :: ab
       real(kind=8)   ,allocatable,dimension(:,:) :: dpx,dpy,dpz
       integer(kind=4)                :: id
       real(kind=8)                   :: pi
       logical  :: first
       data  first/.true./
       data  ab /1,2/

!f2py intent(in)   u,v,w
!f2py intent(inout)px,py,pz
!f2py intent(in)   dxi,dyi,dz,dt
!f2py intent(in)   i0,j0
!f2py intent(in)   nx,ny,nz
!f2py intent(in)   np

       allocate(dpx(np,2))
       allocate(dpy(np,2))
       allocate(dpz(np,2))

       id = ab(1)
       call part_rhs(px,py,pz,dpx(:,id),dpy(:,id),dpz(:,id),u,v,w,dxi,
     &                  dyi,dz,dt,np,i0,j0,nx,ny,nz,np)
       
       if (first) then
!        ab(1) = 1
!        ab(2) = 2
         first = .false.
         dpx(:,ab(2)) = dpx(:,ab(1))
         dpy(:,ab(2)) = dpy(:,ab(1))
         dpz(:,ab(2)) = dpz(:,ab(1))
       endif

       px =  px + 1.5*dpx(:,ab(1)) - 0.5*dpx(:,ab(2))
       py =  py + 1.5*dpy(:,ab(1)) - 0.5*dpy(:,ab(2))
       pz =  pz + 1.5*dpz(:,ab(1)) - 0.5*dpz(:,ab(2))

       ab = mod(ab,2) + 1

       deallocate(dpx)
       deallocate(dpy)
       deallocate(dpz)

       end
!----------------------------------------------------------------------------------------------
       subroutine particlesort(px,py,pz,pi,is,nchx,nchy,nx,ny,np,nch)
       !!
       !! Sort particles into groups corresponding to Eulerian
       !! subdomains
       !!
       implicit none 
!      import/export
       integer(kind=8)                 ,intent(in)   :: nx,ny
       integer(kind=8)                 ,intent(in)   :: np,nch
       integer(kind=8)                 ,intent(in)   :: nchx,nchy
       real(kind=8)   ,dimension(np)   ,intent(inout):: px,py,pz
       integer(kind=4),dimension(np)   ,intent(inout):: pi
       integer(kind=8),dimension(0:nch+1),intent(inout):: is
!      local
       integer(kind=4) :: ip,ich
       integer(kind=4) :: dchx,dchy
       real(kind=8)    :: pxt,pyt,pzt
       integer(kind=4) :: pit,ips

!f2py intent(inout) px,py,pz,pi,is
!f2py intent(in)    nchx,nchy
!f2py intent(in)    nx,ny,np,nch

       dchx = ceiling(1.0*nx/nchx)
       dchy = ceiling(1.0*ny/nchy)

       is      = 1
       is(nch) = np
!      !!! testing
!      call random_number(px(1:is(nch)-1))
!      call random_number(py(1:is(nch)-1))
!      px = px*nx
!      py = py*ny
!      print *,minval(px(1:is(nch)-1)),maxval(px(1:is(nch)-1)),nx
       print *, 'TESTING'
       print *, is
       do ip = 1,is(nch)-1  !! loop over valid particles
          ich = floor(py(ip)/dchy)*nchx + floor(px(ip)/dchx)
          if (ich.lt.0) ich = nch
          print *, 'ip: ',ip, ' ich: ',ich, ' px(ip): ',px(ip)
          pi(ip) = ich
       enddo

       !! make sure the first particle is right-ish
       ich = floor(py(1)/dchy)*nchx + floor(px(1)/dchx)
       if (1.lt.is(ich)) then
          is(0:ich) = 1
          is(ich+1) = 2
       endif

       do ip = 1,is(nch)-1  !! loop over valid particles

          !! find chunk number ich by means of py and px
          !! chunks are counted zero based
          ich = floor(py(ip)/dchy)*nchx + floor(px(ip)/dchx)
          if (ich.lt.0) ich = nch   !! invalid particles have negative positions

          !! sort particles by means of pair-wise switching

          print *,'ip: ', ip,is(ich),is(ich+1),ich,pi(ip)
          if (ip.lt.is(ich)) then
!                 print *,ip, ' up switching with ', is(ich)
             !! example
             !!    is(1)         is(2)   is(3)
             !!    1 1 1 1 2 1 1 2 2 2 2 3 3 3 3
             !!            x ->x
             if (ich.eq.nch) then  !! invalid particle, put at the end
                     print *,'outside'
                is(nch) = is(nch) - 1
                ips = is(nch)
             else
                is(ich+1) = is(ich+1)-1
                ips = is(ich)
             endif
             pxt = px(ip)
             pyt = py(ip)
             pzt = pz(ip)
             pit = pi(ip)

             px(ip) = px(ips)
             py(ip) = py(ips)
             pz(ip) = pz(ips)
             pi(ip) = pi(ips)

             px(ips) = pxt
             py(ips) = pyt
             pz(ips) = pzt
             pi(ips) = pit
          elseif (ip.ge.is(ich+1)) then
             !! 1 1 1 1 1  2 2 2 2   3 1 3 3 3
             !!            x    <-     x
             do while (ip.ge.is(ich+1))
!            print *,ip, ' down switching with ', is(ich+1)
                pxt = px(ip)
                pyt = py(ip)
                pzt = pz(ip)
                pit = pi(ip)

                px(ip) = px(is(ich+1))
                py(ip) = py(is(ich+1))
                pz(ip) = pz(is(ich+1))
                pi(ip) = pi(is(ich+1))

                px(is(ich+1)) = pxt
                py(is(ich+1)) = pyt
                pz(is(ich+1)) = pzt
                pi(is(ich+1)) = pit

                is(ich+1) = is(ich+1)+1
                do while (is(ich+2).lt.is(ich+1))
                    is(ich+2) = is(ich+1)
                    ich = ich+1
                enddo

!               print *,is
!               print *,pi
                ich = floor(py(ip)/dchy)*nchx + floor(px(ip)/dchx)
                if (ich.lt.0) ich = nch
             enddo

          endif

       enddo

       print *, 'is: ', is

       do ip = 1,is(nch)-1  !! loop over relevant particles
          ich = floor(py(ip)/dchy)*nchx + floor(px(ip)/dchx)
          if (ich.lt.0) ich = nch
          print *, 'ip: ',ip, ' ich: ',ich
       enddo


       end
!----------------------------------------------------------------------------------------------
       subroutine int_lin_3d(fi,f,x,y,z)
       !! 3D Linear interpolation
       implicit none 
!      import/export
       integer(kind=8)              ,intent(out):: fi
       real(kind=4),dimension(2,2,2),intent(in) :: f
       real(kind=4)                 ,intent(in) :: x,y,z
!      local 
       real(kind=4),dimension(2,2,2) :: wt

       wt(1,1,1) = (1-x)*(1-y)*(1-z);
       wt(2,1,1) =    x *(1-y)*(1-z);
       wt(1,2,1) = (1-x)*   y *(1-z);
       wt(2,2,1) =    x *   y *(1-z);
       wt(1,1,2) = (1-x)*(1-y)*   z ;
       wt(2,1,2) =    x *(1-y)*   z ;
       wt(1,2,2) = (1-x)*   y *   z ;
       wt(2,2,2) =    x *   y *   z ;
       fi   = sum(f*wt)

       end
!----------------------------------------------------------------------------------------------
       subroutine int_lin_2d(fi,f,x,y)
       !! 2D Linear interpolation
       implicit none 
!      import/export
       integer(kind=8)            ,intent(out):: fi
       real(kind=4),dimension(2,2),intent(in) :: f
       real(kind=4)               ,intent(in) :: x,y
!      local 
       real(kind=4),dimension(2,2) :: wt
       wt(1,1) = (1-x)*(1-y)
       wt(2,1) =    x *(1-y)
       wt(1,2) = (1-x)*   y 
       wt(2,2) =    x *   y 

       fi   = sum(f*wt)

       end
!----------------------------------------------------------------------------------------------
       subroutine int_cubic_3d(fi,f,x,y,z)
       !! 3D Cubic interpolation
       implicit none 
!      import/export
       integer(kind=8)              ,intent(out):: fi
       real(kind=4),dimension(4,4,4),intent(in) :: f
       real(kind=4)                 ,intent(in) :: x,y,z
!      local 
       real(kind=4),dimension(4,4) :: fz,aaa4,aaa3,aaa2,aaa1
       real(kind=4),dimension(4)   :: fy,aa4,aa3,aa2,aa1
       real(kind=4)                :: a4,a3,a2,a1

        aaa4 = -1./6*f(:,:,1) + 0.5*f(:,:,2) - 0.5*f(:,:,3) 
     &                                          + 1./6*f(:,:,4)
        aaa3 =  0.5* f(:,:,1) -     f(:,:,2) + 0.5*f(:,:,3)
        aaa2 = -1./3*f(:,:,1) - 0.5*f(:,:,2) +     f(:,:,3) 
     &                                          - 1./6*f(:,:,4)
        aaa1 =                      f(:,:,2)
        fz= aaa4*z*z*z+ aaa3*z*z + aaa2*z + aaa1;

        aa4 = -1./6*fz(:,1) + 0.5*fz(:,2) - 0.5*fz(:,3) + 1./6*fz(:,4)
        aa3 =  0.5* fz(:,1) -     fz(:,2) + 0.5*fz(:,3)
        aa2 = -1./3*fz(:,1) - 0.5*fz(:,2) +     fz(:,3) - 1./6*fz(:,4)
        aa1 =                     fz(:,2)
        fy= aa4*y*y*y + aa3*y*y + aa2*y + aa1;

        a4 = -1./6*fy(1) + 0.5*fy(2) - 0.5*fy(3) + 1./6*fy(4) 
        a3 =  0.5* fy(1) -     fy(2) + 0.5*fy(3) 
        a2 = -1./3*fy(1) - 0.5*fy(2) +     fy(3) - 1./6*fy(4) 
        a1 =                   fy(2) 

        fi= a4*x*x*x+ a3*x*x + a2*x + a1;

       end
!----------------------------------------------------------------------------------------------
       subroutine calc_omega(u,v,dxi,dyi,dz,w,nx,ny,nz)
       !! 3D Cubic interpolation
       implicit none 
!      import/export
       integer(kind=4)                             ,intent(in) :: nx,ny
     &                                                            ,nz
       real(kind=8),dimension(0:nz+1,0:ny+1,0:nx+1),intent(in) :: u,v
       real(kind=8),dimension(nz,0:ny+1,0:nx+1)    ,intent(in) :: dz
       real(kind=8),dimension(0:ny+1,0:nx+1)       ,intent(in) :: dxi,
     &                                                          dyi
       real(kind=8),dimension(0:nz,0:ny+1,0:nx+1)  ,intent(out):: w
!      local 
       integer(kind=4)                         :: k
       real(kind=4),allocatable,dimension(:,:) :: Fu,Fv

!f2py intent(inout) px,py,pz
!f2py intent(in)    mask
!f2py intent(out)   w
!f2py intent(in)    u,v
!f2py intent(in)    dz
!f2py intent(in)    nx,ny,nz

       print *, nx,ny
       allocate(Fu(ny,0:nx))
       allocate(Fv(0:ny,nx))
       do k = 1,nz
          Fu = 0.25*u(k,1:ny,:)*(dz(k,1:ny,0:nx) + dz(k,1:ny,1:nx+1) )
     &          *(dyi(1:ny,0:nx) + dyi(1:ny,1:nx+1) )
          Fv = 0.25*v(k,:,1:nx)*(dz(k,0:ny,1:nx) + dz(k,1:ny+1,1:nx) )
     &          *(dxi(0:ny,1:nx) + dxi(1:ny+1,1:nx) )
          w(k,1:ny,1:nx) = w(k-1,1:ny,1:nx) - 
     c               ( Fu(1:ny,1:nx) - Fu(1:ny,0:nx) + 
     c                 Fv(1:ny,1:nx) - Fv(0:ny,1:nx) )*
     c                 dxi(1:ny,1:nx)*dyi(1:ny,1:nx)
       enddo

       deallocate(Fu)
       deallocate(Fv)

       end
!----------------------------------------------------------------------------------------------
