      subroutine spline(p0,p1,m0,m1,t,nx,ny,nz,var_int)
       implicit none 
!      import/export
       integer(kind=4)                 ,intent(in)   :: nx,ny,nz
       real(kind=8)   ,dimension(nx,ny,nz),intent(in):: p0,p1,m0,m1 
       real(kind=8)                       ,intent(in):: t
       real(kind=8)                       :: f1,f2,f3,f4
       real(kind=8)   ,dimension(nx,ny,nz)::        var_int
       integer(kind=4)::i,j,k
!      local
!f2py intent(in) p0,p1,m0,m1
!f2py intent(in) t
!f2py intent(out) var_int
       
       f1 = 2*t**3 - 3*t**2 + 1
       f2 = t**3 - 2*t**2 + t
       f3 =  -2*t**3 + 3*t**2
       f4 = t**3 - t**2
       do k=1,nz
          do j=1,ny
             do i=1,nx
                var_int(i,j,k)=f1* p0(i,j,k) + f2 *m0(i,j,k) 
     &                          + f3*p1(i,j,k) + f4 *m1(i,j,k)  
     
             enddo
          enddo
       enddo
      
       end 

