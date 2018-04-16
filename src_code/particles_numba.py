from numba import autojit,prange,jit,njit, vectorize
import numpy as np
import timeit
import os
import multiprocessing as MP
import numba as numba
import spline as fspline
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

#@njit(parallel=True)
def advance3d(px,py,pz,dpx,dpy,dpz,u,v,w,dxi,dyi,dz,dt,imax,jmax,kmax,i0=0,j0=0,first=False):
    npart=len(px)
    #for ip in range(npart):
    for ip in prange(npart):
       # Check 1 offset for i0 etc
       if (~np.isnan(px[ip]*py[ip]*pz[ip])):    
           iu = int(np.floor(px[ip]) - i0)
           jv = int(np.floor(py[ip]) - j0)
           kw = int(np.floor(pz[ip]))

           i = int(np.floor(px[ip] + 0.5))
           j = int(np.floor(py[ip] + 0.5))
           k = int(np.floor(pz[ip] + 0.5))
 
           xu = px[ip] - iu
           yv = py[ip] - jv
           zw = pz[ip] - kw
           
           x = px[ip] - i + 0.5
           y = py[ip] - j + 0.5
           z = pz[ip] - k + 0.5

          ###############################
            # INTERPOLATION
           ###############################
          
           #ONLY INTERPOLATING INBOUNDS, NO IF-STATEMENTS
           pdz =  int_lin_3d(dz[i:i+2,j:j+2,k:k+2],x,y,z)
           pdxi = int_lin_2d(dxi[i:i+2,j:j+2],x,y)
           pdyi = int_lin_2d(dyi[i:i+2,j:j+2],x,y)

           ##########################
           #VELOCITY INTERPOLATIONS
           ########################
           pu = 0
           pv = 0
           pw = 0

           pu = int_lin_3d(u[iu:iu+2,j:j+2,k:k+2],xu,y,z)
           pv = int_lin_3d(v[i:i+2,jv:jv+2,k:k+2],x,yv,z)
           pw = int_lin_3d(w[i:i+2,j:j+2,kw:kw+2],x,y,zw) 
           
           ##############################################
           #Calculate the fraction of index displacement
           ##############################################
           dpxNow = dt*pu/pdxi
           dpyNow = dt*pv/pdyi
           dpzNow = dt*pw/pdz
           #Third order Adams Bashforth
           if first:
              px[ip] =  px[ip] + dpxNow
              py[ip] =  py[ip] + dpyNow
              pz[ip] =  pz[ip] + dpzNow
           else:
              px[ip] =  px[ip] + 1.5*dpxNow - 0.5*dpx[ip]
              py[ip] =  py[ip] + 1.5*dpyNow - 0.5*dpy[ip]
              pz[ip] =  pz[ip] + 1.5*dpzNow - 0.5*dpz[ip]

           dpx[ip]=dpxNow       
           dpy[ip]=dpyNow       
           dpz[ip]=dpzNow       
 
@njit(parallel=True)
def advance2d(px,py,dpx,dpy,u,v,dxi,dyi,dt,imax,jmax,i0=0,j0=0,first=False):
    npart=len(px)
    for ip in prange(npart):
       # Check 1 offset for i0 etc
       if (~np.isnan(px[ip]*py[ip])):    
           i = int(np.floor(px[ip])-i0)
           j = int(np.floor(py[ip])-j0)
           iu= int(np.floor(px[ip]-0.5)-i0)
           jv= int(np.floor(py[ip]-0.5)-j0)
           

           x =  px[ip]-i0-i
           y =  py[ip]-j0-j

           xu=  px[ip]+0.5-i0-iu
           yv=  py[ip]+0.5-j0-jv
          

           ###############################
           # INTERPOLATION
           '''
           Standard linear indexing:  i-1:i+1 (2 pts)
           Back linear indexing:      i:i+2   (2 pts)

           Standard cubic indexing:   i-1:i+3 (3 pts)
           Back cubic indexing:       i-2:i+2 (3 pts)

           Based on where particle is in i,j,k, will utilize one of these
           interpolations with boundary handling implicit in choices
           '''
           ###############################
           bool_lin_std  = (i-1>=0) & (i+1<=imax) & (j-1>=0) & (j+1<=jmax) 
           bool_lin_bck  = ((i>=0) & (i-2<=imax)) & ((j>=0) & (j-2<=jmax))
           bool_cubic_std = (i-1>=0) & (i+3<=imax) & (j-1>=0) & (j+3<=jmax) 
           bool_cubic_bck = ((i-2>=0) & (i+2<=imax)) & ((j-2>=0) & (j+2<=jmax)) 

           #print str(i) + ' ' + str(j) + ' ' + str(k)
           if bool_cubic_std:
              pdxi = int_cubic_2d(dxi[i-1:i+3,j-1:j+3],x,y)
              pdyi = int_cubic_2d(dyi[i-1:i+3,j-1:j+3],x,y)
           elif bool_cubic_std:
              pdxi = int_cubic_2d(dxi[i-2:i+2,j-2:j+2],x,y)
              pdyi = int_cubic_2d(dyi[i-2:i+2,j-2:j+2],x,y)
           elif bool_lin_std:
              pdxi = int_lin_2d(dxi[i-1:i+1,j-1:j+1],x,y)
              pdyi = int_lin_2d(dyi[i-1:i+1,j-1:j+1],x,y)
           elif bool_lin_bck:
              pdxi = int_lin_2d(dxi[i:i+2,j:j+2],x,y)
              pdyi = int_lin_2d(dyi[i:i+2,j:j+2],x,y)
           else:
              pdxi= dxi[i,j]
              pdyi= dyi[i,j]
       

           ##########################
           #VELOCITY INTERPOLATIONS
           ########################
           pu = 0.
           pv = 0.
           imax_u = imax - 1
           jmax_v = jmax - 1
                  
           bool_lin_std_u  = (iu-1>=0) & (iu+1<=imax_u) & (j-1>=0) & (j+1<=jmax) 
           bool_lin_bck_u  = ((iu>=0) & (iu-2<=imax_u)) & ((j>=0) & (j-2<=jmax))
           bool_cubic_std_u = (iu-1>=0) & (iu+3<=imax_u) & (j-1>=0) & (j+3<=jmax) 
           bool_cubic_bck_u = ((iu-2>=0) & (iu+2<=imax_u)) & ((j-2>=0) & (j+2<=jmax))
 
           #u-velocity
           if bool_cubic_std_u:
              pu = int_cubic_2d(u[iu-1:iu+3,j-1:j+3],xu,y)
           elif bool_cubic_bck_u:
              pu = int_cubic_2d(u[iu-2:iu+2,j-2:j+2],xu,y)
           elif bool_lin_std_u:
              pu = int_lin_2d(u[iu-1:iu+1,j-1:j+1],xu,y)
           elif bool_lin_bck_u:
              pu = int_lin_2d(u[iu:iu+2,j:j+2],xu,y)
           else:
              pu = u[iu,j]

           #v-velocity
           bool_lin_std_v  = (i-1>=0) & (i+1<=imax) & (jv-1>=0) & (jv+1<=jmax_v)
           bool_lin_bck_v  = ((i>=0) & (i-2<=imax)) & ((jv>=0) & (jv-2<=jmax_v))
           bool_cubic_std_v = (i-1>=0) & (i+3<=imax) & (jv-1>=0) & (jv+3<=jmax_v)
           bool_cubic_bck_v = ((i-2>=0) & (i+2<=imax)) & ((jv-2>=0) & (jv+2<=jmax_v))
           if bool_cubic_std_v:
              pv = int_cubic_2d(v[i-1:i+3,jv-1:jv+3],x,yv)
           elif bool_cubic_bck_v:
              pv = int_cubic_2d(v[i-2:i+2,jv-2:jv+2],x,yv)
           elif bool_lin_std_v:
              pv = int_lin_2d(v[i-1:i+1,jv-1:jv+1],x,yv)
           elif bool_lin_bck_v:
              pv = int_lin_2d(v[i:i+2,jv:jv+2],x,yv)
           else:
              pv = v[i,jv]

           ##############################################
           #Calculate the fraction of index displacement
           ##############################################
           dpxNow = dt*pu/pdxi
           dpyNow = dt*pv/pdyi
           #Third order Adams Bashforth
           if first:
              px[ip] =  px[ip] + dpxNow
              py[ip] =  py[ip] + dpyNow
           else:
              px[ip] =  px[ip] + 1.5*dpxNow - 0.5*dpx[ip]
              py[ip] =  py[ip] + 1.5*dpyNow - 0.5*dpy[ip]
           
           dpx[ip]=dpxNow       
           dpy[ip]=dpyNow       
          
#----------------------------------------------------------------------------------------------

#@njit
def int_lin_3d(f,x,y,z):
       # 3D Linear interpolation
       wt=np.empty((2,2,2))
       wt[0,0,0] = (1-x)*(1-y)*(1-z)
       wt[1,0,0] =    x *(1-y)*(1-z)
       wt[0,1,0] = (1-x)*   y *(1-z)
       wt[1,1,0] =    x *   y *(1-z)
       wt[0,0,1] = (1-x)*(1-y)*   z 
       wt[1,0,1] =    x *(1-y)*   z 
       wt[0,1,1] = (1-x)*   y *   z 
       wt[1,1,1] =    x *   y *   z 
       for a in range(2):
           for b in range(2):
               for c in range(2):
                   if wt[a,b,c] >1:
                       print 'ERROR WITH WEIGHTS'
                       return np.nan

       return(np.sum(f*wt))

#----------------------------------------------------------------------------------------------
#@njit
def int_lin_2d(f,x,y):
    # 2D Linear interpolation
    wt=np.empty((2,2))
    wt[0,0] = (1-x)*(1-y)
    wt[1,0] =    x *(1-y)
    wt[0,1] = (1-x)*   y 
    wt[1,1] =    x *   y  
    for a in range(2):
        for b in range(2):
            if wt[a,b] >1:
               print 'ERROR WITH WEIGHTS'
               return np.nan

    return(np.sum(f*wt))

#----------------------------------------------------------------------------------------------

@njit
def int_cubic_2d(f,x,y):
       # 2D Cubic interpolation
        fy=np.empty(4)

        fy =     y*y*y* (-1./6*f[:,0] + 0.5*f[:,1] - 0.5*f[:,2] + 1./6*f[:,3])
        fy = fy+ y*y*   (0.5* f[:,0] -     f[:,1] + 0.5*f[:,2])
        fy = fy+ y*     ( -1./3*f[:,0] - 0.5*f[:,1] +     f[:,2] - 1./6*f[:,3])
        fy = fy+                     f[:,1]

        fi =     x*x*x* (-1./6*fy[0] + 0.5*fy[1] - 0.5*fy[2] + 1./6*fy[3])
        fi = fi+ x*x*   ( 0.5* fy[0] -     fy[1] + 0.5*fy[2])
        fi = fi+ x*     (-1./3*fy[0] - 0.5*fy[1] +     fy[2] - 1./6*fy[3])
        fi = fi+                      fy[1] 

        return fi

@njit
def int_cubic_3d(f,x,y,z):
       # 3D Cubic interpolation
        fz=np.empty((4,4))  
        fy=np.empty(4)

        fz =     z*z*z* (-1./6*f[:,:,0] + 0.5*f[:,:,1] - 0.5*f[:,:,2] + 1./6*f[:,:,3])
        fz = fz+ z*z*   (0.5* f[:,:,0] -     f[:,:,1] + 0.5*f[:,:,2])
        fz = fz+ z*     (-1./3*f[:,:,0] - 0.5*f[:,:,1] +     f[:,:,2] - 1./6*f[:,:,3])
        fz = fz+                    f[:,:,1]

        fy =     y*y*y* (-1./6*fz[:,0] + 0.5*fz[:,1] - 0.5*fz[:,2] + 1./6*fz[:,3])
        fy = fy+ y*y*   (0.5* fz[:,0] -     fz[:,1] + 0.5*fz[:,2])
        fy = fy+ y*     ( -1./3*fz[:,0] - 0.5*fz[:,1] +     fz[:,2] - 1./6*fz[:,3])
        fy = fy+                     fz[:,1]

        fi =     x*x*x* (-1./6*fy[0] + 0.5*fy[1] - 0.5*fy[2] + 1./6*fy[3])
        fi = fi+ x*x*   ( 0.5* fy[0] -     fy[1] + 0.5*fy[2])
        fi = fi+ x*     (-1./3*fy[0] - 0.5*fy[1] +     fy[2] - 1./6*fy[3])
        fi = fi+                      fy[1] 

        return fi

@vectorize
def int_splines(p0,p1,m0,m1,t):
    '''
    Cubic-Hermite Spline interpolation
    p0,p1 --> value of variable at t=0 and t=1
    m0,m1 --> tangents of variable at t=0 and t=1
    t--> time to interpoalte to (t in [0,1])
    '''
    return ((2*t**3 - 3*t**2 + 1) * p0)\
           +  ((t**3 - 2*t**2 + t) * m0)\
           +  ((-2*t**3 + 3*t**2)*p1)\
           + ((t**3 - t**2) * m1)
    ########################################

                ###################################################
                #               OMEGA CALCULATION
                ##################################################
@njit
def calc_omega(u,v,z_r,z_w,pm,pn):
    '''
    Offline calculation of omega at a time-step,
    translated directly from ROMS code (omega.F)
    W = [Hz/(m*n)] * omega
    '''
    
    [N, Mm_rho, Lm_rho] = z_r.shape
    W = np.zeros((N+1,Mm_rho, Lm_rho))
    istr = 1
    iend = Lm_rho-1
    jstr = 1
    jend = Mm_rho -1
    Hz = np.zeros((N,Mm_rho,Lm_rho))
    for k in range(N):
        Hz[k,:,:] = z_w[k,:,:] - z_w[k-1,:,:]

    dn_u = np.zeros((Mm_rho, Lm_rho))
    FlxU = np.zeros((N,Mm_rho,Lm_rho-1)) #in fortran (0:Mm+1,1:Lm+1,N)
    for k in prange(N):
        for j in range(jstr,jend):
            for i in range(istr,iend):
                i_u = i -1
                dn_u[j,i] = 2./(pn[j,i] + pn[j,i-1])
                FlxU[k,j,i_u] = 0.5 * (Hz[k,j,i] + Hz[k,j,i-1]) * dn_u[j,i] * u[k,j,i_u]

    dn_v = np.zeros((Mm_rho, Lm_rho))
    FlxV = np.zeros((N,Mm_rho-1,Lm_rho)) #in fortran (1:Mm+1,0:Lm+1,N)
    for k in prange(N):
        for j in range(jstr,jend):
            j_v=j-1
            for i in range(istr,iend):
                dn_v[j,i] = 2./(pm[j,i] + pm[j-1,i])
                j_v = j - 1
                FlxV[k,j_v,i] = 0.5 * (Hz[k,j,i] + Hz[k,j-1,i]) * dn_v[j,i] * v[k,j_v,i]


    wrk = np.zeros((Lm_rho))
    for j in range(jstr,jend):
        j_v = j-1 
        for i in range(istr,iend):
            W[0,j,i] = 0.
        for k_w in range(1,N+1):
            for i in range(istr,iend):
                i_u = i -1
                k_r = k_w-1
                W[k_w,j,i] = W[k_w-1,j,i] - FlxU[k_r,j,i_u+1] + FlxU[k_r,j,i_u]\
                                          - FlxV[k_r,j_v+1,i] + FlxV[k_r,j_v,i]
      
        for i in prange(istr,iend):
            wrk[i] = W[N,j,i] / (z_w[N,j,i] - z_w[0,j,i])
            W[N,j,i] = 0.
    
        for k_w in prange(N-1,0,-1):
            for i in range(istr,iend):
                W[k_w,j,i] = W[k_w,j,i] - wrk[i] * (z_w[k_w,j,i] - z_w[0,j,i])

    # CONVERT FROM volume flux (m^3/s) to velocity (m/s)
    #for j in prange(jstr,jend):
    #    for i in range(istr,iend):
    for k_w in range(N+1):
        W[k_w,:,:] = W[k_w,:,:] * pm * pn



    #SET LATERAL BOUNDARIES
    for j in range(jstr,jend):
        for k in range(N+1):
            W[k,j,istr-1] = W[k,j,istr]
            W[k,j,iend] = W[k,j,iend-1]
    for i in range(istr,iend):
        for k in range(N+1):
            W[k,jstr-1,i] = W[k,jstr,i]
            W[k,jend,i] = W[k,jend-1,i]
    for k in range(N+1):
        W[k,jstr-1,istr-1] = W[k,jstr,istr]
        W[k,jend,istr-1] = W[k,jend-1,istr]
        W[k,jstr-1,iend] = W[k,jstr,iend-1]

    return W
    ######################################



