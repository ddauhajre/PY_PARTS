""" Set of tools to manipulate ROMS data """

def u2rho(u):
   import numpy as np
   """ interpolates from u-points to rho points """
   if u.ndim == 2:
      (ny,nx) = u.shape
      ur = np.zeros((ny,nx+1),order = 'F')
      ur[:,1:nx-1] = 0.5*(u[:,0:nx-2] + u[:,1:nx-1]);
      ur[:, 0] = u[:,   1];
      ur[:,nx] = u[:,nx-1];
   elif u.ndim ==3:
      (nz,ny,nx) = u.shape
      ur = np.zeros((nz,ny,nx+1),order = 'F')
      ur[:,:,1:nx-1] = 0.5*(u[:,:,0:nx-2] + u[:,:,1:nx-1]);
      ur[:,:, 0] = u[:,:,   1];
      ur[:,:,nx] = u[:,:,nx-1];
   else:
      ur = 0
      print 'error in u2rho'

   return ur


def v2rho(v):
   import numpy as np
   """ interpolates from v-points to rho points """

   if v.ndim == 2:
      (ny,nx) = v.shape
      vr = np.zeros((ny+1,nx),order = 'F')
      vr[1:ny-1,:] = 0.5*(v[0:ny-2,:] + v[1:ny-1,:]);
      vr[ 0,:] = v[   1,:];
      vr[ny,:] = v[ny-1,:];
   elif v.ndim ==3:
      (nz,ny,nx) = v.shape
      vr = np.zeros((nz,ny+1,nx),order = 'F')
      vr[:,1:ny-1,:] = 0.5*(v[:,0:ny-2,:] + v[:,1:ny-1,:]);
      vr[:, 0,:] = v[:,   1,:];
      vr[:,ny,:] = v[:,ny-1,:];
   else:
      vr = 0
      print 'error in v2rho'

   return vr

def subsection(px,py,dx,delt,nx,ny):
   import numpy as np
   """ Finds index subrange to move particles around"""

   velmax = 2;  # Could compute this if needed.
   offset = velmax*delt/dx
   i0 = int(np.floor(np.nanmin(px) - offset))
   i0 = max(1,i0)-1
   i1 = int(np.ceil(np.nanmax(px)  + offset))
   i1 = min(nx,i1)-1
   j0 = int(np.floor(np.nanmin(py) - offset))
   j0 = max(1,j0)-1
   j1 = int(np.ceil(np.nanmax(py)  + offset))
   j1 = min(ny,j1)-1

   return [i0,i1,j0,j1]

def cull(px,py,nx,ny):
   import numpy as np
   """ Set particle positions that are out of range to nan"""

   py[px<1]  = np.nan
   px[px<1]  = np.nan
   px[py<1]  = np.nan
   py[py<1]  = np.nan
   py[px>nx] = np.nan
   px[px>nx] = np.nan
   px[py>ny] = np.nan
   py[py>ny] = np.nan

   return [px,py]

def cull_3d(px,py,pz,nx,ny,nz):
   import numpy as np
   """ Set particle positions that are out of range to nan"""

   #X-boundaries
   px[px<0] = np.nan
   py[px<0] = np.nan
   pz[px<0] = np.nan
   px[px>nx] = np.nan
   py[px>nx] = np.nan
   pz[px>nx] = np.nan
   
   #Y-boundaries
   px[py<0] = np.nan
   py[py<0] = np.nan
   pz[py<0] = np.nan

   px[py>ny] = np.nan
   py[py>ny] = np.nan
   pz[py>ny] = np.nan

   return [px,py,pz]

  




