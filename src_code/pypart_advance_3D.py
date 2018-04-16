######################################
#pypart_advance_3D.py
'''
PYTHON EXECUTABLE TO RUN PARTICLE TRACKING
ANALYSIS ON ROMS OUTPUT
BASED ON JEROEN'S CODE AS OF 10/23/2016
   
'''
#DANIEL DAUHAJRE UCLA MARCH 2018
######################################

############################################
#IMPORT MODULES
import os
import sys
sys.path.append('./src_code/f2py_code/')
import scipy.io as sio
import numpy as np
from netCDF4 import Dataset as netcdf
from utils import *
import part  as PY_FORT
import pypart_funcs as PF
import ROMS_solutions_paths as ROMS_out_paths
import pickle as pickle
import ROMS_depths as RD
import particles_numba as PN
import time as clock
import spline as fspline
##############################################

                            ###############################################
                            #              SETUP FILES/ARRAYS
                            ###############################################

#############################################
# LOAD INPUT DICTIONARY TO INITIALIZE ALL 
# PARTICLE RELEASE PARAMETERS
###########################################
path_in = './part_in/'
path,dirs,file_names = os.walk(path_in).next()
init_dict = PF.load_pickle_file(path_in + file_names[0])
##########################################################


###########################################
# OBTIAN ROMS OUTPUT PATHS 
###########################################
print 'Setting ROMS solutions paths for: ' + init_dict['ROMS_ID']
ROMS_obj = ROMS_out_paths.ROMS_run(init_dict['ROMS_ID'])
ROMS_obj.set_paths()
##########################################################

#####################################
#SET OUTPUT AND GRID FILE PATHS/NAMES
####################################
grd_name = ROMS_obj.path_grid + ROMS_obj.grid_name
dat_name = ROMS_obj.path_output +   ROMS_obj.out_base_name
uv_name  = ROMS_obj.path_ts_uv + 'ts_' + ROMS_obj.out_base_name #from mtspline
omega_name = ROMS_obj.path_omega  + 'ts_' + ROMS_obj.omega_name  # from mtspline 
zs_name = ROMS_obj.path_zs +  ROMS_obj.zs_name  # from mtspline 
################################################################


#################################
# READ IN GRID to obtain dx, dy
##################################
nc_grd = netcdf(grd_name,'r')
pm = nc_grd.variables['pm'][:]
pn = nc_grd.variables['pn'][:]
dx  = 1./pm.T
dy  = 1./pn.T
[Lx, Ly] = dx.shape
################################################################


#############################################################
# Make list of filenames with frame numbers of velocity data
#############################################################
t_interp    = init_dict['t_interp']
omega_online = init_dict['omega_online']
tim_opt = init_dict['tim_opt']
fnum = init_dict['fnum']
sub  = init_dict['sub']
frpf =  init_dict['frpf']
nfr  = init_dict['nfr']
dfr  = init_dict['dfr']
fr   = init_dict['fr']
ifr  =   fr;
locfr  = []
files_roms = []
files_uv   = []
files_omega = []
files_zs = []
for it in range(nfr+fr+2):
   locfr.append(ifr)
   try:
     files_roms.append(dat_name + '%04d' %fnum + '.nc')
   except:
       pass
   files_uv.append(uv_name + '%04d' %fnum + '.nc')
   files_omega.append(omega_name +  '%04d' %fnum + '.nc')
   files_zs.append(zs_name +  '%04d' %fnum + '.nc')
   ifr = ifr + dfr 
   if (ifr > frpf-1):
      ifr = 0
      add_on = frpf
      if dfr > frpf:
	 add_on = (frpf * (dfr / frpf))
      fnum = fnum + add_on 
####################################################################

######################################
#OBTAIN INITIAL VELOCITIES AND TIMES
######################################
try:
   nc_roms = netcdf(files_roms[0], 'r')
except:
    pass
nc_uv   = netcdf(files_uv[0], 'r')
if omega_online == False:
   nc_W = netcdf(files_omega[0], 'r')
lfr = locfr[0] 
print '###################################'
print ''
print 'fname: ' ,files_uv[0],' time step: ',lfr
################################################################

###############################
# GET INITIAL DEPTH SPACING
################################
if z_online:
   z_r, z_w = RD.get_zr_zw_tind(nc_roms,nc_grd,lfr,[0,Ly,0,Lx])
else:
   z_r, z_w = PF.get_zs_nc(netcdf(files_zs[0],'r'),lfr, timing=tim_opt)
dz_temp = np.swapaxes(z_w[1:,:,:] - z_w[:-1,:,:],-1,0)
[nx,ny,nz] = dz_temp.shape
dz = np.zeros([nx,ny,nz+2])
#EXTRAPOLATE rho-vertical variable, dz
dz[:,:,1:-1] = np.copy(dz_temp)
dz[:,:,0] = dz_temp[:,:,0]
dz[:,:,-1] = dz_temp[:,:,-1]
################################################
# GRAB APPROPRIATE SURFACE FIELDS
# ACCORDING TO RUN AND NETCDF STORAGE CONVENTION
##################################################
if omega_online:
    if t_interp=='spline':
       u0,v0,W0,du0,dv0,dW0 = PF.get_uvW_nc(nc_uv,lfr,timing=tim_opt,z_r=z_r,z_w=z_w,pm=pm,pn=pn,omega_online=omega_online,t_interp=t_interp)
    if t_interp=='linear':
       u0,v0,W0 = PF.get_uvW_nc(nc_uv,lfr,timing=tim_opt,z_r=z_r,z_w=z_w,pm=pm,pn=pn,omega_online=omega_online,t_interp=t_interp)

else:
   if t_interp == 'spline':
      u0,v0,W0,du0,dv0,dW0 = PF.get_uvW_nc(nc_uv,lfr,timing=tim_opt, nc_omega = nc_W,omega_online=omega_online,t_interp=t_interp)
   if t_interp == 'linear':
      u0,v0,W0 = PF.get_uvW_nc(nc_uv,lfr,timing=tim_opt, nc_omega = nc_W,omega_online=omega_online,t_interp=t_interp)


tim0 = nc_uv.variables['ocean_time'][lfr];

#ACCESS SHAPES
(unx,uny,nz_pad) = u0.shape
(vnx, vny,nz_pad) = v0.shape
(nx,ny,nzW) = W0.shape
nz = nzW - 1
try:
   nc_roms = netcdf(files_roms[1],'r')
except:
    pass
nc_uv = netcdf(files_uv[1],'r')
if omega_online==False:
   nc_W = netcdf(files_omega[1], 'r')
lfr = locfr[1]
print 'fname: ' ,files_uv[1],' time step: ', lfr

if omega_online:
   if z_online:
      z_r, z_w = RD.get_zr_zw_tind(nc_roms,nc_grd,lfr,[0,Ly,0,Lx])
   else:
      z_r, z_w = PF.get_zs_nc(netcdf(files_zs[1],'r'),lfr, timing=tim_opt)
   if t_interp == 'spline':
       u1,v1,W1,du1,dv1,dW1 = PF.get_uvW_nc(nc_uv,lfr,timing=tim_opt,z_r=z_r,z_w=z_w,pm=pm,pn=pn,omega_online=omega_online,t_interp=t_interp)
   if t_interp == 'linear':
       u1,v1,W1 = PF.get_uvW_nc(nc_uv,lfr,timing=tim_opt,z_r=z_r,z_w=z_w,pm=pm,pn=pn,omega_online=omega_online,t_interp=t_interp)
else:
   if t_interp == 'spline':
      u1,v1,W1,du1,dv1,dW1 = PF.get_uvW_nc(nc_uv,lfr,timing=tim_opt, nc_omega=nc_W,omega_online=omega_online,t_interp=t_interp)
   if t_interp == 'linear':
      u1,v1,W1 = PF.get_uvW_nc(nc_uv,lfr,timing=tim_opt, nc_omega=nc_W,omega_online=omega_online,t_interp=t_interp)


tim1 = nc_uv.variables['ocean_time'][lfr];
####################################################################

######################################
# SET UP TIME VARIABLES
######################################
print 'tim0: ' , tim0
print 'tim1: ', tim1
print 'tim1 - tim0 = ', tim1-tim0
print ''
print '#####################################'


time = tim0
time_start = time
idx = 0 
delt  = (tim1-tim0) 
delti = 1./delt
dt = delt/sub
nit = nfr*sub 
#GENERATE TIME VECTOR OF PARTICLE TIME-STEPS
tim_end = netcdf(files_uv[-1],'r').variables['ocean_time'][locfr[-1]]
part_time = np.arange(tim0, tim_end + dt,dt)
######################################################################################


##################################
# Initialize particle arrays
##################################
px = init_dict['px']
py = init_dict['py']
pz = init_dict['pz']
n_releases = init_dict['n_releases']
############################
'''
# temp arrays are to 
# be used in time-stepping
# and will have nans
# for particles that
# have not been released yet
'''
##############################
px_temp = np.copy(px)
py_temp = np.copy(py)
pz_temp = np.copy(pz)

nq = px.shape[0]

dpx = np.zeros(nq)
dpy = np.zeros(nq)
dpz = np.zeros(nq)
####################################################################


##############################################
#    CREATE SEQUENCE OF NETCDF OUTPUT FILES
#############################################
print ''
PF.change_dir_py('part_out')
nc_out_names = PF.setup_out_nc(init_dict['run_name'], init_dict['fnum']+init_dict['fr'], nfr + init_dict['fnum']+init_dict['fr'], frpf, nq,veloc_adv=init_dict['veloc_adv'])
os.chdir('../')
print 'Moving out of directory: ' + './part_out/'
print ''
####################################################################



#######################################
# NaN out particles that are not released
# at initial time
########################################
for n_p in range(1,nq,n_releases):
    px_temp[n_p:n_p + n_releases-1] = np.nan
    py_temp[n_p:n_p + n_releases-1] = np.nan
    pz_temp[n_p:n_p + n_releases-1] = np.nan
####################################################################

######################################
# WRITE INITIAL PARTICLES INTO NETCDF
#######################################
nc_ind = 0
nc_fill = netcdf(os.getcwd() + '/part_out/' + nc_out_names[nc_ind],'r+')
fr = 1
idx_nc = 0
PF.write_data_nc_3d(nc_fill,px_temp,py_temp,pz_temp, time,fr,idx)
####################################################################



############################################
# OBTAIN TIME-INDICES TO ADD NEW PARTICLES
##########################################
tinds_release = range(0,nfr,nfr/n_releases)[0:-1]
npart_release = nq / n_releases
s_num = 2 
####################################################################



         #####################################################################
         #		 	PARTICLE TIME STEPPING 
         ######################################################################
print '#######################################################'
print ''
print '			TIME-STEPPING PARTICLES '
print ''
print '#######################################################'

for it in range(nit):    # particle time steps
    if t_interp=='linear':
       fct = (time+0.5*dt-tim0)*delti
       u = (1-fct)*u0 + fct*u1
       v = (1-fct)*v0 + fct*v1
       W = (1-fct)*W0 + fct*W1
    ####################################
    #SPLINE INTERPOLATION OF VELOCITIES 
    # TO PARTICLE TIME-STEP
    ###################################
    if t_interp == 'spline':
       if tim_opt:
          start_time = clock.time()
       ptime = (part_time[it] - tim0) / (tim1 - tim0)
       u = fspline.spline(u0,u1,du0,du1,ptime)
       v     = fspline.spline(v0,v1,dv0,dv1,ptime)
       W     = fspline.spline(W0,W1,dW0,dW1,ptime)
       if tim_opt:
          print 'spline time: ' + str(clock.time() - start_time)
    ############################
    #ADVANCE PARTICLES
    ############################
    if it == 0:
       scheme_first=True
    else:
       scheme_first=False
    if tim_opt:
        start_time = clock.time()
    PN.advance3d(px_temp,py_temp,pz_temp,dpx,dpy,dpz,u,v,W,dx,dy,dz,dt,nx,ny,nz,0,0,first=scheme_first)
    if tim_opt:
       print 'advance time: ' + str(clock.time() - start_time)
    ################# 
    # CULL PARTICLES
    ##################
    [px_temp,py_temp,pz_temp] = cull_3d(px_temp,py_temp,pz_temp,nx,ny,nz)
  

    if t_interp == 'linear':
       time = time + dt
       bool_write = (time+0.5*dt)>tim1
       time_write = time
    else:
       bool_write = part_time[it] >=tim1
       time_write = part_time[it]
    ######################################
 

    ###################################
    # WRITE PARTICLES AT ROMS FRAME
    ###################################
    if bool_write:
      print 'veloc_adv = ' + init_dict['veloc_adv']
      print 'part_time = ', time
      print 'ocean_time = ', tim1
      print 'writing frame: ' ,idx+1, 'time: ',time
      print 'moving: ', nq,' particles'
      idx = idx+1
      idx_nc = idx_nc + 1
      #######################################
      # SEED NEW PARTICLES IF TIME TO SEED
      ########################################
      if idx in tinds_release and idx>0:
         print 'SEEDING PARTICLEs, seed # ' + str(s_num)
	 s_num+=1
         tind_r = tinds_release.index(idx)
         for n_p in range(tind_r,nq,n_releases):
	     px_temp[n_p] = px[n_p]
             py_temp[n_p] = py[n_p]
             pz_temp[n_p] = pz[n_p] 
      
      fr = fr+1
      if idx_nc < frpf:
         PF.write_data_nc_3d(nc_fill,px_temp,py_temp,pz_temp,time_write,fr,idx_nc)
      else:
          nc_fill.close()
          nc_ind+=1
          idx_nc = 0
          nc_fill = netcdf(os.getcwd() + '/part_out/' + nc_out_names[nc_ind],'r+')
          PF.write_data_nc_3d(nc_fill,px_temp,py_temp,pz_temp,time_write,fr,idx_nc)

      print '   Wrote particle data to: ' + nc_out_names[nc_ind] + ' , idx_nc = ' + str(idx_nc)

      #############################################
      # UPDATE VELOCITY FIELDS FOR NEXT ROMS FRAME
      ############################################
      u0 = u1
      v0 = v1
      w0 = W1
      if t_interp == 'spline':
         du0 = du1
         dv0 = dv1
         dW0 = dW1
      tim0 = tim1
     
      ############################
      #READ IN NEW VELOCITY DATA 
      #############################
      try:
         nc_roms = netcdf(files_roms[fr],'r')
      except:
          pass
      nc_uv = netcdf(files_uv[fr],'r')
      if omega_online == False:
         nc_W = netcdf(files_omega[fr],'r')
      
      print ''
      lfr = locfr[fr] 
      print 'file: ', files_uv[fr] + '  time step: ', locfr[fr]
      #UPDATE VERTICAL SPACING
      if tim_opt:
         start_time = clock.time()
      if z_online:
         z_r, z_w = RD.get_zr_zw_tind(nc_roms,nc_grd,lfr,[0,Ly,0,Lx])
         if tim_opt:
          print 'z calc time: ' + str(clock.time() - start_time)
      else:
         z_r, z_w = PF.get_zs_nc(netcdf(files_zs[fr],'r'),lfr, timing=tim_opt)

      dz_temp = np.swapaxes(z_w[1:,:,:] - z_w[:-1,:,:],-1,0)
      dz = np.zeros([nx,ny,nz+2])
      #EXTRAPOLATE rho-vertical variable, dz
      dz[:,:,1:-1] = np.copy(dz_temp)
      dz[:,:,0] = dz_temp[:,:,0]
      dz[:,:,-1] = dz_temp[:,:,-1]

      tim1 = nc_uv.variables['ocean_time'][lfr];
  
      if omega_online:
         if t_interp == 'spline':
            u1,v1,W1,du1,dv1,dW1 = PF.get_uvW_nc(nc_uv,lfr,timing=tim_opt,z_r=z_r,z_w=z_w,pm=pm,pn=pn,omega_online=omega_online,t_interp=t_interp)
         if t_interp == 'linear':
            u1,v1,W1 = PF.get_uvW_nc(nc_uv,lfr,timing=tim_opt,z_r=z_r,z_w=z_w,pm=pm,pn=pn,omega_online=omega_online,t_interp=t_interp)
      else:
         if t_interp=='spline':
            u1,v1,W1,du1,dv1,dW1 = PF.get_uvW_nc(nc_uv,lfr,timing=tim_opt, nc_omega = nc_W,omega_online=omega_online,t_interp=t_interp)
         if t_interp=='linear':
            u1,v1,W1 = PF.get_uvW_nc(nc_uv,lfr,timing=tim_opt, nc_omega = nc_W,omega_online=omega_online,t_interp=t_interp)
   
      #nc_roms.close()
      #nc_uv.close()
      #nc_W.close()



#######################################
# CLOSE OUTPUT NETCDF FILE
######################################
nc_fill.close()

