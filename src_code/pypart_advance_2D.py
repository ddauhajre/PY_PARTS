######################################
#pypart_main.py
'''
PYTHON EXECUTABLE TO RUN PARTICLE TRACKING
ANALYSIS ON ROMS OUTPUT

BASED ON JEROEN'S CODE AS OF 10/23/2016
   
'''
#DANIEL DAUHAJRE UCLA OCTOBER 2016
######################################

############################################
#IMPORT MODULES
import sys
sys.path.append('./src_code/f2py_code/')
import os
import scipy.io as sio
import numpy as np
from netCDF4 import Dataset as netcdf
from utils import *
from part  import *
import pypart_funcs as PF
import ROMS_solutions_paths as ROMS_out_paths
import pickle as pickle
import particles_numba as PN
import time as clock
import spline as fspline
##############################################

#############################################
# LOAD INPUT DICTIONARY TO INITIALIZE ALL 
# PARTICLE RELEASE PARAMETERS
###########################################
path_in = './part_in/'
path,dirs,file_names = os.walk(path_in).next()
init_dict = PF.load_pickle_file(path_in + file_names[0])

###########################################
# OBTIAN ROMS OUTPUT PATHS 
###########################################
#ROMS_ID = 'L4PV_nowec_P3_1min'
print 'Setting ROMS solutions paths for: ' + init_dict['ROMS_ID']
ROMS_obj = ROMS_out_paths.ROMS_run(init_dict['ROMS_ID'])
ROMS_obj.set_paths()

#####################################
#SET OUTPUT AND GRID FILE PATHS/NAMES
####################################
grd_name = ROMS_obj.path_grid + ROMS_obj.grid_name
dat_name = ROMS_obj.path_output + ROMS_obj.out_base_name
if t_interp == 'spline':
   uv_name = ROMS_obj.path_ts_uv + 'ts_' + ROMS_obj.out_base_name


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
fnum = init_dict['fnum']
sub  = init_dict['sub']
frpf =  init_dict['frpf']
nfr  = init_dict['nfr']
dfr  = init_dict['dfr']
fr   = init_dict['fr']
ifr  =   0;
locfr  = []
files = []


for it in range(nfr+fr+2):
   locfr.append(ifr)
   if t_interp == 'spline':
      files.append(uv_name + '%04d' %fnum + '.nc')
   else:
      files.append(dat_name + '%04d' %fnum + '.nc')
   ifr = ifr + dfr 
   if (ifr > frpf-1):
      ifr = 0
      add_on = frpf
      if dfr > frpf:
	 add_on = (frpf * (dfr / frpf))
      fnum = fnum + add_on 

######################################
#OBTAIN INITIAL VELOCITIES AND TIMES
######################################
nc = netcdf(files[0],'r')
lfr = locfr[0] 
print '###################################'
print ''
print 'fname: ' ,files[0],' time step: ',lfr


################################################
# LOAD VELOCITIES
##################################################
if t_interp=='spline':
   u0,v0,du0,dv0 = PF.get_uv_2D(nc,lfr,t_interp=t_interp)
else:
    u0,v0 = PF.get_uv_2D(nc,lfr)



tim0 = nc.variables['ocean_time'][lfr];
nc.close()

nc = netcdf(files[1],'r')
lfr = locfr[1]
print 'fname: ' ,files[1],' time step: ', lfr

if t_interp=='spline':
   u1,v1,du1,dv1 = PF.get_uv_2D(nc,lfr,t_interp=t_interp)
else:
    u1,v1 = PF.get_uv_2D(nc,lfr)
###############################################################

tim1 = nc.variables['ocean_time'][lfr];
nc.close()
print 'tim0: ' , tim0
print 'tim1: ', tim1
print 'tim1 - tim0 = ', tim1-tim0
print ''
print '#####################################'
#print 'times: ' ,tim1,tim0,tim1-tim0

(uny,unx) = u0.shape
(vny,vnx) = v0.shape
(ny,nx) = pm.shape
##############################################################



################################################
# CREATE TIME VARIABLES FOR ADVANCING PARTICLES
################################################
time = tim0
time_start = time
idx = 0 
delt  = (tim1-tim0) 
delti = 1./delt
dt = delt/sub
nit = nfr*sub
tim_end = netcdf(files[-1],'r').variables['ocean_time'][locfr[-1]]
part_time = np.arange(tim0,tim_end+dt,dt)
##################################
# Initialize particle arrays
##################################
px = init_dict['px']
py = init_dict['py']
n_releases = init_dict['n_releases']
############################
'''
 temp arrays are to 
 be used in time-stepping
 and will have nans
 for particles that
 have not been released yet
'''
##############################
px_temp = np.copy(px)
py_temp = np.copy(py)
nq = px.shape[0]
dpx = np.zeros(nq)
dpy = np.zeros(nq)
############################

##############################################
#    CREATE SEQUENCE OF NETCDF OUTPUT FILES
#############################################
print ''
PF.change_dir_py('part_out')
nc_out_names = PF.setup_out_nc(init_dict['run_name'], init_dict['fnum']+init_dict['fr'], nfr + init_dict['fnum']+init_dict['fr'], frpf, nq,veloc_adv=init_dict['veloc_adv'])
os.chdir('../')
print 'Moving out of directory: ' + './part_out/'
print ''
###################################################

#######################################
# NaN out particles that are not released
# at initial time
########################################
for n_p in range(1,nq,n_releases):
    px_temp[n_p:n_p + n_releases-1] = np.nan
    py_temp[n_p:n_p + n_releases-1] = np.nan
###################################################

######################################
# WRITE INITIAL PARTICLES INTO NETCDF
#######################################
nc_ind = 0
nc_fill = netcdf(os.getcwd() + '/part_out/' + nc_out_names[nc_ind],'r+')
fr = 1
idx_nc = 0
PF.write_data_nc_2d(nc_fill,px_temp,py_temp,time,fr,idx)
###################################################

############################################
# OBTAIN TIME-INDICES TO ADD NEW PARTICLES
##########################################
tinds_release = range(0,nfr,nfr/n_releases)[0:-1]
npart_release = nq / n_releases
s_num = 2 
###################################################

         #####################################################################
         #		 	PARTICLE TIME STEPPING 
         ######################################################################
print '#######################################################'
print ''
print '			TIME-STEPPING PARTICLES '
print ''
print '#######################################################'
for it in range(nit):    # particle time steps
   
    if t_interp == 'linear': 
       fct = (time+0.5*dt-tim0)*delti 
       u = (1-fct)*u0 + fct*u1
       v = (1-fct)*v0 + fct*v1
    else:
       ptime = (part_time[it] - tim0) / tim0
       if tim_opt:
          start_time = clock.time()
       u = fspline.spline(u0,u1,du0,du1,ptime)[:,:,0]
       v = fspline.spline(v0,v1,dv0,dv1,ptime)[:,:,0]
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
    PN.advance2d(px_temp,py_temp,dpx,dpy,u,v,dx,dy,dt,nx,ny,first=scheme_first) 
    if tim_opt:
       print 'advance time: ' + str(clock.time() - start_time)
    #advance(px_temp,py_temp,u,v,dx,dy,dt,nx,ny,nq)
    [px_temp,py_temp] = cull(px_temp,py_temp,nx,ny)

    if t_interp == 'linear':
       time = time + dt
       bool_write = (time+0.5*dt)>tim1
    else:
       bool_write = part_time[it] >=tim1
    ######################################
   
    if bool_write:
       print 'veloc_adv = ' + init_dict['veloc_adv']
       print 'part_time = ', time
       print 'ocean_time = ', tim1
       print 'writing frame: ' ,idx+1, 'time: ',time
#      seed(px,py,nqmx,mask);
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
      
      
       fr = fr+1
       if idx_nc < frpf:
          PF.write_data_nc_2d(nc_fill,px_temp,py_temp,time,fr,idx_nc)
       else:
          nc_fill.close()
          nc_ind+=1
          idx_nc = 0
          nc_fill = netcdf(os.getcwd() + '/part_out/' + nc_out_names[nc_ind],'r+')
          PF.write_data_nc_2d(nc_fill,px_temp,py_temp,time,fr,idx_nc)

       print '   Wrote particle data to: ' + nc_out_names[nc_ind] + ' , idx_nc = ' + str(idx_nc)
       u0 = u1;
       v0 = v1;
       tim0 = tim1;
       if t_interp == 'spline':
          du0 = du1
          dv0 = dv1
       # read new velocity data
       if t_interp=='spline':
          u1,v1,du1,dv1 = PF.get_uv_2D(netcdf(files[fr],'r'),locfr[fr],t_interp=t_interp)
       else:
          u1,v1 = PF.get_uv_2D(netcdf(files[fr],'r'),locfr[fr])

       print ''
       lfr = locfr[fr] 
       print 'file: ', files[fr] + '  time step: ', locfr[fr]
       tim1 = netcdf(files[fr],'r').variables['ocean_time'][lfr];
        

#######################################
# CLOSE NETCDF FILE
######################################
#nc.close()
nc_fill.close()
