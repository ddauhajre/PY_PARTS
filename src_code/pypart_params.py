###################################
#pypart_params.py
'''
SET PARAMETERS FOR PARTICLE RELEASE
SIMULATION
'''
###################################

run_name = 'CC_NEW'


##########################################
# SET ROMS ID AND FILES TO INTEGRATE OVER
"""
ROMS_ID corresponds to cases 
in ROMS_solutions_paths.py class, add on to this
when using your own solution in the class
"""
#########################################
ROMS_ID = 'L4_SBC'


################################
'''
fnum ---> first file index
frpf --> file index spacing (change this by multiples to change temporal resolution)
nfr --> number of frames to integrate over
fr --> starting frame in velocity file (zero index based)
dfr --> time-step increment per netcdf file (default is 1)
sub --> number of time-steps between frames
'''
##################################
fnum    = 432 
frpf    = 48
fr      = 0 
dfr     = 1  
nfr     = 96*10
sub     = 10 

##########################
# timing option
'''
True --> time processes
False --> no timing
'''
###########################
tim_opt = True



#####################################
# TYPE OF VELOCITY TO ADVECT
#####################################
veloc_adv = '3D'
omega_online = True
z_online = True
t_interp = 'linear'
###############################
# set output file name
#################################
#out_part = run_name + '.nc'


############################################
# PARTICLE SEEDING PARAMETERS
##############################################

seed_choice = 'multi_site_circle'
lat_sites = [34.400275]
lon_sites = [-119.7445915]
nq_sites = [50]
rad_sites = [0.2]
n_releases = nfr - 1 
check_seed = False 
pz_seed = 'k'
k_seed = 31

