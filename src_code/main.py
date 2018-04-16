execfile('./src_code/pypart_params.py')
execfile('./src_code/pypart_init.py')
if veloc_adv == '2D':
   execfile('./src_code/pypart_advance_2D.py')
if veloc_adv == '3D':
   execfile('./src_code/pypart_advance_3D.py')




