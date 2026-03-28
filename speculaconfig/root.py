import os.path as op
from os import makedirs

ROOT_DIR = op.dirname(op.abspath(__file__))
calib_dir = op.join(ROOT_DIR,'calib')

iff_path = op.join(calib_dir,'ifunc')
m2c_path = op.join(calib_dir,'m2c')
pupil_path = op.join(calib_dir,'pupilstop')

alias_path = op.join(calib_dir,'aliasing') 
data_path = op.join(calib_dir,'data') 
temp_alias_path = op.join(calib_dir,'scratch_aliasing') 
ogs_path = op.join(calib_dir,'optgains')
sn_path = op.join(calib_dir,'slopenulls') 
frames_path = op.join(calib_dir,'frames')
im_path = op.join(calib_dir,'im')
rec_path = op.join(calib_dir,'rec')


for p in [iff_path, m2c_path, data_path, pupil_path, sn_path, frames_path, im_path, rec_path, alias_path, ogs_path]:
    if not op.exists(p):
        makedirs(p)