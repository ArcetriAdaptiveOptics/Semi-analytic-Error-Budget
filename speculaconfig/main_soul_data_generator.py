import os
import specula
specula.init(0)

import numpy as np
from astropy.io import fits

from speculaconfig.yaml_overrides import write_yaml_overrides
from speculaconfig.utils import get_pupil_mask, read_freq, get_psd, compute_and_save_rec, save_correction_vector

from .root import sn_path, pupil_path, frames_path, im_path, calib_dir, alias_path, ogs_path, temp_alias_path


rMods = np.array([0,1,2,3,4]) #([2,3,4,5,6])
n_subaps = np.array([10,20,40])
n_modes = np.array([54,120,660])
seeings = np.array([0.6,0.8,1.0,1.2,1.4])

max_pup_dist = 48
min_pup_dist = 14

npix = 120

main_config = 'soul_main.yml'


# 1. Calibrate pupdata vs n_subaps
for n_subap in n_subaps:
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    overrides = ("{"
                f"main.root_dir: '{calib_dir}', "
                f"pyr.pup_diam: {n_subap:.1f}, "
                f"pyr.pup_dist: {pup_dist:.1f}, "
                f"pyr_pupdata.output_tag: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                "}")
    write_yaml_overrides(input_string=overrides)
    try:
        os.system(f"specula {main_config} calib_pupdata.yml temp_overrides.yml")
        # specula.main_simul(yml_files=[main_config, 'calib_pupdata.yml'], overrides=overrides)
    except FileExistsError: #OSError:
        pass

# 2. Calibrate sn vs n_subaps, rMods
for n_subap in n_subaps:
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    for rMod in rMods:
        overrides = ("{"
                    f"pyr.pup_diam: {n_subap:.1f}, "
                    f"pyr.pup_dist: {pup_dist:.1f}, "
                    f"pyr.mod_amp: {rMod:.1f}, "
                    f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                    f"pyr_sn.output_tag: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_sn', "
                    f"data_store.store_dir:         '{frames_path}', "  
                    f"data_store.create_tn: false, "
                    f"data_store.inputs.input_list: ['pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_frame_null-ocam.out_pixels'], "
                    "}")
        write_yaml_overrides(input_string=overrides)
        try:
            os.system(f"specula {main_config} calib_sn.yml temp_overrides.yml")
            # specula.main_simul(yml_files=[main_config, 'calib_sn.yml'], overrides=overrides)
        except FileExistsError: #OSError:
            pass

# 2.5 compute sensor throughput
pyr_thrp = np.zeros([len(rMods),len(n_subaps)])
for j,n_subap in enumerate(n_subaps):
    pupdatapath = os.path.join(pupil_path,f'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}.fits')
    pyr_mask = get_pupil_mask(filepath=pupdatapath,npix=npix,pyr=True)
    for i,rMod in enumerate(rMods):
        frame = fits.getdata(os.path.join(frames_path,f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_frame_null.fits'))[0]
        thrp = np.sum(frame[pyr_mask])/np.sum(frame)
        pyr_thrp[i,j] = thrp
        fits.writeto(os.path.join(sn_path,f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_throughput.fits'), np.array([thrp]), overwrite=True)
print(pyr_thrp)

# 3. Calibrate IM vs n_subaps, rMods
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    for rMod in rMods:
        tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}'
        im_tag = tag+'_im'
        overrides = ("{"
                    f"pyr.pup_diam: {n_subap:.1f}, "
                    f"pyr.pup_dist: {pup_dist:.1f}, "
                    f"pyr.mod_amp: {rMod:.1f}, "
                    f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                    f"pyr_im_calibrator.im_tag: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_im', "
                    "}")
        write_yaml_overrides(input_string=overrides)
        try:
            os.system(f"specula {main_config} calib_im.yml temp_overrides.yml")
            # specula.main_simul(yml_files=[main_config, 'calib_im.yml'], overrides=overrides)
        except FileExistsError: #OSError:
            pass
        for N in n_modes[:i+1]:
            rec_tag = tag+f'_{N:1.0f}modes_rec'
            compute_and_save_rec(im_tag=im_tag, rec_tag=rec_tag, Nmodes=N, overwrite=True)



# 3.5 Save correction vectors for SIMPC computation
# These vectors go from max_corr to min_corr linearly over the radial orders of the corrected modes,
# and are 0 for all modes from Ncorrmodes to Nmodes.
# Note that 1 means perfect removal of the mode and 0 means no correction.
for N in n_modes:
    save_correction_vector(tag='s0.6', max_corr=0.99, max_corr=0.4, Ncorrmodes=N, Nmodes=660)
    save_correction_vector(tag='s0.8', max_corr=0.95, max_corr=0.2, Ncorrmodes=N, Nmodes=660)
    save_correction_vector(tag='s1.0', max_corr=0.9, max_corr=0.1, Ncorrmodes=N, Nmodes=660)
    save_correction_vector(tag='s1.2', max_corr=0.85, max_corr=0.1, Ncorrmodes=N, Nmodes=660)
    save_correction_vector(tag='s1.4', max_corr=0.8, max_corr=0.1, Ncorrmodes=N, Nmodes=660)


# 4. Calibrate SIMPC vs n_subap, rMods, r0/correction
fs = read_freq(params_path=f'./{main_config}')
ncycles = 100
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    N = n_modes[i]
    for rMod in rMods:
        im_tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_im'
        im = fits.getdata(os.path.join(im_path,im_tag+'.fits'))
        im = im[:,:N]
        im_norm = np.diag(im.T @ im)
        for seeing in seeings:
            tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.1f}'
            simpc_tag = tag+'_simpc'
            overrides = ("{"
                        f"main.total_time: {N*2*ncycles/fs}, "
                        f"pyr.pup_diam: {n_subap:.1f}, "
                        f"pyr.pup_dist: {pup_dist:.1f}, "
                        f"pyr.mod_amp: {rMod:.1f}, "
                        f"pushpull.nmodes: {N:1.0f}, "
                        f"pyr_im_calibrator.nmodes: {N:1.0f}, "
                        f"scale_random.constant_mul_data: 'correction_vector_{N:1.0f}modes_s{seeing:1.1f}', "
                        f"dm_random.nmodes: {N:1.0f}, "
                        f"dm.nmodes: {N:1.0f}, "
                        f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                        f"seeing_random.constant: {seeing:1.1f}, "
                        f"pyr_im_calibrator.im_tag: '{simpc_tag}', "
                        "}")
            write_yaml_overrides(input_string=overrides)
            try:
                os.system(f"specula {main_config} calib_simpc.yml temp_overrides.yml")
                # specula.main_simul(yml_files=[main_config, 'calib_simpc.yml'], overrides=overrides)
                simpc = fits.getdata(os.path.join(im_path,simpc_tag+'.fits'))
                og = np.diag(simpc.T @ im)/im_norm
                fits.writeto(os.path.join(ogs_path,tag+'_og.fits'),og)
                print('Saved optical gains as: '+tag+'_og')
                # for N in n_modes[:i]:
                #     rec_tag = tag+f'_{N:1.0f}modes_rec'
                #     compute_and_save_rec(root_dir, im_tag=im_tag, rec_tag=rec_tag, Nmodes=N)
            except FileExistsError:
                pass

# 5. Calibrate aliasing vs n_subaps, n_modes, r0
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    for rMod in rMods:
        for seeing in seeings:
            modes_vec = n_modes.copy() if i == len(n_subaps)-1 and seeing == 1.0 else np.array([n_modes[i]])
            for N in modes_vec:
                overrides = ("{"
                            f"pyr.pup_diam: {n_subap:.1f}, "
                            f"pyr.pup_dist: {pup_dist:.1f}, "
                            f"pyr.mod_amp: {rMod:.1f}, "
                            f"pyr_modalrec.recmat_object: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_{N:1.0f}modes_rec', "
                            f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                            f"seeing.constant: {seeing:1.1f}, "
                            f"perfect_dm.nmodes: {N:1.0f}, "
                            f"data_store.store_dir: '{temp_alias_path}', "
                            "}")
                write_yaml_overrides(input_string=overrides)
                try:
                    os.system(f"specula {main_config} calib_aliasing.yml temp_overrides.yml")
                    # specula.main_simul(yml_files=[main_config, 'calib_aliasing.yml'], overrides=overrides) #
                    alias_modes = fits.getdata(os.path.join(temp_alias_path,'pyr_modes.fits'))
                    psd,f = get_psd(alias_modes.T, nperseg=1024, dt=1/fs)
                    tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.1f}_{N:1.0f}modes_alias_PSD'
                    fits.writeto(os.path.join(alias_path,tag+'.fits'),psd,overwrite=True)
                    print('Saved aliasing PSD as: '+tag)
                except FileExistsError:
                    pass



