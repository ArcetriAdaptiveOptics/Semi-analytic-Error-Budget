import os
import yaml
import numpy as np
from astropy.io import fits
from scipy.signal import welch

from specula import cpuArray
from specula.lib.make_mask import make_mask
from specula.lib.mmse_reconstructor import compute_mmse_reconstructor


from .root import im_path, rec_path, data_path

def radial_order(i_mode):
    noll = i_mode + 2
    return int(np.ceil(-3.0/2.0+np.sqrt(1+8*noll)/2.0))

def von_karman_power(k,r0,L0,D):
    C = 0.02289558710855519
    B = k**2 + (D/L0)**2
    return C * (r0/D)**(-5.0/3.0) * B**(-11.0/6.0)

def get_pupil_mask(npix:int, filepath:str='', pyr:bool=True, pupdiam=None, obsratio=0.0):
    if pyr:
        np_size = (npix,npix)
        pup_hdu = fits.open(filepath)
        rad = pup_hdu[2].data
        pup_ids = pup_hdu[1].data
        wfs_mask = np.zeros(np_size)
        for j in range(len(rad)):
            f = np.zeros(npix**2)
            np.put(f, pup_ids[:,j], 1)
            f2d = f.reshape(np_size)
            wfs_mask += f2d
    else:
        wfs_mask = make_mask(np_size=npix, diaratio = pupdiam/npix, obsratio=obsratio)
    return wfs_mask.astype(bool)


def get_psd(data, dt:float, nperseg:int=1024):
    f,psd=welch(data,fs=1/dt,nperseg=nperseg,scaling='density',axis=-1)
    return psd,f


def read_freq(params_path:str, obj_name:str=None):
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
        if obj_name is None:
            fs = 1.0/float(params['main']['time_step'])
        else:
            fs = 1.0/float(params[obj_name]['dt'])
    return fs

def save_correction_vector(tag:str,min_corr:float,max_corr:float,
                        max_rad_order:int=36,Nmodes:int=660,
                        Ncorrmodes:int=None):
    if Ncorrmodes is None:
        Ncorrmodes = Nmodes
    cc = np.linspace(max_corr,min_corr,max_rad_order-2)
    tt = np.hstack([np.repeat(cc[i-2],i) for i in range(2,max_rad_order)])
    residuals = np.zeros(Nmodes)
    residuals[:Ncorrmodes] = tt[:Ncorrmodes]
    
    fname = f'correction_vector_{Ncorrmodes}modes_{tag}.fits'
    filepath = os.path.join(data_path,fname)
    hdr = fits.Header()
    hdr['VERSION'] = 1
    hdr['OBJ_TYPE'] = 'BaseValue'
    hdr['NDARRAY'] = 1
    fits.writeto(filepath, residuals, hdr, overwrite=False)
    print(f'Saved correction vector as {fname}')



def compute_and_save_rec(im_tag:str, rec_tag:str, Nmodes:int, 
                ml:bool=False, slope_null=None, RON:float=0.0, 
                mmse:bool=False, diam:float=None, overwrite:bool=False):
    print(rec_tag,im_tag)
    rec = compute_rec(im_tag, Nmodes, ml=ml, slope_null=slope_null, RON=RON, mmse=mmse, diam=diam)
    save_rec(rec, rec_tag, overwrite=overwrite)


def compute_rec(im_tag:str, Nmodes:int, 
                ml:bool=False, slope_null=None, RON:float=0.0, 
                mmse:bool=False, diam:float=None):    
    im_hdul = fits.open(os.path.join(im_path,im_tag+'.fits'))
    intmat = im_hdul[1].data.copy()
    D = intmat[:,:Nmodes]
    if ml or mmse:
        noise_cov = np.diag((slope_null + RON))
        if mmse:
            k = radial_order(np.arange(Nmodes))/diam
            turb_cov = np.diag(np.sqrt(von_karman_power(k,r0=10e-2,L0=25,D=diam))*(2*np.pi*500))**2
        else:
            turb_cov = np.zeros([Nmodes, Nmodes])
        rec = compute_mmse_reconstructor(interaction_matrix=D, c_atm=turb_cov, c_noise=noise_cov, verbose=True, xp=np, dtype=np.float64)
        # DtCn = D.T @ np.diag(1/(slope_null + RON))
        # rec = np.linalg.pinv(DtCn @ D) @ DtCn
    else:
        U,S,Vt = np.linalg.svd(D,full_matrices=False)
        rec = (Vt.T * 1/S) @ U.T
    return rec


def save_rec(rec, rec_tag:str, overwrite:bool=False):
    path = os.path.join(rec_path)
    if not os.path.exists(path):
        os.mkdir(path)
    filename = os.path.join(path,rec_tag+'.fits')
    hdr = fits.Header()
    hdr['VERSION'] = 1
    hdr['PUP_TAG'] = ''
    hdr['SA_TAG'] = ''
    hdr['NORMFACT'] = 0.0
    hdu = fits.PrimaryHDU(header=hdr)  # main HDU, empty, only header
    hdul = fits.HDUList([hdu])
    hdul.append(fits.ImageHDU(data=cpuArray(rec), name='REC'))
    hdul.writeto(filename, overwrite=overwrite)
    hdul.close()
    print('Reconstructor saved as '+rec_tag)