#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-Aperture Centroid Sensor (FACS) tip/tilt error-budget functions.

FACS is a single-subaperture (1x1) Shack-Hartmann-like sensor: it extracts
tip/tilt from the weighted center-of-gravity (WCoG) of the whole-pupil PSF.
It is treated here as the tip/tilt sensor of a hybrid system in which a
pyramid WFS (see ``Functions.py``) still corrects the higher-order (HO)
modes; FACS only replaces the P-WFS as the *tip/tilt* sensor.

Two properties set FACS apart from the P-WFS branch and shape this module:

* FACS is linear (no pyramid-like sensitivity loss), so there is no optical
  gain to divide by (``c_optg = 1`` implicitly): compare with
  ``Functions.PSD_final_meas``, which does divide by ``c_optg ** 2``.
* Classical spatial-aliasing (WFS undersampling of high-order content) does
  not apply to a full-aperture centroid the way it does to a subaperture
  grid; it is neglected here (see project discussion), so this module has no
  aliasing counterpart to ``Functions.aliasing_variance``.

Noise propagation follows the semi-analytical/numerical route described in
the project drafts (as opposed to the closed-form dual-Gaussian WCoG
formulas). The FACS spot is the long-exposure PSF left by the pyramid's HO
correction (MASTSEL ``longExposurePsf``, residual phase PSD flat inside the
AO control radius and von Karman outside), integrated over the detector
pixels. Per-pixel statistics come from MASTSEL's ``meanVarPixelThr``, the
WCoG gain is calibrated numerically on that PSF, and the resulting
centroid-noise variance is propagated through the closed-loop NTF exactly
like the P-WFS noise term.
"""

from typing import Tuple

import numpy as np
from mastsel.mavisUtilities import meanVarPixelThr, simple2Dgaussian, sigma_from_FWHM
from mastsel.mavisPsf import Field, longExposurePsf

from src.Functions import (
    build_transfer_function,
    compute_noise_PSD_intermediate,
    compute_output_PSD_and_integrate,
    temporal_variance,
)

# Tip/tilt WFE-to-angle conversion: theta[rad] = TILT_TO_ANGLE_FACTOR * a[m] / D,
# for a wavefront tilt of amplitude 'a' (Noll Z2/Z3 normalization).
DEFAULT_TILT_TO_ANGLE_FACTOR = 4.0

# Radians-to-arcsec conversion, 180/pi * 3600.
CRAD2AS = 180.0 / np.pi * 3600.0


def _pixel_grid(n_pixels):
    """Return the (xx, yy) pixel-coordinate meshgrid, centered on the window."""
    coords = np.arange(n_pixels) - (n_pixels - 1) / 2.0
    return np.meshgrid(coords, coords)


# Long-exposure FACS PSF behind the pyramid's HO correction, via MASTSEL's
# longExposurePsf: OTF = OTF_tel * exp(-D_phi/2), with D_phi from a residual
# phase PSD that is flat inside the AO control radius f_cut (HO temporal +
# measurement + aliasing residual) and von Karman-shaped outside it, scaled to
# the fitting-error variance of the budget. Tip/tilt is excluded: within one
# FACS frame it is the measured signal, not a blur.

def facs_long_exposure_psf(
    n_pixels: int, n_samp: int, telescope_diameter: float, wavelength: float,
    sigma2_inband_nm2: float, sigma2_fit_nm2: float, f_cut: float,
    outer_scale: float, x0: float = 0.0, n_pix_pupil: int = None,
) -> np.ndarray:
    """
    Long-exposure PSF of the HO-corrected pupil, integrated over the FACS pixels.

    Parameters
    ----------
    n_pixels : int
        Side of the square FACS pixel window.
    n_samp : int
        Detector sampling, in pixels per lambda/D (>= 2, so that the PSF
        grid supports the full telescope OTF).
    telescope_diameter : float
        Telescope diameter D [m], unobstructed circular pupil.
    wavelength : float
        FACS sensing wavelength [m].
    sigma2_inband_nm2 : float
        HO residual variance inside the control radius [nm^2], spread
        uniformly over |f| < f_cut.
    sigma2_fit_nm2 : float
        Fitting-error variance [nm^2], with a von Karman spectral shape for
        |f| >= f_cut (normalized over the whole outer region, see below).
    f_cut : float
        AO control radius in the pupil-plane spatial frequency [1/m].
    outer_scale : float
        Outer scale L0 [m], only used for the shape of the fitting PSD.
    x0 : float, optional
        Spot center along x, in pixels from the window center (used for the
        numerical WCoG gain calibration).
    n_pix_pupil : int, optional
        Pupil sampling; default puts the PSD grid edge at >= 2 f_cut, with at
        least 128 pixels across the pupil.

    Returns
    -------
    np.ndarray
        ``(n_pixels, n_pixels)`` map, fraction of the total flux per pixel.
        The sum is < 1: light outside the window (halo, and fitting power
        beyond the PSD grid, applied as a uniform exp(-sigma2_far) loss) is
        lost for the WCoG.
    """
    if n_samp < 2:
        raise ValueError(f"n_samp must be >= 2 to support the telescope OTF, got {n_samp}")
    if n_pix_pupil is None:
        # PSD grid edge at >= 2 f_cut, and >= 128 px across the pupil: coarser
        # masks bias the disk area (e.g. +0.6%, hence the PSF peak +0.4%, at 40 px).
        n_pix_pupil = max(2 * int(np.ceil(2.0 * f_cut * telescope_diameter)), 128)
    n_grid = int(round(n_samp * n_pix_pupil))
    if not np.isclose(n_grid, n_samp * n_pix_pupil):
        raise ValueError("n_samp * n_pix_pupil must be an integer")
    pitch = telescope_diameter / n_pix_pupil

    # PSF pixel = wavelength / (n_grid * pitch) = wavelength / (n_samp * D).
    mask = Field(wavelength, n_grid, n_grid * pitch, xp=np)
    mask.setAsTelescopeMask(telescope_diameter / 2)
    otf_field = Field(wavelength, n_grid, n_grid * pitch, xp=np)
    otf_field.sampling = mask.sampling.copy()
    otf_field.pupilToOtf()
    otf_tel = otf_field.sampling / np.max(np.abs(otf_field.sampling))

    dk = 1.0 / (n_grid * pitch)
    freqs = (np.arange(n_grid) - n_grid // 2) * dk
    fxx, fyy = np.meshgrid(freqs, freqs)
    f = np.hypot(fxx, fyy)
    rad2_per_nm2 = (2 * np.pi * 1e-9 / wavelength) ** 2

    # Residual variance per frequency pixel [rad^2]; f = 0 (piston) left empty.
    w_pix = np.zeros((n_grid, n_grid))
    inband = (f > 0) & (f < f_cut)
    w_pix[inband] = sigma2_inband_nm2 * rad2_per_nm2 / np.count_nonzero(inband)
    outband = f >= f_cut
    k0_2 = 1.0 / outer_scale ** 2
    # Integral of (f^2 + k0^2)^(-11/6) over |f| > f_cut, so that the outer
    # region carries exactly sigma2_fit over the infinite plane.
    vk_norm = 6 * np.pi / 5 * (f_cut ** 2 + k0_2) ** (-5 / 6)
    w_pix[outband] = (sigma2_fit_nm2 * rad2_per_nm2 * dk ** 2 / vk_norm
                      * (f[outband] ** 2 + k0_2) ** (-11 / 6))
    sigma2_far = max(sigma2_fit_nm2 * rad2_per_nm2 - w_pix[outband].sum(), 0.0)

    # longExposurePsf zero-pads the PSD to n_pad and gets the phase
    # covariance as (kk * freq_range)^2 * ifft2(psd): this scaling makes the
    # zero-lag covariance equal to sum(w_pix).
    psd = Field(wavelength, n_grid, n_grid * dk, unit='rad', xp=np)
    n_pad = n_grid + 2 * (n_grid // 2)
    psd.sampling = w_pix * n_pad ** 2 / (psd.kk * psd.width) ** 2

    # The FFT PSF is centered on grid pixel n_grid//2; the WCoG window center
    # sits half a pixel away for even n_pixels. A phase ramp on the OTF moves
    # the spot exactly (no interpolation).
    offset = 0.5 if n_pixels % 2 == 0 else 0.0
    lags = np.arange(n_grid) - n_grid // 2
    uu, vv = np.meshgrid(lags, lags)
    ramp = np.exp(2j * np.pi * (uu * (x0 - offset) + vv * (-offset)) / n_grid)
    # Pixel integration: OTF times the pixel MTF, sinc(rho / (n_samp D)) per
    # axis, i.e. sinc(lag / n_grid); the PSF sampled at pixel centers then
    # gives the flux integrated over each pixel.
    mtf_pixel = np.sinc(uu / n_grid) * np.sinc(vv / n_grid)

    psf = longExposurePsf(mask, psd, otf_tel=otf_tel * ramp * mtf_pixel).sampling
    psf = np.clip(psf, 0.0, None)
    psf = psf / psf.sum() * np.exp(-sigma2_far)

    start = n_grid // 2 - n_pixels // 2
    return psf[start:start + n_pixels, start:start + n_pixels]


# Gaussian WCoG weighting function Fw(x,y) (Eq. 4 in the P. draft / Eq. 4 FACS
# draft). Reuses MASTSEL's own Gaussian and FWHM-to-sigma helpers.

def wcog_weight(n_pixels: int, fwhm_weight: float,
                x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """
    Build a Gaussian WCoG weighting function on a pixel grid.

    Parameters
    ----------
    n_pixels : int
        Side of the square pixel window.
    fwhm_weight : float
        FWHM of the weighting function, in pixels (N_w in the drafts).
    x0, y0 : float, optional
        Weighting-function center offset from the window center, in pixels.

    Returns
    -------
    np.ndarray
        ``(n_pixels, n_pixels)`` weighting map.
    """
    xx, yy = _pixel_grid(n_pixels)
    sigma_w = sigma_from_FWHM(fwhm_weight)
    return simple2Dgaussian(xx, yy, x0=x0, y0=y0, sg=sigma_w)


# Numerical WCoG gain (push-pull on the actual spot): the closed-form
# N_W^2/(N_T^2+N_W^2) of Thomas et al. only holds for a Gaussian spot.

def wcog_gain_numerical(map_plus: np.ndarray, map_minus: np.ndarray,
                        weight_map: np.ndarray, shift_pix: float) -> float:
    """
    Small-signal WCoG response from two spots shifted by +/- shift_pix along x.

    Parameters
    ----------
    map_plus, map_minus : np.ndarray
        Spot intensity maps with the spot center at +shift_pix and -shift_pix.
    weight_map : np.ndarray
        WCoG weighting function, same shape as the maps.
    shift_pix : float
        Half the push-pull displacement, in pixels.

    Returns
    -------
    float
        Measured shift per true shift; gamma = 1 / gain gives unit response.
    """
    xx, _ = _pixel_grid(map_plus.shape[0])

    def wcog_x(spot):
        return np.sum(xx * spot * weight_map) / np.sum(spot * weight_map)

    return float((wcog_x(map_plus) - wcog_x(map_minus)) / (2 * shift_pix))


# Maréchal approximation for the Strehl ratio. Used as a diagnostic: it is the
# fraction of light left in the diffraction-limited core of the FACS PSF.

def strehl_ratio_marechal(residual_rms_nm: float, wavelength: float) -> float:
    """
    Estimate the Strehl ratio from a residual WFE RMS (Maréchal approximation).

    Only accurate for S >= ~0.1-0.2; callers should treat lower values as
    qualitative, since the true PSF develops a significant halo that the
    Maréchal approximation does not capture.

    Parameters
    ----------
    residual_rms_nm : float
        Residual wavefront-error RMS, in nm.
    wavelength : float
        Wavelength at which the Strehl ratio is evaluated, in meters.

    Returns
    -------
    float
        Strehl ratio S = exp(-sigma_rad^2).
    """
    sigma_rad = residual_rms_nm * 1e-9 * 2 * np.pi / wavelength
    return float(np.exp(-sigma_rad ** 2))


# Per-pixel mean/variance including photon, read-out, background and excess
# noise: thin wrapper around MASTSEL's meanVarPixelThr (Sec. 2.2.2 FACS
# draft). ``excess_noise_factor2`` is F^2 (meanVarPixelThr convention), while
# value_for_F_excess_noise in this project's YAML files is F: square it first.

def facs_pixel_noise_variance(flux_map: np.ndarray, ron: float,
                              sky_bkg: float = 0.0, dark_curr: float = 0.0,
                              excess_noise_factor2: float = 1.0,
                              threshold: float = -np.inf, order: int = 30,
                              npoints: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel mean and variance for a noisy intensity map.

    Parameters
    ----------
    flux_map : np.ndarray
        Noiseless per-pixel photoelectron count for one frame.
    ron : float
        Read-out noise, RMS electrons per pixel.
    sky_bkg, dark_curr : float, optional
        Sky-background and dark-current photoelectrons per pixel per frame.
    excess_noise_factor2 : float, optional
        Excess-noise power factor F^2 (1.0 for a standard CCD/CMOS).
    threshold, order, npoints : optional
        Passed through to ``mastsel.mavisUtilities.meanVarPixelThr``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(mean_map, var_map)``, same shape as ``flux_map``.
    """
    mean_map, var_map = meanVarPixelThr(
        flux_map, ron=ron, bg=sky_bkg + dark_curr, thresh=threshold,
        order=order, excess=excess_noise_factor2, npoints=npoints,
    )
    return mean_map, var_map


# Weighted centroid noise variance, Eq. (17)/(18) in the FACS draft. Under
# circular symmetry (round spot and weighting function on a square grid) the
# x and y noise variances are equal, so this single value is reused for both
# tip and tilt.

def facs_centroid_noise_variance_pix(flux_map: np.ndarray, weight_map: np.ndarray,
                                     sigma2_map: np.ndarray, gamma: float) -> float:
    """
    Compute the WCoG centroid-noise variance from per-pixel statistics.

    Parameters
    ----------
    flux_map : np.ndarray
        Noiseless per-pixel photoelectron count (the WCoG denominator uses
        the expected signal, per the error-propagation law).
    weight_map : np.ndarray
        WCoG weighting function Fw(x,y), same shape as ``flux_map``.
    sigma2_map : np.ndarray
        Per-pixel noise variance, same shape as ``flux_map``.
    gamma : float
        Unit-gain WCoG calibration coefficient (1 / ``wcog_gain_numerical``).

    Returns
    -------
    float
        Centroid-noise variance along one axis, in pixel^2.
    """
    n_pixels = flux_map.shape[0]
    xx, _ = _pixel_grid(n_pixels)

    numerator = np.sum((xx ** 2) * (weight_map ** 2) * sigma2_map)
    denominator = np.sum(flux_map * weight_map) ** 2

    return float(gamma ** 2 * numerator / denominator)


# Convert a centroid-noise variance (pixel^2) into a tip/tilt WFE variance
# (nm^2), through the intermediate angular jitter (arcsec^2). The nm<->arcsec
# leg uses theta[arcsec] = a[nm] * TILT_TO_ANGLE_FACTOR*1e-9/D * CRAD2AS,
# i.e. the standard Noll Z2/Z3 tip/tilt WFE-to-angle relation.

def facs_centroid_variance_pix_to_tiptilt_nm2(
    sigma2_pix: float, pixel_scale_arcsec: float, telescope_diameter: float,
    tilt_to_angle_factor: float = DEFAULT_TILT_TO_ANGLE_FACTOR,
) -> float:
    """
    Convert a WCoG centroid-noise variance to a tip/tilt WFE variance.

    Parameters
    ----------
    sigma2_pix : float
        Centroid-noise variance, in pixel^2 (``facs_centroid_noise_variance_pix``).
    pixel_scale_arcsec : float
        Detector plate scale, in arcsec per pixel.
    telescope_diameter : float
        Telescope diameter D, in meters.
    tilt_to_angle_factor : float, optional
        Noll Z2/Z3 tip/tilt WFE-to-angle conversion factor (default 4.0).

    Returns
    -------
    float
        Tip/tilt WFE variance, in nm^2.
    """
    sigma2_arcsec = sigma2_pix * pixel_scale_arcsec ** 2
    nm_per_arcsec = telescope_diameter / (tilt_to_angle_factor * 1e-9 * CRAD2AS)
    return sigma2_arcsec * nm_per_arcsec ** 2


# Pixel-domain -> nm^2 chain for one frame, given the spot map and its gain.

def facs_slope_noise_variance_nm2(
    flux_map: np.ndarray, weight_map: np.ndarray, gamma: float, ron: float,
    sky_bkg: float, dark_curr: float, excess_noise_factor2: float,
    pixel_scale_arcsec: float, telescope_diameter: float,
    threshold: float = -np.inf, order: int = 30, npoints: int = 1000,
    tilt_to_angle_factor: float = DEFAULT_TILT_TO_ANGLE_FACTOR,
) -> float:
    """
    Compute the FACS tip/tilt measurement-noise variance for one frame.

    Per-pixel noise statistics come from ``facs_pixel_noise_variance``
    (MASTSEL's ``meanVarPixelThr``); the WCoG centroid-noise variance
    (Eq. 17/18), referred to unit gain through ``gamma``, is converted to a
    tip/tilt WFE variance. sigma_x^2 == sigma_y^2 for a round spot, so the
    returned value applies to both tip and tilt.

    Parameters
    ----------
    flux_map : np.ndarray
        Expected photoelectrons per pixel per frame (e.g. total flux times
        ``facs_long_exposure_psf``).
    weight_map : np.ndarray
        WCoG weighting function (``wcog_weight``).
    gamma : float
        Unit-response calibration, 1 / ``wcog_gain_numerical``.
    ron : float
        Read-out noise, RMS electrons per pixel.
    sky_bkg, dark_curr : float
        Sky-background and dark-current photoelectrons per pixel per frame.
    excess_noise_factor2 : float
        Excess-noise power factor F^2.
    pixel_scale_arcsec : float
        Detector plate scale, in arcsec per pixel.
    telescope_diameter : float
        Telescope diameter D, in meters.
    threshold, order, npoints : optional
        Passed through to ``meanVarPixelThr``.
    tilt_to_angle_factor : float, optional
        Noll Z2/Z3 tip/tilt WFE-to-angle conversion factor (default 4.0).

    Returns
    -------
    float
        Tip/tilt measurement-noise variance, in nm^2.
    """
    _, sigma2_map = facs_pixel_noise_variance(
        flux_map, ron, sky_bkg, dark_curr, excess_noise_factor2,
        threshold, order, npoints,
    )
    sigma2_pix = facs_centroid_noise_variance_pix(flux_map, weight_map, sigma2_map, gamma)

    return facs_centroid_variance_pix_to_tiptilt_nm2(
        sigma2_pix, pixel_scale_arcsec, telescope_diameter, tilt_to_angle_factor,
    )


# Propagates the FACS measurement-noise variance through the tip/tilt
# closed-loop NTF. Equivalent to Functions.measure_variance, but without the
# optical-gain division (FACS is assumed linear, c_optg = 1).

def facs_measurement_variance(
    transf_funct: np.ndarray, omega_temp_freq_interval: np.ndarray,
    sigma2_tt_nm2: float, verbose: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Propagate the FACS tip/tilt measurement noise through the NTF.

    Parameters
    ----------
    transf_funct : np.ndarray
        Noise transfer function H_n for the 2 tip/tilt modes, shape (2, N_freq).
    omega_temp_freq_interval : np.ndarray
        Angular temporal frequency vector [rad/s].
    sigma2_tt_nm2 : float
        Tip/tilt measurement-noise variance in nm^2 (assumed equal for tip
        and tilt), from ``facs_slope_noise_variance_nm2``.
    verbose : bool, optional
        Print open-loop/closed-loop RMS if True.

    Returns
    -------
    tuple[float, float, np.ndarray, np.ndarray]
        ``(variance_OL, variance_CL, PSD_output, PSD_input)``, summed over
        the 2 tip/tilt modes.
    """
    actuators_number = 2
    sigma2_w = np.full(actuators_number, sigma2_tt_nm2)

    PSD_input = compute_noise_PSD_intermediate(omega_temp_freq_interval, actuators_number, sigma2_w)
    variance_OL, variance_CL, PSD_output = compute_output_PSD_and_integrate(
        actuators_number, transf_funct, PSD_input, omega_temp_freq_interval,
    )

    if verbose:
        print(f"FACS tip/tilt measurement variance OL: {variance_OL} nm^2")
        print(f"FACS tip/tilt measurement standard deviation OL: {np.sqrt(variance_OL)} nm")
        print(f"FACS tip/tilt measurement variance CL: {variance_CL} nm^2")
        print(f"FACS tip/tilt measurement standard deviation CL: {np.sqrt(variance_CL)} nm")

    return variance_OL, variance_CL, PSD_output, PSD_input


# Integrator-gain sweep for the FACS tip/tilt loop. Same policy as the
# pyramid's optimize_gain_blocks: the cost is the closed-loop variance only
# (temporal + measurement), with no stability-margin constraint, so the
# optimum can sit at the top of the sweep, i.e. close to the stability limit.

def facs_gain_sweep(
    gain_values: np.ndarray, omega_temp_freq_interval: np.ndarray, t_0: float,
    plant_num: np.ndarray, plant_den: np.ndarray, psd_atmo_tt: np.ndarray,
    psd_vib_tt: np.ndarray, sigma2_tt_nm2: float,
) -> Tuple[float, np.ndarray]:
    """
    Pick the integrator gain minimizing the FACS tip/tilt closed-loop variance.

    Parameters
    ----------
    gain_values : np.ndarray
        Candidate integrator gains (same gain for tip and tilt).
    omega_temp_freq_interval : np.ndarray
        Angular temporal frequency vector of the FACS loop [rad/s].
    t_0 : float
        FACS sampling period [s].
    plant_num, plant_den : np.ndarray
        FACS plant polynomials (delay included in ``plant_den``).
    psd_atmo_tt, psd_vib_tt : np.ndarray
        Turbulence and vibration PSDs for tip and tilt, shape (2, N_freq).
    sigma2_tt_nm2 : float
        FACS tip/tilt measurement-noise variance in nm^2, per axis
        (``facs_slope_noise_variance_nm2``); gain-independent.

    Returns
    -------
    tuple[float, np.ndarray]
        ``(best_gain, variances)``; ``variances`` has shape (N_gain, 2) with
        the closed-loop temporal and measurement variances [nm^2], each summed
        over tip and tilt.
    """
    variances = np.zeros((len(gain_values), 2))
    for k, gain in enumerate(gain_values):
        H_r, H_n = build_transfer_function(
            omega_temp_freq_interval, t_0, 2, plant_num, plant_den, gain=np.full(2, gain),
        )
        _, var_temp, _, _ = temporal_variance(psd_atmo_tt, psd_vib_tt, H_r, 2, omega_temp_freq_interval)
        _, var_meas, _, _ = facs_measurement_variance(H_n, omega_temp_freq_interval, sigma2_tt_nm2)
        variances[k] = np.real(var_temp), np.real(var_meas)

    best = int(np.argmin(variances.sum(axis=1)))
    return float(gain_values[best]), variances


# Combines the HO (pyramid) and TT (FACS) contributions into one system
# total. HO fitting/temporal/measurement/aliasing are expected to already be
# summed over the HO modes only (i.e. excluding the 2 tip/tilt modes, which
# are replaced end-to-end by the FACS branch); TT has no aliasing term
# (neglected for a full-aperture sensor, see module docstring).

def facs_total_variance(
    var_fit_ho: float, var_temp_ho: float, var_meas_ho: float, var_alias_ho: float,
    var_temp_tt: float, var_meas_tt: float, verbose: bool = False,
) -> Tuple[float, float, float]:
    """
    Combine the pyramid HO budget and the FACS tip/tilt budget.

    Parameters
    ----------
    var_fit_ho : float
        Fitting-error variance, computed with the HO mode count only
        (excludes the 2 tip/tilt modes handled by FACS).
    var_temp_ho, var_meas_ho, var_alias_ho : float
        Pyramid HO temporal, measurement-noise and aliasing variances,
        summed over the HO modes only.
    var_temp_tt, var_meas_tt : float
        FACS tip/tilt temporal and measurement-noise variances.
    verbose : bool, optional
        Print the breakdown in nm RMS if True.

    Returns
    -------
    tuple[float, float, float]
        ``(var_total, var_ho, var_tt)``, all in nm^2.
    """
    var_ho = np.real(var_fit_ho) + np.real(var_temp_ho) + np.real(var_meas_ho) + np.real(var_alias_ho)
    var_tt = np.real(var_temp_tt) + np.real(var_meas_tt)
    var_total = var_ho + var_tt

    if verbose:
        print(f"HO (pyramid) fitting RMS [nm]: {np.sqrt(np.real(var_fit_ho))}")
        print(f"HO (pyramid) temporal RMS [nm]: {np.sqrt(np.real(var_temp_ho))}")
        print(f"HO (pyramid) measurement RMS [nm]: {np.sqrt(np.real(var_meas_ho))}")
        print(f"HO (pyramid) aliasing RMS [nm]: {np.sqrt(np.real(var_alias_ho))}")
        print(f"HO (pyramid) total RMS [nm]: {np.sqrt(var_ho)}")
        print(f"TT (FACS) temporal RMS [nm]: {np.sqrt(np.real(var_temp_tt))}")
        print(f"TT (FACS) measurement RMS [nm]: {np.sqrt(np.real(var_meas_tt))}")
        print(f"TT (FACS) total RMS [nm]: {np.sqrt(var_tt)}")
        print(f"TOTAL (HO + TT) RMS [nm]: {np.sqrt(var_total)}")

    return var_total, var_ho, var_tt
