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
formulas): a diffraction-limited spot and a Gaussian weighting function are
built on a pixel grid, per-pixel statistics are obtained from MASTSEL's
``meanVarPixelThr``, and the resulting centroid-noise variance is propagated
through the closed-loop NTF exactly like the P-WFS noise term.
"""

from typing import Tuple

import numpy as np
from mastsel.mavisUtilities import meanVarPixelThr, simple2Dgaussian, sigma_from_FWHM

from src.Functions import compute_noise_PSD_intermediate, compute_output_PSD_and_integrate

# Tip/tilt WFE-to-angle conversion: theta[rad] = TILT_TO_ANGLE_FACTOR * a[m] / D,
# for a wavefront tilt of amplitude 'a' (Noll Z2/Z3 normalization).
DEFAULT_TILT_TO_ANGLE_FACTOR = 4.0

# Radians-to-arcsec conversion, 180/pi * 3600.
CRAD2AS = 180.0 / np.pi * 3600.0


def _pixel_grid(n_pixels):
    """Return the (xx, yy) pixel-coordinate meshgrid, centered on the window."""
    coords = np.arange(n_pixels) - (n_pixels - 1) / 2.0
    return np.meshgrid(coords, coords)


# Diffraction-limited PSF model, Eq. (16) in the FACS draft: a separable
# sinc^2 spot sampled at n_samp pixels per lambda/D. Normalized to unit sum
# so it can be scaled by any total photon flux.

def diffraction_limited_spot(n_pixels: int, n_samp: float,
                             x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """
    Build a normalized diffraction-limited (sinc^2) spot on a pixel grid.

    Parameters
    ----------
    n_pixels : int
        Side of the square pixel window.
    n_samp : float
        Detector sampling in pixels per lambda/D (e.g. 2 for Nyquist).
    x0, y0 : float, optional
        Spot center offset from the window center, in pixels.

    Returns
    -------
    np.ndarray
        ``(n_pixels, n_pixels)`` intensity map, normalized to unit sum.
    """
    xx, yy = _pixel_grid(n_pixels)
    spot = np.sinc((xx - x0) / n_samp) ** 2 * np.sinc((yy - y0) / n_samp) ** 2
    return spot / np.sum(spot)


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


# Unit-gain WCoG normalization coefficient, Eq. (3) in the FACS draft.

def wcog_gamma(fwhm_spot: float, fwhm_weight: float) -> float:
    """
    Compute the WCoG unit-response calibration coefficient gamma.

    Parameters
    ----------
    fwhm_spot : float
        FWHM of the spot intensity distribution, in pixels (N_T).
    fwhm_weight : float
        FWHM of the weighting function, in pixels (N_w).

    Returns
    -------
    float
        gamma = (N_T^2 + N_w^2) / N_w^2.
    """
    return (fwhm_spot ** 2 + fwhm_weight ** 2) / fwhm_weight ** 2


# Maréchal approximation for the Strehl ratio, used to account for the
# pyramid's own HO residual broadening the FACS PSF (a full-aperture sensor
# looks at the same, not-perfectly-corrected, focal-plane image).

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


# Approximate PSF-core broadening from a Strehl-ratio degradation, applied
# consistently to both the sinc^2 spot map (via n_samp) and the WCoG gamma
# calibration (via fwhm_spot_pix).

def broaden_spot_for_strehl(n_samp: float, fwhm_spot_pix: float,
                            strehl: float) -> Tuple[float, float]:
    """
    Scale the diffraction-limited sampling and spot FWHM for a given Strehl.

    Uses ``FWHM_eff = FWHM_diffraction / sqrt(S)``: for a roughly Gaussian
    core that conserves total energy, peak intensity scales as S and peak
    scales as 1/FWHM^2, so FWHM scales as 1/sqrt(S). This is an
    approximation of the true core+halo PSF shape (consistent in spirit
    with the pyramid's own Strehl-derived sensitivity-loss factor, not a
    substitute for a full PSF simulation from the residual spatial PSD),
    and becomes increasingly rough as the Strehl drops much below ~0.2.

    Parameters
    ----------
    n_samp : float
        Diffraction-limited detector sampling, in pixels per lambda/D.
    fwhm_spot_pix : float
        Diffraction-limited spot FWHM, in pixels.
    strehl : float
        Strehl ratio, in (0, 1]. 1.0 leaves both inputs unchanged.

    Returns
    -------
    tuple[float, float]
        ``(n_samp_effective, fwhm_spot_pix_effective)``.
    """
    if not 0.0 < strehl <= 1.0:
        raise ValueError(f"strehl must be in (0, 1], got {strehl}")

    broadening = 1.0 / np.sqrt(strehl)
    return n_samp * broadening, fwhm_spot_pix * broadening


# Per-pixel mean/variance including photon, read-out, background and excess
# noise: thin wrapper around MASTSEL's meanVarPixelThr (Sec. 2.2.2 FACS
# draft). ``excess_noise_factor2`` is F^2, the same convention already used
# by wavefront_sensor.value_for_F_excess_noise in this project's YAML files.

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
        Unit-gain WCoG calibration coefficient (``wcog_gamma``).

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


# Orchestrates the whole pixel-domain -> nm^2 chain for one frame.

def facs_slope_noise_variance_nm2(
    photon_flux_total: float, ron: float, sky_bkg: float, dark_curr: float,
    excess_noise_factor2: float, n_pixels: int, n_samp: float,
    fwhm_spot_pix: float, fwhm_weight_pix: float, pixel_scale_arcsec: float,
    telescope_diameter: float, threshold: float = -np.inf, order: int = 30,
    npoints: int = 1000, tilt_to_angle_factor: float = DEFAULT_TILT_TO_ANGLE_FACTOR,
) -> float:
    """
    Compute the FACS tip/tilt measurement-noise variance for one frame.

    Builds the diffraction-limited spot and Gaussian WCoG weighting function
    on a pixel grid, obtains per-pixel noise statistics from
    ``facs_pixel_noise_variance`` (MASTSEL's ``meanVarPixelThr``), and
    converts the resulting centroid-noise variance to a tip/tilt WFE
    variance in nm^2. By construction sigma_x^2 == sigma_y^2 (circular
    symmetry), so the returned value applies to both tip and tilt.

    ``n_samp``/``fwhm_spot_pix`` are taken at face value as the spot's
    actual sampling/FWHM: if the HO residual left by the pyramid degrades
    the FACS PSF below the diffraction limit, pre-broaden them with
    ``broaden_spot_for_strehl`` before calling this function.

    Parameters
    ----------
    photon_flux_total : float
        Total detected photoelectrons per frame over the whole pupil.
    ron : float
        Read-out noise, RMS electrons per pixel.
    sky_bkg, dark_curr : float
        Sky-background and dark-current photoelectrons per pixel per frame.
    excess_noise_factor2 : float
        Excess-noise power factor F^2.
    n_pixels : int
        Side of the square pixel window used for the WCoG.
    n_samp : float
        Detector sampling in pixels per lambda/D.
    fwhm_spot_pix, fwhm_weight_pix : float
        FWHM (pixels) of the spot and of the WCoG weighting function.
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
    spot = diffraction_limited_spot(n_pixels, n_samp)
    flux_map = spot * photon_flux_total
    weight_map = wcog_weight(n_pixels, fwhm_weight_pix)
    gamma = wcog_gamma(fwhm_spot_pix, fwhm_weight_pix)

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
