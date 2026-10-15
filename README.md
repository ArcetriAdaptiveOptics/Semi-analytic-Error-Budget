# Semi-analytic-Error-Budget

A semi-analytic simulator for the Pyramid Wavefront Sensor (P-WFS) Error Budget, based on the theoretical framework presented in *Agapito et al. 2019* ("Semianalytical error budget for adaptive optics systems with pyramid wavefront sensors").

## Overview

This code computes the total residual variance of an Adaptive Optics (AO) system by evaluating its four main components:
1. **Fitting Error**
2. **Temporal Error** (Atmospheric turbulence + Windshake)
3. **Aliasing Error**
4. **Measurement / Noise Error**

The project has been modularized to separate core mathematical functions, visualization tools, and execution scripts, and it is fully configurable via YAML files.

## Hybrid budget: pyramid HO + full-aperture tip/tilt sensor (FACS)

`Total_Variance_FACS.py` computes the error budget of a system in which the pyramid corrects the higher-order (HO) modes and a **Full-Aperture Centroid Sensor** (FACS: a 1x1 Shack-Hartmann, i.e. a weighted center of gravity, WCoG, of the whole-pupil PSF) senses tip/tilt:

$$\sigma^2_{tot} = \sigma^2_{fit}(N_{HO}) + \sum_{i \in HO}\left(\sigma^2_{temp} + \sigma^2_{meas} + \sigma^2_{alias}\right)_{pyr} + \sigma^2_{temp,TT} + \sigma^2_{meas,TT}$$

* **HO (pyramid)**: the standard pipeline runs unchanged on all `control.n_modes`; tip and tilt (modes 0-1) are discarded. The fitting error uses $N_{HO}$ = `n_modes` - 2.
* **TT (FACS)**: independent 2-mode closed loop with its own gain, sampling time and delay (`control_facs`).
  * FACS spot: the long-exposure PSF left by the pyramid's HO correction, computed with MASTSEL `longExposurePsf`.
    The residual phase PSD is flat inside the AO control radius $f_c = \sqrt{N_{HO}/\pi}/D$ (HO temporal + measurement + aliasing variance).
    Outside $f_c$ it has a von Kármán shape, scaled to the HO fitting variance.
    The PSF is integrated over pixels of λ/(`n_samp`·D) (pixel MTF applied to the OTF); light outside the `n_pixels` window is lost.
  * Per-pixel statistics come from MASTSEL `meanVarPixelThr`. The WCoG gain is calibrated numerically (push-pull) on that PSF and feeds the WCoG centroid noise.
  * Conversion to WFE uses $\theta[\mathrm{arcsec}] = 4\,a[\mathrm{nm}]\,10^{-9}/D \cdot 206265$.
  * No optical gain (FACS is linear) and no aliasing term (neglected).
* **Photon budget** (`wavefront_sensor_facs.flux_mode`):
  * `shared`: same band, beamsplitter. FACS gets a fraction `flux_split` and the pyramid keeps the rest.
  * `independent`: different band, dichroic. FACS photometry is set in `guide_star_facs`.
* `wavefront_sensor_facs.wavelength` is in **nm**.

Known limitations:
* The FACS gain is chosen by a simple sweep, as for the pyramid blocks: minimum variance up to
  `gain_max_margin` × the integrator stability limit (`control.gain_max_margin`, default 0.8, i.e. a gain
  margin of at least 1.9 dB; optionally overridden by `control_facs.gain_max_margin`). The optimum usually
  sits at that upper bound.
* The in-band residual is assumed spatially flat. If it is concentrated at low spatial frequencies, more halo falls inside the FACS window.
* FACS aliasing and the frame-to-frame speckle motion of the centroid are neglected (only the long-exposure PSF is used).
* The FACS flux uses the full circular pupil area, πD²/4, with no central obstruction.

Run with:
```bash
python Total_Variance_FACS.py   # reads params_ANDES_FACS.yaml
```
Core functions live in `src/facs_functions.py`; tests in `tests/test_facs_functions.py`.

## Project Structure

* **`src/facs_functions.py`**: Full-Aperture Centroid Sensor (FACS) tip/tilt functions: spot/weighting model, MASTSEL-based pixel noise, WCoG centroid noise, unit conversion, HO+TT total.
* **`src/Functions.py`**: The core library containing all mathematical and physical definitions (transfer functions, PSD calculations, optical gain interpolation, variance integrations, etc.).
* **`src/plots.py`**: Visualization library containing functions to generate PSD plots, transfer function plots, and closed-loop summary displays.
* **`src/fits_file`**: Directory with data files.
* **`examples/plot_psds.py`**: A standalone script dedicated to performing and plotting the spectral analysis (PSDs) in closed loop for specific Zernike modes.
* **`examples/verify_aliasing_energy.py`**: A validation script that checks the aliasing energy conservation by comparing the variance calculated from slopes vs. the numerical integral of the SA PSD.
* **`scripts/main_saeb.py`**: Command-line entrypoint to run the error budget simulation with a YAML file.
* **`Total_variance.py`**: A script designed to run the error budget simulation.
* **`Total_Variance_FACS.py`** / **`params_ANDES_FACS.yaml`**: Hybrid budget, pyramid for HO modes + FACS for tip/tilt (see above).

### Configuration Files


*These files control telescope properties, atmospheric conditions (seeing, wind speed, $r_0$, $L_0$), WFS characteristics, data paths (reconstruction matrices, optical gains), and AO control parameters.*

## Dependencies

The code relies on several standard scientific Python libraries. Ensure you have the following installed:
* `numpy`
* `scipy`
* `matplotlib`
* `astropy` (for reading `.fits` files)
* `pyyaml` (for parsing `.yaml` configurations)
* `sympy` (for symbolic transfer function derivations)
* `arte` (for atmospheric and turbulence PSD generation)
* `mastsel` (per-pixel noise statistics for the FACS tip/tilt sensor)

## How to Run

1. **Configure your simulation**: Edit `params_mod4.yaml` (or your chosen config file) to set up the telescope, atmosphere, and file paths. Make sure your `.fits` files are correctly located in the `src/file_fits/...` directories.
2. **Run a basic test**:
   ```bash
   python Total_Variance.py
3. **Run a full-scale simulation (e.g., 4000 modes)**:
   ```bash
   python script/main_saeb.py run params_mod4_4000modes.yaml
4. **Plot specific PSDs**:
   ```bash
   python plot_psds.py
5. **Verify Aliasing**:
   ```bash
    python verify_aliasing_energy.py
