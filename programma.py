from os import path
from urllib import request
import matplotlib.pyplot as plt
import numpy as np
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib


lam = np.loadtxt('SRGe0011F16.DAT', usecols=0, dtype=float)
galaxy = np.loadtxt('SRGe0011F16.DAT', usecols=1, dtype=float)
#plt.plot(lam, galaxy)

R = 500
FWHM_gal = 1e4*np.sqrt(0.381*0.887)/R
print( f"FWHM_gal: {FWHM_gal:.1f} Å")

c = 299792.458                      # speed of light in km/s
sigma_inst = c/(R*2.355)
print( f"sigma_inst: {sigma_inst:.0f} km/s")

z = 0.137                      # Initial estimate of the galaxy redshift
lam /= (1 + z)               # Compute approximate restframe wavelength
FWHM_gal /= (1 + z)     # Adjust resolution in Angstrom
print(f"de-redshifted NIRSpec G235H/F170LP resolution FWHM in Å: {FWHM_gal:.1f}")

galaxy = galaxy/np.median(galaxy)       # Normalize spectrum to avoid numerical issues

# Create coordinates centred on the brightest spectrum
c = 299792.458  # speed of light in km/s
velscale = c*np.diff(np.log(lam[-2:]))  # Smallest velocity step
lam_range_temp = [np.min(lam), np.max(lam)]
galaxy, lam, velscale = util.log_rebin(lam_range_temp, galaxy, velscale=velscale)
lam = np.exp(lam)
noise = np.full_like(galaxy, 0.05)

FWHM_temp = 2.51
sps_name = 'fsps'

# НЕ ПОНЯТНО
ppxf_dir = path.dirname(path.realpath(lib.__file__))
basename = f"spectra_{sps_name}_9.0.npz"
filename = path.join(ppxf_dir, 'sps_models', basename)
if not path.isfile(filename):
    url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
    request.urlretrieve(url, filename)

sps = lib.sps_lib(filename, velscale, norm_range=[5070, 5950], age_range=[0, 2.2])
reg_dim = sps.templates.shape[1:]
stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

lam_range_gal = [np.min(lam), np.max(lam)]
gas_templates, gas_names, line_wave = util.emission_lines(sps.ln_lam_temp, lam_range_gal, FWHM_gal, tie_balmer=1)
templates = np.column_stack([stars_templates, gas_templates])

c = 299792.458
start = [1200, 200.]     # (km/s), starting guess for [V, sigma]
n_stars = stars_templates.shape[1]
n_gas = len(gas_names)
component = [0]*n_stars + [1]*n_gas
gas_component = np.array(component) > 0
moments = [2, 2]
start = [start, start]

pp = ppxf(templates, galaxy, noise, velscale[0], start,
          moments=moments, degree=-1, mdegree=-1, lam=lam, lam_temp=sps.lam_temp,
          reg_dim=reg_dim, component=component, gas_component=gas_component,
          reddening=0, gas_reddening=0, gas_names=gas_names)
plt.figure(figsize=(15, 5))
pp.plot()
plt.show()

plt.figure(figsize=(15, 5))
pp.plot(gas_clip=1)
plt.xlim([0.42, 0.52]);
plt.show()

lam_med = np.median(lam)  # Angstrom
sigma_gal = c*FWHM_gal/lam_med/2.355  # in km/s
sigma_temp = c*FWHM_temp/lam_med/2.355
sigma_obs = pp.sol[0][1]   # sigma is second element of first kinematic component
sigma_diff2 = sigma_gal**2 - sigma_temp**2   # eq. (5) of Cappellari (2017)
sigma = np.sqrt(sigma_obs**2 - sigma_diff2)
print(f"sigma stars corrected: {sigma:.0f} km/s")

errors = pp.error[0]*np.sqrt(pp.chi2)      # assume the fit is good
print("Formal errors:")
print("   dV   dsigma")
print("".join("%6.2g" % f for f in errors))

vpec = pp.sol[0][0]                         # This is the fitted residual velocity in km/s
znew = (1 + z)*np.exp(vpec/c) - 1           # eq.(5c) Cappellari (2023)
dznew = (1 + znew)*errors[0]/c              # eq.(5d) Cappellari (2023)
print(f"Best-fitting redshift z = {znew:#.6f} +/- {dznew:#.2g}")
