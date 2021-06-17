import sys
sys.path.append('/mnt/projects/lensing/lens_codes_v3.7')
import numpy as np
from astropy.io import fits
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)
from fit_profiles_curvefit import *
from multiprocessing import Pool
from multiprocessing import Process

zhalos = fits.open('../catalogs/halo_props/out_8_2_plus.fits')[1].data.z_v
profiles = np.loadtxt('../catalogs/halo_props/out_profile_8_2.bz2',skiprows=1,delimiter=',')
ncores = 32


def fit_profile(pro,z):
    
         r = np.arange(16)*(pro[1]/15.)  
         r = (r[:-1] + np.diff(r)/2.)/1000
         
         mr = 0.7*pro[1]
         
         rho   = pro[2:17][mr]
         rho_E = pro[17:32][mr]
         S     = pro[32:47][mr]
         S_E   = pro[47:][mr]

        r      = r[mr]
        
         erho = np.ones(15)*50.
         
         rho_f    = rho_fit(r[rho>0],rho[rho>0]/(1.e6**3),erho[rho>0],z,cosmo,True)
         rho_E_f    = rho_fit(r[rho_E>0],rho_E[rho_E>0]/(1.e6**3),erho[rho_E>0],z,cosmo,True)
         S_f      = Sigma_fit(r[S>0],S[S>0]/(1.e6**2),erho[S>0],z,cosmo,True)
         S_E_f      = Sigma_fit(r[S_E>0],S_E[S_E>0]/(1.e6**2),erho[S_E>0],z,cosmo,True)
         
         return [np.log10(rho_f.M200),rho_f.c200,
                 np.log10(rho_E_f.M200),rho_E_f.c200,
                 np.log10(S_f.M200),S_f.c200,
                 np.log10(S_E_f.M200),S_E_f.c200]
                 

def run_fit_profile(index):
    
    output_fits = np.zeros((len(index),8))
    
    for j in index:
        print(j)
        output_fits[j] = fit_profile(profiles[j],zhalos[j])
        


index = np.arange(len(zhalos))

slicer = int(round(len(index)/float(ncores), 0))
slices = ((np.arange(ncores-1)+1)*slicer).astype(int)
slices = slices[(slices <= len(index))]

index_splitted = np.split(index,slices)

pool = Pool(processes=(ncores))
salida = pool.map(run_fit_profile, index_splitted)
pool.terminate()
