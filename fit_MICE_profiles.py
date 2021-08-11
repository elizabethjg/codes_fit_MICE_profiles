import sys
sys.path.append('/mnt/projects/lensing/lens_codes_v3.7')
import numpy as np
from astropy.io import fits
from astropy.cosmology import LambdaCDM, z_at_value
from fit_profiles_curvefit import *
from multiprocessing import Pool
from multiprocessing import Process
import astropy.units as u
import pandas as pd
part = '4_4'
cosmo = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)

# hn = fits.open('../catalogs/halo_props/halo_props2_'+part+'_2_main_plus.fits')[1].data.Halo_number
# zhalos = fits.open('../catalogs/halo_props/halo_props2_'+part+'_2_main_plus.fits')[1].data.z_v
# profiles = np.loadtxt('../catalogs/halo_props/halo_props2_'+part+'.csv_profile.bz2',skiprows=1,delimiter=',')

ncores = 32
main = pd.read_csv('/home/elizabeth/lightconedir_129/halo_props2_'+part+'_main.csv.bz2') 
profiles = np.loadtxt('/home/elizabeth/lightconedir_129/halo_props2_'+part+'_pro.csv.bz2',skiprows=1,delimiter=',')

mp = 2.927e10
zhalos = main.z_halo


nrings = 10

index = np.arange(len(profiles))

avance = np.linspace(0,len(profiles)/ncores,10)

def fit_profile(pro,z,plot=False):
    
         r   = pro[2:2+nrings]/1.e3
         mr = r > 0.
         
         mpV = mp/((4./3.)*np.pi*(rbins[1:]**3 - rbins[:-1]**3)) # mp/V
         
         rho   = pro[2+nrings:2+2*nrings][mr]
         rho_E = pro[2+2*nrings:2+3*nrings][mr]
         S     = pro[2+3*nrings:2+4*nrings][mr]
         S_E   = pro[2+4*nrings:][mr]

         r      = r[mr]

         mrho = rho > 0.
         mS = S > 0.
         mrhoe = rho_E > 0.
         mSe = S_E > 0.

         
         if mrho.sum() > 0. and mS.sum() > 0. and mrhoe.sum() > 0. and mSe.sum() > 0.:
         
             rho_f    = rho_fit(r[rho>0],rho[rho>0],np.sqrt(mpV/rho)[rho>0],z,cosmo,True)
             rho_E_f    = rho_fit(r[rho_E>0],rho_E[rho_E>0],np.sqrt(mpV/rho_E)[rho_E>0],z,cosmo,True)
             S_f      = Sigma_fit(r[S>0],S[S>0],np.sqrt(mpV/S)[S>0],z,cosmo,True)
             S_E_f      = Sigma_fit(r[S_E>0],S_E[S_E>0],np.sqrt(mpV/S_E)[S_E>0],z,cosmo,True)
             
             if plot:
             
                 f,ax = plt.subplots()
                 f2,ax2 = plt.subplots()
                 
                 m = rho_f.xplot > r.min()
                 m1 = rho_E_f.xplot >  r.min()
                 m2 = S_f.xplot >  r.min()
                 m3 = S_E_f.xplot >  r.min()
             
             
                 ax.plot(r,rho,'C7',lw=2)
                 ax.plot(rho_f.xplot[m],rho_f.yplot[m],'k')
                 ax.plot(r,rho_E,'C7--',lw=2)
                 ax.plot(rho_E_f.xplot[m1],rho_E_f.yplot[m1],'k--')
                 
                 ax2.plot(r,S,'C7',lw=2)
                 ax2.plot(r,S_E,'C7--',lw=2)
                 ax2.plot(S_f.xplot[m2],S_f.yplot[m2],'k')
                 ax2.plot(S_E_f.xplot[m3],S_E_f.yplot[m3],'k--')
             
                 ax.set_xscale('log')
                 ax.set_yscale('log')
                 ax2.set_xscale('log')
                 ax2.set_yscale('log')
                 ax.set_ylabel(r'$\rho [M_\odot h^2/Mpc^3]$')
                 ax.set_xlabel('$r[Mpc/h]$')
                 ax2.set_ylabel(r'$\Sigma [M_\odot h/Mpc^2]$')
                 ax2.set_xlabel('$R[Mpc/h]$')
             
             
                 print(0.7*(pro[1]/1000.))
                 print(np.log10(rho_f.M200),rho_f.c200)
             
                 input("Press Enter to continue...")
             
                 plt.close('all')
             
             
             return [np.log10(rho_f.M200),rho_f.c200,
                     np.log10(rho_E_f.M200),rho_E_f.c200,
                     np.log10(S_f.M200),S_f.c200,
                     np.log10(S_E_f.M200),S_E_f.c200]
         
         else:
             
             return np.zeros(9)
                 

def run_fit_profile(index):
    
    
    output_fits = np.zeros((len(index),8))
    
    a = '='
    
    for i in range(len(index)):
        j = index[i]
        if j in avance:
            print(a)
        output_fits[i] = fit_profile(profiles[j],zhalos[j])
        a += '='
        
    return output_fits
        


slicer = int(round(len(index)/float(ncores), 0))
slices = ((np.arange(ncores-1)+1)*slicer).astype(int)
slices = slices[(slices <= len(index))]

# run_fit_profile(index[c_re <0.1])

# '''
index_splitted = np.split(index,slices)

pool = Pool(processes=(ncores))
salida = pool.map(run_fit_profile, np.array(index_splitted).T)
pool.terminate()

M_r = []
M_re = []
M_S = []
M_Se = []
c_r = []
c_re = []
c_S = []
c_Se = []

for fitted in salida:
    
    fitted = fitted.T
    
    M_r  = np.append(M_r,fitted[0])
    M_re = np.append(M_re,fitted[2])
    M_S  = np.append(M_S,fitted[3])
    M_Se = np.append(M_Se,fitted[6])
    c_r  = np.append(c_r,fitted[1]) 
    c_re = np.append(c_re,fitted[3])
    c_S  = np.append(c_S ,fitted[5])
    c_Se = np.append(c_Se,fitted[7])
    

table = [fits.Column(name='halo_id', format='E', array=hn),
             fits.Column(name='lM200_rho', format='E', array=M_r),
             fits.Column(name='c200_rho', format='E', array=c_r),
             fits.Column(name='lM200_rho_E', format='E', array=M_re),
             fits.Column(name='c200_rho_E', format='E', array=c_re),
             fits.Column(name='lM200_S', format='E', array=M_S),
             fits.Column(name='c200_S', format='E', array=c_S),
             fits.Column(name='lM200_S_E', format='E', array=M_Se),
             fits.Column(name='c200_S_E', format='E', array=c_Se)]
             
tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(table))
primary_hdu = fits.PrimaryHDU()

hdul = fits.HDUList([primary_hdu,tbhdu])

# hdul.writeto('../catalogs/halo_props/fitted_mass_'+part+'.fits',overwrite=True)
hdul.writeto('../catalogs/halo_props/fitted_mass_'+part+'.fits',overwrite=True)
# '''
