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
main = pd.read_csv('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_main.csv.bz2')[:100]
profiles = np.loadtxt('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_pro.csv.bz2',skiprows=1,delimiter=',')[:100]

mp = 2.927e10
zhalos = main.redshift


nrings = 10

index = np.arange(len(profiles))

avance = np.linspace(0,len(profiles)/ncores,10)

def fit_profile(pro,z,plot=False):
    
         r   = pro[2:2+nrings]/1.e3         
         
         rbins = (np.arange(nrings+1)*(pro[1]/float(nrings)))/1000
         mpV = mp/((4./3.)*np.pi*(rbins[1:]**3 - rbins[:-1]**3)) # mp/V
         
         rho   = pro[2+nrings:2+2*nrings]
         rho_E = pro[2+2*nrings:2+3*nrings]
         S     = pro[2+3*nrings:2+4*nrings]
         S_E   = pro[2+4*nrings:]
         
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
             
             
             return [np.log10(rho_f.M200),rho_f.c200,rho_f.chi2,
                     np.log10(rho_E_f.M200),rho_E_f.c200,rho_E_f.chi2,
                     np.log10(S_f.M200),S_f.c200,S_f.chi2,
                     np.log10(S_E_f.M200),S_E_f.c200,S_E_f.chi2]
         
         else:
             
             return np.zeros(12)
                 

def run_fit_profile(index):
    
    
    output_fits = np.zeros((len(index),12))
    
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

output = salida[0]

for fitted in salida[1:]:
    
    
    output  = np.vstack((output,fitted))
    
hn = main['column_halo_id']

output = np.colum_stack(hn,output)
    
out_file = '/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_mass.csv.bz2'

head = 'column_halo_id,lgM200_rho,c200_rho,chi2_rho,lgM200_rho_E,c200_rho_E,chi2_rho_E,lgM200_S,c200_S,chi2_S,lgM200_S_E,c200_S_E,chi2_S_E'

np.savetxt(out_file,output,fmt=['%10d']+['%5.2f']*12,header=head,comments='',delimiter=',')
