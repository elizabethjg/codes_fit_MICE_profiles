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
cosmo = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)

part = '3_2'

main = pd.read_csv('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_main.csv.bz2') 
profiles = np.loadtxt('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_pro.csv.bz2',skiprows=1,delimiter=',')
# masses = fits.open('../catalogs/halo_props/fitted_mass_'+part+'.fits')[1].data
zhalos = main.redshift

rc = np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2)

plots_path = '/home/elizabeth/plots/'

f1,ax1 = plt.subplots()
f2,ax2 = plt.subplots()
        
def fit_profile(pro,z):
    
        nrings = 10
        mp = 2.927e10
        r   = pro[2:2+nrings]/1.e3
        
        
        rbins = (np.arange(nrings+1)*(pro[1]/float(nrings)))/1000
        mpV = mp/((4./3.)*np.pi*(rbins[1:]**3 - rbins[:-1]**3)) # mp/V
        
        rho   = pro[2+nrings:2+2*nrings]
        rho_E = pro[2+2*nrings:2+3*nrings]
        S     = pro[2+3*nrings:2+4*nrings]
        S_E   = pro[2+4*nrings:]
        
        r      = r
        
        
        
        ax1.plot(r[rho>0],rho[rho>0],'C7',alpha=0.2)
        ax1.plot(r[rho_E>0],rho_E[rho_E>0],'C3',alpha=0.2)
        
        ax2.plot(r[S>0],S[S>0],'C7',alpha=0.2)
        ax2.plot(r[S_E>0],S_E[S_E>0],'C3',alpha=0.2)
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax1.set_ylabel(r'$\rho [M_\odot h^2/Mpc^3]$')
        ax1.set_xlabel('$r[Mpc/h]$')
        ax2.set_ylabel(r'$\Sigma [M_\odot h/Mpc^2]$')
        ax2.set_xlabel('$R[Mpc/h]$')
        
indexr = np.random.uniform(0,len(zhalos),1000).astype(int)

for j in indexr:
    
    out = fit_profile(profiles[j],zhalos[j])
    
f1.savefig(plots_path+'rho_profile_'+part+'.png')
f2.savefig(plots_path+'S_profile_'+part+'.png')

s    = main.c3D/main.a3D
q    = main.b3D/main.a3D
q2d  = main.b2D/main.a2D
sr   = main.c3Dr/main.a3Dr
qr   = main.b3Dr/main.a3Dr
q2dr = main.b2Dr/main.a2Dr


index = np.arange(len(s))

mrelax = (rc/main.r_max < 0.05)


k2 = (main.EKin)
u2 = (main.EPot)

m = (main.lgM > 12.5)#*mrelax
plt.figure()
plt.plot(zhalos[~m],((2.*k2)/abs(u2))[~m],',',alpha=0.5,zorder=1)
plt.scatter(zhalos[m],((2.*k2)/abs(u2))[m],c=(rc/main.r_max)[m],alpha=0.3,s=4,vmax=0.3,zorder=2)
# plt.plot(zhalos[m],Eratio[m],',')
plt.xlabel('$z$')
plt.ylabel('$(2K)/U$')
plt.ylim([0.5,3])
plt.axhline(1.35)
plt.colorbar()
plt.savefig(plots_path+'Eratio_'+part+'.png')


plt.figure()
plt.scatter(sr,s,c=(rc/main.r_max),alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.xlabel('$s_r$')
plt.ylabel('$s$')
plt.colorbar()
plt.savefig(plots_path+'s_'+part+'.png')

plt.figure()
plt.scatter(qr,q,c=(rc/main.r_max),alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.xlabel('$q_r$')
plt.ylabel('$q$')
plt.colorbar()
plt.savefig(plots_path+'q_'+part+'.png')

plt.figure()
plt.scatter(q2dr,q2d,c=(rc/main.r_max),alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.xlabel('$q2d_r$')
plt.ylabel('$q2d$')
plt.colorbar()
plt.savefig(plots_path+'q2d_'+part+'.png')
