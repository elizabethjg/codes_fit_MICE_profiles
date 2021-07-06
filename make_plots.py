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
part = '8_5'
cosmo = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)

part = '8_5'

main = pd.read_csv('../catalogs/halo_props/halo_props2_8_5_2_main.csv.bz2') 
profiles = np.loadtxt('../catalogs/halo_props/halo_props2_'+part+'_2_pro.csv.bz2',skiprows=1,delimiter=',')
masses = fits.open('../catalogs/halo_props/fitted_mass_'+part+'_2.fits')[1].data
zhalos = masses.zhalo

rc = np.sqrt((main.xc_fof - main.xc_rc)**2 + (main.yc_fof - main.yc_rc)**2 + (main.zc_fof - main.zc_rc)**2)

plots_path = '/home/elizabeth/plots/'


def q_75(y):
    return np.quantile(y, 0.75)

def q_25(y):
    return np.quantile(y, 0.25)

def binned(x,y,nbins=10):
    
    
    bined = stats.binned_statistic(x,y,statistic='median', bins=nbins)
    x_b = 0.5*(bined.bin_edges[:-1] + bined.bin_edges[1:])
    q50     = bined.statistic
    
    bined = stats.binned_statistic(x,y,statistic=q_25, bins=nbins)
    q25     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic=q_75, bins=nbins)
    q75     = bined.statistic
    
    dig   = np.digitize(x,bined.bin_edges)
    mz    = np.ones(len(x))
    for j in range(nbins):
        mbin = dig == (j+1)
        mz[mbin] = y[mbin] >= q50[j]   
    mz = mz.astype(bool)
    return x_b,q50,q25,q75,mz
            

def plot_fig(x,y,nbins,ax=plt,color = 'sienna', style = '',label=''):
                
        X,q50,q25,q75,mz = binned(x,y,nbins)
        ax.plot(X,q50,style,color=color,label=label)
        ax.plot(X,q75,style,color=color,alpha=0.2)
        ax.plot(X,q25,style,color=color,alpha=0.2)
        ax.fill_between(X,q75,q25,color = color,alpha=0.1)
        
def fit_profile(pro,z,plot=True):
    
         nrings = 20
         mp = 2.927e10
         rbins = (np.arange(nrings+1)*(pro[1]/float(nrings)))/1000
         r = (rbins[:-1] + np.diff(rbins)/2.)
         
         # mr = (r < 0.7*(pro[1]/1000.))*(r > r[1])
         mr = r > 0.
         
         mpV = mp/((4./3.)*np.pi*(rbins[1:]**3 - rbins[:-1]**3)) # mp/V
         
         rho   = pro[2:2+nrings][mr]
         rho_E = pro[2+nrings:2+2*nrings][mr]
         S     = pro[2+2*nrings:2+3*nrings][mr]
         S_E   = pro[2+3*nrings:][mr]

         r      = r[mr]

         erho = np.ones(mr.sum())*100.

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
             
             
             
             
             return [z,np.log10(rho_f.M200),rho_f.c200,
                     np.log10(rho_E_f.M200),rho_E_f.c200,
                     np.log10(S_f.M200),S_f.c200,
                     np.log10(S_E_f.M200),S_E_f.c200]
         
         else:
             
             return np.zeros(9)
                 
# r     = []
# rho   = []
# rho_E = []
# S     = []
# S_E   = []


# for j in range(len(zhalos)):
    
    # out = fit_profile(profiles[j],zhalos[j])
    # r     = np.append(r,out[0])
    # rho   = np.append(rho,out[1])
    # rho_E = np.append(rho_E,out[2])
    # S     = np.append(S,out[3])
    # S_E   = np.append(S_E,out[4])
    

s = main.c3D_mod/main.a3D_mod
q = main.b2D_mod/main.a2D_mod
sr = main.c3Dr_mod/main.a3Dr_mod
qr = main.b2Dr_mod/main.a2Dr_mod


index = np.arange(len(s))

mrelax = (rc/main.r_max < 0.05)

m = main.lMfof > 13.0

j = index[mrelax*(s < 0.4)*(main.lMfof > 14.)][6]

fit_profile(profiles[j],zhalos[j])
ax.set_xscale('log')
ax.set_yscale('log')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax.set_ylabel(r'$\rho [M_\odot h^2/Mpc^3]$')
ax.set_xlabel('$r[Mpc/h]$')
ax2.set_ylabel(r'$\Sigma [M_\odot h/Mpc^2]$')
ax2.set_xlabel('$R[Mpc/h]$')
f.savefig(plots_path+'rho_profile_'+part+'.png')
f2.savefig(plots_path+'Sigma_profile_'+part+'.png')


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
part = '8_5'
cosmo = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)

part = '8_5'

main = pd.read_csv('../catalogs/halo_props/halo_props2_8_5_2_main.csv.bz2') 
profiles = np.loadtxt('../catalogs/halo_props/halo_props2_'+part+'_2_pro.csv.bz2',skiprows=1,delimiter=',')
masses = fits.open('../catalogs/halo_props/fitted_mass_'+part+'_2.fits')[1].data
zhalos = masses.zhalo

rc = np.sqrt((main.xc_fof - main.xc_rc)**2 + (main.yc_fof - main.yc_rc)**2 + (main.zc_fof - main.zc_rc)**2)

plots_path = '/home/elizabeth/plots/'


def q_75(y):
    return np.quantile(y, 0.75)

def q_25(y):
    return np.quantile(y, 0.25)

def binned(x,y,nbins=10):
    
    
    bined = stats.binned_statistic(x,y,statistic='median', bins=nbins)
    x_b = 0.5*(bined.bin_edges[:-1] + bined.bin_edges[1:])
    q50     = bined.statistic
    
    bined = stats.binned_statistic(x,y,statistic=q_25, bins=nbins)
    q25     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic=q_75, bins=nbins)
    q75     = bined.statistic
    
    dig   = np.digitize(x,bined.bin_edges)
    mz    = np.ones(len(x))
    for j in range(nbins):
        mbin = dig == (j+1)
        mz[mbin] = y[mbin] >= q50[j]   
    mz = mz.astype(bool)
    return x_b,q50,q25,q75,mz
            

def plot_fig(x,y,nbins,ax=plt,color = 'sienna', style = '',label=''):
                
        X,q50,q25,q75,mz = binned(x,y,nbins)
        ax.plot(X,q50,style,color=color,label=label)
        ax.plot(X,q75,style,color=color,alpha=0.2)
        ax.plot(X,q25,style,color=color,alpha=0.2)
        ax.fill_between(X,q75,q25,color = color,alpha=0.1)
        
def fit_profile(pro,z,plot=True):
    
         nrings = 20
         mp = 2.927e10
         rbins = (np.arange(nrings+1)*(pro[1]/float(nrings)))/1000
         r = (rbins[:-1] + np.diff(rbins)/2.)
         
         # mr = (r < 0.7*(pro[1]/1000.))*(r > r[1])
         mr = r > 0.
         
         mpV = mp/((4./3.)*np.pi*(rbins[1:]**3 - rbins[:-1]**3)) # mp/V
         
         rho   = pro[2:2+nrings][mr]
         rho_E = pro[2+nrings:2+2*nrings][mr]
         S     = pro[2+2*nrings:2+3*nrings][mr]
         S_E   = pro[2+3*nrings:][mr]

         r      = r[mr]

         erho = np.ones(mr.sum())*100.

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
             
             
             
             
             return [z,np.log10(rho_f.M200),rho_f.c200,
                     np.log10(rho_E_f.M200),rho_E_f.c200,
                     np.log10(S_f.M200),S_f.c200,
                     np.log10(S_E_f.M200),S_E_f.c200]
         
         else:
             
             return np.zeros(9)
                 
# r     = []
# rho   = []
# rho_E = []
# S     = []
# S_E   = []


# for j in range(len(zhalos)):
    
    # out = fit_profile(profiles[j],zhalos[j])
    # r     = np.append(r,out[0])
    # rho   = np.append(rho,out[1])
    # rho_E = np.append(rho_E,out[2])
    # S     = np.append(S,out[3])
    # S_E   = np.append(S_E,out[4])
    

s = main.c3D_mod/main.a3D_mod
q = main.b2D_mod/main.a2D_mod
sr = main.c3Dr_mod/main.a3Dr_mod
qr = main.b2Dr_mod/main.a2Dr_mod


index = np.arange(len(s))

mrelax = (rc/main.r_max < 0.05)

m = main.lMfof > 13.0

j = index[mrelax*(s < 0.4)*(main.lMfof > 14.)][6]

fit_profile(profiles[j],zhalos[j])
ax.set_xscale('log')
ax.set_yscale('log')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax.set_ylabel(r'$\rho [M_\odot h^2/Mpc^3]$')
ax.set_xlabel('$r[Mpc/h]$')
ax2.set_ylabel(r'$\Sigma [M_\odot h/Mpc^2]$')
ax2.set_xlabel('$R[Mpc/h]$')
f.savefig(plots_path+'rho_profile_'+part+'.png')
f2.savefig(plots_path+'Sigma_profile_'+part+'.png')




plt.figure()
plt.scatter(zhalos[m],10**(masses.lM200_rho - main.lMfof)[m],c=(rc/main.r_max)[m],alpha=0.3,s=20,vmax=0.3)
plt.xlabel('$z$')
plt.ylabel('$M_{200}/M_{FOF}$')
plt.ylim([0,1.2])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'M_comparison_3D_'+part+'.png')

plt.figure()
plt.scatter(zhalos[m],10**(masses.lM200_S - main.lMfof)[m],c=(rc/main.r_max)[m],alpha=0.3,s=20,vmax=0.3)
plt.xlabel('$z$')
plt.ylim([0,1.2])
plt.ylabel('$M^{2D}_{200}/M_{FOF}$')
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'M_comparison_2D_'+part+'.png')

plt.figure()
plt.scatter(s[m],10**(masses.lM200_rho - masses.lM200_rho_E)[m],c=(rc/main.r_max)[m],alpha=0.3,s=20,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$M_{200}/M_{200E}$')
plt.ylim([0,3])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'M_comparison_3D_elliptical_'+part+'.png')

plt.figure()
plt.scatter(q[m],10**(masses.lM200_rho-masses.lM200_rho_E)[m],c=(rc/main.r_max)[m],alpha=0.3,s=20,vmax=0.3)
plt.xlabel('$q=b/a$')
plt.ylabel('$M^{2D}_{200}/M^{2D}_{200E}$')
plt.ylim([0,3])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'M_comparison_2D_elliptical_'+part+'.png')

plt.figure()
plt.scatter(s[m],(masses.c200_rho/masses.c200_rho_E)[m],c=(rc/main.r_max)[m],alpha=0.3,s=20,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$c_{200}/c_{200E}$')
plt.ylim([-0.5,6])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'c200_comparison_3D_elliptical_'+part+'.png')

plt.figure()
plt.scatter(q[m],(masses.c200_rho/masses.c200_rho_E)[m],c=(rc/main.r_max)[m],alpha=0.3,s=20,vmax=0.3)
plt.xlabel('$q=b/a$')
plt.ylabel('$c^{2D}_{200}/c^{2D}_{200E})$')
plt.ylim([-0.5,6])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'c200_comparison_2D_elliptical_'+part+'.png')
