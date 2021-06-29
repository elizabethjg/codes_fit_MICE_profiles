import sys
sys.path.append('/mnt/projects/lensing/lens_codes_v3.7')
import numpy as np
from astropy.io import fits
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)
from fit_profiles_curvefit import *
from multiprocessing import Pool
from multiprocessing import Process
from scipy import stats

part = '8_5'

zhalos = fits.open('../catalogs/halo_props/halo_props2_'+part+'_plus.fits')[1].data.z_v
profiles = np.loadtxt('../catalogs/halo_props/halo_props2_'+part+'.csv_profile.bz2',skiprows=1,delimiter=',')
haloid = fits.open('../catalogs/halo_props/halo_props2_'+part+'_plus.fits')[1].data['unique_halo_id']

masses = fits.open('../catalogs/halo_props/fitted_mass_'+part+'.fits')[1].data
plus = fits.open('../catalogs/halo_props/halo_props2_'+part+'_plus.fits')[1].data
plots_path = '/home/elizabeth/plots/'

f,ax = plt.subplots()
f2,ax2 = plt.subplots()

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
        
def fit_profile(pro,z):
    
         r = np.arange(16)*(pro[1]/15.)  
         r = (r[:-1] + np.diff(r)/2.)/1000
         
         mr = (r < 0.7*pro[1])*(r > 0.1)
         
         rho   = pro[2:17][mr]
         rho_E = pro[17:32][mr]
         S     = pro[32:47][mr]
         S_E   = pro[47:][mr]

         r      = r[mr]
        
         erho = np.ones(mr.sum())*50.
         
         rho_f    = rho_fit(r[rho>0],rho[rho>0],erho[rho>0],z,cosmo,True)
         rho_E_f    = rho_fit(r[rho_E>0],rho_E[rho_E>0],erho[rho_E>0],z,cosmo,True)
         S_f      = Sigma_fit(r[S>0],S[S>0],erho[S>0],z,cosmo,True)
         S_E_f      = Sigma_fit(r[S_E>0],S_E[S_E>0],erho[S_E>0],z,cosmo,True)
         
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
         
         return [r,rho,rho_E,S,S_E]
                 
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
    

s = plus.c3D_mod/plus.a3D_mod
q = plus.b2D_mod/plus.a2D_mod
sr = plus.c3Dr_mod/plus.a3Dr_mod
qr = plus.b2Dr_mod/plus.a2Dr_mod

s_old = plus.s3d
sr_old = plus.s3dr
q_old = plus.q2d
qr_old = plus.q2dr

index = np.arange(len(s))

mrelax = (plus.rc/plus.r_max < 0.05)

j = index[mrelax*(s < 0.4)][6]

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
plt.plot(s,s_old,'.',alpha=0.5,label='standard')
plt.plot(sr,sr_old,'.',alpha=0.5,label='reduced')
plt.plot([0,1],[0,1])
plt.legend()
plt.xlabel('S=c/a')
plt.ylabel('S_old')
plt.savefig(plots_path+'s_comparison_'+part+'.png')

plt.figure()
plt.plot(q,q_old,'.',alpha=0.5,label='standard')
plt.plot(qr,qr_old,'.',alpha=0.5,label='reduced')
plt.plot([0,1],[0,1])
plt.xlabel('q=b/a')
plt.ylabel('q_old')
plt.legend()
plt.savefig(plots_path+'q_comparison_'+part+'.png')

plt.figure()
plt.scatter(zhalos,10**(masses.lM200_rho-plus.lgm),c=plus.rc/plus.r_max,alpha=0.3,s=20,vmax=0.3)
plt.xlabel('$z$')
plt.ylabel('$M_{200}/M_{FOF}$')
plt.ylim([0,1.2])
plt.colorbar()
plt.savefig(plots_path+'M_comparison_3D_'+part+'.png')

plt.figure()
plt.scatter(zhalos,10**(masses.lM200_S-plus.lgm),c=plus.rc/plus.r_max,alpha=0.3,s=20,vmax=0.3)
plt.xlabel('$z$')
plt.ylim([0,1.2])
plt.ylabel('$M^{2D}_{200}/M_{FOF}$')
plt.colorbar()
plt.savefig(plots_path+'M_comparison_2D_'+part+'.png')

plt.figure()
plt.scatter(s,10**(masses.lM200_rho-masses.lM200_rho_E),c=plus.rc/plus.r_max,alpha=0.3,s=20,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$M_{200}/M_{200E}$')
plt.ylim([0,3])
plt.colorbar()
plt.savefig(plots_path+'M_comparison_3D_elliptical_'+part+'.png')

plt.figure()
plt.scatter(q,10**(masses.lM200_rho-masses.lM200_rho_E),c=plus.rc/plus.r_max,alpha=0.3,s=20,vmax=0.3)
plt.xlabel('$q=b/a$')
plt.ylabel('$M^{2D}_{200}/M^{2D}_{200E}$')
plt.ylim([0,3])
plt.colorbar()
plt.savefig(plots_path+'M_comparison_2D_elliptical_'+part+'.png')

plt.figure()
plt.scatter(s,masses.c200_rho/masses.c200_rho_E,c=plus.rc/plus.r_max,alpha=0.3,s=20,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$c_{200}/c_{200E}$')
plt.ylim([-0.5,6])
plt.colorbar()
plt.savefig(plots_path+'c200_comparison_3D_elliptical_'+part+'.png')

plt.figure()
plt.scatter(q,masses.c200_rho/masses.c200_rho_E,c=plus.rc/plus.r_max,alpha=0.3,s=20,vmax=0.3)
plt.xlabel('$q=b/a$')
plt.ylabel('$c^{2D}_{200}/c^{2D}_{200E})$')
plt.ylim([-0.5,6])
plt.colorbar()
plt.savefig(plots_path+'c200_comparison_2D_elliptical_'+part+'.png')
