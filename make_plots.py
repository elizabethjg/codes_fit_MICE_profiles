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

part = '4_4'

main = pd.read_csv('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_main.csv.bz2') 
profiles = np.loadtxt('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_pro.csv.bz2',skiprows=1,delimiter=',')
masses = pd.read_csv('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_mass.csv.bz2') 
zhalos = main.redshift

rc = np.array(np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2))

plots_path = '/home/elizabeth/plots/'

        
def fit_profile(pro,z,plot=True):
    
         nrings = 10
         mp = 2.927e10
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

                 
                 m = rho_f.xplot > r.min()
                 m1 = rho_E_f.xplot >  r.min()
                 m2 = S_f.xplot >  r.min()
                 m3 = S_E_f.xplot >  r.min()
             
                 f,ax = plt.subplots()
                 ax.plot(r,rho,'C7o',lw=2)
                 ax.plot(r,rho,'C7',lw=2)
                 ax.plot(rho_f.xplot[m],rho_f.yplot[m],'k')
                 ax.plot(r,rho_E,'C7x',lw=2)
                 ax.plot(r,rho_E,'C7--',lw=2)
                 ax.plot(rho_E_f.xplot[m1],rho_E_f.yplot[m1],'k--')
                 
                 f2,ax2 = plt.subplots()                 
                 ax2.plot(r,S,'C7',lw=2)
                 ax2.plot(r,S,'C7o',lw=2)
                 ax2.plot(r,S_E,'C7--',lw=2)
                 ax2.plot(r,S_E,'C7x',lw=2)
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
             
             
                 f.savefig(plots_path+'rho_profile_'+part+'.png')
                 f2.savefig(plots_path+'Sigma_profile_'+part+'.png')

             
             
             return [z,np.log10(rho_f.M200),rho_f.c200,
                     np.log10(rho_E_f.M200),rho_E_f.c200,
                     np.log10(S_f.M200),S_f.c200,
                     np.log10(S_E_f.M200),S_E_f.c200]
         
         else:
             
             return np.zeros(9)
                 
lMrho   = np.array(masses.lgM200_rho).astype(float)
lMrhoE  = np.array(masses.lgM200_rho_E).astype(float)
elMrho  = np.array(masses.e_lgM200_rho).astype(float)
elMrhoE = np.array(masses.e_lgM200_rho_E).astype(float)

crho   = np.array(masses.c200_rho).astype(float)
crhoE  = np.array(masses.c200_rho_E).astype(float)
ecrho  = np.array(masses.e_c200_rho).astype(float)
ecrhoE = np.array(masses.e_c200_rho_E).astype(float)

lMS     = np.array(masses.lgM200_S).astype(float)
lMSE    = np.array(masses.lgM200_S_E).astype(float)
elMS    = np.array(masses.e_lgM200_S).astype(float)
elMSE   = np.array(masses.e_lgM200_S_E).astype(float)

cS     = np.array(masses.c200_S).astype(float)
cSE    = np.array(masses.c200_S_E).astype(float)
ecS    = np.array(masses.e_c200_S).astype(float)
ecSE   = np.array(masses.e_c200_S).astype(float)

mrho   = (elMrho/lMrho < 0.5)*(ecrho/crho < 0.5)
mrhoE  = (elMrhoE/lMrhoE < 0.5)*(ecrhoE/crhoE < 0.5)
mS      = (elMS/lMS < 0.5)*(ecS/cS < 0.5)
mSE     = (elMSE/lMSE < 0.5)*(ecSE/cSE < 0.5)
    

s    = main.c3D/main.a3D
q    = main.b3D/main.a3D
q2d  = main.b2D/main.a2D
sr   = main.c3Dr/main.a3Dr
qr   = main.b3Dr/main.a3Dr
q2dr = main.b2Dr/main.a2Dr

index = np.arange(len(s))

mrelax = (rc/main.r_max < 0.05)

mR3D = np.isfinite(masses.R3D.astype(float))
mR2D = np.isfinite(masses.R2D.astype(float))

R3D  = np.array(masses.R3D.astype(float))
R2D  = np.array(masses.R2D.astype(float))


j = index[mrelax*(q < 0.4)*(main.lgM > 13.5)][6]

# fit_profile(profiles[j],zhalos[j])

k2 = np.array(main.Ekin)
u2 = np.array(main.Epot)

mrelax2 = ((2.*k2)/abs(u2)) < 1.35

mcut = (main.lgM > 12.5)#*mrelax
lmcut = 'M125'

m = mcut

plt.figure()
plt.scatter(zhalos[m],((2.*k2)/abs(u2))[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.xlabel('$z$')
plt.ylabel('$(2K)/U$')
plt.ylim([0.5,3])
plt.axhline(1.35)
plt.colorbar()
plt.savefig(plots_path+'Eratio_'+part+'_'+lmcut+'.png')


plt.figure()
plt.scatter(sr[m],s[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.xlabel('$s_r$')
plt.ylabel('$s$')
plt.colorbar()
plt.savefig(plots_path+'s_'+part+'_'+lmcut+'.png')

plt.figure()
plt.scatter(qr[m],q[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.xlabel('$q_r$')
plt.ylabel('$q$')
plt.colorbar()
plt.savefig(plots_path+'q_'+part+'_'+lmcut+'.png')

plt.figure()
plt.scatter(q2dr[m],q2d[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.xlabel('$q2d_r$')
plt.ylabel('$q2d$')
plt.colorbar()
plt.savefig(plots_path+'q2d_'+part+'_'+lmcut+'.png')

m = mrho*mcut
plt.figure()
plt.scatter(zhalos[m],10**(lMrho[m] - np.array(main.lgM)[m]),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$z$')
plt.ylabel('$M_{200}/M_{FOF}$')
plt.ylim([0,1.2])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'M_comparison_3D_'+part+'_'+lmcut+'.png')

m = mS*mcut
plt.figure()
plt.scatter(zhalos[m],np.array(10**(lMS[m] - np.array(main.lgM)[m])),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$z$')
plt.ylim([0,1.2])
plt.ylabel('$M^{2D}_{200}/M_{FOF}$')
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'M_comparison_2D_'+part+'_'+lmcut+'.png')

m = (mrho*mrhoE)*mcut
plt.figure()
plt.scatter(s[m],np.array(10**(lMrho[m] - lMrhoE[m])),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$M_{200}/M_{200E}$')
plt.ylim([0,3])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'M_comparison_3D_elliptical_'+part+'_'+lmcut+'.png')

plt.figure()
plt.scatter(s[m],(crho/crhoE)[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$c_{200}/c_{200E}$')
plt.ylim([-0.5,6])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'c200_comparison_3D_elliptical_'+part+'_'+lmcut+'.png')


m = (mS*mSE)*mcut
plt.figure()
plt.scatter(q[m],np.array(10**(lMS[m] - lMSE[m])),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$q=b/a$')
plt.ylabel('$M^{2D}_{200}/M^{2D}_{200E}$')
plt.ylim([0,3])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'M_comparison_2D_elliptical_'+part+'_'+lmcut+'.png')


plt.figure()
plt.scatter(q[m],(cS/cSE)[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$q=b/a$')
plt.ylabel('$c^{2D}_{200}/c^{2D}_{200E}$')
plt.ylim([-0.5,6])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'c200_comparison_2D_elliptical_'+part+'_'+lmcut+'.png')


m = (mrho*mS)*mcut
plt.figure()
plt.scatter(s[m],np.array(10**(lMrho[m] - lMS[m])),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$M_{200}/M^{2D}_{200}$')
plt.ylim([0,3])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'M_comparison_project_'+part+'_'+lmcut+'.png')

plt.figure()
plt.scatter(s[m],(crho/cS)[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$c_{200}/c^{2D}_{200}$')
plt.ylim([-0.5,6])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'c200_comparison_project_'+part+'_'+lmcut+'.png')

m = (mrhoE*mSE)*mcut
plt.figure()
plt.scatter(s[m],np.array(10**(lMrhoE[m] - lMSE[m])),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$M_{200}/M^{2D}_{200}$')
plt.ylim([0,3])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'M_comparison_project_ellip_'+part+'_'+lmcut+'.png')

plt.figure()
plt.scatter(s[m],(crhoE/cSE)[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$c_{200}/c^{2D}_{200}$')
plt.ylim([-0.5,6])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'c200_comparison_project_ellip_'+part+'_'+lmcut+'.png')


plt.figure()
plt.hist(R3D[mcut*~mrelax*mR3D],np.linspace(0,100,50),histtype='step',density=True,color='C0')
plt.hist(R3D[mcut*mrelax*mR3D],np.linspace(0,100,50),histtype='step',density=True,label='relaxed',color='C1')
plt.legend()
plt.hist(R2D[mcut*~mrelax*mR2D],np.linspace(0,100,50),histtype='step',density=True,lw=2,color='C0')
plt.hist(R2D[mcut*mrelax*mR2D],np.linspace(0,100,50),histtype='step',density=True,label='relaxed',lw=2,color='C1')
plt.xlabel('$N$')
plt.ylabel('$R$')
plt.savefig(plots_path+'res_'+part+'_'+lmcut+'.png')


plt.figure()
plt.hist((ecrho/crho)[mcut*~mrelax],np.linspace(0,0.2,50),histtype='step',density=True,color='C0')
plt.hist((ecrho/crho)[mcut*mrelax],np.linspace(0,0.2,50),histtype='step',density=True,label='relaxed',color='C1')
plt.legend()
plt.hist((ecS/cS)[mcut*~mrelax],np.linspace(0,0.2,50),histtype='step',density=True,lw=2,color='C0')
plt.hist((ecS/cS)[mcut*mrelax],np.linspace(0,0.2,50),histtype='step',density=True,label='relaxed',lw=2,color='C1')
plt.xlabel('$N$')
plt.ylabel('$\epsilon_{c_{200}}$')
plt.savefig(plots_path+'epsilon_c200_'+part+'_'+lmcut+'.png')

plt.figure()
plt.hist((elMrho/lMrho)[mcut],np.linspace(0,0.004,50),histtype='step',density=True,color='C0')
plt.hist((elMrho/lMrho)[mcut*mrelax],np.linspace(0,0.004,50),histtype='step',density=True,label='relaxed',color='C1')
plt.legend()
plt.hist((elMS/lMS)[mcut],np.linspace(0,0.004,50),histtype='step',density=True,lw=2,color='C0')
plt.hist((elMS/lMS)[mcut*mrelax],np.linspace(0,0.004,50),histtype='step',density=True,label='relaxed',lw=2,color='C1')
plt.xlabel('$N$')
plt.ylabel('$\epsilon_{M_{200}}$')
plt.savefig(plots_path+'epsilon_M200_'+part+'_'+lmcut+'.png')

plt.figure()
plt.hist((ecrhoE/crhoE)[mcut],np.linspace(0,0.2,50),histtype='step',density=True,color='C0')
plt.hist((ecrhoE/crhoE)[mcut*mrelax],np.linspace(0,0.2,50),histtype='step',density=True,label='relaxed',color='C1')
plt.legend()
plt.hist((ecSE/cSE)[mcut],np.linspace(0,0.2,50),histtype='step',density=True,lw=2,color='C0')
plt.hist((ecSE/cSE)[mcut*mrelax],np.linspace(0,0.2,50),histtype='step',density=True,label='relaxed',lw=2,color='C1')
plt.xlabel('$N$')
plt.ylabel('$\epsilon_{c_{200}}$')
plt.savefig(plots_path+'epsilon_c200_ellip_'+part+'_'+lmcut+'.png')

plt.figure()
plt.hist((elMrhoE/lMrhoE)[mcut],np.linspace(0,0.02,50),histtype='step',density=True,color='C0')
plt.hist((elMrhoE/lMrhoE)[mcut*mrelax],np.linspace(0,0.02,50),histtype='step',density=True,label='relaxed',color='C1')
plt.legend()
plt.hist((elMSE/lMSE)[mcut],np.linspace(0,0.02,50),histtype='step',density=True,lw=2,color='C0')
plt.hist((elMSE/lMSE)[mcut*mrelax],np.linspace(0,0.02,50),histtype='step',density=True,label='relaxed',lw=2,color='C1')
plt.xlabel('$N$')
plt.ylabel('$\epsilon_{M_{200}}$')
plt.savefig(plots_path+'epsilon_M200_ellip_'+part+'_'+lmcut+'.png')
