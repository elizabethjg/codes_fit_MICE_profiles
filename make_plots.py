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
from models_profiles import *
cosmo = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)
from scipy import stats

# part = '4_4'
part = '8_5_2'

main2 = pd.read_csv('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_main.csv.bz2') 
profiles2 = np.loadtxt('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_pro.csv.bz2',skiprows=1,delimiter=',')
masses2 = pd.read_csv('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_mass.csv.bz2') 
zhalos = main.redshift

rc = np.array(np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2))

plots_path = '/home/elizabeth/plots/'

        
index = np.arange(len(profiles))

mp = 2.927e10
zhalos = np.array(main.redshift)


nrings = 10

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
            



def fit_profile(pro,z,plot=True,halo=''):
    
         roc_mpc = cosmo.critical_density(z).to(u.Msun/(u.Mpc)**3).value
     
         r   = pro[2:2+nrings]/1.e3         
         
         rbins = (np.arange(nrings+1)*(pro[1]/float(nrings)))/1000
         mpV = mp/((4./3.)*np.pi*(rbins[1:]**3 - rbins[:-1]**3)) # mp/V
         mpA = mp/(np.pi*(rbins[1:]**2 - rbins[:-1]**2)) # mp/A
         

         
         rho   = pro[2+nrings:2+2*nrings]
         rho_E = pro[2+2*nrings:2+3*nrings]
         S     = pro[2+3*nrings:2+4*nrings]
         S_E   = pro[2+4*nrings:]

         Vsum = (4./3.)*np.pi*(rbins[1:]**3)
         Msum = np.cumsum(rho/mpV)*mp
         j200 = np.argmin(abs((Msum/Vsum)/(200.*roc_mpc) - 1))
         MDelta = Msum[j200]
         Delta  = ((Msum/Vsum)/roc_mpc)[j200]
         
         mrho = (rho > 0.)*(r < 0.7*pro[1]*1.e-3)#*(r > 0.1*pro[1]*1.e-3)
         mS = (S > 0.)*(r < 0.7*pro[1]*1.e-3)#*(r > 0.1*pro[1]*1.e-3)
         mrhoe = (rho_E > 0.)*(r < 0.7*pro[1]*1.e-3)#*(r > 0.1*pro[1]*1.e-3)
         mSe = (S_E > 0.)*(r < 0.7*pro[1]*1.e-3)#*(r > 0.1*pro[1]*1.e-3)
         
         
         # error = 1.e12*np.ones(len(r))
         
         rho_f    = rho_fit(r[mrho],rho[mrho],mpV[mrho],z,cosmo,True)
         rho_E_f    = rho_fit(r[mrhoe],rho_E[mrhoe],mpV[mrhoe],z,cosmo,True)
         S_f      = Sigma_fit(r[mS],S[mS],mpA[mS],z,cosmo,True)
         S_E_f      = Sigma_fit(r[mSe],S_E[mSe],mpA[mSe],z,cosmo,True)
         
         if plot:
             
             
             m = rho_f.xplot > r.min()
             m1 = rho_E_f.xplot >  r.min()
             m2 = S_f.xplot >  r.min()
             m3 = S_E_f.xplot >  r.min()
             
         
             f,ax = plt.subplots()                              
             ax.fill_between(r,rho+mpV*0.5,rho-mpV*0.5,color='C0',alpha=0.5)
             ax.plot(r,rho,'C7',lw=2,label=halo)
             ax.plot(rho_f.xplot[m],rho_f.yplot[m],'k')
             ax.fill_between(r,rho_E+mpV*0.5,rho_E-mpV*0.5,color='C1',alpha=0.5)
             ax.plot(r,rho_E,'C7--',lw=2)
             ax.plot(rho_E_f.xplot[m1],rho_E_f.yplot[m1],'k--')
             ax.legend()
             ax.axvline(0.7*pro[1]*1.e-3)
             
             f2,ax2 = plt.subplots()                 
             ax2.fill_between(r,S+mpA*0.5,S-mpA*0.5,color='C0',alpha=0.5)
             ax2.plot(r,S,'C7',lw=2,label=halo)
             ax2.fill_between(r,S_E+mpA*0.5,S_E-mpA*0.5,color='C1',alpha=0.5)
             ax2.plot(r,S_E,'C7--',lw=2)
             ax2.plot(S_f.xplot[m2],S_f.yplot[m2],'k')
             ax2.plot(S_E_f.xplot[m3],S_E_f.yplot[m3],'k--')
             ax2.legend()
             ax2.axvline(0.7*pro[1]*1.e-3)
             
             ax.set_xscale('log')
             ax.set_yscale('log')
             ax2.set_xscale('log')
             ax2.set_yscale('log')
             ax.set_ylabel(r'$\rho [M_\odot h^2/Mpc^3]$')
             ax.set_xlabel('$r[Mpc/h]$')
             ax2.set_ylabel(r'$\Sigma [M_\odot h/Mpc^2]$')
             ax2.set_xlabel('$R[Mpc/h]$')
         
             f.savefig(plots_path+'profile_rho_'+part+'_'+halo+'.png')
             f2.savefig(plots_path+'profile_S_'+part+'_'+halo+'.png')
         
             print(0.7*(pro[1]/1000.))
             print(np.log10(rho_f.M200),rho_f.c200)
         
         
                 
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
    
r200 = R200_NFW(10**lMrho,zhalos,cosmo=cosmo)*1000.

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


k2 = np.array(main.EKin)
u2 = np.array(main.EPot)

mrelax2 = ((2.*k2)/abs(u2)) < 1.35

mcut = (main.lgM > 12.5)#*mrelax
lmcut = 'M125'


# RELAXATION 
m = mcut#*(zhalos < 0.7)

x,q50,q25,q75,nada = binned(zhalos[m],(rc/main.r_max)[m],20)
plt.figure()
plt.scatter(zhalos[m],(rc/main.r_max)[m],c=((2.*k2)/abs(u2))[m],alpha=0.3,s=1,vmax=1.35,zorder=2)
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.xlabel('$z$')
plt.ylabel('$r_c/r_{max}$')
plt.ylim([0.,0.3])
plt.colorbar()
plt.savefig(plots_path+'01_rc_z_'+part+'_'+lmcut+'.png')

x,q50,q25,q75,nada = binned(zhalos[m],((2.*k2)/abs(u2))[m],20)
plt.figure()
plt.scatter(zhalos[m],((2.*k2)/abs(u2))[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.xlabel('$z$')
plt.ylabel('$(2K)/U$')
plt.ylim([0.5,3])
plt.axhline(1.35)
plt.colorbar()
plt.savefig(plots_path+'02_Eratio_z_'+part+'_'+lmcut+'.png')


x,q50,q25,q75,nada = binned(main.lgM[m],(rc/main.r_max)[m],20)
plt.figure()
plt.scatter(main.lgM[m],(rc/main.r_max)[m],c=((2.*k2)/abs(u2))[m],alpha=0.3,s=2,vmax=1.35,zorder=2)
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.xlabel(r'$\log M_{FOF}$')
plt.ylabel('$r_c/r_{max}$')
plt.ylim([0.,0.6])
plt.colorbar()
plt.savefig(plots_path+'03_rc_lgM_'+part+'_'+lmcut+'.png')

x,q50,q25,q75,nada = binned(main.lgM[m],((2.*k2)/abs(u2))[m],20)
plt.figure()
plt.scatter(main.lgM[m],((2.*k2)/abs(u2))[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.xlabel(r'$\log M_{FOF}$')
plt.ylabel('$(2K)/U$')
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.ylim([0.5,3])
plt.axhline(1.35)
plt.colorbar()
plt.savefig(plots_path+'04_Eratio_lgM_'+part+'_'+lmcut+'.png')

x,q50,q25,q75,nada = binned((rc/main.r_max)[m],((2.*k2)/abs(u2))[m],20)
plt.figure()
plt.plot((rc/main.r_max)[m],((2.*k2)/abs(u2))[m],'.',alpha=0.2)
plt.axis([0,0.7,0.7,2])
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.ylabel('$(2K)/U$')
plt.xlabel('$r_c/r_{max}$')
plt.savefig(plots_path+'05_rc_Eratio_'+part+'_'+lmcut+'.png')


# SHAPES

plt.figure()
plt.scatter(sr[m],s[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.xlabel('$s_r$')
plt.ylabel('$s$')
plt.colorbar()
plt.savefig(plots_path+'06_s_'+part+'_'+lmcut+'.png')

plt.figure()
plt.scatter(qr[m],q[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.xlabel('$q_r$')
plt.ylabel('$q$')
plt.colorbar()
plt.savefig(plots_path+'07_q_'+part+'_'+lmcut+'.png')

plt.figure()
plt.scatter(q2dr[m],q2d[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.xlabel('$q2d_r$')
plt.ylabel('$q2d$')
plt.colorbar()
plt.savefig(plots_path+'08_q2d_'+part+'_'+lmcut+'.png')

# PROFILES 
msel  = (rc/main.r_max < 0.05)*(main.lgM > 13.5)*(R3D>0.4)
j =  index[msel][0]
fit_profile(profiles[j],zhalos[j],plot=True,halo='relaxed')

msel  = (rc/main.r_max > 0.4)*(main.lgM > 13.5)*(R3D<0.2)
j =  index[msel][0]
fit_profile(profiles[j],zhalos[j],plot=True,halo='unrelaxed')


msel  = (main.lgM > 13.5)*(masses2.R3D.astype(float)>1.0)
j =  index[msel][0]

plt.figure()
plt.scatter(main.lgM[m],R2D[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3,zorder=2)
plt.xlabel(r'$\log M_{FOF}$')
plt.ylabel('$R_{3D}$')
plt.ylim([0.,1])
plt.colorbar()
plt.savefig(plots_path+'09_Res_'+part+'_'+lmcut+'.png')

x,q50,q25,q75,nada = binned(zhalos[m],R3D[m],20)
plt.figure()
plt.scatter(zhalos[m],R3D[m],c=(rc/main.r_max)[m],alpha=0.3,s=4,vmax=0.3,zorder=2)
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.xlabel(r'$z$')
plt.ylabel('$R_{3D}$')
plt.ylim([0.,0.4])
plt.colorbar()
plt.savefig(plots_path+'09_Res_z_'+part+'_'+lmcut+'.png')


x,q50,q25,q75,nada = binned((rc/main.r_max)[m],R3D[m],20)
plt.figure()
plt.scatter((rc/main.r_max)[m],R3D[m],c=(rc/main.r_max)[m],alpha=0.3,s=4,vmax=0.3,zorder=2)
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.xlabel('$r_c/r_{max}$')
plt.ylabel('$R_{3D}$')
plt.ylim([0.,1.])
plt.colorbar()
plt.savefig(plots_path+'09_Res_rc_'+part+'_'+lmcut+'.png')


# MASS COMPARISON
# with FOF
m = mrho*mcut

x,q50,q25,q75,nada = binned(zhalos[m],10**(lMrho[m] - np.array(main.lgM)[m]),20)
plt.figure()
plt.scatter(zhalos[m],10**(lMrho[m] - np.array(main.lgM)[m]),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$z$')
plt.ylabel('$M_{200}/M_{FOF}$')
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.ylim([0,1.2])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'10_M_comparison_3D_'+part+'_'+lmcut+'.png')

m = mrho*mcut*(masses.Delta > 180)*(masses.Delta < 220)

x,q50,q25,q75,nada = binned(zhalos[m],10**(lMrho[m] - np.array(masses.lgMDelta.astype(float))[m]),20)
plt.figure()
plt.scatter(zhalos[m],10**(lMrho[m] - np.array(masses.lgMDelta.astype(float))[m]),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$z$')
plt.ylabel('$M_{200}/M_{\Delta}$')
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.ylim([0.2,2])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'11_M_comparison_Delta_'+part+'_'+lmcut+'.png')

m = mS*mcut

x,q50,q25,q75,nada = binned(zhalos[m],np.array(10**(lMS[m] - np.array(main.lgM)[m])),20)
plt.figure()
plt.scatter(zhalos[m],np.array(10**(lMS[m] - np.array(main.lgM)[m])),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$z$')
plt.ylim([0,1.2])
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.ylabel('$M^{2D}_{200}/M_{FOF}$')
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'12_M_comparison_2D_'+part+'_'+lmcut+'.png')

# 3D vs 2D

m = (mrho*mS)*mcut

x,q50,q25,q75,nada = binned(zhalos[m],np.array(10**(lMrho[m] - lMS[m])),20)
plt.figure()
plt.scatter(zhalos[m],np.array(10**(lMrho[m] - lMS[m])),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$z$')
plt.ylabel('$M_{200}/M^{2D}_{200}$')
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.ylim([0,3])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'13_z_M_comparison_project_'+part+'_'+lmcut+'.png')

x,q50,q25,q75,nada = binned(s[m],np.array(10**(lMrho[m] - lMS[m])),20)
plt.figure()
plt.scatter(s[m],np.array(10**(lMrho[m] - lMS[m])),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$M_{200}/M^{2D}_{200}$')
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.ylim([0,3])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'13_M_comparison_project_'+part+'_'+lmcut+'.png')

x,q50,q25,q75,nada = binned(s[m],(crho/cS)[m],20)
plt.figure()
plt.scatter(s[m],(crho/cS)[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$c_{200}/c^{2D}_{200}$')
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.ylim([-0.5,2])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'14_c200_comparison_project_'+part+'_'+lmcut+'.png')

x,q50,q25,q75,nada = binned(zhalos[m],(crho/cS)[m],20)
plt.figure()
plt.scatter(zhalos[m],(crho/cS)[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$z$')
plt.ylabel('$c_{200}/c^{2D}_{200}$')
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.ylim([-0.5,2])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'14_c200_z_comparison_project_'+part+'_'+lmcut+'.png')


m = (mrhoE*mSE)*mcut

x,q50,q25,q75,nada = binned(s[m],np.array(10**(lMrhoE[m] - lMSE[m])),20)
plt.figure()
plt.scatter(s[m],np.array(10**(lMrhoE[m] - lMSE[m])),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$M_{200}/M^{2D}_{200}$')
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.ylim([0,3])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'15_M_comparison_project_ellip_'+part+'_'+lmcut+'.png')

x,q50,q25,q75,nada = binned(s[m],(crhoE/cSE)[m],20)
plt.figure()
plt.scatter(s[m],(crhoE/cSE)[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$c_{200}/c^{2D}_{200}$')
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.ylim([-0.5,3])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'16_c200_comparison_project_ellip_'+part+'_'+lmcut+'.png')

# SPHERICAL VS ELLIPTICAL

m = (mrho*mrhoE)*mcut

x,q50,q25,q75,nada = binned(s[m],np.array(10**(lMrho[m] - lMrhoE[m])),20)
plt.figure()
plt.scatter(s[m],np.array(10**(lMrho[m] - lMrhoE[m])),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$M_{200}/M_{200E}$')
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.ylim([0,3])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'17_M_comparison_3D_elliptical_'+part+'_'+lmcut+'.png')

x,q50,q25,q75,nada = binned(s[m],(crho/crhoE)[m],20)
plt.figure()
plt.scatter(s[m],(crho/crhoE)[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$S=c/a$')
plt.ylabel('$c_{200}/c_{200E}$')
plt.ylim([-0.5,6])
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'18_c200_comparison_3D_elliptical_'+part+'_'+lmcut+'.png')


m = (mS*mSE)*mcut

x,q50,q25,q75,nada = binned(q[m],np.array(10**(lMS[m] - lMSE[m])),20)
plt.figure()
plt.scatter(q[m],np.array(10**(lMS[m] - lMSE[m])),c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.xlabel('$q=b/a$')
plt.ylabel('$M^{2D}_{200}/M^{2D}_{200E}$')
plt.ylim([0,3])
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'19_M_comparison_2D_elliptical_'+part+'_'+lmcut+'.png')


x,q50,q25,q75,nada = binned(q[m],(cS/cSE)[m],20)
plt.figure()
plt.scatter(q[m],(cS/cSE)[m],c=(rc/main.r_max)[m],alpha=0.3,s=2,vmax=0.3)
plt.plot(x,q50,'C3')
plt.plot(x,q25,'C3--')
plt.plot(x,q75,'C3--')
plt.xlabel('$q=b/a$')
plt.ylabel('$c^{2D}_{200}/c^{2D}_{200E}$')
plt.ylim([-0.5,6])
plt.axhline(1)
plt.colorbar()
plt.savefig(plots_path+'20_c200_comparison_2D_elliptical_'+part+'_'+lmcut+'.png')


