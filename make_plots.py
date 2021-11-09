import sys
import numpy as np
from astropy.io import fits
from astropy.cosmology import LambdaCDM, z_at_value
from multiprocessing import Pool
from multiprocessing import Process
import astropy.units as u
import pandas as pd
from fit_models import *
from scipy import stats
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('-file', action='store', dest='file',default='2_2')
args = parser.parse_args()
part = args.file

# part = '2_2'

print(part)

main = pd.read_csv('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_main.csv.bz2') 
profiles = np.loadtxt('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_pro.csv.bz2',skiprows=1,delimiter=',')
masses = pd.read_csv('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_mass.csv.bz2')


# overlap = main0.column_halo_id[np.in1d(main0.column_halo_id,masses.column_halo_id)]

# main     = main0.loc[overlap]
# profiles = profiles0[overlap]

# u,i = np.unique(masses.column_halo_id,return_index=True)
# masses   = masses.loc[i]

zhalos = main.redshift

rc = np.array(np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2))

plots_path = '/home/elizabeth/plots/'

        
index = np.arange(len(profiles))


zhalos = np.array(main.redshift)

mp = 2.927e10
nrings = 25

def c200_duffy(M,z,model = 'NFW'):
    #calculo de c usando la relacion de Duffy et al 2008
    if model == 'NFW':
        return 5.71*((M/2.e12)**-0.084)*((1.+z)**-0.47)
    elif model == 'Einasto':
        return 6.4*((M/2.e12)**-0.108)*((1.+z)**-0.62)


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
            

def make_plot(X,Y,Z,zlim=0.3,nbins=20):
    plt.figure()
    x,q50,q25,q75,nada = binned(X,Y,nbins)
    plt.scatter(X,Y, c=Z, alpha=0.3,s=1,vmax=zlim)
    plt.plot(x,q50,'C3')
    plt.plot(x,q25,'C3--')
    plt.plot(x,q75,'C3--')
    plt.colorbar()

def fit_profile(pro,z,plot=True,halo=''):
     
         r   = pro[2:2+nrings]/1.e3         
         
         a_t = 1./(1.+ z)
         
         rin = 10.
         rbins = ((np.arange(nrings+1)*((1000.-rin)/float(nrings)))+rin)/1000.
         mpV = mp/((4./3.)*np.pi*(rbins[1:]**3 - rbins[:-1]**3)) # mp/V
         mpA = mp/(np.pi*(rbins[1:]**2 - rbins[:-1]**2)) # mp/A
         

         
         rho   = pro[2+nrings:2+2*nrings]
         rho_E = pro[2+2*nrings:2+3*nrings]
         S     = pro[2+3*nrings:2+4*nrings]
         S_E   = pro[2+4*nrings:]
         
         mrho = (rho > 0.)*(r > 0.05)
         mS = (S > 0.)*(r > 0.05)
         mrhoe = (rho_E > 0.)*(r > 0.05)
         mSe = (S_E > 0.)*(r > 0.05)         
         
         
         if mrho.sum() > 4. and mS.sum() > 4. and mrhoe.sum() > 4. and mSe.sum() > 4.:

            rho_f    = rho_fit(r[mrho],rho[mrho],mpV[mrho],z)
            rho_E_f    = rho_fit(r[mrhoe],rho_E[mrhoe],mpV[mrhoe],z)
            S_f      = Sigma_fit(r[mS],S[mS],mpA[mS],z)
            S_E_f      = Sigma_fit(r[mSe],S_E[mSe],mpA[mSe],z)
            rho_f_E    = rho_fit(r[mrho],rho[mrho],mpV[mrho],z,'Einasto',rho_f.M200,rho_f.c200)
            rho_E_f_E    = rho_fit(r[mrhoe],rho_E[mrhoe],mpV[mrhoe],z,'Einasto',rho_f.M200,rho_f.c200)
            S_f_E      = Sigma_fit(r[mS],S[mS],mpA[mS],z,'Einasto',rho_f.M200,rho_f.c200)
            S_E_f_E      = Sigma_fit(r[mSe],S_E[mSe],mpA[mSe],z,'Einasto',rho_f.M200,rho_f.c200)
            
            if plot:
                
                
                m = rho_f.xplot > r[r>0.].min()
                m1 = rho_E_f.xplot >  r[r>0.].min()
                m2 = S_f.xplot >  r[r>0.].min()
                m3 = S_E_f.xplot > r[r>0.].min()
                
            
                f,ax = plt.subplots()                              
                ax.fill_between(r[mrho],(rho+mpV*0.5)[mrho],(rho-mpV*0.5)[mrho],color='C0',alpha=0.5)
                ax.plot(r[mrho],rho[mrho],'C7',lw=4)
                ax.plot(rho_f.xplot[m],rho_f.yplot[m],'C2',label='NFW')
                ax.plot(rho_f_E.xplot[m],rho_f_E.yplot[m],'C3',label='Einasto')
                ax.fill_between(r[mrhoe],(rho_E+mpV*0.5)[mrhoe],(rho_E-mpV*0.5)[mrhoe],color='C1',alpha=0.5)
                ax.plot(r[mrhoe],rho_E[mrhoe],'C7--',lw=4)
                ax.plot(rho_E_f.xplot[m1],rho_E_f.yplot[m1],'C2--')
                ax.plot(rho_E_f_E.xplot[m1],rho_E_f_E.yplot[m1],'C3--')
                ax.axvline(0.7*a_t*pro[1]*1.e-3)
                
                f2,ax2 = plt.subplots()                 
                ax2.fill_between(r[mS],(S+mpA*0.5)[mS],(S-mpA*0.5)[mS],color='C0',alpha=0.5)
                ax2.plot(r[mS],S[mS],'C7',lw=4)
                ax2.fill_between(r[mSe],(S_E+mpA*0.5)[mSe],(S_E-mpA*0.5)[mSe],color='C1',alpha=0.5)
                ax2.plot(r[mSe],S_E[mSe],'C7--',lw=4)
                ax2.plot(S_f.xplot[m2],S_f.yplot[m2],'C2',label='NFW')
                ax2.plot(S_f_E.xplot[m2],S_f_E.yplot[m2],'C3',label='Einasto')
                ax2.plot(S_E_f.xplot[m3],S_E_f.yplot[m3],'C2--')
                ax2.plot(S_E_f_E.xplot[m3],S_E_f_E.yplot[m3],'C3--')
                ax2.axvline(0.7*a_t*pro[1]*1.e-3)
            
                ax.legend()
                ax2.legend()
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                ax.set_ylim([1.e11,5.e15])
                ax2.set_ylim([1.e11,1.e15])
                ax.set_ylabel(r'$\rho [M_\odot h^2/Mpc^3]$')
                ax.set_xlabel('$r[Mpc/h]$')
                ax2.set_ylabel(r'$\Sigma [M_\odot h/Mpc^2]$')
                ax2.set_xlabel('$R[Mpc/h]$')            
            
                print(0.7*(pro[1]/1000.))
                print(np.log10(rho_f.M200),rho_f.c200)
         
                f.savefig(plots_path+'profile_rho_'+part+'_'+halo+'.png')
                f2.savefig(plots_path+'profile_S_'+part+'_'+halo+'.png')
         
   
         
s    = main.c3D/main.a3D
q    = main.b3D/main.a3D
q2d  = main.b2D/main.a2D
sr   = main.c3Dr/main.a3Dr
qr   = main.b3Dr/main.a3Dr
q2dr = main.b2Dr/main.a2Dr

index = np.arange(len(s))

mrelax = (rc/main.r_max < 0.05)


k2 = np.array(main.Ekin)
u2 = np.array(main.Epot)

mrelax2 = ((2.*k2)/abs(u2)) < 1.35

mcut = np.array(main.lgM > 13.5)#*mrelax
lmcut = '13.5'


# RELAXATION 
m = mcut#*(zhalos < 0.7)


make_plot(zhalos[m],(rc/main.r_max)[m],((2.*k2)/abs(u2))[m],1.35)
plt.xlabel('$z$')
plt.ylabel('$r_c/r_{max}$')
plt.ylim([0.,0.3])
plt.savefig(plots_path+'01_rc_z_'+part+'_'+lmcut+'.png')

make_plot(zhalos[m],((2.*k2)/abs(u2))[m],(rc/main.r_max)[m])
plt.xlabel('$z$')
plt.ylabel('$(2K)/U$')
plt.ylim([0.5,3])
plt.axhline(1.35)
plt.savefig(plots_path+'02_Eratio_z_'+part+'_'+lmcut+'.png')

make_plot(main.lgM[m],(rc/main.r_max)[m],((2.*k2)/abs(u2))[m],1.35)
plt.xlabel(r'$\log M_{FOF}$')
plt.ylabel('$r_c/r_{max}$')
plt.ylim([0.,0.6])
plt.savefig(plots_path+'03_rc_lgM_'+part+'_'+lmcut+'.png')

make_plot(main.lgM[m],((2.*k2)/abs(u2))[m],(rc/main.r_max)[m])
plt.xlabel(r'$\log M_{FOF}$')
plt.ylabel('$(2K)/U$')
plt.ylim([0.5,3])
plt.axhline(1.35)
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

plt.plot(sr[m*mrelax],s[m*mrelax],'C1.',alpha=0.3)

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

mfit = masses.lgMDelta > 0

lMNFW_rho   = np.array(masses.lgMNFW_rho).astype(float)[mfit]
lMNFW_rhoE  = np.array(masses.lgMNFW_rho_E).astype(float)[mfit]
lMEin_rho   = np.array(masses.lgMEin_rho).astype(float)[mfit]
lMEin_rhoE  = np.array(masses.lgMEin_rho_E).astype(float)[mfit]

cNFW_rho   = np.array(masses.cNFW_rho).astype(float)[mfit]
cNFW_rhoE  = np.array(masses.cNFW_rho_E).astype(float)[mfit]
cEin_rho   = np.array(masses.cEin_rho).astype(float)[mfit]
cEin_rhoE  = np.array(masses.cEin_rho_E).astype(float)[mfit]

lMNFW_S     = np.array(masses.lgMNFW_S).astype(float)[mfit]
lMNFW_SE    = np.array(masses.lgMNFW_S_E).astype(float)[mfit]
lMEin_S     = np.array(masses.lgMEin_S).astype(float)[mfit]
lMEin_SE    = np.array(masses.lgMEin_S_E).astype(float)[mfit]

cNFW_S     = np.array(masses.cNFW_S).astype(float)[mfit]
cNFW_SE    = np.array(masses.cNFW_S_E).astype(float)[mfit]
cEin_S     = np.array(masses.cEin_S).astype(float)[mfit]
cEin_SE    = np.array(masses.cEin_S_E).astype(float)[mfit]

alpha_rho   = np.array(masses.alpha_rho).astype(float)[mfit]
alpha_rhoE  = np.array(masses.alpha_rho_E).astype(float)[mfit]
alpha_S     = np.array(masses.alpha_S).astype(float)[mfit]
alpha_SE    = np.array(masses.alpha_S_E).astype(float)[mfit]


R3Dnfw  = np.array(masses.resNFW_rho.astype(float))[mfit]
R2Dnfw  = np.array(masses.resNFW_S.astype(float))[mfit]
R3Dein  = np.array(masses.resEin_rho.astype(float))[mfit]
R2Dein  = np.array(masses.resEin_S.astype(float))[mfit]

R3Dnfw_E  = np.array(masses.resNFW_rho_E.astype(float))[mfit]
R2Dnfw_E  = np.array(masses.resNFW_S_E.astype(float))[mfit]
R3Dein_E  = np.array(masses.resEin_rho_E.astype(float))[mfit]
R2Dein_E  = np.array(masses.resEin_S_E.astype(float))[mfit]

lgMfof = np.array(main.lgM)[mfit]
rratio = np.array(rc/main.r_max)[mfit]
mrelax = rratio < 0.05

plt.figure()
plt.hist(lgMfof,histtype='step',lw=2)
plt.xlabel('$M_{FOF}$')
plt.ylabel('$N$')
plt.savefig(plots_path+'09_Mfof_fitted_'+part+'_'+lmcut+'.png')

plt.figure()
plt.hist(zhalos[mfit],histtype='step',lw=2)
plt.xlabel('$z$')
plt.ylabel('$N$')
plt.savefig(plots_path+'09_zhalos_fitted_'+part+'_'+lmcut+'.png')



msel  = (rratio > 0.4)*(lgMfof > 13.0)*(R3Dnfw<0.2)
j =  index[mfit][msel][1]
fit_profile(profiles[j],zhalos[j],plot=True,halo='unrelaxed')

msel  = (rratio < 0.05)*(lgMfof > 13.)*(R3Dnfw<0.1)
j =  index[mfit][msel][0]
fit_profile(profiles[j],zhalos[j],plot=True,halo='relaxed')


plt.figure()
plt.hist(R3Dnfw,np.linspace(0,1,50),color='C0',label='NFW',histtype='step')
plt.hist(R3Dein,np.linspace(0,1,50),color='C1',label='Einasto',histtype='step')
plt.legend()
plt.hist(R2Dnfw,np.linspace(0,1,50),color='C0',label='NFW',lw=2,histtype='step')
plt.hist(R2Dein,np.linspace(0,1,50),color='C1',label='Einasto',lw=2,histtype='step')
plt.xlabel('$Residuals$')
plt.ylabel('$N$')
plt.savefig(plots_path+'10_Res_'+part+'_'+lmcut+'.png')


# MASS COMPARISON
# with FOF

m = np.isfinite(lMNFW_rho)
make_plot(zhalos[mfit][m],10**(lMNFW_rho[m] - lgMfof[m]),rratio[m])
plt.xlabel('$z$')
plt.ylabel('$M_{200}/M_{FOF}$')
plt.ylim([0,1.2])
plt.axhline(1)
plt.savefig(plots_path+'11_M_NFW_comparison_3D_'+part+'_'+lmcut+'.png')

m = np.isfinite(lMEin_rho)
make_plot(zhalos[mfit][m],10**(lMEin_rho[m] - lgMfof[m]),rratio[m])
plt.xlabel('$z$')
plt.ylabel('$M_{200}/M_{FOF}$')
plt.ylim([0,1.2])
plt.axhline(1)
plt.savefig(plots_path+'12_M_Ein_comparison_3D_'+part+'_'+lmcut+'.png')

m = np.isfinite(lMEin_rho)*np.isfinite(lMNFW_rho)
plt.figure()
plt.hist(10**(lMNFW_rho[m] - lgMfof[m]),np.linspace(0,2,50),color='C0',label='NFW',histtype='step')
plt.hist(10**(lMEin_rho[m] - lgMfof[m]),np.linspace(0,2,50),color='C1',label='Einasto',histtype='step')
plt.legend()
plt.xlabel('$M_{200}/M_{FOF}$')
plt.ylabel('$N$')
plt.savefig(plots_path+'13_Mrho_Mfof_'+part+'_'+lmcut+'.png')

plt.figure()
plt.hist(10**(lMNFW_S[m] - lgMfof[m]),np.linspace(0,2,50),color='C0',label='NFW',lw=2,histtype='step')
plt.hist(10**(lMEin_S[m] - lgMfof[m]),np.linspace(0,2,50),color='C1',label='Einasto',lw=2,histtype='step')
plt.legend()
plt.xlabel('$M^{2D}_{200}/M_{FOF}$')
plt.ylabel('$N$')
plt.savefig(plots_path+'14_MS_Mfof_'+part+'_'+lmcut+'.png')

m = np.isfinite(lMEin_rho)*np.isfinite(lMNFW_rho)*mrelax
plt.figure()
plt.hist(10**(lMNFW_rho[m] - lgMfof[m]),np.linspace(0,2,50),color='C0',label='NFW',histtype='step')
plt.hist(10**(lMEin_rho[m] - lgMfof[m]),np.linspace(0,2,50),color='C1',label='Einasto',histtype='step')
plt.legend()
plt.xlabel('$M_{200}/M_{FOF}$')
plt.ylabel('$N$')
plt.savefig(plots_path+'13_Mrho_Mfof_relaxed_'+part+'_'+lmcut+'.png')

plt.figure()
plt.hist(10**(lMNFW_S[m] - lgMfof[m]),np.linspace(0,2,50),color='C0',label='NFW',lw=2,histtype='step')
plt.hist(10**(lMEin_S[m] - lgMfof[m]),np.linspace(0,2,50),color='C1',label='Einasto',lw=2,histtype='step')
plt.legend()
plt.xlabel('$M^{2D}_{200}/M_{FOF}$')
plt.ylabel('$N$')
plt.savefig(plots_path+'14_MS_Mfof_relaxed_'+part+'_'+lmcut+'.png')



md = (masses.Delta[mfit] > 190)*(masses.Delta[mfit] < 200)
m = np.isfinite(lMEin_rho)*np.isfinite(lMNFW_rho)*md
plt.figure()
plt.hist(10**(lMNFW_rho[m] - np.array(masses.lgMDelta)[mfit][m]),np.linspace(0,2,50),color='C0',label='NFW',lw=2,histtype='step')
plt.hist(10**(lMEin_rho[m] - np.array(masses.lgMDelta)[mfit][m]),np.linspace(0,2,50),color='C1',label='Einasto',lw=2,histtype='step')
plt.legend()
plt.xlabel('$M_{200}/M_{\Delta}$')
plt.ylabel('$N$')
plt.savefig(plots_path+'15_Mrho_MDelta_'+part+'_'+lmcut+'.png')


# 3D vs 2D
# M_RATIO

m = np.isfinite(lMNFW_S)*np.isfinite(lMNFW_rho)*np.isfinite(R2Dnfw)
make_plot(R2Dnfw[m],np.array(10**(lMNFW_rho[m] - lMNFW_S[m])),(rc/np.array(main.r_max))[mfit][m])
plt.ylabel('$M_{200}/M^{2D}_{200}$')
plt.xlabel('$R2D$')
plt.ylim([0,3])
plt.axhline(1)
plt.savefig(plots_path+'16_MNFW_R2D_comparison_project_'+part+'_'+lmcut+'.png')


m = np.isfinite(lMEin_S)*np.isfinite(lMEin_rho)*np.isfinite(R2Dein)*(R2Dein < 1.)
make_plot(R2Dein[m],np.array(10**(lMEin_rho[m] - lMEin_S[m])),rratio[m])
plt.ylabel('$M_{200}/M^{2D}_{200}$')
plt.xlabel('$R2D$')
plt.ylim([0,3])
plt.axhline(1)
plt.savefig(plots_path+'17_MEin_R2D_comparison_project_'+part+'_'+lmcut+'.png')

m = np.isfinite(lMNFW_S)*np.isfinite(lMNFW_rho)
make_plot(np.array(s)[mfit][m],np.array(10**(lMNFW_rho[m] - lMNFW_S[m])),(rc/np.array(main.r_max))[mfit][m])
plt.ylabel('$M_{200}/M^{2D}_{200}$')
plt.xlabel('$s=c/a$')
plt.ylim([0,3])
plt.xlim([0,1])
plt.axhline(1)
plt.savefig(plots_path+'18_MNFW_S_comparison_project_'+part+'_'+lmcut+'.png')


m = np.isfinite(lMEin_S)*np.isfinite(lMEin_rho)*np.isfinite(R2Dein)*(R2Dein < 1.)
make_plot(np.array(s)[mfit][m],np.array(10**(lMEin_rho[m] - lMEin_S[m])),rratio[m])
plt.ylabel('$M_{200}/M^{2D}_{200}$')
plt.xlabel('$s=c/a$')
plt.ylim([0,3])
plt.axhline(1)
plt.savefig(plots_path+'19_MEin_S_comparison_project_'+part+'_'+lmcut+'.png')



# C RATIO

m = np.isfinite(cNFW_rho/cNFW_S)*(R2Dnfw > 0)
make_plot(R2Dnfw[m],(cNFW_rho/cNFW_S)[m],rratio[m])
plt.ylabel('$c_{200}/c^{2D}_{200}$')
plt.xlabel('$R2D$')
plt.ylim([0,1.5])
plt.axhline(1)
plt.savefig(plots_path+'20_cNFW_R2D_comparison_project_'+part+'_'+lmcut+'.png')


m = np.isfinite(cEin_rho/cEin_S)*(R2Dein < 0.8)*(R2Dein > 0)
make_plot(R2Dein[m],(cEin_rho/cEin_S)[m],rratio[m])
plt.ylabel('$c_{200}/c^{2D}_{200}$')
plt.xlabel('$R2D$')
plt.ylim([0,1.5])
plt.axhline(1)
plt.savefig(plots_path+'21_cEin_R2D_comparison_project_'+part+'_'+lmcut+'.png')

m = np.isfinite(cNFW_rho/cNFW_S)
make_plot(np.array(s)[mfit][m],(cNFW_rho/cNFW_S)[m],rratio[m])
plt.ylabel('$c_{200}/c^{2D}_{200}$')
plt.xlabel('$S$')
plt.ylim([0,1.5])
plt.axhline(1)
plt.savefig(plots_path+'22_cNFW_s_comparison_project_'+part+'_'+lmcut+'.png')


m = np.isfinite(cEin_rho/cEin_S)
make_plot(np.array(s)[mfit][m],(cEin_rho/cEin_S)[m],rratio[m])
plt.ylabel('$c_{200}/c^{2D}_{200}$')
plt.xlabel('$S$')
plt.ylim([0,1.5])
plt.axhline(1)
plt.savefig(plots_path+'19_cEin_S_comparison_project_'+part+'_'+lmcut+'.png')



# ALPHA RATIO
alpha_ratio = np.array(masses.alpha_rho[mfit]).astype(float)/np.array(masses.alpha_S[mfit]).astype(float)
m = np.isfinite(alpha_ratio)*(R2Dein < 0.8)*(R2Dein > 0)
make_plot(R2Dein[m],alpha_ratio[m],rratio[m])
plt.ylabel(r'$\alpha/ \alpha^{2D}$')
plt.xlabel('$R2D$')
plt.ylim([0,2.5])
plt.axhline(1)
plt.savefig(plots_path+'23_alpha_R2D_comparison_project_'+part+'_'+lmcut+'.png')



# SPHERICAL VS ELLIPTICAL
# MASSES
m = np.isfinite(lMNFW_rho)*np.isfinite(lMNFW_rhoE)
make_plot(np.array(s)[mfit][m],np.array(10**(lMNFW_rho[m] - lMNFW_rhoE[m])),rratio[m])
plt.xlabel('$S=c/a$')
plt.ylabel('$M_{200}/M_{200E}$')
plt.axhline(1)
plt.ylim([0.,1.5])
plt.savefig(plots_path+'21_M_NFW_comparison_3D_elliptical_'+part+'_'+lmcut+'.png')

m = np.isfinite(lMEin_rho)*np.isfinite(lMEin_rhoE)
make_plot(np.array(s)[mfit][m],np.array(10**(lMEin_rho[m] - lMEin_rhoE[m])),rratio[m])
plt.xlabel('$S=c/a$')
plt.ylabel('$M_{200}/M_{200E}$')
plt.axhline(1)
plt.ylim([0.,1.5])
plt.savefig(plots_path+'21_M_Ein_comparison_3D_elliptical_'+part+'_'+lmcut+'.png')


#CONCENTRATIONS

m = np.isfinite(cNFW_rho/cNFW_rhoE)
make_plot(np.array(s)[mfit][m],(cNFW_rho/cNFW_rhoE)[m],rratio[m])
plt.xlabel('$S=c/a$')
plt.ylabel('$c_{200}/c_{200E}$')
plt.axhline(1)
plt.ylim([0.,2.5])
plt.savefig(plots_path+'22_c_NFW_comparison_3D_elliptical_'+part+'_'+lmcut+'.png')

m = np.isfinite(cEin_rho/cEin_rhoE)
make_plot(np.array(s)[mfit][m],(cEin_rho[m]/cEin_rhoE[m]),(rc/np.array(main.r_max))[mfit][m])
plt.xlabel('$S=c/a$')
plt.ylabel('$c_{200}/c_{200E}$')
plt.axhline(1)
plt.ylim([0.,2.5])
plt.savefig(plots_path+'22_c_Ein_comparison_3D_elliptical_'+part+'_'+lmcut+'.png')


m = np.isfinite(alpha_rho/alpha_rhoE)
make_plot(np.array(s)[mfit][m],(alpha_rho[m]/alpha_rhoE[m]),(rc/np.array(main.r_max))[mfit][m])
plt.xlabel('$S=c/a$')
plt.ylabel(r'$\alpha/\alpha_{E}$')
plt.axhline(1)
plt.ylim([0.,2.5])
plt.savefig(plots_path+'22_alpha_Ein_comparison_3D_elliptical_'+part+'_'+lmcut+'.png')


# Mass/concentration
# M_RATIO

m = np.isfinite(cNFW_rho)*np.isfinite(lMNFW_rho)*(lMNFW_rho<15.)*(lMNFW_rho>11.)*(zhalos[mfit] < np.median(zhalos[mfit])+0.05)*(zhalos[mfit] > np.median(zhalos[mfit])-0.05)
make_plot(np.array(lMNFW_rho)[m],cNFW_rho[m],(rc/np.array(main.r_max))[mfit][m])
plt.plot(np.arange(13,15,0.1),c200_duffy(10**np.arange(13.,15,0.1),np.median(zhalos[mfit][m])))
plt.xlabel('$\log M_{200}$')
plt.ylabel('$c_{200}$')
plt.ylim([0,6])
plt.savefig(plots_path+'23_MNFW_c200_comparison_project_'+part+'_'+lmcut+'.png')

m = np.isfinite(cNFW_rho)*np.isfinite(lMNFW_rho)*(lMNFW_rho<14.)*(lMNFW_rho>13.)
make_plot(np.array(zhalos)[mfit][m],cNFW_rho[m],(rc/np.array(main.r_max))[mfit][m])
plt.plot(np.arange(0.2,1.5,0.1),c200_duffy(np.median(10**lMNFW_rho[m]),np.arange(0.2,1.5,0.1)))
plt.xlabel('$z$')
plt.ylabel('$c_{200}$')
plt.ylim([0,6])
plt.savefig(plots_path+'23_z_cNFW_comparison_project_'+part+'_'+lmcut+'.png')

m = np.isfinite(cEin_rho)*np.isfinite(lMEin_rho)*(lMEin_rho<15.)*(lMEin_rho>11.)*(zhalos[mfit] < np.median(zhalos[mfit])+0.05)*(zhalos[mfit] > np.median(zhalos[mfit])-0.05)*(cEin_rho > 0)#*(R3Dein < 0.1)
make_plot(np.array(lMEin_rho)[m],np.log10(cEin_rho)[m],(rc/np.array(main.r_max))[mfit][m])
plt.plot(np.arange(11,15,0.1),np.log10(c200_duffy(10**np.arange(11.,15,0.1),np.median(zhalos[mfit][m]),'Einasto')))
plt.plot(np.arange(11,15,0.1),np.log10(c200_duffy(10**np.arange(11.,15,0.1),2.,'Einasto')))
plt.xlabel('$\log M_{200}$')
plt.ylabel('$\log c_{200}$')
plt.ylim([0.,1])
plt.savefig(plots_path+'24_MEin_c200_comparison_project_'+part+'_'+lmcut+'.png')

m = np.isfinite(cEin_rho)*np.isfinite(lMEin_rho)*(lMEin_rho<14.)*(lMEin_rho>13.)
make_plot(np.array(zhalos)[mfit][m],cEin_rho[m],(rc/np.array(main.r_max))[mfit][m])
plt.plot(np.arange(0.2,1.5,0.1),c200_duffy(np.median(10**lMEin_rho[m]),np.arange(0.2,1.5,0.1),'Einasto'))
plt.xlabel('$z$')
plt.ylabel('$c_{200}$')
plt.ylim([0,6])
plt.savefig(plots_path+'24_z_cEin_comparison_project_'+part+'_'+lmcut+'.png')

m = np.isfinite(alpha_rhoE)*np.isfinite(lMEin_rhoE)*(lMEin_rhoE<15.)*(lMEin_rhoE>11.)*(zhalos[mfit] < np.median(zhalos[mfit])+0.05)*(zhalos[mfit] > np.median(zhalos[mfit])-0.05)#*(R3Dein < 0.1)
make_plot(np.array(lMEin_rhoE)[m],alpha_rhoE[m],(rc/np.array(main.r_max))[mfit][m])
plt.xlabel('$\log M_{200}$')
plt.ylabel(r'$\alpha$')
plt.ylim([0,2])


# COMPARISON NFW vs Einasto

m = np.isfinite(cEin_rho/cNFW_rho)*np.isfinite(R3Dnfw)*(R3Dnfw > 0)
make_plot(R3Dnfw[m],(cEin_rho[m]/cNFW_rho[m]),(rc/np.array(main.r_max))[mfit][m])
plt.ylabel('$c^{Ein}_{200}/c^{NFW}_{200}$')
plt.xlabel('$R^{NFW}_{3D}$')
plt.axhline(1)
plt.ylim([0.5,2.])
plt.savefig(plots_path+'25_cEinNFW_comparison_'+part+'_'+lmcut+'.png')

m = np.isfinite(lMEin_rho)*np.isfinite(lMNFW_rho)*(lMEin_rho>10)*(lMNFW_rho>10)
make_plot(R3Dnfw[m],10**(lMEin_rho[m]-lMNFW_rho[m]),(rc/np.array(main.r_max))[mfit][m])
plt.ylabel('$M^{Ein}_{200}/M^{NFW}_{200}$')
plt.xlabel('$R^{NFW}_{3D}$')
plt.axhline(1)
plt.ylim([0.5,2.])
plt.savefig(plots_path+'26_MEinNFW_comparison_'+part+'_'+lmcut+'.png')

