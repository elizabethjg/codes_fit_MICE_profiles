import numpy as np
import pandas as pd
from astropy.io import fits
sys.path.append('/home/elizabeth/lens_codes_v3.7')
from fit_models import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from models_profiles import *

# main = pd.read_csv('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv1.0/catalogs/halo_props2_8_5_pru_main.csv.bz2')
# profiles = np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv1.0/catalogs/halo_props2_8_5_pru_pro.csv.bz2',skiprows=1,delimiter=',')
main = fits.open('../halo_subset_filt.fits')[1].data

lmap  = fits.open('/home/elizabeth/maps/map_halo_subset.fits')[1].data
lmap2 = fits.open('/home/elizabeth/maps/map_halo_subset_lr.fits')[1].data
pro   = fits.open('/home/elizabeth/profiles/profile_halo_subset.fits')[1].data
zmean  = fits.open('/home/elizabeth/profiles/profile_halo_subset.fits')[0].header['Z_MEAN']
fit_NFW = fits.open('/home/elizabeth/profiles/fitresults_2h_2q_350_5000_profile_halo_subset.fits')[0].header
fit_Ein = fits.open('/home/elizabeth/profiles/fitresults_2h_2q_Ein_350_5000_profile_halo_subset.fits')[0].header

mlmap = (np.abs(lmap.xmpc) < 6.) & (np.abs(lmap.ympc) < 6.)
mlmap2 = (np.abs(lmap2.xmpc) < 6.) & (np.abs(lmap2.ympc) < 6.)
lmap = lmap[mlmap]
lmap2 = lmap2[mlmap2]

ides = main.row_id.astype(int)

offset = main.offset

Eratio = (2.*main.ekin/abs(main.epot))


relax = (offset < 0.1)*(Eratio < 1.35)

nrings = 50


# for halo in range(len(list_n)):

Xp  = np.array([])
Yp  = np.array([])
    
X   = np.array([])
Y   = np.array([])
Z   = np.array([])
    
Xe  = np.array([])
Ye  = np.array([])
Ze  = np.array([])

Xep = np.array([])
Yep = np.array([])

RMAX = np.array([])
zhalos = np.array([])


nhalos = 0

for j in range(len(ides)):

    # if relax[j]:
    
        part0 = np.loadtxt('../ind_halos/'+str(main.cat_x[j])+'_'+str(main.cat_y[j])+'/coords'+str(ides[j])).T  
    
    
        x,y,z = part0[0],part0[1],part0[2]
        xp,yp = part0[6],part0[7]
        xe,ye,ze = part0[3],part0[4],part0[5]
        xep,yep = part0[8],part0[9]
        
        a_t = 1/(1.+main.redshift[j])
    
        RMAX = np.append(RMAX,a_t*main.r_max[j])
        zhalos = np.append(zhalos,main.redshift[j])
    
        
        Xp = np.append(Xp,a_t*xp)
        Yp = np.append(Yp,a_t*yp)
    
        X = np.append(X,a_t*x)
        Y = np.append(Y,a_t*y)
        Z = np.append(Z,a_t*z)
    
        Xep = np.append(Xep,a_t*xep)
        Yep = np.append(Yep,a_t*yep)
    
        Xe = np.append(Xe,a_t*xe)
        Ye = np.append(Ye,a_t*ye)
        Ze = np.append(Ze,a_t*ze)
        
        nhalos +=1
    # else:
        # continue
    
    

rin = 10.
mp = 2.927e10
step = (1000.-rin)/float(nrings)

s = 1.
q = 1.
q2 = 1.

rhop = np.zeros(nrings)

Sp = np.zeros(nrings)
rp = np.zeros(nrings)
mpV = np.zeros(nrings)
mpA = np.zeros(nrings)


Ntot = 0

ring = 0

rmax = np.max([0.7*np.mean(RMAX),220.])

while ring < (nrings-1) and (rin+step < rmax):
    
    abin_in = rin/(q*s)**(1./3.)
    bbin_in = abin_in*q
    cbin_in = abin_in*s

    abin_out = (rin+step)/(q*s)**(1./3.)
    bbin_out = abin_out*q
    cbin_out = abin_out*s
    
    rp[ring] = (rin + 0.5*step)/1.e3
    
    rpart_E_in = (X**2/abin_in**2 + Y**2/bbin_in**2 + Z**2/cbin_in**2)
    rpart_E_out = (X**2/abin_out**2 + Y**2/bbin_out**2 + Z**2/cbin_out**2)
    
    V    = (4./3.)*np.pi*(((rin+step)/1.e3)**3 - (rin/1.e3)**3)
    mask = (rpart_E_in >= 1)*(rpart_E_out < 1)
    rhop[ring] = (mask.sum()*mp)/V
    mpV[ring] = mp/V

    # print(mask.sum())

    abin_in = rin/np.sqrt(q2) 
    bbin_in = abin_in*q2

    abin_out = (rin+step)/np.sqrt(q2) 
    bbin_out = abin_out*q2

    rpart_E_in = (Xp**2/abin_in**2 + Yp**2/bbin_in**2)
    rpart_E_out = (Xp**2/abin_out**2 + Yp**2/bbin_out**2)
        
    A    = np.pi*(((rin+step)/1.e3)**2 - (rin/1.e3)**2)
    mask = (rpart_E_in >= 1)*(rpart_E_out < 1)

    Sp[ring] = (mask.sum()*mp)/A
       
    mpA[ring] = mp/A
    rin += step
    ring += 1
    
z = np.mean(zhalos)   

mr = rp > 0.

rhof    = rho_fit(rp[mr],rhop[mr]/nhalos,mpV[mr]/nhalos,z)
Sf      = Sigma_fit(rp[mr],Sp[mr]/nhalos,mpA[mr]/nhalos,z)
rhof_E    = rho_fit(rp[mr],rhop[mr]/nhalos,mpV[mr]/nhalos,z,'Einasto',rhof.M200,rhof.c200)
Sf_E      = Sigma_fit(rp[mr],Sp[mr]/nhalos,mpA[mr]/nhalos,z,'Einasto',rhof.M200,rhof.c200)
# rhof_Ea    = rho_fit(rp[mr],rhop[mr]/nhalos,mpV[mr]/nhalos,z,'Einasto',rhof.M200,rhof.c200,False)
# Sf_Ea      = Sigma_fit(rp[mr],Sp[mr]/nhalos,mpA[mr]/nhalos,z,'Einasto',rhof.M200,rhof.c200,False)

mr2 = rp > 0.05

rhof2    = rho_fit(rp[mr2],rhop[mr2]/nhalos,mpV[mr2]/nhalos,z)
Sf2      = Sigma_fit(rp[mr2],Sp[mr2]/nhalos,mpA[mr2]/nhalos,z)
rhof_E2    = rho_fit(rp[mr2],rhop[mr2]/nhalos,mpV[mr2]/nhalos,z,'Einasto',rhof2.M200,rhof2.c200)
Sf_E2      = Sigma_fit(rp[mr2],Sp[mr2]/nhalos,mpA[mr2]/nhalos,z,'Einasto',rhof2.M200,rhof2.c200)
# rhof_E2a    = rho_fit(rp[mr2],rhop[mr2]/nhalos,mpV[mr2]/nhalos,z,'Einasto',rhof2.M200,rhof2.c200,False)
# Sf_E2a      = Sigma_fit(rp[mr2],Sp[mr2]/nhalos,mpA[mr2]/nhalos,z,'Einasto',rhof2.M200,rhof2.c200,False)

DS_NFW_2h   = Delta_Sigma_NFW_2h_parallel(pro.Rp,zmean,M200 = 10**fit_NFW['lM200'],c200=fit_NFW['c200'],cosmo_params=params,terms='2h',ncores=30)
DS_NFW_1h   = Delta_Sigma_NFW_2h_parallel(pro.Rp,zmean,M200 = 10**fit_NFW['lM200'],c200=fit_NFW['c200'],cosmo_params=params,terms='1h',ncores=30)
DS_Ein_2h   = Delta_Sigma_Ein_2h_parallel(pro.Rp,zmean,M200 = 10**fit_Ein['lM200'],c200=fit_Ein['c200'],cosmo_params=params,terms='2h',ncores=30,alpha=fit_Ein['alpha'])
DS_Ein_1h   = Delta_Sigma_Ein_2h_parallel(pro.Rp,zmean,M200 = 10**fit_Ein['lM200'],c200=fit_Ein['c200'],cosmo_params=params,terms='1h',ncores=30,alpha=fit_Ein['alpha'])


Xp = Xp*1.e-3
Yp = Yp*1.e-3
X  = X*1.e-3
Y  = Y*1.e-3
Z  = Z*1.e-3

fig = plt.figure(figsize=(12,12))    
# f.subplots_adjust(hspace=0,wspace=0)

ax = [fig.add_subplot(3,2,2),fig.add_subplot(3,2,4)]

ax[0].plot(rp[mr],1.e-12*rhop[mr]/nhalos,'C7',lw=4)
ax[0].plot(rp[mr2],1.e-12*rhop[mr2]/nhalos,'k--',lw=4)
# ax[0].plot(rhof.xplot,1.e-12*rhof2.yplot,'orangered',label='$\log{M_{200}}$ = '+str(np.round(np.log10(rhof.M200),1))+', $c_{200}$ = '+str(np.round(rhof.c200,1)),lw=1.5)
ax[0].plot(rhof2.xplot,1.e-12*rhof2.yplot,'-',color='orangered',label='NFW - $\log{M_{200}}$ = '+str(np.round(np.log10(rhof2.M200),2))+', $c_{200}$ = '+str(np.round(rhof2.c200,1)),lw=1.5)
# ax[0].plot(rhof_E.xplot,1.e-12*rhof_E.yplot,'seagreen',label='$\log{M_{200}}$ = '+str(np.round(np.log10(rhof_E.M200),1))+', $c_{200}$ = '+str(np.round(rhof_E.c200,1))+r', $\alpha$ = '+str(np.round(rhof_E.alpha,1)),lw=1.5)
ax[0].plot(rhof_E2.xplot,1.e-12*rhof_E2.yplot,'-',color='seagreen',label='Eintasto - $\log{M_{200}}$ = '+str(np.round(np.log10(rhof_E2.M200),2))+', $c_{200}$ = '+str(np.round(rhof_E2.c200,1))+r', $\alpha$ = '+str(np.round(rhof_E2.alpha,1)),lw=1.5)
# ax[0].plot(rhof_E2a.xplot,rhof_E2a.yplot,'--',color='gold',label='$\log{M_{200}}$ = '+str(np.round(np.log10(rhof_E2a.M200),1))+', $c_{200}$ = '+str(np.round(rhof_E2a.c200,1))+r', $\alpha$ = '+str(np.round(rhof_E2a.alpha,1)),lw=1.5)

ax[1].plot(rp[mr],1.e-12*Sp[mr]/nhalos,'C7',lw=4)
ax[1].plot(rp[mr2],1.e-12*Sp[mr2]/nhalos,'k--',lw=4)
# ax[1].plot(Sf.xplot,1.e-12*Sf.yplot,'orangered',label='$\log{M_{200}}$ = '+str(np.round(np.log10(Sf.M200),1))+', $c_{200}$ = '+str(np.round(Sf.c200,1)),lw=1.5)
ax[1].plot(rhof2.xplot,1.e-12*Sf2.yplot,'-',color='orangered',label='NFW - $\log{M_{200}}$ = '+str(np.round(np.log10(Sf2.M200),2))+', $c_{200}$ = '+str(np.round(Sf2.c200,1)),lw=1.5)
# ax[1].plot(rhof_E.xplot,1.e-12*Sf_E.yplot,'seagreen',label='$\log{M_{200}}$ = '+str(np.round(np.log10(Sf_E.M200),1))+', $c_{200}$ = '+str(np.round(Sf_E.c200,1))+r', $\alpha$ = '+str(np.round(Sf_E.alpha,1)),lw=1.5)
ax[1].plot(rhof_E2.xplot,1.e-12*Sf_E2.yplot,'-',color='seagreen',label='Einasto - $\log{M_{200}}$ = '+str(np.round(np.log10(Sf_E2.M200),2))+', $c_{200}$ = '+str(np.round(Sf_E2.c200,1))+r', $\alpha$ = '+str(np.round(Sf_E2.alpha,1)),lw=1.5)
# ax[1].plot(rhof_E2a.xplot,Sf_E2a.yplot,'--',color='gold',label='$\log{M_{200}}$ = '+str(np.round(np.log10(Sf_E2a.M200),1))+', $c_{200}$ = '+str(np.round(Sf_E2a.c200,1))+r', $\alpha$ = '+str(np.round(Sf_E2a.alpha,1)),lw=1.5)

# ax[0].set_xlim([0.01,1.0])
# ax[1].set_xlim([0.01,1.0])
ax[0].set_xlim([0.03,1.0])
ax[1].set_xlim([0.03,1.0])
ax[0].set_ylim([1,1.e4]) 
ax[1].set_ylim([5,1.e3]) 
# ax[1].set_xlabel('$r[Mpc/h]$')
ax[0].set_ylabel(r'$\rho [M_\odot h^2/pc^3]$')
ax[1].set_ylabel(r'$\Sigma [M_\odot h/pc^2]$')
ax[0].loglog()
ax[1].loglog() 
ax[0].legend(loc=3,frameon=False) 
ax[1].legend(loc=3,frameon=False) 

ax = [fig.add_subplot(3,2,1,projection='3d'),fig.add_subplot(3,2,3)]

lim = 1.199
m   = (abs(X) < lim) & (abs(Y) < lim) & (abs(Z) < lim)


ax[0].plot(X[m],Y[m],Z[m], ',', alpha = 0.5,color='C7')
ax[0].plot(X[m],Y[m],Z[m], ',', alpha = 0.04,color='seagreen')
ax[0].plot(X[m],Y[m],Z[m], ',', alpha = 0.006,color='orangered')

ax[1].plot(Xp,Yp, ',', alpha = 0.5,color='C7')
ax[1].plot(Xp,Yp, ',', alpha = 0.04,color='seagreen')
ax[1].plot(Xp,Yp, ',', alpha = 0.006,color='orangered')

ax[0].set_xlim([-1.2,1.2])
ax[0].set_ylim([-1.2,1.2])
ax[0].set_zlim([-1.2,1.2])
ax[1].set_xlim([-1.8,1.8])
ax[1].set_ylim([-1,1])

ax[0].set_xlabel('$X [Mpc/h]$')
ax[0].set_ylabel('$Y [Mpc/h]$')
ax[0].set_zlabel('$Z [Mpc/h]$')

ax[1].set_xlabel('$x [Mpc/h]$')
ax[1].set_ylabel('$y [Mpc/h]$')


ax = [fig.add_subplot(3,2,5),fig.add_subplot(3,2,6)]

ax[0].scatter(lmap2.xmpc,lmap2.ympc,c=np.log10(lmap2.GT),s=800,vmin=0,vmax=2.,cmap='autumn')
im = ax[0].scatter(lmap.xmpc,lmap.ympc,c=np.log10(lmap.GT),s=50,vmin=0,vmax=2.,cmap='autumn')
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.1)
fig.colorbar(im, cax=cax, orientation='vertical')
ax[0].set_xlabel('$x [Mpc/h]$')
ax[0].set_ylabel('$y [Mpc/h]$')


ax[1].plot(pro.Rp,pro.DSigma_T,'C7',lw=3)
ax[1].plot(pro.Rp,DS_NFW_1h+DS_NFW_2h,'-',color='orangered',label='NFW - $\log{M_{200}}$ = '+str(np.round(fit_NFW['lM200'],2))+', $c_{200}$ = '+str(np.round(fit_NFW['c200'],1)),lw=1.5)
ax[1].plot(pro.Rp,DS_NFW_2h,':',color='orangered',lw=1.5)
ax[1].plot(pro.Rp,DS_NFW_1h,'--',color='orangered',lw=1.5)
ax[1].plot(pro.Rp,DS_Ein_1h+DS_Ein_2h,'-',color='seagreen',label='Einasto - $\log{M_{200}}$ = '+str(np.round(fit_Ein['lM200'],2))+', $c_{200}$ = '+str(np.round(fit_Ein['c200'],1))+r', $\alpha$ = '+str(np.round(fit_Ein['alpha'],1)),lw=1.5)
ax[1].plot(pro.Rp,DS_Ein_1h,'--',color='seagreen',lw=1.5)
ax[1].plot(pro.Rp,DS_Ein_2h,':',color='seagreen',lw=1.5)


ax[1].set_xlabel('$r[Mpc/h]$')
ax[1].set_ylabel(r'$\Delta \Sigma [M_\odot h/pc^2]$')
ax[1].set_xlim([0.1,10])
ax[1].set_ylim([2,200]) 
ax[1].loglog()
ax[1].legend(loc=3,frameon=False) 
# fig.tight_layout(pad=5.0)
fig.subplots_adjust(wspace=0.35)
fig.savefig('../profile_stack.png',bbox_inches='tight')

