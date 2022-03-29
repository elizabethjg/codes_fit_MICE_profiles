import numpy as np
import pandas as pd
sys.path.append('/mnt/projects/lensing/lens_codes_v3.7')
sys.path.append('/mnt/projects/lensing/HALO_SHAPE/MICEv2.0/codes_fit_MICE_profiles')
from fit_models import *


# list_n = np.loadtxt('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/ind_halos_lM/cord.list',dtype=str)
# main = pd.read_csv('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/halo_props2_8_5_pru_lM_main.csv.bz2')
# profiles = np.loadtxt('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/halo_props2_8_5_pru_lM_pro.csv.bz2',skiprows=1,delimiter=',')

# list_n = np.loadtxt('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/ind_halos/cord.list',dtype=str)
# main = pd.read_csv('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/halo_props2_8_5_pru_main.csv.bz2')
# profiles = np.loadtxt('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/halo_props2_8_5_pru_pro.csv.bz2',skiprows=1,delimiter=',')

list_n = np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv1.0/catalogs/ind_halos/cord.list',dtype=str)
main = pd.read_csv('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv1.0/catalogs/halo_props2_8_5_pru_main.csv.bz2')
profiles = np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv1.0/catalogs/halo_props2_8_5_pru_pro.csv.bz2',skiprows=1,delimiter=',')


ides = np.array([])

for j in range(len(list_n)):
    ides = np.append(ides,int(list_n[j][6:]))

ides = np.sort(ides).astype(int)

rc = np.array(np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2))/np.array(main.r_max)
Eratio = (2.*main.EKin/abs(main.EPot))

s = np.array(main.c3D/main.a3D)

relax = (rc < 1)*(Eratio < 1.35)*(np.array(main.redshift) > 1.25)#*(s < 0.4)

# nrings = 500
nrings = 25


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
# for j in range(len(list_n)):
for j in [67]:
# for j in [69]:


    
    if relax[j]:
    
        # part0 = np.loadtxt('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/ind_halos_lM/coords'+str(ides[j])).T  
        # part0 = np.loadtxt('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/ind_halos/coords'+str(ides[j])).T  
        part0 = np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv1.0/catalogs/ind_halos/coords'+str(ides[j])).T  
    

    
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
    else:
        continue
    
    

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

f, ax = plt.subplots(2,1, figsize=(5,8),sharex=True)    
f.subplots_adjust(hspace=0,wspace=0)

ax[0].plot(rp[mr],1.e-12*rhop[mr]/nhalos,'C7',lw=4)
ax[0].plot(rp[mr2],1.e-12*rhop[mr2]/nhalos,'k--',lw=4)
ax[0].plot(rhof.xplot,1.e-12*rhof.yplot,'orangered',label='$\log{M_{200}}$ = '+str(np.round(np.log10(rhof.M200),1))+', $c_{200}$ = '+str(np.round(rhof.c200,1)),lw=1.5)
ax[0].plot(rhof2.xplot,1.e-12*rhof2.yplot,'--',color='orangered',label='$\log{M_{200}}$ = '+str(np.round(np.log10(rhof2.M200),1))+', $c_{200}$ = '+str(np.round(rhof2.c200,1)),lw=1.5)
ax[0].plot(rhof_E.xplot,1.e-12*rhof_E.yplot,'seagreen',label='$\log{M_{200}}$ = '+str(np.round(np.log10(rhof_E.M200),1))+', $c_{200}$ = '+str(np.round(rhof_E.c200,1))+r', $\alpha$ = '+str(np.round(rhof_E.alpha,1)),lw=1.5)
ax[0].plot(rhof_E2.xplot,1.e-12*rhof_E2.yplot,'--',color='seagreen',label='$\log{M_{200}}$ = '+str(np.round(np.log10(rhof_E2.M200),1))+', $c_{200}$ = '+str(np.round(rhof_E2.c200,1))+r', $\alpha$ = '+str(np.round(rhof_E2.alpha,1)),lw=1.5)
# ax[0].plot(rhof_E2a.xplot,rhof_E2a.yplot,'--',color='gold',label='$\log{M_{200}}$ = '+str(np.round(np.log10(rhof_E2a.M200),1))+', $c_{200}$ = '+str(np.round(rhof_E2a.c200,1))+r', $\alpha$ = '+str(np.round(rhof_E2a.alpha,1)),lw=1.5)

ax[1].plot(rp[mr],1.e-12*Sp[mr]/nhalos,'C7',lw=4)
ax[1].plot(rp[mr2],1.e-12*Sp[mr2]/nhalos,'k--',lw=4)
ax[1].plot(Sf.xplot,1.e-12*Sf.yplot,'orangered',label='$\log{M_{200}}$ = '+str(np.round(np.log10(Sf.M200),1))+', $c_{200}$ = '+str(np.round(Sf.c200,1)),lw=1.5)
ax[1].plot(rhof2.xplot,1.e-12*Sf2.yplot,'--',color='orangered',label='$\log{M_{200}}$ = '+str(np.round(np.log10(Sf2.M200),1))+', $c_{200}$ = '+str(np.round(Sf2.c200,1)),lw=1.5)
ax[1].plot(rhof_E.xplot,1.e-12*Sf_E.yplot,'seagreen',label='$\log{M_{200}}$ = '+str(np.round(np.log10(Sf_E.M200),1))+', $c_{200}$ = '+str(np.round(Sf_E.c200,1))+r', $\alpha$ = '+str(np.round(Sf_E.alpha,1)),lw=1.5)
ax[1].plot(rhof_E2.xplot,1.e-12*Sf_E2.yplot,'--',color='seagreen',label='$\log{M_{200}}$ = '+str(np.round(np.log10(Sf_E2.M200),1))+', $c_{200}$ = '+str(np.round(Sf_E2.c200,1))+r', $\alpha$ = '+str(np.round(Sf_E2.alpha,1)),lw=1.5)
# ax[1].plot(rhof_E2a.xplot,Sf_E2a.yplot,'--',color='gold',label='$\log{M_{200}}$ = '+str(np.round(np.log10(Sf_E2a.M200),1))+', $c_{200}$ = '+str(np.round(Sf_E2a.c200,1))+r', $\alpha$ = '+str(np.round(Sf_E2a.alpha,1)),lw=1.5)

# ax[0].set_xlim([0.01,1.0])
# ax[1].set_xlim([0.01,1.0])
ax[1].set_xlim([0.03,1.0])
ax[0].set_ylim([1,5.e4]) 
ax[1].set_ylim([5,5.e3]) 
ax[1].set_xlabel('$r[Mpc/h]$')
ax[0].set_ylabel(r'$\rho [M_\odot h^2/pc^3]$')
ax[1].set_ylabel(r'$\Sigma [M_\odot h/pc^2]$')
ax[0].loglog()
ax[1].loglog() 
ax[0].legend(loc=3,frameon=False) 
ax[1].legend(loc=3,frameon=False) 
plot_path = '/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/plot_paper_HSMice/'
f.savefig(plot_path+'profile_67.pdf',bbox_inches='tight')
# f.savefig(plot_path+'profile_stack.pdf',bbox_inches='tight')
