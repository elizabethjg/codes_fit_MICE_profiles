import numpy as np
import pandas as pd
sys.path.append('/mnt/projects/lensing/lens_codes_v3.7')
sys.path.append('/mnt/projects/lensing/HALO_SHAPE/MICEv2.0/codes_fit_MICE_profiles')
sys.path.append('/home/eli/lens_codes_v3.7')
sys.path.append('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/codes_fit_MICE_profiles')
import argparse
from fit_profiles_curvefit import *
from models_profiles import *
from fit_models import *
from scipy import stats
from colossus.cosmology import cosmology  
from colossus.lss import peaks
from astropy.cosmology import LambdaCDM
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('MICE', params)
cosmo = cosmology.setCosmology('MICE')
cosmo_as = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)

# plot_path = '/home/elizabeth/plot_paper_HSMice'
plot_path = '/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/plot_paper_HSMice/'

mp = 2.927e10
nrings = 25

def q_75(y):
    return np.quantile(y, 0.75)

def q_25(y):
    return np.quantile(y, 0.25)


def binned(x,y,nbins=10):
    
    
    bined = stats.binned_statistic(x,y,statistic='mean', bins=nbins)
    x_b = 0.5*(bined.bin_edges[:-1] + bined.bin_edges[1:])
    ymean     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic='median', bins=nbins)
    x_b = 0.5*(bined.bin_edges[:-1] + bined.bin_edges[1:])
    q50     = bined.statistic
    
    bined = stats.binned_statistic(x,y,statistic=q_25, bins=nbins)
    q25     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic=q_75, bins=nbins)
    q75     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic='count', bins=nbins)
    N     = bined.statistic

    bined = stats.binned_statistic(x,y,statistic='std', bins=nbins)
    sigma = bined.statistic
    
    dig   = np.digitize(x,bined.bin_edges)
    mz    = np.ones(len(x))
    for j in range(nbins):
        mbin = dig == (j+1)
        mz[mbin] = y[mbin] >= q50[j]   
    mz = mz.astype(bool)
    return x_b,q50,q25,q75,mz,ymean,sigma/np.sqrt(N)
            

def make_plot(X,Y,Z,zlim=0.3,nbins=20,plt=plt,error = False):
    plt.figure()
    x,q50,q25,q75,nada,ymean,ers = binned(X,Y,nbins)
    plt.scatter(X,Y, c=Z, alpha=0.3,s=1,vmax=zlim)
    if error:
        plt.plot(x,ymean,'C3')
        plt.plot(x,ymean+ers,'C3--')
        plt.plot(x,ymean-ers,'C3--')    
    else:
        plt.plot(x,q50,'C3')
        plt.plot(x,q25,'C3--')
        plt.plot(x,q75,'C3--')
    plt.colorbar()

def make_plot2(X,Y,color='C0',nbins=20,plt=plt,label='',error = False,lw=1,lt='-'):
    x,q50,q25,q75,nada,ymean,ers = binned(X,Y,nbins)
    if error:
        plt.plot(x,ymean,lt,color=color,label=label,lw=lw)
        plt.fill_between(x,ymean+ers,ymean-ers,color=color,alpha=0.2)
    else:
        plt.plot(x,q50,lt,color=color,label=label,lw=lw)
        plt.fill_between(x,q75,q25,color=color,alpha=0.2)


params = np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/HALO_Props_MICE.cat').T
lgMvir, Rvir, Cvir = np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/HALO_Props_vir.cat')

halo_id = params[0]
Npart   = params[1]
lgM     = params[2]
xc      = params[3]
yc      = params[4]
zc      = params[5]
xc_rc   = params[6]
yc_rc   = params[7]
zc_rc   = params[8]
z       = params[9]
r_max   = params[10]
vxc     = params[11]
vyc     = params[12]
vzc     = params[13]
Jx      = params[14]
Jy      = params[15]
Jz      = params[16]
K       = params[17]
U       = params[18]
a2D     = params[19]
b2D     = params[20]
a2Dx    = params[21]
a2Dy    = params[22]
b2Dx    = params[23]
b2Dy    = params[24]
a2Dr    = params[25]
b2Dr    = params[26]
a2Drx   = params[27]
a2Dry   = params[28]
b2Drx   = params[29]
b2Dry   = params[30]
a3D     = params[31]
b3D     = params[32]
c3D     = params[33]
a3Dx    = params[34]
a3Dy    = params[35]
a3Dz    = params[36]
b3Dx    = params[37]
b3Dy    = params[38]
b3Dz    = params[39]
c3Dx    = params[40]
c3Dy    = params[41]
c3Dz    = params[42]
a3Dr    = params[43]
b3Dr    = params[44]
c3Dr    = params[45]
a3Drx   = params[46]
a3Dry   = params[47]
a3Drz   = params[48]
b3Drx   = params[49]
b3Dry   = params[50]
b3Drz   = params[51]
c3Drx   = params[52]
c3Dry   = params[53]
c3Drz   = params[54]
R       = params[55:80]/1.e3
rho     = params[80:105]
rhoE    = params[105:130]
S       = params[130:155]
SE      = params[155:180]
lgMDelta     = params[180]
Delta        = params[181]
lgMNFW_rho   = params[182]
cNFW_rho     = params[183]
resNFW_rho   = params[184]
nb_rho       = params[185]
lgMNFW_rho_E = params[186]
cNFW_rho_E   = params[187]
resNFW_rho_E = params[188]
nb_rho_E     = params[189]
lgMNFW_S     = params[190]
cNFW_S       = params[191]
resNFW_S     = params[192]
nb_S         = params[193]
lgMNFW_S_E   = params[194]
cNFW_S_E     = params[195]
resNFW_S_E   = params[196]
nb_S_E       = params[197]
lgMEin_rho   = params[198]
cEin_rho     = params[199]
alpha_rho    = params[200]
resEin_rho   = params[201]
lgMEin_rho_E = params[202]
cEin_rho_E   = params[203]
alpha_rho_E  = params[204]
resEin_rho_E = params[205]
lgMEin_S     = params[206]
cEin_S       = params[207]
alpha_S      = params[208]
resEin_S     = params[209]
lgMEin_S_E   = params[210]
cEin_S_E     = params[211]
alpha_S_E    = params[212]
resEin_S_E   = params[213]

a_t = 1./(1.+ z)
roc_mpc = cosmo_as.critical_density(z).to(u.Msun/(u.Mpc)**3).value

Eratio = (2.*K/abs(U))

nu = peaks.peakHeight(10**lgMvir, z)

rc = np.array(np.sqrt((xc - xc_rc)**2 + (yc - yc_rc)**2 + (zc - zc_rc)**2))
offset = rc/r_max
index = np.arange(len(params.T))

s    = c3D/a3D
q    = b3D/a3D
q2d  = b2D/a2D
sr   = c3Dr/a3Dr
qr   = b3Dr/a3Dr
q2dr = b2Dr/a2Dr

T = (1. - q**2)/(1. - s**2)
Tr = (1. - qr**2)/(1. - sr**2)

V = (4./3.)*np.pi*(((a_t*r_max*1.e-3)**3))
dens3d = ((mp*Npart)/V)/roc_mpc

Ve = (4./3.)*np.pi*(((a_t*r_max*1.e-3)**3)*q*s)
dens3de = ((mp*Npart)/Ve)/roc_mpc

mfit_NFW = (cNFW_rho > 1.)*(cNFW_S > 1.)*(cNFW_S_E > 1.)*(cNFW_rho_E > 1.)*(lgMNFW_rho > 12)*(lgMNFW_S > 12)*(lgMNFW_S_E > 12)*(lgMNFW_rho_E > 12)*(cNFW_rho < 15)*(cNFW_S < 15)*(cNFW_S_E < 15)*(cNFW_rho_E < 15)
mfit_Ein = (cEin_rho > 1.)*(cEin_S > 1.)*(cEin_S_E > 1.)*(cEin_rho_E > 1.)*(lgMEin_rho > 12)*(lgMEin_S > 12)*(lgMEin_S_E > 12)*(lgMEin_rho_E > 12)*(cEin_rho < 15)*(cEin_S < 15)*(cEin_S_E < 15)*(cEin_rho_E < 15)*(alpha_rho > 0.)*(alpha_rho_E > 0.)

mfit = mfit_NFW*mfit_Ein#*(~mraro)

mrelax = (offset < 0.1)*(Eratio < 1.35)

mres1 = mfit*(dens3d > 20)*(dens3d < 40)
mres2 = mfit*(dens3de > 100)*(dens3de < 150)

'''
'''

f, ax = plt.subplots(4,2, figsize=(6,9))
f.subplots_adjust(hspace=0,wspace=0)


def plt_fig9(samp,msel,clab):

        
    m = msel*mrelax
    # make_plot2(s[m],(resNFW_rho/resNFW_rho_E)[m],clab,7,label=samp,error=False,lw=1,plt=ax[0,0],lt='-')
    make_plot2(s[m],(resEin_rho/resEin_rho_E)[m],clab,7,label='Einasto',error=False,lw=1,plt=ax[0,0],lt='--')
    # make_plot2(q2d[m],(resNFW_S/resNFW_S_E)[m],clab,7,label='Einasto',error=False,plt=ax[0,1],lt='-')
    make_plot2(q2d[m],(resEin_S/resEin_S_E)[m],clab,7,label='Einasto',error=False,plt=ax[0,1],lt='--')
        
    m = msel*mrelax
    # make_plot2(s[m],10**(lgMNFW_rho-lgMNFW_rho_E)[m],clab,7,label='NFW',error=False,lw=1,plt=ax[1,0],lt='-')
    make_plot2(s[m],10**(lgMEin_rho-lgMEin_rho_E)[m],clab,7,label='Einasto',error=False,plt=ax[1,0],lt='--')
    # make_plot2(q2d[m],10**(lgMNFW_S-lgMNFW_S_E)[m],clab,7,label='NFW',error=False,plt=ax[1,1],lt='-')
    make_plot2(q2d[m],10**(lgMEin_S-lgMEin_S_E)[m],clab,7,label='Einasto',error=False,plt=ax[1,1],lt='--')
        
    m = msel*mrelax
    # make_plot2(s[m],(cNFW_rho[m]/cNFW_rho_E[m]),clab,7,label='NFW',error=False,lw=1,plt=ax[2,0],lt='-')
    make_plot2(s[m],(cEin_rho[m]/cEin_rho_E[m]),clab,7,label='Einasto',error=False,lw=1,plt=ax[2,0],lt='--')
    # make_plot2(q2d[m],(cNFW_S[m]/cNFW_S_E[m]),clab,7,label='NFW',error=False,plt=ax[2,1],lt='-')
    make_plot2(q2d[m],(cEin_S[m]/cEin_S_E[m]),clab,7,label='Einasto',error=False,plt=ax[2,1],lt='--')
        
    m = msel*mrelax
    make_plot2(s[m],(alpha_rho[m]/alpha_rho_E[m]),clab,7,label='Einasto',error=False,lw=1,plt=ax[3,0],lt='--')
    make_plot2(q2d[m],(alpha_S[m]/alpha_S_E[m]),clab,7,label='Einasto',plt=ax[3,1],lt='--')
    
    ax[0,0].text(0.3,1.2,'3D')
    ax[0,1].text(0.4,1.2,'2D')
    
    ax[0,0].set_ylim([0.35,1.3])
    ax[0,1].set_ylim([0.35,1.3])
    ax[1,0].set_ylim([0.78,1.05])
    ax[1,1].set_ylim([0.78,1.05])
    ax[2,0].set_ylim([0.8,1.35])
    ax[2,1].set_ylim([0.8,1.35])
    ax[3,0].set_ylim([0.45,1.3])
    ax[3,1].set_ylim([0.45,1.3])
    
    for j in range(4):
        ax[j,0].plot([0,1.4],[1,1],'C7',lw=0.5)
        ax[j,1].plot([0,1.4],[1,1],'C7',lw=0.5)
        ax[j,1].set_yticklabels([])
        ax[j,0].set_xlim([0.25,0.95])
        ax[j,1].set_xlim([0.25,0.95])
    
    ax[3,0].set_xlabel('$S$')
    ax[3,1].set_xlabel('$q$')
    ax[0,0].set_ylabel('$\delta/\delta^E$')
    ax[1,0].set_ylabel('$M_{200}/M^E_{200}$')
    ax[2,0].set_ylabel('$c_{200}/c^E_{200}$')
    ax[3,0].set_ylabel(r'$\alpha/\alpha^E$')
    # f.savefig(plot_path+'spherical_elliptical'+samp+'.png',bbox_inches='tight')

msel = mfit*(s<1.)*(s>0.25)*(lgM>14)#*(lgM<14.)#*mres2

# plt_fig9('',msel)

plt_fig9('_z_00_02',msel*(z < 0.2),'C0')
plt_fig9('_z_02_03',msel*(z > 0.2)*(z < 0.3),'C1')
plt_fig9('_z_03_04',msel*(z > 0.3)*(z < 0.4),'C2')
plt_fig9('_z_04_05',msel*(z > 0.4)*(z < 0.5),'C3')
plt_fig9('_z_05_12',msel*(z > 0.5),'C4')

'''
##########################

prev = pd.read_csv('../halo_props_1_1.csv.bz2')
new = pd.read_csv('../MICEv2.0/catalogs/halo_props2_1_1_main.csv.bz2')

index = np.arange(len(prev))
m = np.random.choice(index[prev.Npart > 100],size=10000)

offset = np.array(np.sqrt((new.xc - new.xc_rc)**2 + (new.yc - new.yc_rc)**2 + (new.zc - new.zc_rc)**2))/new.r_max

s3Dr = new.c3Dr/new.a3Dr
s3D  = new.c3D/new.a3D
q3Dr = new.b3Dr/new.a3Dr
q3D  = new.b3D/new.a3D

q2Dr = new.b2Dr/new.a2Dr
q2D  = new.b2D/new.a2D

T = (1 - q3D**2)/(1 - s3D**2)
Tr = (1 - q3Dr**2)/(1 - s3Dr**2)

T_prev = (1 - prev.q3D**2)/(1 - prev.s3D**2)
Tr_prev = (1 - prev.q3Dr**2)/(1 - prev.s3Dr**2)



# xedges = np.linspace(0.3,0.8,50)
# yedges = np.linspace(0.3,0.8,50)

# xcenters = (xedges[:-1] + xedges[1:]) / 2.
# ycenters = (yedges[:-1] + yedges[1:]) / 2.

# X,Y = np.meshgrid(xcenters,ycenters)

# H, xedges, yedges = np.histogram2d(s3Dr,prev.s3Dr, bins=(xedges, yedges))
# plt.contour(X, Y, H.T,cmap='Reds')

f, ax = plt.subplots(3,2, figsize=(6,10), sharex=True, sharey=True)
# f.subplots_adjust(hspace=0,wspace=0)

ax = ax.T

ax[0,0].hexbin(s3D[m],prev.s3D[m],gridsize=150,vmin=0,vmax=5,cmap='YlGn')
ax[1,0].hexbin(s3Dr[m],prev.s3Dr[m],gridsize=150,vmin=0,vmax=5,cmap='PuRd')

ax[0,1].hexbin(T[m],T_prev[m],gridsize=150,vmin=0,vmax=5,cmap='YlGn')
ax[1,1].hexbin(Tr[m],Tr_prev[m],gridsize=150,vmin=0,vmax=5,cmap='PuRd')

ax[0,2].hexbin(q2D[m],prev.q2D[m],gridsize=150,vmin=0,vmax=5,cmap='YlGn')
ax[1,2].hexbin(q2Dr[m],prev.q2Dr[m],gridsize=150,vmin=0,vmax=5,cmap='PuRd')

ax[0,0].set_xlabel('$S$')
ax[0,0].set_ylabel('$S^{off}$')
        
ax[1,0].set_xlabel('$S_r$')
ax[1,0].set_ylabel('$S^{off}_r$')
        
ax[0,1].set_xlabel('$T$')
ax[0,1].set_ylabel('$T^{off}$')
        
ax[1,1].set_xlabel('$T_r$')
ax[1,1].set_ylabel('$T^{off}_r$')
        
ax[0,2].set_xlabel('$q$')
ax[0,2].set_ylabel('$q^{off}$')
        
ax[1,2].set_xlabel('$q_r$')
ax[1,2].set_ylabel('$q^{off}_r$')

for j in range(2):
    for i in range(3):
        ax[j,i].plot([0,1],[0,1],'C7--')

######################
f, ax = plt.subplots(3,2, figsize=(6,10), sharex=True, sharey=True)
# f.subplots_adjust(hspace=0,wspace=0)
f2,ax2 = plt.subplots()

im0 = ax2.scatter(s3D[m],prev.s3D[m],c=offset[m],cmap='cividis',s=1,vmax=0.3,vmin=0.)

ax = ax.T

ax[0,0].scatter(s3D[m],prev.s3D[m]/s3D[m],c=offset[m],cmap='cividis', alpha=0.3,s=1,vmax=0.3,vmin=0.)
ax[1,0].scatter(s3Dr[m],prev.s3Dr[m]/s3Dr[m],c=offset[m],cmap='cividis', alpha=0.3,s=1,vmax=0.3,vmin=0.)
        
ax[0,1].scatter(T[m],T_prev[m]/T[m],c=offset[m],cmap='cividis', alpha=0.3,s=1,vmax=0.3,vmin=0.)
ax[1,1].scatter(Tr[m],Tr_prev[m]/Tr[m],c=offset[m],cmap='cividis', alpha=0.3,s=1,vmax=0.3,vmin=0.)
        
ax[0,2].scatter(q2D[m],prev.q2D[m]/q2D[m],c=offset[m],cmap='cividis', alpha=0.3,s=1,vmax=0.3,vmin=0.)
ax[1,2].scatter(q2Dr[m],prev.q2Dr[m]/q2Dr[m],c=offset[m],cmap='cividis', alpha=0.3,s=1,vmax=0.3,vmin=0.)

ax[0,0].set_xlabel('$S^{SS}$')
ax[0,0].set_ylabel('$S^{CM}/S^{SS}$')
        
ax[1,0].set_xlabel('$S^{SS}_r$')
ax[1,0].set_ylabel('$S^{CM}_r/S^{SS}_r$')
        
ax[0,1].set_xlabel('$T^{SS}$')
ax[0,1].set_ylabel('$T^{CM}/T^{SS}$')
        
ax[1,1].set_xlabel('$T^{SS}_r$')
ax[1,1].set_ylabel('$T^{CM}_r/T^{SS}_r$')
        
ax[0,2].set_xlabel('$q^{SS}$')
ax[0,2].set_ylabel('$q^{CM}/q^{SS}$')
        
ax[1,2].set_xlabel('$q^{SS}_r$')
ax[1,2].set_ylabel('$q^{CM}_r/q^{SS}_r$')

for j in range(2):
    for i in range(3):
        ax[j,i].plot([0,1],[1,1],'C7--')

ax[0,2].set_ylim([0.3,1.7])

f.colorbar(im0, ax=ax, orientation='horizontal', fraction=.05,alpha=1,label = '$r_c/r_{MAX}$',pad=0.1)
f.savefig(plot_path+'shape_offset.pdf',bbox_inches='tight')
'''
