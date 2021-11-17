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
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('MICE', params)
cosmo = cosmology.setCosmology('MICE')

# plot_path = '/home/elizabeth/plot_paper_HSMice'
plot_path = '/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/plot_paper_HSMice/'

mp = 2.927e10
nrings = 25

def c200_duffy(M,z,model = 'NFW',relax=False):
    #calculo de c usando la relacion de Duffy et al 2008
    if model == 'NFW' and not relax:
        return 5.71*((M/2.e12)**-0.084)*((1.+z)**-0.47)
    if model == 'NFW' and relax:
        return 6.71*((M/2.e12)**-0.091)*((1.+z)**-0.44)
    if model == 'Einasto' and not relax:
        return 6.4*((M/2.e12)**-0.108)*((1.+z)**-0.62)
    if model == 'Einasto' and relax:
        return 7.74*((M/2.e12)**-0.123)*((1.+z)**-0.60)


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


def fit_profile(pro,z,plot=True,halo=''):
     
         r,rho,rho_E,S,S_E   = pro
         
         a_t = 1./(1.+ z)
         
         rin = 10.
         rbins = ((np.arange(nrings+1)*((1000.-rin)/float(nrings)))+rin)/1000.
         mpV = mp/((4./3.)*np.pi*(rbins[1:]**3 - rbins[:-1]**3)) # mp/V
         mpA = mp/(np.pi*(rbins[1:]**2 - rbins[:-1]**2)) # mp/A
                  
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
                # ax.axvline(0.7*a_t*pro[1]*1.e-3)
                
                f2,ax2 = plt.subplots()                 
                ax2.fill_between(r[mS],(S+mpA*0.5)[mS],(S-mpA*0.5)[mS],color='C0',alpha=0.5)
                ax2.plot(r[mS],S[mS],'C7',lw=4)
                ax2.fill_between(r[mSe],(S_E+mpA*0.5)[mSe],(S_E-mpA*0.5)[mSe],color='C1',alpha=0.5)
                ax2.plot(r[mSe],S_E[mSe],'C7--',lw=4)
                ax2.plot(S_f.xplot[m2],S_f.yplot[m2],'C2',label='NFW')
                ax2.plot(S_f_E.xplot[m2],S_f_E.yplot[m2],'C3',label='Einasto')
                ax2.plot(S_E_f.xplot[m3],S_E_f.yplot[m3],'C2--')
                ax2.plot(S_E_f_E.xplot[m3],S_E_f_E.yplot[m3],'C3--')
                # ax2.axvline(0.7*a_t*pro[1]*1.e-3)
            
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
            
                print(np.log10(rho_f.M200),rho_f.c200)
         
                f.savefig(plot_path+'profile_rho_'+part+'_'+halo+'.png')
                f2.savefig(plot_path+'profile_S_'+part+'_'+halo+'.png')



'''
# list_n = np.loadtxt('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/ind_halos/cord.list',dtype=str)
# main = pd.read_csv('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/halo_props2_8_5_pru_main.csv.bz2')
# profiles = np.loadtxt('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/halo_props2_8_5_pru_pro.csv.bz2',skiprows=1,delimiter=',')

list_n = np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv1.0/catalogs/ind_halos/cord.list',dtype=str)
main = pd.read_csv('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv1.0/catalogs/halo_props2_8_5_pru_main.csv.bz2')
profiles = np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv1.0/catalogs/halo_props2_8_5_pru_pro.csv.bz2',skiprows=1,delimiter=',')

rc = np.array(np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2))

ides = np.array([])

for j in range(len(list_n)):
    ides = np.append(ides,int(list_n[j][6:]))

ides = np.sort(ides).astype(int)

j = 0#np.argsort(rc)[-5]

# part0 = np.loadtxt('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/ind_halos/coords'+str(ides[j])).T  
part0 = np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv1.0/catalogs/ind_halos/coords'+str(ides[j])).T  

a_t = 1/(1.+main.redshift[j])

xc = main.xc[j]*1.e-3
yc = main.yc[j]*1.e-3
zc = main.zc[j]*1.e-3

xc_rc = main.xc_rc[j]*1.e-3
yc_rc = main.yc_rc[j]*1.e-3
zc_rc = main.zc_rc[j]*1.e-3

x0 = a_t*(xc_rc - xc)
y0 = a_t*(yc_rc - yc)
z0 = a_t*(zc_rc - zc)

x,y,z = a_t*part0[0]*1.e-3+x0,a_t*part0[1]*1.e-3+y0,a_t*part0[2]*1.e-3+z0
xp,yp = a_t*part0[6],a_t*part0[7]
xe,ye,ze = a_t*part0[3],a_t*part0[4],a_t*part0[5]
xep,yep = a_t*part0[8],a_t*part0[9]

f, ax = plt.subplots(1,3, figsize=(12.7,3.7))
plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.1)

# f.subplots_adjust(hspace=0,wspace=0)
ax[0].plot(x,y,'C7.',alpha=0.3)
ax[1].plot(x,z,'C7.',alpha=0.3)
ax[2].plot(y,z,'C7.',alpha=0.3)
ax[0].plot(x0,y0,'C3o')
ax[1].plot(x0,z0,'C3o')
ax[2].plot(y0,z0,'C3o')
ax[0].plot(0,0,'C1o')
ax[1].plot(0,0,'C1o')
ax[2].plot(0,0,'C1o')

ax[0].set_xlim([-1.01,1.01])
ax[0].set_ylim([-1.01,1.01])
ax[1].set_xlim([-1.01,1.01])
ax[1].set_ylim([-1.01,1.01])
ax[2].set_xlim([-1.01,1.01])
ax[2].set_ylim([-1.01,1.01])

ax[0].set_yticks([-1,-0.5,0,0.5,1])
ax[0].set_yticklabels(['-1.0','-0.5','0.0','0.5','1.0'])
ax[1].set_yticks([-1,-0.5,0,0.5,1])
ax[1].set_yticklabels(['-1.0','-0.5','0.0','0.5','1.0'])
ax[2].set_yticks([-1,-0.5,0,0.5,1])
ax[2].set_yticklabels(['-1.0','-0.5','0.0','0.5','1.0'])
ax[0].set_xticks([-1,-0.5,0,0.5,1])
ax[0].set_xticklabels(['-1.0','-0.5','0.0','0.5','1.0'])
ax[1].set_xticks([-1,-0.5,0,0.5,1])
ax[1].set_xticklabels(['-1.0','-0.5','0.0','0.5','1.0'])
ax[2].set_xticks([-1,-0.5,0,0.5,1])
ax[2].set_xticklabels(['-1.0','-0.5','0.0','0.5','1.0'])

ax[0].set_xlabel('x [kpc]')
ax[0].set_ylabel('y [kpc]')

ax[1].set_xlabel('x [kpc]')
ax[1].set_ylabel('z [kpc]')

ax[2].set_xlabel('y [kpc]')
ax[2].set_ylabel('z [kpc]')

f.savefig(plot_path+'coords.pdf',bbox_inches='tight')
'''

params = np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/HALO_Props_MICE.cat').T
lgMvir, Rvir, Cvir = np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/HALO_Props_vir.cat')
# params= np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/HALO_Props_MICE_1_1.cat').T

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

s    = c3D/a3D
q    = b3D/a3D
q2d  = b2D/a2D
sr   = c3Dr/a3Dr
qr   = b3Dr/a3Dr
q2dr = b2Dr/a2Dr

T = (1. - q**2)/(1. - s**2)
Eratio = (2.*K/abs(U))

nu = peaks.peakHeight(10**lgMvir, z)

rc = np.array(np.sqrt((xc - xc_rc)**2 + (yc - yc_rc)**2 + (zc - zc_rc)**2))
offset = rc/r_max
index = np.arange(len(params.T))

mfit_NFW = (cNFW_rho > 1.)*(cNFW_S > 1.)*(cNFW_S_E > 1.)*(cNFW_rho_E > 1.)*(lgMNFW_rho > 12)*(lgMNFW_S > 12)*(lgMNFW_S_E > 12)*(lgMNFW_rho_E > 12)*(cNFW_rho < 15)*(cNFW_S < 15)*(cNFW_S_E < 15)*(cNFW_rho_E < 15)
mfit_Ein = (cEin_rho > 1.)*(cEin_S > 1.)*(cEin_S_E > 1.)*(cEin_rho_E > 1.)*(lgMEin_rho > 12)*(lgMEin_S > 12)*(lgMEin_S_E > 12)*(lgMEin_rho_E > 12)*(cEin_rho < 15)*(cEin_S < 15)*(cEin_S_E < 15)*(cEin_rho_E < 15)*(alpha_rho > 0.)*(alpha_rho_E > 0.)

mfit = mfit_NFW*mfit_Ein

mrelax = (offset < 0.1)*(Eratio < 1.35)

#################
# RESIDUALS
#################
#-------------------
# WITH Npart
# 3D
'''
m = mfit
f, ax = plt.subplots(1,2, figsize=(6,3),sharex=True,sharey=True)
f.subplots_adjust(hspace=0,wspace=0)
make_plot2(np.log10(Npart)[m],resNFW_rho[m],'orangered',10,plt=ax[0],label='NFW')
make_plot2(np.log10(Npart)[m],resEin_rho[m],'seagreen',10,plt=ax[0],label='Einasto')
make_plot2(np.log10(Npart)[m],resNFW_rho_E[m],'orangered',10,plt=ax[1])
make_plot2(np.log10(Npart)[m],resEin_rho_E[m],'seagreen',10,plt=ax[1])
ax[0].legend(loc=2,frameon=False)
ax[0].set_ylim([0.03,0.25])
ax[0].text(0.02,0.2,'Spherical')
ax[1].text(0.02,0.2,'Elliptical')
ax[0].set_xlabel(r'$\log N_{PART}$')
ax[1].set_xlabel(r'$\log N_{PART}$')
ax[0].set_ylabel('$\delta_{3D}$')
f.savefig(plot_path+'res_npart1.pdf',bbox_inches='tight')
# 2D
m = mfit
f, ax = plt.subplots(1,2, figsize=(6,3),sharex=True,sharey=True)
f.subplots_adjust(hspace=0,wspace=0)
make_plot2(np.log10(Npart[m]),resNFW_S[m],'orangered',10,plt=ax[0],label='NFW')
make_plot2(np.log10(Npart[m]),resEin_S[m],'seagreen',10,plt=ax[0],label='Einasto')
make_plot2(np.log10(Npart[m]),resNFW_S_E[m],'orangered',10,plt=ax[1])
make_plot2(np.log10(Npart[m]),resEin_S_E[m],'seagreen',10,plt=ax[1])
ax[0].text(0.02,0.2,'Spherical')
ax[1].text(0.02,0.2,'Elliptical')
ax[0].set_ylim([0.03,0.25])
ax[0].set_xlabel(r'$\log N_{PART}$')
ax[1].set_xlabel(r'$\log N_{PART}$')
ax[0].set_ylabel('$\delta_{2D}$')
f.savefig(plot_path+'res_npart2.pdf',bbox_inches='tight')
#-------------------
# WITH concentration
# 3D
m = mfit*(Npart>15000)*(cNFW_rho > 0.)*(cNFW_rho_E > 0.)*(cEin_rho > 0.)*(cEin_rho_E > 0.)*(cNFW_rho < 10.)*(cNFW_rho_E < 10.)*(cEin_rho < 10.)*(cEin_rho_E < 10.)
f, ax = plt.subplots(1,2, figsize=(6,3),sharex=True,sharey=True)
f.subplots_adjust(hspace=0,wspace=0)
make_plot2((cNFW_rho)[m],resNFW_rho[m],'orangered',10,plt=ax[0],label='NFW')
make_plot2((cEin_rho)[m],resEin_rho[m],'seagreen',10,plt=ax[0],label='Einasto')
make_plot2((cNFW_rho_E)[m],resNFW_rho_E[m],'orangered',10,plt=ax[1])
make_plot2((cEin_rho_E)[m],resEin_rho_E[m],'seagreen',10,plt=ax[1])
ax[0].legend(loc=2,frameon=False)
ax[0].set_ylim([0.03,0.25])
ax[0].text(0.35,0.45,'Spherical')
ax[1].text(0.35,0.45,'Elliptical')
ax[0].set_xlabel(r'$c_{200}$')
ax[1].set_xlabel(r'$c_{200}$')
ax[0].set_ylabel('$\delta_{3D}$')
f.savefig(plot_path+'res_con1.pdf',bbox_inches='tight')
# 2D
m = mfit*(Npart>1000)*(Npart<1100)*(cNFW_S > 0.)*(cNFW_S_E > 0.)*(cEin_S > 0.)*(cEin_S_E > 0.)*(cNFW_S < 10.)*(cNFW_S_E < 10.)*(cEin_S < 10.)*(cEin_S_E < 10.)
f, ax = plt.subplots(1,2, figsize=(6,3),sharex=True,sharey=True)
f.subplots_adjust(hspace=0,wspace=0)
make_plot2((cNFW_S[m]),resNFW_S[m],'orangered',10,plt=ax[0],label='NFW')
make_plot2((cEin_S[m]),resEin_S[m],'seagreen',10,plt=ax[0],label='Einasto')
make_plot2((cNFW_S_E[m]),resNFW_S_E[m],'orangered',10,plt=ax[1])
make_plot2((cEin_S_E[m]),resEin_S_E[m],'seagreen',10,plt=ax[1])
ax[0].text(0.35,0.45,'Spherical')
ax[1].text(0.35,0.45,'Elliptical')
ax[0].set_ylim([0.03,0.25])
ax[0].set_xlabel(r'$c_{200}$')
ax[1].set_xlabel(r'$c_{200}$')
ax[0].set_ylabel('$\delta_{2D}$')
f.savefig(plot_path+'res_con2.pdf',bbox_inches='tight')
'''
#-------------------
# WITH OFFSET
# m = mfit*(Npart>10000)*(Npart>11000)*(offset < 0.4)*(cNFW_rho >2)*(cNFW_rho < 6)
mres = mfit*(Npart>10000)*(Npart>12000)
f1, ax1 = plt.subplots(1,4, figsize=(12,2.5),sharey=True)
f2, ax2 = plt.subplots(1,4, figsize=(12,2.5),sharey=True)
f1.subplots_adjust(hspace=0,wspace=0)
f2.subplots_adjust(hspace=0,wspace=0)
m = mres*(offset < 0.4)
make_plot2(offset[m],resNFW_rho[m],'orangered',8,plt=ax1[0],label='NFW')
make_plot2(offset[m],resEin_rho[m],'seagreen',10,plt=ax1[0],label='Einasto')
make_plot2(offset[m],resNFW_S[m],'orangered',10,plt=ax2[0],label='NFW')
make_plot2(offset[m],resEin_S[m],'seagreen',10,plt=ax2[0],label='Einasto')
ax1[0].axvline(0.1,color='gold')
ax2[0].axvline(0.1,color='gold')
ax1[0].set_xlabel(r'$r_c/r_{MAX}$')
ax2[0].set_xlabel(r'$r_c/r_{MAX}$')
ax1[0].set_ylabel('$\delta_{3D}$')
ax2[0].set_ylabel('$\delta_{2D}$')
# ------------------------
# WITH ERATIO
m = mres*(Eratio < 1.5)
# m = mfit*(Eratio < 1.5)*(Npart>10000)*(Npart<11000)*(cNFW_rho >2)*(cNFW_rho < 6)
make_plot2(Eratio[m],resNFW_rho[m],'orangered',10,plt=ax1[1],label='NFW')
make_plot2(Eratio[m],resEin_rho[m],'seagreen',10,plt=ax1[1],label='Einasto')
ax[0,1].legend(loc=2,frameon=False)
make_plot2(Eratio[m],resNFW_S[m],'orangered',10,plt=ax2[1],label='NFW')
make_plot2(Eratio[m],resEin_S[m],'seagreen',10,plt=ax2[1],label='Einasto')
ax1[1].axvline(1.35,color='gold')
ax2[1].axvline(1.35,color='gold')
ax1[1].set_xlim([1.07,1.49])
ax2[1].set_xlim([1.07,1.49])
ax1[1].set_xlabel(r'$2K/|W|$')
ax2[1].set_xlabel(r'$2K/|W|$')
# ------------------------
# WITH s
# 3D
m = mres*mrelax
make_plot2(s[m],resNFW_rho[m],'orangered',8,plt=ax1[2],label='NFW',lt='--')
make_plot2(s[m],resEin_rho[m],'seagreen',8,plt=ax1[2],label='Einasto',lt='--')
make_plot2(q[m],resNFW_S[m],'orangered',8,plt=ax2[2],label='NFW',lt='--')
make_plot2(q[m],resEin_S[m],'seagreen',8,plt=ax2[2],label='Einasto',lt='--')
ax1[2].set_xlim([0.185,0.8])
ax2[2].set_xlim([0.25,0.95])
ax1[2].set_xlabel(r'$S$')
ax2[2].set_xlabel(r'$q$')
# ------------------------
# WITH z
# 3D
m = mres*mrelax
make_plot2(z[m],resNFW_rho[m],'orangered',8,plt=ax1[3],label='NFW',lt='--')
make_plot2(z[m],resEin_rho[m],'seagreen',8,plt=ax1[3],label='Einasto',lt='--')
make_plot2(z[m],resNFW_S[m],'orangered',8,plt=ax2[3],label='NFW',lt='--')
make_plot2(z[m],resEin_S[m],'seagreen',8,plt=ax2[3],label='Einasto',lt='--')
ax1[3].set_xlim([0.07,1.4])
ax2[3].set_xlim([0.07,1.4])
ax1[3].set_xlabel(r'$z$')
ax2[3].set_xlabel(r'$z$')
ax1[3].set_ylim([0.02,0.2])
ax2[3].set_ylim([0.02,0.25])
f1.savefig(plot_path+'res_relax_3D.pdf',bbox_inches='tight')
f2.savefig(plot_path+'res_relax_2D.pdf',bbox_inches='tight')
#########################
# SPHERICAL VS ELLIPTICAL
#########################
m = mfit
f, ax = plt.subplots(2,1, figsize=(4,6),sharey=True)
# f.subplots_adjust(hspace=0,wspace=0)
make_plot2(s[m],(resNFW_rho/resNFW_rho_E)[m],'orangered',10,label='NFW',lw=2,plt=ax[0])
make_plot2(s[m],(resEin_rho/resEin_rho_E)[m],'seagreen',10,label='Einasto',lw=2,plt=ax[0])
ax[0].legend(loc=4,frameon=False)
make_plot2(q[m],(resNFW_S/resNFW_S_E)[m],'orangered',10,label='NFW',lw=2,plt=ax[1])
make_plot2(q[m],(resEin_S/resEin_S_E)[m],'seagreen',10,label='NFW',lw=2,plt=ax[1])
m = mfit*mrelax
make_plot2(s[m],(resNFW_rho/resNFW_rho_E)[m],'orangered',10,label='NFW',lt='--',lw=2,plt=ax[0])
make_plot2(s[m],(resEin_rho/resEin_rho_E)[m],'seagreen',10,label='NFW',lt='--',lw=2,plt=ax[0])
make_plot2(q[m],(resNFW_S/resNFW_S_E)[m],'orangered',10,label='NFW',lt='--',lw=2,plt=ax[1])
make_plot2(q[m],(resEin_S/resEin_S_E)[m],'seagreen',10,label='NFW',lt='--',lw=2,plt=ax[1])
ax[0].set_xlabel('$S$')
ax[1].set_xlabel('$q$')
ax[0].set_ylabel('$\delta_{3D} / \delta^E_{3D}$')
ax[1].set_ylabel('$\delta_{2D} / \delta^E_{2D}$')
f.savefig(plot_path+'res_ellip.pdf',bbox_inches='tight')

f, ax = plt.subplots(3,1, figsize=(4,7.5),sharex=True)
f.subplots_adjust(hspace=0,wspace=0)
m = mfit*(s>0.3)#(10**(lgMNFW_rho-lgMNFW_rho_E)< 5.)*(10**(lgMNFW_rho-lgMNFW_rho_E) > 1/5.)
make_plot2(s[m],10**(lgMNFW_rho-lgMNFW_rho_E)[m],'orangered',10,label='NFW',error=False,lw=1,plt=ax[0])
m = mfit*(s>0.3)#(10**(lgMEin_rho-lgMEin_rho_E)< 5.)*(10**(lgMEin_rho-lgMEin_rho_E) > 1/5.)
make_plot2(s[m],10**(lgMEin_rho-lgMEin_rho_E)[m],'seagreen',10,label='Einasto',error=False,lw=1,plt=ax[0])
ax[0].legend(frameon=False)

m = mfit*mrelax*(s>0.3)
make_plot2(s[m],10**(lgMNFW_rho-lgMNFW_rho_E)[m],'orangered',10,label='NFW',error=False,lt='--',plt=ax[0])
m = mfit*mrelax*(s>0.3)
make_plot2(s[m],10**(lgMEin_rho-lgMEin_rho_E)[m],'seagreen',10,label='Einasto',error=False,lt='--',plt=ax[0])

m = mfit*(s>0.3)
make_plot2(s[m],(cNFW_rho[m]/cNFW_rho_E[m]),'orangered',10,label='NFW',error=False,lw=1,plt=ax[1])
m = mfit*(s>0.3)
make_plot2(s[m],(cEin_rho[m]/cEin_rho_E[m]),'seagreen',10,label='Einasto',error=False,lw=1,plt=ax[1])


m = mfit*mrelax*(s>0.3)
make_plot2(s[m],(cNFW_rho[m]/cNFW_rho_E[m]),'orangered',10,label='NFW',error=False,lt='--',plt=ax[1])
m = mfit*mrelax*(s>0.3)
make_plot2(s[m],(cEin_rho[m]/cEin_rho_E[m]),'seagreen',10,label='Einasto',error=False,lt='--',plt=ax[1])

m = mfit*(s>0.3)
make_plot2(s[m],(alpha_rho[m]/alpha_rho_E[m]),'seagreen',10,label='Einasto',error=False,lw=1,plt=ax[2])
m = mfit*mrelax*(s>0.3)
make_plot2(s[m],(alpha_rho[m]/alpha_rho_E[m]),'seagreen',10,label='Einasto',lt='--',error=False,lw=1,plt=ax[2])

ax[0].plot([0,1.4],[1,1],'C7',lw=0.5)
ax[1].plot([0,1.4],[1,1],'C7',lw=0.5)
ax[2].plot([0,1.4],[1,1],'C7',lw=0.5)

ax[0].set_xlim([0.29,1])

ax[2].set_xlabel('$S$')
ax[1].set_ylabel('$c_{200}/c^E_{200}$')
ax[0].set_ylabel('$M_{200}/M^E_{200}$')
ax[2].set_ylabel(r'$\alpha/\alpha^E$')
f.savefig(plot_path+'spherical_elliptical.pdf',bbox_inches='tight')
#################
# FITTED PARAMETERS
#################
#-------------------
# alpha
plt.figure(figsize=(4,3))
plt.plot([-1,0],[-1,0],'k:',label='$13.5 < M_{200} < 13.7$')
plt.plot([-1,0],[-1,0],'k--',label='$14.0 < M_{200} < 14.3$')
plt.plot([-1,0],[-1,0],'k',label='$M_{200} > 14.5$')
plt.legend(frameon=False)
# m = mfit*mrelax*(Npart > 1000)*(Npart < 1500)
m = mfit*mrelax*(lgMEin_rho > 13.5)*(lgMEin_rho > 13.7)
make_plot2(z[m],alpha_rho[m],nbins=15,lw=2,lt=':',color='seagreen')
m = mfit*mrelax*(lgMEin_rho_E > 13.5)*(lgMEin_rho_E > 13.7)
make_plot2(z[m],alpha_rho_E[m],nbins=15,lw=2,lt=':',color='orangered')
# m = mfit*mrelax*(Npart > 10000)*(Npart < 11000)
m = mfit*mrelax*(lgMEin_rho > 14.)*(lgMEin_rho > 14.3)
make_plot2(z[m],alpha_rho[m],nbins=10,lw=2,lt='--',color='seagreen')
m = mfit*mrelax*(lgMEin_rho_E > 14.)*(lgMEin_rho_E > 14.3)
make_plot2(z[m],alpha_rho_E[m],nbins=10,lw=2,lt='--',color='orangered')
m = mfit*mrelax*(lgMEin_rho > 14.5)
make_plot2(z[m],alpha_rho[m],nbins=10,lw=2,lt='-',color='seagreen')
m = mfit*mrelax*(lgMEin_rho_E > 14.5)
make_plot2(z[m],alpha_rho_E[m],nbins=10,lw=2,lt='-',color='orangered')
plt.xlabel(r'$z$')
plt.ylabel(r'$\alpha$')
plt.axis([0.05,1.4,0.15,1])
plt.savefig(plot_path+'alpha_z.pdf',bbox_inches='tight')


plt.figure(figsize=(4,3))
m = mfit*mrelax*(lgMEin_rho > 14.5)*(nu < 4.5)
make_plot2(nu[m],alpha_rho[m],label='Spherical',nbins=10,lw=2,lt='-',color='seagreen')
m = mfit*mrelax*(lgMEin_rho_E > 14.5)*(nu < 4.5)
make_plot2(nu[m],alpha_rho_E[m],label='Elliptical',nbins=10,lw=2,lt='-',color='orangered')
plt.plot(np.arange(2,4.5,0.2),0.155+0.0095*(np.arange(2,4.5,0.2))**2,'C8',lw=4,label='Gao et al. 2008')
plt.legend(loc=2,frameon=False)
# m = mfit*mrelax*(Npart > 1000)*(Npart < 1500)
m = mfit*mrelax*(lgMEin_rho > 13.5)*(lgMEin_rho > 13.7)*(nu < 4.5)
make_plot2(nu[m],alpha_rho[m],nbins=15,lw=2,lt=':',color='seagreen')
m = mfit*mrelax*(lgMEin_rho_E > 13.5)*(lgMEin_rho_E > 13.7)*(nu < 4.5)
make_plot2(nu[m],alpha_rho_E[m],nbins=15,lw=2,lt=':',color='orangered')
# m = mfit*mrelax*(Npart > 10000)*(Npart < 11000)
m = mfit*mrelax*(lgMEin_rho > 14.)*(lgMEin_rho > 14.3)*(nu < 4.5)
make_plot2(nu[m],alpha_rho[m],nbins=10,lw=2,lt='--',color='seagreen')
m = mfit*mrelax*(lgMEin_rho_E > 14.)*(lgMEin_rho_E > 14.3)*(nu < 4.5)
make_plot2(nu[m],alpha_rho_E[m],nbins=10,lw=2,lt='--',color='orangered')
plt.xlabel(r'$\nu$')
plt.ylabel(r'$\alpha$')
plt.savefig(plot_path+'alpha_nu.pdf',bbox_inches='tight')

plt.figure(figsize=(4,3))
m = mfit*mrelax*(lgMEin_rho > 14.5)*(nu < 4.)*(z > 0.9)*(z < 1)*(nu > 2.5)
make_plot2(nu[m],alpha_rho[m],label='Spherical',nbins=10,lw=2,lt='-',color='seagreen')
m = mfit*mrelax*(lgMEin_rho_E > 14.5)*(nu < 4)*(z > 0.9)*(z < 1)*(nu > 2.5)
make_plot2(nu[m],alpha_rho_E[m],label='Elliptical',nbins=10,lw=2,lt='-',color='orangered')
plt.plot(np.arange(2,4.5,0.2),0.155+0.0095*(np.arange(2,4.5,0.2))**2,'C8',lw=4,label='Gao et al. 2008')
plt.legend(loc=2,frameon=False)
# m = mfit*mrelax*(Npart > 1000)*(Npart < 1500)
m = mfit*mrelax*(lgMEin_rho > 13.5)*(lgMEin_rho > 13.7)*(nu < 4)*(z > 0.9)*(z < 1)*(nu > 2.5)
make_plot2(nu[m],alpha_rho[m],nbins=15,lw=2,lt=':',color='seagreen')
m = mfit*mrelax*(lgMEin_rho_E > 13.5)*(lgMEin_rho_E > 13.7)*(nu < 4)*(z > 0.9)*(z < 1)*(nu > 2.5)
make_plot2(nu[m],alpha_rho_E[m],nbins=15,lw=2,lt=':',color='orangered')
# m = mfit*mrelax*(Npart > 10000)*(Npart < 11000)
m = mfit*mrelax*(lgMEin_rho > 14.)*(lgMEin_rho > 14.3)*(nu < 4)*(z > 0.9)*(z < 1)*(nu > 2.5)
make_plot2(nu[m],alpha_rho[m],nbins=10,lw=2,lt='--',color='seagreen')
m = mfit*mrelax*(lgMEin_rho_E > 14.)*(lgMEin_rho_E > 14.3)*(nu < 4)*(z > 0.9)*(z < 1)*(nu > 2.5)
make_plot2(nu[m],alpha_rho_E[m],nbins=10,lw=2,lt='--',color='orangered')
plt.xlabel(r'$\nu$')
plt.ylabel(r'$\alpha$')
plt.savefig(plot_path+'alpha_nu_Hz.pdf',bbox_inches='tight')


# plt.figure(figsize=(4,3))
m = mfit*mrelax*(lgMEin_rho > 14.5)*(nu < 3.2)*(z < 0.2)
make_plot2(nu[m],alpha_rho[m],label='Spherical',nbins=10,lw=2,lt='-',color='seagreen')
m = mfit*mrelax*(lgMEin_rho_E > 14.5)*(nu < 3.2)*(z < 0.2)
make_plot2(nu[m],alpha_rho_E[m],label='Elliptical',nbins=10,lw=2,lt='-',color='orangered')
plt.plot(np.arange(2,4.5,0.2),0.155+0.0095*(np.arange(2,4.5,0.2))**2,'C8',lw=4,label='Gao et al. 2008')
plt.legend(loc=2,frameon=False)
# m = mfit*mrelax*(Npart > 1000)*(Npart < 1500)
m = mfit*mrelax*(lgMEin_rho > 13.5)*(lgMEin_rho > 13.7)*(nu < 3.2)*(z < 0.2)
make_plot2(nu[m],alpha_rho[m],nbins=15,lw=2,lt=':',color='seagreen')
m = mfit*mrelax*(lgMEin_rho_E > 13.5)*(lgMEin_rho_E > 13.7)*(nu < 3.2)*(z < 0.2)
make_plot2(nu[m],alpha_rho_E[m],nbins=15,lw=2,lt=':',color='orangered')
# m = mfit*mrelax*(Npart > 10000)*(Npart < 11000)
m = mfit*mrelax*(lgMEin_rho > 14.)*(lgMEin_rho > 14.3)*(nu < 3.2)*(z < 0.2)
make_plot2(nu[m],alpha_rho[m],nbins=10,lw=2,lt='--',color='seagreen')
m = mfit*mrelax*(lgMEin_rho_E > 14.)*(lgMEin_rho_E > 14.3)*(nu < 3.2)*(z < 0.2)
make_plot2(nu[m],alpha_rho_E[m],nbins=10,lw=2,lt='--',color='orangered')
plt.xlabel(r'$\nu$')
plt.ylabel(r'$\alpha$')
plt.savefig(plot_path+'alpha_nu_Lz.pdf',bbox_inches='tight')


#################
# concentration with mass
f, ax = plt.subplots(2,2, figsize=(8,6),sharex=True,sharey=True)
f.subplots_adjust(hspace=0,wspace=0)

m = mfit*(cNFW_rho > 0)*(cNFW_rho < 10)*(z > 0.1)*(z < 0.25)*(lgMNFW_rho > 13.5)*mrelax*(lgMNFW_rho < 15)
make_plot2(lgMNFW_rho[m],cNFW_rho[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[0,0])
m = mfit*(cEin_rho> 0)*(cEin_rho < 10)*(z > 0.1)*(z < 0.25)*(lgMEin_rho > 13.5)*mrelax*(lgMEin_rho < 15)
make_plot2(lgMEin_rho[m],cEin_rho[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[0,0])
ax[0,0].legend(frameon=False)

m = mfit*(cNFW_rho > 0)*(cNFW_rho < 10)*(z > 0.1)*(z < 0.25)*(lgMNFW_rho > 13.5)*mrelax*(resNFW_rho<0.05)*(lgMNFW_rho < 15)
make_plot2(lgMNFW_rho[m],cNFW_rho[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[0,0],lt='--')
m = mfit*(cEin_rho> 0)*(cEin_rho < 10)*(z > 0.1)*(z < 0.25)*(lgMEin_rho > 13.5)*mrelax*(resEin_rho<0.05)*(lgMEin_rho < 15)
make_plot2(lgMEin_rho[m],cEin_rho[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[0,0],lt='--')

m = mfit*(cNFW_rho_E > 0)*(cNFW_rho_E < 10)*(z > 0.1)*(z < 0.25)*(lgMNFW_rho_E > 13.5)*mrelax*(resNFW_rho_E<0.05)*(lgMNFW_rho_E < 15)
make_plot2(lgMNFW_rho_E[m],cNFW_rho_E[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[0,1],lt='--')
m = mfit*(cEin_rho_E > 0)*(cEin_rho_E < 10)*(z > 0.1)*(z < 0.25)*(lgMEin_rho_E > 13.5)*mrelax*(resEin_rho_E<0.05)*(lgMEin_rho_E < 15)
make_plot2(lgMEin_rho_E[m],cEin_rho_E[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[0,1],lt='--')

ax[0,0].set_ylabel(r'$c_{200}$')
ax[1,0].set_ylabel(r'$c_{200}$')
ax[0,0].set_xlim([13.55,14.9])
ax[0,0].set_ylim([2.3,5.])

m = mfit*(cNFW_rho > 0)*(cNFW_rho < 10)*(z > 0.9)*(z < 1)*(lgMNFW_rho > 13.5)*(lgMNFW_rho < 15)*mrelax
make_plot2(lgMNFW_rho[m],cNFW_rho[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[1,0])
m = mfit*(cEin_rho> 0)*(cEin_rho < 10)*(z > 0.9)*(z < 1)*(lgMEin_rho > 13.5)*mrelax*(lgMEin_rho < 15)
make_plot2(lgMEin_rho[m],cEin_rho[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[1,0])

m = mfit*(cNFW_rho_E > 0)*(cNFW_rho_E < 10)*(z > 0.9)*(z < 1)*(lgMNFW_rho_E > 13.5)*mrelax*(lgMNFW_rho_E < 15)
make_plot2(lgMNFW_rho_E[m],cNFW_rho_E[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[1,1])
m = mfit*(cEin_rho_E > 0)*(cEin_rho_E < 10)*(z > 0.9)*(z < 1)*(lgMEin_rho_E > 13.5)*mrelax*(lgMEin_rho_E < 15)
make_plot2(lgMEin_rho_E[m],cEin_rho_E[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[1,1])

m = mfit*(cNFW_rho > 0)*(cNFW_rho < 10)*(z > 0.9)*(z < 1)*(lgMNFW_rho > 13.5)*mrelax*(resNFW_rho<0.05)*(lgMNFW_rho < 15)
make_plot2(lgMNFW_rho[m],cNFW_rho[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[1,0],lt='--')
m = mfit*(cEin_rho> 0)*(cEin_rho < 10)*(z > 0.9)*(z < 1)*(lgMEin_rho > 13.5)*mrelax*(resEin_rho<0.05)*(lgMEin_rho < 15)
make_plot2(lgMEin_rho[m],cEin_rho[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[1,0],lt='--')

m = mfit*(cNFW_rho_E > 0)*(cNFW_rho_E < 10)*(z > 0.9)*(z < 1)*(lgMNFW_rho_E > 13.5)*mrelax*(resNFW_rho_E<0.05)*(lgMNFW_rho_E < 15)
make_plot2(lgMNFW_rho_E[m],cNFW_rho_E[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[1,1],lt='--')
m = mfit*(cEin_rho_E > 0)*(cEin_rho_E < 10)*(z > 0.9)*(z < 1)*(lgMEin_rho_E > 13.5)*mrelax*(resEin_rho_E<0.05)*(lgMEin_rho_E < 15)
make_plot2(lgMEin_rho_E[m],cEin_rho_E[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[1,1],lt='--')

ax[1,1].set_xlabel(r'$\log (M_{200})$')
ax[1,0].set_xlabel(r'$\log (M_{200})$')

ax[0,0].text(14.,2.7,'Spherical - $0.1 < z < 0.25$')
ax[0,1].text(14.,2.7,'Elliptical - $0.1 < z < 0.25$')
ax[1,0].text(14.,2.7,'Spherical - $0.9 < z < 1.0$')
ax[1,1].text(14.,2.7,'Elliptical - $0.9 < z < 1.0$')

f.savefig(plot_path+'concentration_mass.pdf',bbox_inches='tight')

# concentration with redshift
f, ax = plt.subplots(2,2, figsize=(8,6),sharex=True,sharey=True)
f.subplots_adjust(hspace=0,wspace=0)

m = mfit*(cNFW_rho > 0)*(cNFW_rho < 10)*(z > 0.07)*(z < 1.2)*(lgMNFW_rho > 13.5)*mrelax*(lgMNFW_rho < 14)
make_plot2(z[m],cNFW_rho[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[0,0])
m = mfit*(cEin_rho> 0)*(cEin_rho < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho > 13.5)*mrelax*(lgMEin_rho < 14)
make_plot2(z[m],cEin_rho[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[0,0])
ax[0,0].legend(frameon=False)

m = mfit*(cNFW_rho_E > 0)*(cNFW_rho_E < 10)*(z > 0.07)*(z < 1.2)*(lgMNFW_rho_E > 13.5)*mrelax*(lgMNFW_rho_E < 14)
make_plot2(z[m],cNFW_rho_E[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[0,1])
m = mfit*(cEin_rho_E > 0)*(cEin_rho_E < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho_E > 13.5)*mrelax*(lgMEin_rho_E < 14)
make_plot2(z[m],cEin_rho_E[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[0,1])

m = mfit*(cNFW_rho > 0)*(cNFW_rho < 10)*(z > 0.07)*(z < 1.2)*(lgMNFW_rho > 13.5)*mrelax*(resNFW_rho<0.05)*(lgMNFW_rho < 14)
make_plot2(z[m],cNFW_rho[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[0,0],lt='--')
m = mfit*(cEin_rho> 0)*(cEin_rho < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho > 13.5)*mrelax*(resEin_rho<0.05)*(lgMEin_rho < 14)
make_plot2(z[m],cEin_rho[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[0,0],lt='--')

m = mfit*(cNFW_rho_E > 0)*(cNFW_rho_E < 10)*(z > 0.07)*(z < 1.2)*(lgMNFW_rho_E > 13.5)*mrelax*(resNFW_rho_E<0.05)*(lgMNFW_rho_E < 14)
make_plot2(z[m],cNFW_rho_E[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[0,1],lt='--')
m = mfit*(cEin_rho_E > 0)*(cEin_rho_E < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho_E > 13.5)*mrelax*(resEin_rho_E<0.05)*(lgMEin_rho_E < 14)
make_plot2(z[m],cEin_rho_E[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[0,1],lt='--')

ax[0,0].set_ylabel(r'$c_{200}$')
ax[1,0].set_ylabel(r'$c_{200}$')
ax[0,0].set_xlim([0.07,1.2])
ax[0,0].set_ylim([2.3,5.])

m = mfit*(cNFW_rho > 0)*(cNFW_rho < 10)*(z > 0.07)*(z < 1.2)*(lgMNFW_rho > 14)*(lgMNFW_rho < 15)*mrelax
make_plot2(z[m],cNFW_rho[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[1,0])
m = mfit*(cEin_rho> 0)*(cEin_rho < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho > 14)*mrelax*(lgMEin_rho < 15)
make_plot2(z[m],cEin_rho[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[1,0])

m = mfit*(cNFW_rho_E > 0)*(cNFW_rho_E < 10)*(z > 0.07)*(z < 1.2)*(lgMNFW_rho_E > 14)*mrelax*(lgMNFW_rho_E < 15)
make_plot2(z[m],cNFW_rho_E[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[1,1])
m = mfit*(cEin_rho_E > 0)*(cEin_rho_E < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho_E > 14)*mrelax*(lgMEin_rho_E < 15)
make_plot2(z[m],cEin_rho_E[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[1,1])

m = mfit*(cNFW_rho > 0)*(cNFW_rho < 10)*(z > 0.07)*(z < 1.2)*(lgMNFW_rho > 14)*mrelax*(resNFW_rho<0.05)*(lgMNFW_rho < 15)
make_plot2(z[m],cNFW_rho[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[1,0],lt='--')
m = mfit*(cEin_rho> 0)*(cEin_rho < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho > 14)*mrelax*(resEin_rho<0.05)*(lgMEin_rho < 15)
make_plot2(z[m],cEin_rho[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[1,0],lt='--')

m = mfit*(cNFW_rho_E > 0)*(cNFW_rho_E < 10)*(z > 0.07)*(z < 1.2)*(lgMNFW_rho_E > 14)*mrelax*(resNFW_rho_E<0.05)*(lgMNFW_rho_E < 15)
make_plot2(z[m],cNFW_rho_E[m],'orangered',5,label='NFW',error=True,lw=2,plt=ax[1,1],lt='--')
m = mfit*(cEin_rho_E > 0)*(cEin_rho_E < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho_E > 14)*mrelax*(resEin_rho_E<0.05)*(lgMEin_rho_E < 15)
make_plot2(z[m],cEin_rho_E[m],'seagreen',5,label='Einasto',error=True,lw=2,plt=ax[1,1],lt='--')

ax[1,1].set_xlabel(r'$z$')
ax[1,0].set_xlabel(r'$z$')

ax[0,0].text(0.2,2.7, 'Spherical - $13.5 < \log(M_{200}) < 14.5$')
ax[0,1].text(0.2,2.7,'Elliptical - $13.5 < \log(M_{200}) < 14.5$')
ax[1,0].text(0.2,2.7, 'Spherical - $14.0 < \log(M_{200}) < 15.0$')
ax[1,1].text(0.2,2.7,'Elliptical - $14.0 < \log(M_{200}) < 15.0$')

f.savefig(plot_path+'concentration_z.pdf',bbox_inches='tight')


#################
# MASS FOF
f, ax = plt.subplots(2,1, figsize=(4,6),sharex=True,sharey=True)
f.subplots_adjust(hspace=0,wspace=0)

m = mfit*(z > 0.07)*(z < 1.2)*(lgMNFW_rho > 13.5)*mrelax*(lgMNFW_rho < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMNFW_rho - lgM)[m],'orangered',10,label='NFW',error=True,lw=2,plt=ax[0])
m = mfit*(z > 0.07)*(z < 1.2)*(lgMEin_rho > 13.5)*mrelax*(lgMEin_rho < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMEin_rho - lgM)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[0])
ax[0].legend(frameon=False)

m = mfit*(z > 0.07)*(z < 1.2)*(lgMNFW_rho_E > 13.5)*mrelax*(lgMNFW_rho_E < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMNFW_rho_E - lgM)[m],'orangered',10,label='NFW',error=True,lw=2,plt=ax[1])
m = mfit*(cEin_rho_E > 0)*(cEin_rho_E < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho_E > 13.5)*mrelax*(lgMEin_rho_E < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMEin_rho_E - lgM)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[1])

m = mfit*(z > 0.07)*(z < 1.2)*(lgMNFW_rho > 13.5)*mrelax*(resNFW_rho<0.05)*(lgMNFW_rho < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMNFW_rho - lgM)[m],'orangered',10,label='NFW',error=True,lw=2,plt=ax[0],lt='--')
m = mfit*(cEin_rho> 0)*(cEin_rho < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho > 13.5)*mrelax*(resEin_rho<0.05)*(lgMEin_rho < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMEin_rho - lgM)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[0],lt='--')

m = mfit*(z > 0.07)*(z < 1.2)*(lgMNFW_rho_E > 13.5)*mrelax*(resNFW_rho_E<0.05)*(lgMNFW_rho_E < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMNFW_rho_E - lgM)[m],'orangered',10,label='NFW',error=True,lw=2,plt=ax[1],lt='--')
m = mfit*(z > 0.07)*(z < 1.2)*(lgMEin_rho_E > 13.5)*mrelax*(resEin_rho_E<0.05)*(lgMEin_rho_E < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMEin_rho_E - lgM)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[1],lt='--')

ax[0].set_ylabel(r'$M_{200}/M_{FOF}$')
ax[1].set_ylabel(r'$M_{200}/M_{FOF}$')
ax[0].set_xlim([0.3,0.95])
ax[0].set_ylim([0.7,0.97])
ax[1].set_xlabel(r'$S$')

f.savefig(plot_path+'MFOF_S.pdf',bbox_inches='tight')

# 2D vs 3D
f, ax = plt.subplots(2,1, figsize=(4,6),sharex=True,sharey=True)
f.subplots_adjust(hspace=0,wspace=0)

m = mfit*(z > 0.07)*(z < 1.2)*(lgMNFW_rho > 13.5)*mrelax*(lgMNFW_rho < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMNFW_rho - lgMNFW_S)[m],'orangered',10,label='NFW',error=True,lw=2,plt=ax[0])
m = mfit*(z > 0.07)*(z < 1.2)*(lgMEin_rho > 13.5)*mrelax*(lgMEin_rho < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMEin_rho - lgMEin_S)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[0])
ax[0].legend(frameon=False)

m = mfit*(z > 0.07)*(z < 1.2)*(lgMNFW_rho_E > 13.5)*mrelax*(lgMNFW_rho_E < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMNFW_rho_E - lgMNFW_S_E)[m],'orangered',10,label='NFW',error=True,lw=2,plt=ax[1])
m = mfit*(cEin_rho_E > 0)*(cEin_rho_E < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho_E > 13.5)*mrelax*(lgMEin_rho_E < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMEin_rho_E - lgMEin_S_E)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[1])

m = mfit*(z > 0.07)*(z < 1.2)*(lgMNFW_rho > 13.5)*mrelax*(resNFW_rho<0.05)*(lgMNFW_rho < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMNFW_rho - lgMNFW_S)[m],'orangered',10,label='NFW',error=True,lw=2,plt=ax[0],lt='--')
m = mfit*(cEin_rho> 0)*(cEin_rho < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho > 13.5)*mrelax*(resEin_rho<0.05)*(lgMEin_rho < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMEin_rho - lgMNFW_S)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[0],lt='--')

m = mfit*(z > 0.07)*(z < 1.2)*(lgMNFW_rho_E > 13.5)*mrelax*(resNFW_rho_E<0.05)*(lgMNFW_rho_E < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMNFW_rho_E - lgMNFW_S_E)[m],'orangered',10,label='NFW',error=True,lw=2,plt=ax[1],lt='--')
m = mfit*(z > 0.07)*(z < 1.2)*(lgMEin_rho_E > 13.5)*mrelax*(resEin_rho_E<0.05)*(lgMEin_rho_E < 14)*(s > 0.3)
make_plot2(s[m],10**(lgMEin_rho_E - lgMNFW_S_E)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[1],lt='--')

ax[0].set_ylabel(r'$M_{200}/M^{2D}_{200}$')
ax[1].set_ylabel(r'$M_{200}/M^{2D}_{200}$')
ax[1].set_xlabel(r'$S$')

f.savefig(plot_path+'M3D2D_S.pdf',bbox_inches='tight')

f, ax = plt.subplots(2,1, figsize=(4,6),sharex=True,sharey=True)
f.subplots_adjust(hspace=0,wspace=0)

m = mfit*(z > 0.07)*(z < 1.2)*(lgMNFW_rho > 13.5)*mrelax*(lgMNFW_rho < 14)*(s > 0.3)
make_plot2(s[m],(cNFW_rho/cNFW_S)[m],'orangered',10,label='NFW',error=True,lw=2,plt=ax[0])
m = mfit*(z > 0.07)*(z < 1.2)*(lgMEin_rho > 13.5)*mrelax*(lgMEin_rho < 14)*(s > 0.3)
make_plot2(s[m],(cEin_rho/cEin_S)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[0])
ax[0].legend(frameon=False)

m = mfit*(z > 0.07)*(z < 1.2)*(lgMNFW_rho_E > 13.5)*mrelax*(lgMNFW_rho_E < 14)*(s > 0.3)
make_plot2(s[m],(cNFW_rho_E/cNFW_S_E)[m],'orangered',10,label='NFW',error=True,lw=2,plt=ax[1])
m = mfit*(cEin_rho_E > 0)*(cEin_rho_E < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho_E > 13.5)*mrelax*(lgMEin_rho_E < 14)*(s > 0.3)
make_plot2(s[m],(cEin_rho_E/cEin_S_E)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[1])

m = mfit*(z > 0.07)*(z < 1.2)*(lgMNFW_rho > 13.5)*mrelax*(resNFW_rho<0.05)*(lgMNFW_rho < 14)*(s > 0.3)
make_plot2(s[m],(cNFW_rho/cNFW_S)[m],'orangered',10,label='NFW',error=True,lw=2,plt=ax[0],lt='--')
m = mfit*(cEin_rho> 0)*(cEin_rho < 10)*(z > 0.07)*(z < 1.2)*(lgMEin_rho > 13.5)*mrelax*(resEin_rho<0.05)*(lgMEin_rho < 14)*(s > 0.3)
make_plot2(s[m],(cEin_rho/cEin_S)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[0],lt='--')

m = mfit*(z > 0.07)*(z < 1.2)*(lgMNFW_rho_E > 13.5)*mrelax*(resNFW_rho_E<0.05)*(lgMNFW_rho_E < 14)*(s > 0.3)
make_plot2(s[m],(cNFW_rho_E/cNFW_S_E)[m],'orangered',10,label='NFW',error=True,lw=2,plt=ax[1],lt='--')
m = mfit*(z > 0.07)*(z < 1.2)*(lgMEin_rho_E > 13.5)*mrelax*(resEin_rho_E<0.05)*(lgMEin_rho_E < 14)*(s > 0.3)
make_plot2(s[m],(cEin_rho_E/cEin_S_E)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[1],lt='--')

ax[0].set_ylabel(r'$c_{200}/M_{FOF}$')
ax[1].set_ylabel(r'$c_{200}/M_{FOF}$')
ax[1].set_xlabel(r'$S$')

f.savefig(plot_path+'c3D2D_S.pdf',bbox_inches='tight')

f, ax = plt.subplots(2,1, figsize=(4,6),sharex=True,sharey=True)
f.subplots_adjust(hspace=0,wspace=0)

m = mfit*(z > 0.07)*(z < 1.2)*(lgMEin_rho > 13.5)*mrelax*(lgMEin_rho < 14)*(s > 0.3)*(alpha_rho> 0)*(alpha_rho < 1)
make_plot2(s[m],(cEin_rho/cEin_S)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[0])

m = mfit*(alpha_rho_E > 0)*(alpha_rho_E < 1)*(z > 0.07)*(z < 1.2)*(lgMEin_rho_E > 13.5)*mrelax*(lgMEin_rho_E < 14)*(s > 0.3)
make_plot2(s[m],(cEin_rho_E/cEin_S_E)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[1])

m = mfit*(alpha_rho> 0)*(alpha_rho < 1)*(z > 0.07)*(z < 1.2)*(lgMEin_rho > 13.5)*mrelax*(resEin_rho<0.05)*(lgMEin_rho < 14)*(s > 0.3)
make_plot2(s[m],(cEin_rho/cEin_S)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[0],lt='--')

m = mfit*(z > 0.07)*(z < 1.2)*(lgMEin_rho_E > 13.5)*mrelax*(resEin_rho_E<0.05)*(lgMEin_rho_E < 14)*(s > 0.3)*(alpha_rho_E > 0)*(alpha_rho_E < 1)
make_plot2(s[m],(cEin_rho_E/cEin_S_E)[m],'seagreen',10,label='Einasto',error=True,lw=2,plt=ax[1],lt='--')

ax[0].set_ylabel(r'$\alpha_{200}/\alpha^{2D}_{200}$')
ax[1].set_ylabel(r'$\alpha_{200}/\alpha^{2D}_{200}$')
ax[1].set_xlabel(r'$S$')
ax[0].set_ylim([1,1.4])

f.savefig(plot_path+'alpha3D2D_S.pdf',bbox_inches='tight')
