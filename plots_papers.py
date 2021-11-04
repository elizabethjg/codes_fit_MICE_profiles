import numpy as np
import pandas as pd
sys.path.append('/mnt/projects/lensing/lens_codes_v3.7')
sys.path.append('/mnt/projects/lensing/HALO_SHAPE/MICEv2.0/codes_fit_MICE_profiles')
sys.path.append('/home/eli/lens_codes_v3.7')
sys.path.append('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/codes_fit_MICE_profiles')
import argparse
from astropy.cosmology import LambdaCDM 
from fit_profiles_curvefit import *
from models_profiles import *
from fit_models import *
cosmo = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)

# plot_path = '/home/elizabeth/plot_paper_HSMice'
plot_path = '/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/plot_paper_HSMice/'

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

j = np.argsort(rc)[-5]

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
