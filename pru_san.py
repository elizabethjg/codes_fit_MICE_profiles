import numpy as np
import pandas as pd
from colossus.cosmology import cosmology  
from colossus.lss import peaks
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('MICE', params)
cosmo = cosmology.setCosmology('MICE')

path = '/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv1.0/catalogs/'


list_n = np.loadtxt(path+'ind_halos/cord.list',dtype=str)
main = pd.read_csv(path+'halo_props2_8_5_pru_main.csv.bz2')

# distancia entre centros - rc bajo, halo lindo
rc = np.array(np.sqrt((main.xc - main.xc_rc)**2 + (main.yc - main.yc_rc)**2 + (main.zc - main.zc_rc)**2))

ides = np.array([])

for j in range(len(list_n)):
    ides = np.append(ides,int(list_n[j][6:]))

ides = np.sort(ides).astype(int)

j = np.argsort(rc)[30]

# part0 = np.loadtxt('/mnt/projects/lensing/HALO_SHAPE/MICEv1.0/catalogs/ind_halos/coords'+str(ides[j])).T  
part0 = np.loadtxt(path+'ind_halos/coords'+str(ides[j])).T  

nparthalo = main.Npart.loc[j]

x,y,z,xp,yp,xe,ye,ze,xep,yep = part0

plt.plot(xp,yp,'.')
plt.axis([-2000,2000,-2000,2000])
plt.plot([0,1000.*np.array(main.a2Dx)[j]],[0,-1000.*np.array(main.a2Dy)[j]],'C2')
plt.plot([0,1000.*np.array(main.a2Dx)[j]],[0,1000.*np.array(main.a2Dy)[j]],'C3')
