import numpy as np
import pandas as pd
sys.path.append('/mnt/projects/lensing/lens_codes_v3.7')
sys.path.append('/mnt/projects/lensing/HALO_SHAPE/MICEv2.0/codes_fit_MICE_profiles')
sys.path.append('/home/eli/lens_codes_v3.7')
sys.path.append('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/codes_fit_MICE_profiles')
from colossus.cosmology import cosmology  
from colossus.lss import peaks
from colossus.halo import mass_defs
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('MICE', params)
cosmo = cosmology.setCosmology('MICE')

# plot_path = '/home/elizabeth/plot_paper_HSMice'
plot_path = '/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/plot_paper_HSMice/'

params = np.loadtxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/HALO_Props_MICE_old.cat').T

z            = params[9]
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


Mvir = np.array([])
Cvir = np.array([])
Rvir = np.array([])

for j in range(len(z)):
    print(j/float(len(z)))
    mvir, rvir, cvir = mass_defs.changeMassDefinition(10**lgMNFW_rho[j], cNFW_rho[j], z[j], '200c', 'vir')
    Mvir = np.append(Mvir,mvir)
    Cvir = np.append(Cvir,cvir)
    Rvir = np.append(Rvir,rvir)

np.savetxt('/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/HALO_Props_vir_old.cat',[np.log10(Mvir),Rvir,Cvir],fmt='%12.6f')
