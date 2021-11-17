import sys
import numpy as np


# path = '/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv2.0/catalogs/'
path = '/home/elizabeth/halo_props2/lightconedir_129/'
join = np.ones((1,214))


for i in np.arange(1,11,1):

    for j in np.arange(1,11,1):
        
        part = str(i)+'_'+str(j)
        
        print(part)
        print(len(join))
        
        try:
            main = np.loadtxt(path+'halo_props2_'+part+'_main.csv.bz2',skiprows=1,delimiter=',')
            profiles = np.loadtxt(path+'halo_props2_'+part+'_pro.csv.bz2',skiprows=1,delimiter=',')
            masses = np.loadtxt(path+'halo_props2_'+part+'_mass.csv.bz2',skiprows=1,delimiter=',')
        
        except:
            continue
            
        mask = main.T[1] > 1000.
        join0 = np.concatenate((main[mask,:].T,profiles[mask,2:].T,masses[mask,1:].T)).T
        join = np.concatenate((join,join0))
        
# join = join[1:,:]

# np.savetxt('/home/elizabeth/HALO_Props_MICE.cat',join,fmt='%12.6f')
