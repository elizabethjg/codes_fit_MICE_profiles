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
# part = '8_5'
part = '4_4'
cosmo = LambdaCDM(H0=100, Om0=0.25, Ode0=0.75)

# hn = fits.open('../catalogs/halo_props/halo_props2_'+part+'_2_main_plus.fits')[1].data.Halo_number
# zhalos = fits.open('../catalogs/halo_props/halo_props2_'+part+'_2_main_plus.fits')[1].data.z_v
# profiles = np.loadtxt('../catalogs/halo_props/halo_props2_'+part+'.csv_profile.bz2',skiprows=1,delimiter=',')

ncores = 32
main = pd.read_csv('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_main.csv.bz2')
profiles = np.loadtxt('/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_pro.csv.bz2',skiprows=1,delimiter=',')

# j = np.argsort(np.array(main0.lgM))[-10000:]
# main = main0.loc[j]
# profiles = profiles0[j]


mp = 2.927e10
zhalos = np.array(main.redshift)


nrings = 10

index = np.arange(len(profiles))

avance = np.linspace(0,len(profiles)/ncores,10)

def fit_profile(pro,z,plot=False):
    
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
         
         mrho = (rho > 0.)*(r < 0.7*pro[1]*1.e-3)
         mS = (S > 0.)*(r < 0.7*pro[1]*1.e-3)
         mrhoe = (rho_E > 0.)*(r < 0.7*pro[1]*1.e-3)
         mSe = (S_E > 0.)*(r < 0.7*pro[1]*1.e-3)
         
         
         # error = 1.e12*np.ones(len(r))
         
         if mrho.sum() > 0. and mS.sum() > 0. and mrhoe.sum() > 0. and mSe.sum() > 0.:

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
                ax.plot(r,rho,'C7',lw=2)
                ax.plot(rho_f.xplot[m],rho_f.yplot[m],'k')
                ax.fill_between(r,rho_E+mpV*0.5,rho_E-mpV*0.5,color='C1',alpha=0.5)
                ax.plot(r,rho_E,'C7--',lw=2)
                ax.plot(rho_E_f.xplot[m1],rho_E_f.yplot[m1],'k--')
                
                
                f2,ax2 = plt.subplots()                 
                ax2.fill_between(r,S+mpA*0.5,S-mpA*0.5,color='C0',alpha=0.5)
                ax2.plot(r,S,'C7',lw=2)
                ax2.fill_between(r,S_E+mpA*0.5,S_E-mpA*0.5,color='C1',alpha=0.5)
                ax2.plot(r,S_E,'C7--',lw=2)
                ax2.plot(S_f.xplot[m2],S_f.yplot[m2],'k')
                ax2.plot(S_E_f.xplot[m3],S_E_f.yplot[m3],'k--')
            
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                ax.set_ylabel(r'$\rho [M_\odot h^2/Mpc^3]$')
                ax.set_xlabel('$r[Mpc/h]$')
                ax2.set_ylabel(r'$\Sigma [M_\odot h/Mpc^2]$')
                ax2.set_xlabel('$R[Mpc/h]$')
            
            
                print(0.7*(pro[1]/1000.))
                print(np.log10(rho_f.M200),rho_f.c200)
            
                input("Press Enter to continue...")
            
                plt.close('all')
            
            
            return [np.log10(MDelta),Delta,
                    np.log10(rho_f.M200),rho_f.error_M200/(rho_f.M200*np.log(10.)),
                    rho_f.c200,rho_f.error_c200,rho_f.res,mrho.sum(),
                    np.log10(rho_E_f.M200),rho_E_f.error_M200/(rho_E_f.M200*np.log(10.)),
                    rho_E_f.c200,rho_E_f.error_c200,rho_E_f.res,mrhoe.sum(),
                    np.log10(S_f.M200),S_f.error_M200/(S_f.M200*np.log(10.)),
                    S_f.c200,S_f.error_c200,S_f.res,mS.sum(),
                    np.log10(S_E_f.M200),S_E_f.error_M200/(S_E_f.M200*np.log(10.)),
                    S_E_f.c200,S_E_f.error_c200,S_E_f.res,mSe.sum()]
                    
         else:
             
            return np.zeros(26)
                          

def run_fit_profile(index):
    
    
    output_fits = np.zeros((len(index),26))
    
    a = '='
    
    for i in range(len(index)):
        j = index[i]
        if j in avance:
            print(a)
        output_fits[i] = fit_profile(profiles[j],zhalos[j])
        a += '='
        
    return output_fits
        


slicer = int(round(len(index)/float(ncores), 0))
slices = ((np.arange(ncores-1)+1)*slicer).astype(int)
slices = slices[(slices <= len(index))]

# run_fit_profile(index[c_re <0.1])

# '''
index_splitted = np.split(index,slices)

pool = Pool(processes=(ncores))
salida = pool.map(run_fit_profile, np.array(index_splitted).T)
pool.terminate()

output = salida[0]

for fitted in salida[1:]:
    
    
    output  = np.vstack((output,fitted))
    
hn = main['column_halo_id']

output = np.column_stack((hn,output))
    
out_file = '/home/elizabeth/halo_props2/lightconedir_129/halo_props2_'+part+'_mass.csv.bz2'

head = 'column_halo_id,lgMDelta,Delta,lgM200_rho,e_lgM200_rho,c200_rho,e_c200_rho,R3D,nb_rho,lgM200_rho_E,e_lgM200_rho_E,c200_rho_E,e_c200_rho_E,R3D_E,nb_rho_E,lgM200_S,e_lgM200_S,c200_S,e_c200_S,R2D,nb_S,lgM200_S_E,e_lgM200_S_E,c200_S_E,e_c200_S_E,R2D_E,nb_S_E'

np.savetxt(out_file,output,fmt=['%10d']+['%5.2f']*26,header=head,comments='',delimiter=',')
