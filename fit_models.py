import sys, os
import numpy as np
import sys
from scipy.optimize import curve_fit
from models_profiles import *
from colossus.cosmology import cosmology  
params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('MICE', params)
cosmo = cosmology.setCosmology('MICE')
from colossus.halo import profile_nfw
from colossus.halo import profile_einasto


class Sigma_fit:
	# R en Mpc, Sigma M_Sun/Mpc2
	

    def __init__(self,R,Sigma,err,z):


        xplot   = np.arange(0.001,R.max()+1.,0.001)

        p = profile_nfw.NFWProfile(M = 1.e13, c = 4., z = z, mdef = '200c')

        try:
        
            out = p.fit(R*1000., Sigma/(1.e3**2), 'Sigma', q_err = err/(1.e3**2))
            
            rhos,rs = out['x']
            
            prof = profile_nfw.NFWProfile(rhos = rhos, rs = rs)
            
            ajuste = prof.surfaceDensity(R*1000.)*(1.e3**2)
            yplot  = prof.surfaceDensity(xplot*1000.)*(1.e3**2)
            
            BIN= len(Sigma)
            
            res=np.sqrt(((((np.log10(ajuste)-np.log10(Sigma))**2)).sum())/float(BIN-2))

            M200 = prof.MDelta(z,'200c')
            c200 = prof.RDelta(z,'200c')/rs
        
        except:
            yplot = xplot
            res   = -999.
            M200  = -999.
            c200  = -999.
        

        self.xplot = xplot
        self.yplot = yplot
        self.res  = res
        self.M200 = M200
        self.c200 = c200

class rho_fit:
	# R en Mpc, rho M_Sun/Mpc3
	

    def __init__(self,R,rho,err,z):


        xplot   = np.arange(0.001,R.max()+1.,0.001)

        p = profile_nfw.NFWProfile(M = 1.e13, c = 4., z = z, mdef = '200c')

        try:
        
            out = p.fit(R*1000., rho/(1.e3**3), 'rho', q_err = err/(1.e3**3))
            rhos,rs = out['x']
            
            prof = profile_nfw.NFWProfile(rhos = rhos, rs = rs)
            
            ajuste = prof.density(R*1000.)*(1.e3**3)
            yplot  = prof.density(xplot*1000.)*(1.e3**3)
            
            BIN= len(rho)
            
            res=np.sqrt(((((np.log10(ajuste)-np.log10(rho))**2)).sum())/float(BIN-2))
            
            M200 = prof.MDelta(z,'200c')
            c200 = prof.RDelta(z,'200c')/rs
        
        except:
            yplot = xplot
            res   = -999.
            M200  = -999.
            c200  = -999.
        

        self.xplot = xplot
        self.yplot = yplot
        self.res  = res
        self.M200 = M200
        self.c200 = c200
