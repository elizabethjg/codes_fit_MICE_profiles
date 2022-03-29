import sys
import numpy as np

path = '/home/elizabeth/halo_props2/lightconedir_129/'
# path = '/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/MICE/MICEv2.0/catalogs/raro/'
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
        params = np.concatenate((main[mask,:].T,profiles[mask,2:].T,masses[mask,1:].T)).T
        join = np.concatenate((join,params))
        
join = join[1:,:]

np.savetxt('/home/elizabeth/HALO_Props_MICE.cat',join,fmt='%12.6f')


ra = np.rad2deg(np.arctan(mch.xgal/mch.ygal))
ra[mch.ygal==0] = 90.
dec = np.rad2deg(np.arcsin(mch.zgal/sqrt(mch.xgal**2 + mch.ygal**2 + mch.zgal**2)))


ra_rc = np.rad2deg(np.arctan(xc_rc/yc_rc))
ra_rc[yc_rc==0] = 90.
dec_rc = np.rad2deg(np.arcsin(zc_rc/sqrt(xc_rc**2 + yc_rc**2 + zc_rc**2)))

s    = c3D/a3D
q    = b3D/a3D
q2d  = b2D/a2D
sr   = c3Dr/a3Dr
qr   = b3Dr/a3Dr
q2dr = b2Dr/a2Dr

table = [fits.Column(name='halo_id',format='E',array= halo_id),
        fits.Column(name='Npart',format='E',array= Npart),
        fits.Column(name='lgM',format='E',array= lgM),
        fits.Column(name='ra_rc',format='E',array= ra_rc),
        fits.Column(name='dec_rc',format='E',array= dec_rc),
        fits.Column(name='xc',format='E',array= xc),
        fits.Column(name='yc',format='E',array= yc),
        fits.Column(name='zc',format='E',array= zc),
        fits.Column(name='xc_rc',format='E',array= xc_rc),
        fits.Column(name='yc_rc',format='E',array= yc_rc),
        fits.Column(name='zc_rc',format='E',array= zc_rc),
        fits.Column(name='offset',format='E',array= offset),
        fits.Column(name='z',format='E',array= z),
        fits.Column(name='r_max',format='E',array= r_max),
        fits.Column(name='vxc',format='E',array= vxc),
        fits.Column(name='vyc',format='E',array= vyc),
        fits.Column(name='vzc',format='E',array= vzc),
        fits.Column(name='Jx',format='E',array= Jx),
        fits.Column(name='Jy',format='E',array= Jy),
        fits.Column(name='Jz',format='E',array= Jz),
        fits.Column(name='K',format='E',array= K),
        fits.Column(name='U',format='E',array= U),
        fits.Column(name='s',format='E',array= s),
        fits.Column(name='q',format='E',array= q),
        fits.Column(name='q2d',format='E',array= q2d),
        fits.Column(name='sr',format='E',array= sr),
        fits.Column(name='qr',format='E',array= qr),
        fits.Column(name='q2dr',format='E',array= q2dr),
        fits.Column(name='a2D',format='E',array= a2D),
        fits.Column(name='b2D',format='E',array= b2D),
        fits.Column(name='a2Dx',format='E',array= a2Dx),
        fits.Column(name='a2Dy',format='E',array= a2Dy),
        fits.Column(name='b2Dx',format='E',array= b2Dx),
        fits.Column(name='b2Dy',format='E',array= b2Dy),
        fits.Column(name='a2Dr',format='E',array= a2Dr),
        fits.Column(name='b2Dr',format='E',array= b2Dr),
        fits.Column(name='a2Drx',format='E',array= a2Drx),
        fits.Column(name='a2Dry',format='E',array= a2Dry),
        fits.Column(name='b2Drx',format='E',array= b2Drx),
        fits.Column(name='b2Dry',format='E',array= b2Dry),
        fits.Column(name='a3D',format='E',array= a3D),
        fits.Column(name='b3D',format='E',array= b3D),
        fits.Column(name='c3D',format='E',array= c3D),
        fits.Column(name='a3Dx',format='E',array= a3Dx),
        fits.Column(name='a3Dy',format='E',array= a3Dy),
        fits.Column(name='a3Dz',format='E',array= a3Dz),
        fits.Column(name='b3Dx',format='E',array= b3Dx),
        fits.Column(name='b3Dy',format='E',array= b3Dy),
        fits.Column(name='b3Dz',format='E',array= b3Dz),
        fits.Column(name='c3Dx',format='E',array= c3Dx),
        fits.Column(name='c3Dy',format='E',array= c3Dy),
        fits.Column(name='c3Dz',format='E',array= c3Dz),
        fits.Column(name='a3Dr',format='E',array= a3Dr),
        fits.Column(name='b3Dr',format='E',array= b3Dr),
        fits.Column(name='c3Dr',format='E',array= c3Dr),
        fits.Column(name='a3Drx',format='E',array= a3Drx),
        fits.Column(name='a3Dry',format='E',array= a3Dry),
        fits.Column(name='a3Drz',format='E',array= a3Drz),
        fits.Column(name='b3Drx',format='E',array= b3Drx),
        fits.Column(name='b3Dry',format='E',array= b3Dry),
        fits.Column(name='b3Drz',format='E',array= b3Drz),
        fits.Column(name='c3Drx',format='E',array= c3Drx),
        fits.Column(name='c3Dry',format='E',array= c3Dry),
        fits.Column(name='c3Drz',format='E',array= c3Drz),
        fits.Column(name='lgMDelta',format='E',array= lgMDelta),
        fits.Column(name='Delta',format='E',array= Delta),
        fits.Column(name='lgMNFW_rho',format='E',array= lgMNFW_rho),
        fits.Column(name='cNFW_rho',format='E',array= cNFW_rho),
        fits.Column(name='resNFW_rho',format='E',array= resNFW_rho),
        fits.Column(name='nb_rho',format='E',array= nb_rho),
        fits.Column(name='lgMNFW_rho_E',format='E',array= lgMNFW_rho_E),
        fits.Column(name='cNFW_rho_E',format='E',array= cNFW_rho_E  ),
        fits.Column(name='resNFW_rho_E',format='E',array= resNFW_rho_E),
        fits.Column(name='nb_rho_E',format='E',array= nb_rho_E    ),
        fits.Column(name='lgMNFW_S',format='E',array= lgMNFW_S    ),
        fits.Column(name='cNFW_S',format='E',array= cNFW_S      ),
        fits.Column(name='resNFW_S',format='E',array= resNFW_S    ),
        fits.Column(name='nb_S',format='E',array= nb_S        ),
        fits.Column(name='lgMNFW_S_E',format='E',array= lgMNFW_S_E  ),
        fits.Column(name='cNFW_S_E',format='E',array= cNFW_S_E    ),
        fits.Column(name='resNFW_S_E',format='E',array= resNFW_S_E  ),
        fits.Column(name='nb_S_E',format='E',array= nb_S_E      ),
        fits.Column(name='lgMEin_rho',format='E',array= lgMEin_rho  ),
        fits.Column(name='cEin_rho',format='E',array= cEin_rho    ),
        fits.Column(name='alpha_rho',format='E',array= alpha_rho   ),
        fits.Column(name='resEin_rho',format='E',array= resEin_rho  ),
        fits.Column(name='lgMEin_rho_E',format='E',array= lgMEin_rho_E),
        fits.Column(name='cEin_rho_E',format='E',array= cEin_rho_E  ),
        fits.Column(name='alpha_rho_E',format='E',array= alpha_rho_E ),
        fits.Column(name='resEin_rho_E',format='E',array= resEin_rho_E),
        fits.Column(name='lgMEin_S',format='E',array= lgMEin_S    ),
        fits.Column(name='cEin_S',format='E',array= cEin_S      ),
        fits.Column(name='alpha_S',format='E',array= alpha_S     ),
        fits.Column(name='resEin_S',format='E',array= resEin_S    ),
        fits.Column(name='lgMEin_S_E',format='E',array= lgMEin_S_E  ),
        fits.Column(name='cEin_S_E',format='E',array= cEin_S_E    ),
        fits.Column(name='alpha_S_E',format='E',array= alpha_S_E   ),
        fits.Column(name='resEin_S_E',format='E',array= resEin_S_E  ),
        fits.Column(name='lgMvir',format='E',array= lgMvir),
        fits.Column(name='Rvir',format='E',array= Rvir),
        fits.Column(name='Cvir',format='E',array= Cvir),]

tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(table))

tbhdu.writeto('HALO_Props_MICE.fits',overwrite=True)
