import sys
import numpy as np
from astropy.io import fits
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

params = join.T
params = np.loadtxt('/home/elizabeth/HALO_Props_MICE.cat').T

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


rc = np.array(np.sqrt((xc - xc_rc)**2 + (yc - yc_rc)**2 + (zc - zc_rc)**2))
offset = rc/r_max

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
        fits.Column(name='resEin_S_E',format='E',array= resEin_S_E  )]
        # fits.Column(name='lgMvir',format='E',array= lgMvir),
        # fits.Column(name='Rvir',format='E',array= Rvir),
        # fits.Column(name='Cvir',format='E',array= Cvir),]

tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(table))

tbhdu.writeto('HALO_Props_MICE.fits',overwrite=True)
