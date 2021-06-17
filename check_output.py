import numpy as np
import pandas as pd
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-file_name', action='store', dest='file_name',default='output')
args = parser.parse_args()

check_cat = pd.read_csv('output_check.csv.bz2')
cat    = pd.read_csv(args.file_name+'.bz2')

print(check_cat.shape)
print(cat.shape)

for j in range(len(check_cat.columns)):
    col_check = check_cat.columns[j]
    col_cat   = cat.columns[j]
    m = np.abs(check_cat[col_check] > 0.) 
    print(col_cat,np.max(((cat[col_cat]/check_cat[col_check])[m])))

'''

cat = check_cat 

rc = np.sqrt((cat.xc_fof-cat.xc_rc)**2 + (cat.yc_fof-cat.yc_rc)**2 + (cat.zc_fof-cat.zc_rc)**2)

xold = cat.xc_fof - cat.xc_rc
yold = cat.yc_fof - cat.yc_rc
zold = cat.zc_fof - cat.zc_rc

lM = cat['log10(mass)']
idhalo = cat['Halo number']


part0 = np.loadtxt('ind_halos/particles_halo'+str(int(idpru))).T

f, ax = plt.subplots(1, 3, figsize=(12,4),sharex=True,sharey=True)
f.subplots_adjust(hspace=0,wspace=0)

ax[0].plot(part0[3],part0[4],'C7.')
ax[1].plot(part0[3],part0[5],'C7.')
ax[2].plot(part0[4],part0[5],'C7.')

ax[0].plot(part0[3],part0[4],'C7.')
ax[1].plot(part0[3],part0[5],'C7.')
ax[2].plot(part0[4],part0[5],'C7.')

ax[0].plot(0,0,'C3o')
ax[1].plot(0,0,'C3o')
ax[2].plot(0,0,'C3o')

ax[0].plot(xold[idhalo == idpru],yold[idhalo == idpru],'C1o')
ax[1].plot(xold[idhalo == idpru],zold[idhalo == idpru],'C1o')
ax[2].plot(yold[idhalo == idpru],zold[idhalo == idpru],'C1o')
'''




# R = np.zeros((nhalos,15))

for j in range(nhalos):
    
    r = np.arange(16)*(pro[1,j]/15.) 
    R[j,:] = r[:-1] + np.diff(r)/2. 

plt.figure()
for j in range(nhalos):
    plt.plot(R[j,:],pro[2:17,j],'C7',alpha=0.3)
    plt.plot(R[j,:],pro[17:32,j],'C3',alpha=0.3)

# for j in range(nhalos):
    # plt.plot(R[j,:],pro[32:47,j],'C7',alpha=0.3)
    # plt.plot(R[j,:],pro[47:,j],'C3',alpha=0.3)


rpart = np.sqrt(x**2 + y**2 + z**2)/1.e3
rin = 0
step = rmax/15.

rho = np.zeros(15)
rp = np.zeros(15)

Ntot = 0
for ring in range(15):
    V    = (4./3.)*np.pi*((rin+step)**3 - rin**3)
    mask = (rpart <= rin+step)*(rpart > rin)
    rho[ring] = (mask.sum()*mp)/V
    rp[ring] = rin + 0.5*step
    Ntot += mask.sum()
    rin = rin+step

# x = np.array([])
# y = np.array([])
# z = np.array([])
# x_r = np.array([])
# y_r = np.array([])
# z_r = np.array([])

# x2d = np.array([])
# y2d = np.array([])
# x2d_r = np.array([])
# y2d_r = np.array([])

# for halo in idhalos:
    # x0,y0,z0,x0_r,y0_r,z0_r,x02d,y02d,x02d_r,y02d_r = np.loadtxt('ind_halos/coords'+str(halo)).T

    # x = np.append(x,x0)
    # y = np.append(y,y0)
    # z = np.append(z,z0)

    # x_r = np.append(x_r,x0_r)
    # y_r = np.append(y_r,y0_r)
    # z_r = np.append(z_r,z0_r)

    # x2d = np.append(x2d,x02d)
    # y2d = np.append(y2d,y02d)

    # x2d_r = np.append(x2d_r,x02d_r)
    # y2d_r = np.append(y2d_r,y02d_r)


# f, ax = plt.subplots(1, 3, figsize=(12,4),sharex=True,sharey=True)
# ax[0].set_xlabel('x')
# ax[0].set_ylabel('y')
# ax[0].plot(x0,y0,'C7.')

# ax[1].set_xlabel('x')
# ax[1].set_ylabel('z')
# ax[1].plot(x0,z0,'C7.')

# ax[2].set_xlabel('y')
# ax[2].set_ylabel('z')
# ax[2].plot(y0,z0,'C7.')
# plt.savefig('halo0_xyz.png')

# f, ax = plt.subplots(1, 3, figsize=(12,4),sharex=True,sharey=True)
# ax[0].set_xlabel('a')
# ax[0].set_ylabel('b')
# ax[0].plot(x0_r,y0_r,'C7.')

# ax[1].set_xlabel('a')
# ax[1].set_ylabel('c')
# ax[1].plot(x0_r,z0_r,'C7.')

# ax[2].set_xlabel('b')
# ax[2].set_ylabel('c')
# ax[2].plot(y0_r,z0_r,'C7.')
# plt.savefig('halo0_abc.png')

# f, ax = plt.subplots(1, 1, figsize=(4,4),sharex=True,sharey=True)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.plot(x02d,y02d,'C7.')
# plt.savefig('halo0_xy.png')

# f, ax = plt.subplots(1, 1, figsize=(4,4),sharex=True,sharey=True)
# ax.set_xlabel('a')
# ax.set_ylabel('b')
# ax.plot(x02d_r,y02d_r,'C7.')
# plt.savefig('halo0_ab.png')

# #######

# f, ax = plt.subplots(1, 3, figsize=(12,4),sharex=True,sharey=True)
# ax[0].set_xlabel('x')
# ax[0].set_ylabel('y')
# ax[0].plot(x,y,'C7,')
# ax[0].set_xlim([-4000.,4000.])
# ax[0].set_ylim([-4000.,4000.])

# ax[1].set_xlabel('x')
# ax[1].set_ylabel('z')
# ax[1].plot(x,z,'C7,')

# ax[2].set_xlabel('y')
# ax[2].set_ylabel('z')
# ax[2].plot(y,z,'C7,')
# plt.savefig('halos_xyz.png')

# f, ax = plt.subplots(1, 3, figsize=(12,4),sharex=True,sharey=True)
# ax[0].set_xlabel('a')
# ax[0].set_ylabel('b')
# ax[0].plot(x_r,y_r,'C7,')
# ax[0].set_xlim([-4000.,4000.])
# ax[0].set_ylim([-4000.,4000.])


# ax[1].set_xlabel('a')
# ax[1].set_ylabel('c')
# ax[1].plot(x_r,z_r,'C7,')

# ax[2].set_xlabel('b')
# ax[2].set_ylabel('c')
# ax[2].plot(y_r,z_r,'C7,')
# plt.savefig('halos_abc.png')


