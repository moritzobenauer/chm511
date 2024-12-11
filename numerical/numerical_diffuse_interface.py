import numpy as np
import matplotlib.pyplot as plt

# Initial Setup

dx = .5/2**3
z = np.arange(-50,50,dx)
index_left = np.where(z == -10.)[0][0]
print(index_left)
index_right = np.where(z == 10.)[0][0]
print(index_right)
psi = np.zeros((len(z)))
psi[index_left:index_right] = 1.

dens = np.full((len(z)),0.8)
dens[index_left:index_right] = 1.

fig, ax1 = plt.subplots()
ax1.plot(z,psi, linewidth=3)
ax2 = ax1.twinx()
ax2.plot(z,dens, linewidth=3, color='red')
ax2.set_ylim(0.5,1.2)
fig.tight_layout()
# plt.show()
plt.clf()


# Define values
rhos = 1.
rhol = 0.7
A = 5
B = 5

def dpsi2dz2(psi, dx):
    diff = np.gradient(np.gradient(psi, dx), dx)
    return diff

def dfdpsi(psi, rho, dx):
    seconddiv = dpsi2dz2(psi,dx)
    f2 = 0.5*B*(rho-rhos)**2
    f1 = 0.5*A*(rho-rhol)**2
    f2 = np.nan_to_num(f2, nan=0.0)
    f1 = np.nan_to_num(f1, nan=0.0)
    return -seconddiv + 2*psi*(1-psi)**2-2*psi**2*(1-psi)+(6*psi-6*psi**2)*(f2-f1)

def dfdrho(psi,rho):
    df1drho = A*(rho-rhol)
    df2drho = B*(rho-rhos)
    return df1drho + (3*psi**2-2*psi**3)*(df2drho-df1drho)

def chi(psi,rho,dx):
    df_drho = dfdrho(psi,rho)
    nabla_df_drho = np.gradient(df_drho, dx)
    func = (1-psi+0.1*psi)*(nabla_df_drho)
    func = np.nan_to_num(func, nan=0.0)
    return np.gradient(func, dx)

# Setting up the solver
t = 600.
dt = 1e-3
psi_t = np.zeros((int(t/dt),int(len(z))))
psi_t[0,:] = psi

dens_t = np.zeros((int(t/dt),int(len(z))))
dens_t[0,:] = dens

time = np.arange(0,int(t),dt)

for i,ts in enumerate(range(psi_t.shape[0]-1)):
    dens_t[ts+1,:] = dens_t[ts,:] + dt*chi(psi_t[ts,:],dens_t[ts,:],dx)
    psi_t[ts+1,:] = psi_t[ts,:] - dt*dfdpsi(psi_t[ts,:], dens_t[ts,:],dx)

    psi_t[ts+1,0] = 0.
    psi_t[ts+1,-1] = 0.
    dens_t[ts+1,0] = 0.8
    dens_t[ts+1,-1] = 0.8

    if i % 1000 == 0:
        print(i)



print(len(psi_t))

plt.plot(z,dens_t[0,:])
plt.plot(z,dens_t[-1,:])
plt.show()
plt.clf()
plt.plot(z,psi_t[0,:])
plt.plot(z,psi_t[-1,:])
plt.show()

np.save('orderparm.npy', psi_t)
np.save('dens.npy', dens_t)
