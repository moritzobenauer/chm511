# Analysis of the numerical results

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dark = sns.color_palette(palette='Paired')

dx = .5/2**3
z = np.arange(-50,50,dx)

psi_t = np.load('orderparm.npy')
dens_t = np.load('dens.npy')

def interface_analysis():
    def lin_solution(t,dl,ds,S,x0):
        val = S*(dl-ds)*t + x0
        for i in range(len(val)):
            if val[i] > 50:
                val[i] = 50.0
            elif val[i] < -50:
                val[i] = -50.0
        return val

    time = []
    i1 = []
    i2 = []


    for k in range(0,psi_t.shape[0],50):
        z1 = np.argmax(np.abs(np.gradient(psi_t[k,:])))
        mask = z > 0
        z2 = np.argmax(np.abs(np.gradient(psi_t[k,:]))[mask])
        i1.append(z[z1])
        i2.append(z[z2])
        time.append(k*0.001)

    time = np.array(time)
    plt.plot(time, i1, marker='o', markersize='4', linestyle='none', color=dark[0], label=r'Diffuse Interface $z_1$')
    plt.plot(time,lin_solution(time,1,0.1,-0.1,-10), linewidth=3, color=dark[4], label=r'Sharp Interface $\zeta_1$', linestyle='dashed')
    plt.plot(time, np.array(i2)+50, marker='o', markersize='4', linestyle='none', color=dark[2], label=r'Diffuse Interface $z_2$')
    plt.plot(time,lin_solution(time,1,0.1,0.1,10), linewidth=3, color=dark[6], label=r'Sharp Interface $\zeta_2$', linestyle='dashed')
    plt.xlabel(r'Time $t$', fontsize=15)
    plt.ylabel(r'Interface Position $z$ and $\zeta$', fontsize=15)
    plt.minorticks_on()
    plt.tick_params('both',direction='in',top=True,pad=10,labelsize=12)
    plt.legend(fontsize=12)
    plt.ylim(-52,52)
    plt.yticks([-50,-25,0,25,50])
    plt.xlim(0,500)
    plt.tight_layout()
    plt.savefig('profiles.pdf', dpi=300)
    plt.savefig('profiles.png', dpi=300)
    plt.show()
    plt.clf()

def plot_versus_time():
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.set_ylim(0.5,1.2)
    fig.tight_layout()
    ax1.plot(z, psi_t[0,:], color=dark[0], label=r'$\psi (t=0)$', linewidth=3)
    ax2.plot(z, dens_t[0,:], color=dark[1], label=r'$\rho (t=0)$', linewidth=3, linestyle='dotted')
    ax1.plot(z, psi_t[100000,:], color=dark[2], label=r'$\psi (t=100)$', linewidth=3)
    ax2.plot(z, dens_t[100000,:], color=dark[3], label=r'$\rho (t=100)$', linewidth=3, linestyle='dotted')
    ax1.plot(z, psi_t[200000,:], color=dark[4], label=r'$\psi (t=200)$', linewidth=3)
    ax2.plot(z, dens_t[200000,:], color=dark[5], label=r'$\rho (t=200)$', linewidth=3, linestyle='dotted')
    ax1.legend(loc=2, fontsize=12)
    ax2.legend(loc=1, fontsize=12)
    ax1.set_ylabel(r'Order Parameter $\psi$', fontsize=15)
    ax2.set_ylabel(r'Density $\rho$', fontsize=15)
    ax1.set_xlabel(r'Position $z$', fontsize=15)
    plt.show()
    fig.savefig('versust.pdf', dpi=300)
    fig.savefig('versust.png', dpi=300)
    plt.clf()

plot_versus_time()
interface_analysis()
