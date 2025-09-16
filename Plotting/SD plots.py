import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sps
import scipy.integrate as spi

#DEFINITION OF SIZE DISTRIBUTIONS
def GammaSD(a, b, r): #As described in Mishchenko and Travis 1998 Eq. (39).
    var1 = r/(a*b)
    var2 = (1-3*b)/b
    var3 = sps.gamma((1-2*b)/b)

    n_r = np.exp(-var1)*(var1**var2)/(a*b*var3)
    
    return n_r

def LogNormalSD(r_eff, nu_eff, r): #As described in Mishchenko and Travis 1994 Eqs (13), (16), (17).
    sigma_g = np.sqrt(np.log(nu_eff+1))
    r_g = r_eff/np.exp(2.5*(sigma_g**2))

    var1 = np.sqrt(2*np.pi)*r*sigma_g
    var2 = ((np.log(r)-np.log(r_g))**2)/(2*(sigma_g)**2)

    n_r = np.exp(-var2)/var1

    return n_r

def SD(a, b, r, SD_choice):
    if (SD_choice == 1):
        return GammaSD(a,b,r)
    elif (SD_choice == 2):
        return LogNormalSD(a,b,r)
    
#Set which distribution to be plotted
SD_choice = 2 #Use a GammaSD -> SD=1. Use a LogNormalSD -> SD=2

# Establish parameters of SDs
# For GammaSD: a=a, b=b. For LogNormalSD: a=r_eff, b=nu_eff
a = (1.5,3.0,4.5)
b = (0.1,0.25,0.4)

#Plots the size distribution
if(SD_choice == 1):
    SD_name = f'Gamma SD'
elif(SD_choice == 2):
    SD_name = f'Log-Normal SD'

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
x_axis = np.linspace(1e-6,10,num=200)
ax[0].plot(x_axis, SD(a[0], b[0], x_axis, SD_choice), linewidth=2, color='red', label=r'$\nu _{eff}=$'f'{b[0]}')
ax[0].plot(x_axis, SD(a[0], b[1], x_axis, SD_choice), linewidth=2, color='green', label=r'$\nu _{eff}=$'f'{b[1]}')
ax[0].plot(x_axis, SD(a[0], b[2], x_axis, SD_choice), linewidth=2, color='blue', label=r'$\nu _{eff}=$'f'{b[2]}')
ax[0].set_title(f'{SD_name}, 'r'$r_{eff}=$'f'{a[0]}'r'$\mu$m')
ax[0].set_xlabel(r'r $(\mu m)$')
ax[0].set_ylabel('n(r)')
ax[0].set_ylim(0,1.4)
ax[0].set_xlim(-0.1,5)
ax[0].legend(loc='best')

ax[1].plot(x_axis, SD(a[0], b[1], x_axis, SD_choice), linewidth=2, color='red', label=r'$r _{eff}=$'f'{a[0]}')
ax[1].plot(x_axis, SD(a[1], b[1], x_axis, SD_choice), linewidth=2, color='green', label=r'$r _{eff}=$'f'{a[1]}')
ax[1].plot(x_axis, SD(a[2], b[1], x_axis, SD_choice), linewidth=2, color='blue', label=r'$r _{eff}=$'f'{a[2]}')
ax[1].set_title(f'{SD_name}, 'r'$\nu _{eff}=$'f'{b[1]}')
ax[1].set_xlabel(r'r $(\mu m)$')
ax[1].set_ylabel('n(r)')
ax[1].set_ylim(0,1.4)
ax[1].set_xlim(-0.1,10)
ax[1].legend(loc='best')

plt.show()