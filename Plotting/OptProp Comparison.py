import pandas as pd
import matplotlib.pyplot as plt

#Names
OptProp1_name = r"Cylinders. $r_{eff}=1.5\mu m$ $\nu _{eff}=0.3$" 
OptProp2_name = r"Hexahedra. $r_{eff}=1.5\mu m$ $\nu _{eff}=0.3$"

#OptProp1_name = r"Log-Normal SD. $\mu = 1.5 \mu m$ $\sigma ² = 0.3$" 
#OptProp2_name = r"Gamma SD. $\mu = 1.5 \mu m$ $\sigma ² = 0.3$" 

#Data files
OptProp1_vis = pd.read_csv("Wolff Files/Original Files/datadir TM OptProps 1,5um vis.txt", sep='\\s+', comment="#", header=None) #Reads opt. props. 4 columns files: wvl, qext, ssa, g
OptProp1_ir = pd.read_csv("Wolff Files/Original Files/datadir TM OptProps 1,5um ir.txt", sep='\\s+', comment="#", header=None)
OptProp2_vis = pd.read_csv("Averaged OptProps Wolff SW GammaSD reff=1.5 nueff=0.3.txt", sep='\\s+', comment="#", header=None)
OptProp2_ir = pd.read_csv("Averaged OptProps Wolff LW GammaSD reff=1.5 nueff=0.3.txt", sep='\\s+', comment="#", header=None)

#Plot the optical properties
fig, ax1 = plt.subplots(3, 1, figsize=(12, 7))
ax1[0].plot(OptProp1_vis.iloc[:,0], OptProp1_vis.iloc[:,1], 'o', markersize=2, color='red', label=OptProp1_name)
ax1[0].plot(OptProp1_ir.iloc[:,0], OptProp1_ir.iloc[:,1], 'o', markersize=2, color='red')
ax1[0].plot(OptProp2_vis.iloc[:,0], OptProp2_vis.iloc[:,1], 'o', markersize=2, color='blue', label=OptProp2_name)
ax1[0].plot(OptProp2_ir.iloc[:,0], OptProp2_ir.iloc[:,1], 'o', markersize=2, color='blue')

ax1[0].plot(OptProp1_vis.iloc[:,0], OptProp1_vis.iloc[:,1], linewidth=1, markersize=2, color='red')
ax1[0].plot(OptProp1_ir.iloc[:,0], OptProp1_ir.iloc[:,1], linewidth=1, markersize=2, color='red')
ax1[0].plot(OptProp2_vis.iloc[:,0], OptProp2_vis.iloc[:,1], linewidth=1, markersize=2, color='blue')
ax1[0].plot(OptProp2_ir.iloc[:,0], OptProp2_ir.iloc[:,1], linewidth=1, markersize=2, color='blue')

ax1[0].set_xscale('log')
ax1[0].set_ylim(0, 3)
ax1[0].set_xlim(0.2, 100)
ax1[0].set_ylabel(r'$Q_{ext}$')
ax1[0].set_xlabel(r'Wavelength ($\mu$m)')
ax1[0].set_title(r'Extinction efficiency factor, $Q_{ext}$')

ax1[1].plot(OptProp1_vis.iloc[:,0], OptProp1_vis.iloc[:,2], 'o', markersize=2, color='red', label=OptProp1_name)
ax1[1].plot(OptProp1_ir.iloc[:,0], OptProp1_ir.iloc[:,2], 'o', markersize=2, color='red')
ax1[1].plot(OptProp2_vis.iloc[:,0], OptProp2_vis.iloc[:,2], 'o', markersize=2, color='blue', label=OptProp2_name)
ax1[1].plot(OptProp2_ir.iloc[:,0], OptProp2_ir.iloc[:,2], 'o', markersize=2, color='blue')

ax1[1].plot(OptProp1_vis.iloc[:,0], OptProp1_vis.iloc[:,2], linewidth=1, markersize=2, color='red')
ax1[1].plot(OptProp1_ir.iloc[:,0], OptProp1_ir.iloc[:,2], linewidth=1, markersize=2, color='red')
ax1[1].plot(OptProp2_vis.iloc[:,0], OptProp2_vis.iloc[:,2], linewidth=1, markersize=2, color='blue')
ax1[1].plot(OptProp2_ir.iloc[:,0], OptProp2_ir.iloc[:,2], linewidth=1, markersize=2, color='blue')

ax1[1].set_ylim(0, 1)
ax1[1].set_xlim(0.2, 100)
ax1[1].set_xscale('log')
ax1[1].set_ylabel(r'$\omega$')
ax1[1].set_xlabel(r'Wavelength ($\mu$m)')
ax1[1].set_title(r'Single scattering albedo, $\omega$')

ax1[2].plot(OptProp1_vis.iloc[:,0], OptProp1_vis.iloc[:,3], 'o', markersize=2, color='red', label=OptProp1_name)
ax1[2].plot(OptProp1_ir.iloc[:,0], OptProp1_ir.iloc[:,3], 'o', markersize=2, color='red')
ax1[2].plot(OptProp2_vis.iloc[:,0], OptProp2_vis.iloc[:,3], 'o', markersize=2, color='blue', label=OptProp2_name)
ax1[2].plot(OptProp2_ir.iloc[:,0], OptProp2_ir.iloc[:,3], 'o', markersize=2, color='blue')

ax1[2].plot(OptProp1_vis.iloc[:,0], OptProp1_vis.iloc[:,3], linewidth=1, markersize=2, color='red')
ax1[2].plot(OptProp1_ir.iloc[:,0], OptProp1_ir.iloc[:,3], linewidth=1, markersize=2, color='red')
ax1[2].plot(OptProp2_vis.iloc[:,0], OptProp2_vis.iloc[:,3], linewidth=1, markersize=2, color='blue')
ax1[2].plot(OptProp2_ir.iloc[:,0], OptProp2_ir.iloc[:,3], linewidth=1, markersize=2, color='blue')

ax1[2].set_xscale('log')
ax1[2].set_ylim(0, 1)
ax1[2].set_xlim(0.2, 100)
ax1[2].set_ylabel('g')
ax1[2].set_title(r'Asymmetry factor, g')
ax1[2].set_xlabel(r'Wavelength ($\mu$m)')

for j in range(3):
    ax1[j].legend(loc='best') 

#ax1[1].legend(loc='lower left')

plt.tight_layout()
plt.show()     



