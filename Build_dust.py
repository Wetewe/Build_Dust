import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sps
from scipy.interpolate import Akima1DInterpolator
from matplotlib.ticker import LogLocator, ScalarFormatter

#SIZE DISTRIBUTIONS AND AUXILIARY FUNCTIONS
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
    
def OptPropAxis(OptProp: pd.DataFrame):
    #Read from isca.dat the wvl axis
    k = 0
    wvls = []
    for i in range(len(OptProp[0]) - 1):
        if i == 0:
            wvls.append(OptProp.iloc[0,0])
            wvl_temp = wvls[k]
        if wvl_temp != OptProp.iloc[i+1,0]:
            wvls.append(OptProp.iloc[i+1,0])
            k = k + 1
            wvl_temp = wvls[k]
            
    wvls = np.array(wvls)
    wvls_num = len(wvls)

    #Read from isca.dat the size axis
    sizes = []
    for i in range(len(OptProp[1]) - 1):
        if i == 0:
            sizes.append(OptProp.iloc[0,1])
        if OptProp.iloc[i+1,1] not in sizes:
            sizes.append(OptProp.iloc[i+1,1])

    sizes = np.array(sizes)
    sizes_num = len(sizes)

    #Obtain rmie axis averaged over wvls
    # This is only really used in OptPropExtract 
    parea_temp = OptProp[3]
    parea = np.zeros((wvls_num,sizes_num))
    rmie = np.zeros((wvls_num,sizes_num))

    k = 0
    for i in range(wvls_num):
        for j in range(sizes_num):
                parea[i,j] = parea_temp[k]
                rmie[i,j] = np.sqrt(parea_temp[k]/np.pi)
                k = k+1
    rmie_avg = np.mean(rmie, axis=0)

    return wvls, wvls_num, rmie_avg, sizes_num

def OptPropArrays(OptProp: pd.DataFrame):
    #Read axis
    wvls, wvls_num, sizes, sizes_num = OptPropAxis(OptProp)

    #Initialize arrays
    parea = np.zeros((wvls_num, sizes_num))
    qext = np.zeros((wvls_num, sizes_num)) #Size dependent optical properties
    ssa = np.zeros((wvls_num, sizes_num))
    g = np.zeros((wvls_num, sizes_num))

    #Read values
    parea_temp = OptProp.iloc[:,3] #Projected surface area
    qext_temp = OptProp.iloc[:,4] #Extinction efficiency
    ssa_temp = OptProp.iloc[:,5] #Single Scattering albedo
    g_temp = OptProp.iloc[:,6] #Asymmetry factor

    #Creates 2-dim arrays of optical properties. First index indicates wavelength, second size bin
    k=0 #k ensures that when reading the TAMU output we copy only the unique wavelengths
    for i in range(wvls_num):
        for j in range(sizes_num):
            parea[i,j] = parea_temp[k]
            qext[i,j] = qext_temp[k]
            ssa[i,j] = ssa_temp[k]
            g[i,j] = g_temp[k]
            k=k+1
    
    return parea, qext, ssa, g

########################################

def WriteGCMFormat(sizes: np.array, wvls: np.array, header: str, files: list):
    #Some necessary actions and definitions
    wvls_num = len(wvls)
    sizes_num = len(sizes)
    files_num = len(files)
    sizes = sizes * 1e-6 #Change from um to m (LMD-GCM reads meters)
    wvls = wvls * 1e-6
    output_path = "outputWriteGCM.dat"

    #Foolproofing
    if files_num != sizes_num:
        return print(f"Error: number of files given ({files_num}) is not equal to number of sizes ({sizes_num})")

    #Write header, nº of wvls and nº of radii in file
    with open(output_path, "w") as f:
        f.write(f"{header}\n")
        f.write(f"# Number of wavelengths (nwvl):\n  {wvls_num}\n")
        f.write(f"# Number of radius (nsize):\n  {sizes_num}\n")

    #Write wvls axis
    with open(output_path, "a") as f:
        f.write("# Wavelength axis (wvl):\n")
        k = 0
        groups = wvls_num//5 #Number of groups of 5 lines
        lastline = wvls_num - groups*5 #Number of elements in last line either 1,2,3 or 4
        for i in range(groups):
            f.write(f" {wvls[0+k]:.6E} {wvls[1+k]:.6E} {wvls[2+k]:.6E} {wvls[3+k]:.6E} {wvls[4+k]:.6E}\n")
            k = k + 5
        if lastline != 0:
            for i in range(lastline):
                f.write(f" {wvls[k+i]:.6E}")
            f.write("\n")

    #Write sizes axis
    with open(output_path, "a") as f:
        f.write("# Particle size axis (radius):\n")
        k = 0
        groups = sizes_num//5 #Number of groups of 5 lines
        lastline = sizes_num - groups*5 #Number of elements in last line either 1,2,3 or 4
        for i in range(groups):
            f.write(f" {sizes[0+k]:.6E} {sizes[1+k]:.6E} {sizes[2+k]:.6E} {sizes[3+k]:.6E} {sizes[4+k]:.6E}\n")
            k = k + 5
        if lastline != 0:
            for i in range(lastline):
                f.write(f" {sizes[k+i]:.6E}")
            f.write("\n")

    #Write extinction coefficient
    with open(output_path, "a") as f:
        f.write("# Extinction coef. Qext (ep):\n")
        groups = wvls_num//5
        lastline = wvls_num - groups*5        
        for i in range(sizes_num):
            k = 0
            f.write(f"# Radius number     {i+1}\n")
            OptProps = pd.read_csv(files[i], sep='\\s+', comment="#", header=None)
            Q_ext = np.array(OptProps.iloc[:,1])
            for j in range(groups):
                f.write(f" {Q_ext[0+k]:.6E} {Q_ext[1+k]:.6E} {Q_ext[2+k]:.6E} {Q_ext[3+k]:.6E} {Q_ext[4+k]:.6E}\n")
                k = k + 5
            if lastline != 0:
                for l in range(lastline):
                    f.write(f" {Q_ext[l+k]:.6E}")
                f.write("\n")

    #Write single scattering albedo                
    with open(output_path, "a") as f:
        f.write("# Single Scat Albedo (omeg):\n")
        groups = wvls_num//5
        lastline = wvls_num - groups*5        
        for i in range(sizes_num):
            k = 0
            f.write(f"# Radius number     {i+1}\n")
            OptProps = pd.read_csv(files[i], sep='\\s+', comment="#", header=None)
            ssa = np.array(OptProps.iloc[:,2])
            for j in range(groups):
                f.write(f" {ssa[0+k]:.6E} {ssa[1+k]:.6E} {ssa[2+k]:.6E} {ssa[3+k]:.6E} {ssa[4+k]:.6E}\n")
                k = k + 5
            if lastline != 0:
                for l in range(lastline):
                    f.write(f" {ssa[l+k]:.6E}")
                f.write("\n")
    
    #Write asymmetry parameter
    with open(output_path, "a") as f:
        f.write("# Assymetry Factor (gfactor):\n")
        groups = wvls_num//5
        lastline = wvls_num - groups*5        
        for i in range(sizes_num):
            k = 0
            f.write(f"# Radius number     {i+1}\n")
            OptProps = pd.read_csv(files[i], sep='\\s+', comment="#", header=None)
            g = np.array(OptProps.iloc[:,3])
            for j in range(groups):
                f.write(f" {g[0+k]:.6E} {g[1+k]:.6E} {g[2+k]:.6E} {g[3+k]:.6E} {g[4+k]:.6E}\n")
                k = k + 5
            if lastline != 0:
                for l in range(lastline):
                    f.write(f" {g[l+k]:.6E}")
                f.write("\n")    

    return

def OptPropAve(OptProp_path: str, r_eff: float, v_eff: float, SD_choice: int, plot: bool = True):

    OptProp = pd.read_csv(OptProp_path, sep='\\s+', comment="#", header=None)

    #Read axis
    wvls, wvls_num, sizes, sizes_num = OptPropAxis(OptProp)

    #Read essential arrays
    parea, qext, ssa, g = OptPropArrays(OptProp)

    #Initialize arrays
    qsca = np.zeros((wvls_num, sizes_num))
    rmie_TAMU = np.zeros((wvls_num, sizes_num))

    qext_avg = np.zeros((wvls_num)) #Size-averaged optical properties
    ssa_avg = np.zeros((wvls_num))
    g_avg = np.zeros((wvls_num))

    n_A_qext = np.zeros((wvls_num, sizes_num))
    n_A = np.zeros((wvls_num, sizes_num))
    n_A_qsca = np.zeros((wvls_num, sizes_num))
    n_A_qsca_g = np.zeros((wvls_num, sizes_num))

    #Here we transform the projected surface area outputed by TAMUdust into rmie
    for i in range(wvls_num):
        for j in range(sizes_num):
            rmie_TAMU[i,j] = np.sqrt(parea[i,j]/np.pi) 

    #Calculates the scattering efficiency Qsca from ssa and Qext
    for i in range(wvls_num):
        for j in range(sizes_num):
            qsca[i,j] = ssa[i,j]*qext[i,j]

    #Averaging!
    for i in range(wvls_num):
        #<Qext>
        n_A_qext[i,:] = SD(r_eff, v_eff, rmie_TAMU[i,:], SD_choice)*parea[i,:]*qext[i,:]
        n_A[i,:] = SD(r_eff, v_eff, rmie_TAMU[i,:], SD_choice)*parea[i,:]
    
        numer_qext = np.trapezoid(n_A_qext[i,:],rmie_TAMU[i,:])
        denom_qext = np.trapezoid(n_A[i,:],rmie_TAMU[i,:])

        qext_avg[i] = numer_qext/denom_qext

        if qext_avg[i] <= 0:
            qext_avg[i] = 0

        #<ssa>
        n_A_qsca[i,:] = SD(r_eff, v_eff, rmie_TAMU[i,:], SD_choice)*parea[i,:]*qsca[i,:]

        numer_ssa = np.trapezoid(n_A_qsca[i,:],rmie_TAMU[i,:])
        denom_ssa = numer_qext

        ssa_avg[i] = numer_ssa/denom_ssa

        if ssa_avg[i] <= 0:
            ssa_avg[i] = 0

        #<g>
        n_A_qsca_g[i,:] = SD(r_eff, v_eff, rmie_TAMU[i,:], SD_choice)*parea[i,:]*qsca[i,:]*g[i,:]

        numer_g = np.trapezoid(n_A_qsca_g[i,:],rmie_TAMU[i,:])
        denom_g = numer_ssa

        g_avg[i] = numer_g/denom_g

    #Save averaged values in .txt file
    output_path = f"OptPropAve_reff{r_eff}_veff{v_eff}.dat"
    output = pd.DataFrame({"#Wavelength(um)": wvls, "Qext": qext_avg, "ssa": ssa_avg, "g": g_avg})
    output.to_csv(output_path, sep="\t", index=False, float_format='%.15f')

    #Plots the size distribution
    if plot == True:
        if(SD_choice == 1):
            SD_name = f'GammaSD reff={r_eff} nueff={v_eff}'
        elif(SD_choice == 2):
            SD_name = f'Log-NormalSD reff={r_eff}um nueff={v_eff}'

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        x_axis = np.linspace(1e-6,35,num=200)
        ax.plot(x_axis, SD(r_eff, v_eff, x_axis, SD_choice), linewidth=2, color='purple', label=SD_name)
        ax.set_title('Size distributions used for averaging the optical properties')
        ax.set_xlabel('Surface area equivalent sphere radius, r_A (um)')
        ax.legend(loc='best')

        plt.show()

    #Plot the averaged optical properties
    if plot == True:
        fig, ax1 = plt.subplots(3, 1, figsize=(12, 7))
        ax1[0].plot(wvls, qext_avg, 'o', markersize=2, color='red', label='Qext')
        ax1[0].plot(wvls, qext_avg, linewidth=1, color='red')
        ax1[0].set_ylabel('Extinction factor, Q_ext')
        ax1[0].set_title('Size-averaged optical properties')

        ax1[1].plot(wvls, ssa_avg, 'o', markersize=2, color='red', label='SSA')
        ax1[1].plot(wvls, ssa_avg, linewidth=1, color='red')
        ax1[1].set_ylabel('Single scattering albedo, w')

        ax1[2].plot(wvls, g_avg, 'o', markersize=2, color='red', label='g')
        ax1[2].plot(wvls, g_avg, linewidth=1, color='red')
        ax1[2].set_ylabel('Asymmetry factor, g')
        ax1[2].set_xlabel('Wavelength (um)')

        for j in range(3):
            ax1[j].legend(loc='best')

        plt.show()     

    return

def OptPropAve_Disc(OptProp_path: str, SD_path: str, plot: bool = True):
    #Open files
    OptProp = pd.read_csv(OptProp_path, sep='\\s+', comment="#", header=None)
    SD = pd.read_csv(SD_path, sep='\\s+', comment="#", header=None)

    #Read SD bins from discretely measured SD
    SD_num = len(SD[0])
    SD_bins = SD_num - 1
    rmie_SD = SD.iloc[:,0].values #rmie from the SD is the sizes at which we want to interpolate
    n_SD = SD.iloc[:,1].values

    #Read axis
    wvls, wvls_num, sizes, sizes_num = OptPropAxis(OptProp)

    #Read essential arrays
    parea, qext, ssa, g = OptPropArrays(OptProp)

    #Array initialization
    rmie_TAMU = np.zeros((wvls_num, sizes_num))

    qext_inter = np.zeros((wvls_num, SD_num))
    ssa_inter = np.zeros((wvls_num, SD_num))
    g_inter = np.zeros((wvls_num, SD_num))
    qsca_inter = np.zeros((wvls_num, SD_num))
    parea_inter = np.zeros((wvls_num, SD_num))

    height = np.zeros((SD_bins))
    width = np.zeros((SD_bins))
    weigth = np.zeros((SD_bins))
    midpoint = np.zeros((SD_bins))

    qext_avg = np.zeros((wvls_num)) #Size-averaged optical properties
    ssa_avg = np.zeros((wvls_num))
    g_avg = np.zeros((wvls_num))

    #Here we transform the projected surface area outputed by TAMUdust into rmie
    for i in range(wvls_num):
        for j in range(sizes_num):
            rmie_TAMU[i,j] = np.sqrt(parea[i,j]/np.pi)

    #Interpolate
    for i in range(wvls_num):
        qext_Akima = Akima1DInterpolator(rmie_TAMU[i,:],qext[i,:])
        ssa_Akima = Akima1DInterpolator(rmie_TAMU[i,:],ssa[i,:])
        g_Akima = Akima1DInterpolator(rmie_TAMU[i,:],g[i,:])

        qext_inter[i,:] = qext_Akima(rmie_SD)
        ssa_inter[i,:] = ssa_Akima(rmie_SD)
        g_inter[i,:] = g_Akima(rmie_SD)
        parea_inter[i,:] = np.pi*(rmie_SD**2)

    #Calculates the scattering efficiency Qsca from ssa and Qext
    for i in range(wvls_num):
        for j in range(SD_num):
            qsca_inter[i,j] = ssa_inter[i,j]*qext_inter[i,j]

    #Obtain n(r) weights by Riemann midpoint rule
    for i in range(SD_bins):
        height[i] = (n_SD[i]+n_SD[i+1])/2.0
        width[i] = rmie_SD[i+1]-rmie_SD[i]
        weigth[i] = height[i]*width[i]
        midpoint[i] = (rmie_SD[i+1]+rmie_SD[i])/2.0

    #Normalize the size distribution
    sum = 0
    for i in range(SD_bins): #Calculate normalization constant
        sum = sum + weigth[i]
    A = 1.0/sum #Normalization constant

    for i in range(SD_bins): #Normalization
        weigth[i]=A*weigth[i]

    #Averaging!
    for i in range(wvls_num):
        #<Qext>
        n_A_qext = 0 #Initialization
        n_A = 0
        for j in range(SD_bins):
            n_A_qext = n_A_qext + weigth[j]*((parea_inter[i,j+1]+parea_inter[i,j])/2)*((qext_inter[i,j+1]+qext_inter[i,j])/2) #This value is used as well for <ssa>
            n_A = n_A + weigth[j]*((parea_inter[i,j+1]+parea_inter[i,j])/2)
        qext_avg[i] = n_A_qext / n_A

        #<ssa>
        n_A_qsca = 0
        for j in range(SD_bins):
            n_A_qsca = n_A_qsca + weigth[j]*((parea_inter[i,j+1]+parea_inter[i,j])/2)*((qsca_inter[i,j+1]+qsca_inter[i,j])/2) #This values is used as well for <g>
        ssa_avg[i] = n_A_qsca / n_A_qext

        #<g>
        n_A_qsca_g = 0
        for j in range(SD_bins):
            n_A_qsca_g = n_A_qsca_g + weigth[j]*((parea_inter[i,j+1]+parea_inter[i,j])/2)*((qsca_inter[i,j+1]+qsca_inter[i,j])/2)*((g_inter[i,j+1]+g_inter[i,j])/2)
        g_avg[i] = n_A_qsca_g / n_A_qsca

    #Save averaged values in .txt file
    output = pd.DataFrame({"#Wavelength(um)": wvls, "Qext": qext_avg, "ssa": ssa_avg, "g": g_avg})
    output.to_csv("outputOptPropAve_Disc.txt", sep="\t", index=False, float_format='%.15f')

    #Plot if wanted
    if plot == True:
        #Plots the discrete size distribution and SD_bins
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(rmie_SD, n_SD, 'o', markersize=2, color='red', label='n(r)')
        ax.plot(rmie_SD, n_SD, linewidth=1, color='orange')
        ax.bar(midpoint[:], height[:], width=width, color='blue', alpha=0.2)
        ax.set_title('Size distribution used for averaging the optical properties')
        ax.set_xlabel('Surface area equivalent sphere radius, r_A (um)')
        ax.legend(loc='best')

        plt.show()

        #Plot the averaged optical properties
        fig, ax1 = plt.subplots(3, 1, figsize=(12, 7))
        ax1[0].plot(wvls, qext_avg, 'o', markersize=2, color='red', label='Qext')
        ax1[0].plot(wvls, qext_avg, linewidth=1, color='red')
        ax1[0].set_ylabel('Extinction factor, Q_ext')
        ax1[0].set_title('Size-averaged optical properties')

        ax1[1].plot(wvls, ssa_avg, 'o', markersize=2, color='red', label='SSA')
        ax1[1].plot(wvls, ssa_avg, linewidth=1, color='red')
        ax1[1].set_ylabel('Single scattering albedo, w')

        ax1[2].plot(wvls, g_avg, 'o', markersize=2, color='red', label='g')
        ax1[2].plot(wvls, g_avg, linewidth=1, color='red')
        ax1[2].set_ylabel('Asymmetry factor, g')
        ax1[2].set_xlabel('Wavelength (um)')

        for j in range(3):
            ax1[j].legend(loc='best') 

        plt.show()

    return


def InterpolateRI(RI_path: str, wvls_int: np.array, plot: bool = False):

    #RI file is expected to be 3 columns: 1-wvl, 2-real part, 3-im. part
    RI = pd.read_csv(RI_path, sep='\\s+', comment="#", header=None)

    #Read columns:  Complex refractive index: m = n + ik
    wvls_ori = RI.iloc[:, 0].values
    n = RI.iloc[:, 1].values
    k = RI.iloc[:, 2].values

    #Interpolate
    inter_n = Akima1DInterpolator(wvls_ori,n)
    inter_k = Akima1DInterpolator(wvls_ori,k)

    #Write interpolation
    inter_RI = pd.DataFrame({"# Wvl (um) visible": wvls_int, "  # n": inter_n(wvls_int), "  # k": inter_k(wvls_int)})
    inter_RI.to_csv("outputInterpolateRI.txt", sep="\t", index=False, float_format="%.10f")

    #Plot if needed
    if plot == True:
        fig, ax = plt.subplots(2, 1, figsize=(5, 7))
        ax[0].plot(wvls_ori, n, 'o', markersize=2, color='red', label='n')
        ax[0].plot(wvls_ori, n, linewidth=1, color='red')
        ax[0].plot(wvls_int, inter_n(wvls_int), linewidth=1, color='orange', label='interpolation')
        ax[0].plot(wvls_int, inter_n(wvls_int), 'x', markersize=2, color='black')
        ax[0].set_ylabel('Real part of RI')
        ax[0].set_title('Refractive index: m=n+ik')
        ax[0].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        ax[1].plot(wvls_ori, k, 'o', markersize=1, color='purple', label='k')
        ax[1].plot(wvls_ori, k, linewidth=1, color='purple')
        ax[1].plot(wvls_int, inter_k(wvls_int), linewidth=1, color='blue', label='interpolation')
        ax[1].plot(wvls_int, inter_k(wvls_int), 'x', markersize=2, color='black')
        ax[1].set_ylabel('Imaginary part of RI')
        ax[1].set_xlabel('Wavelength (m)')
        ax[1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        for j in range(2):
            ax[j].legend(loc='best')   

        plt.show()

    return


def OptPropPlotTAMU(*OptProp_path: str, mode: str = "wvls", value: float):

    fig, ax = plt.subplots(3, 1, figsize=(12,7))

    #Loop to plot files
    for i in OptProp_path:
        OptProp = pd.read_csv(i, sep='\\s+', comment="#", header=None)

        #Read axis
        wvls, wvls_num, sizes, sizes_num = OptPropAxis(OptProp)

        #Initialize arrays
        qext = np.zeros((wvls_num, sizes_num)) #Size dependent optical properties
        ssa = np.zeros((wvls_num, sizes_num))
        g = np.zeros((wvls_num, sizes_num))
        rmie_TAMU = np.zeros((wvls_num, sizes_num))

        #Read values
        parea_temp = OptProp.iloc[:,3] #Surface area equivalent radius
        qext_temp = OptProp.iloc[:,4] #Extinction efficiency
        ssa_temp = OptProp.iloc[:,5] #Single Scattering albedo
        g_temp = OptProp.iloc[:,6] #Asymmetry factor

        #Creates 2-dim arrays of optical properties. First index indicates wavelength, second size bin
        k=0 #k ensures that when reading the TAMU output we copy only the unique wavelengths
        for j in range(wvls_num):
            for l in range(sizes_num):
                qext[j,l] = qext_temp[k]
                ssa[j,l] = ssa_temp[k]
                g[j,l] = g_temp[k]
                rmie_TAMU[j,l] = np.sqrt(parea_temp[k]/np.pi) #Here we transform the projected surface area outputed by TAMUdust into rmie
                k=k+1

        #Choose wvl or size to plot
        if mode == "wvls":
            plot_idx = np.argmin(np.abs(rmie_TAMU[0,:] - value))

            ax[0].plot(wvls, qext[:,plot_idx], linewidth=1, markersize=2, label=f"{i}, rmie={rmie_TAMU[0,plot_idx]:.2f}um")
            ax[0].plot(wvls, qext[:,plot_idx], "o", markersize=2, color="black")
            ax[0].set_xscale('log')
            ax[0].set_ylim(0, 3)
            ax[0].set_ylabel(r'$Q_{ext}$')
            ax[0].set_xlabel(r'Wavelength ($\mu$m)')
            ax[0].set_title(r'Extinction efficiency factor, $Q_{ext}$')

            ax[1].plot(wvls, ssa[:,plot_idx], linewidth=1, markersize=2, label=f"{i}, rmie={rmie_TAMU[0,plot_idx]:.2f}um")
            ax[1].plot(wvls, ssa[:,plot_idx], "o", markersize=2, color="black")
            ax[1].set_ylim(0, 1)
            ax[1].set_xscale('log')
            ax[1].set_ylabel(r'$\omega$')
            ax[1].set_xlabel(r'Wavelength ($\mu$m)')
            ax[1].set_title(r'Single scattering albedo, $\omega$')

            ax[2].plot(wvls, g[:,plot_idx], linewidth=1, markersize=2, label=f"{i}, rmie={rmie_TAMU[0,plot_idx]:.2f}um")
            ax[2].plot(wvls, g[:,plot_idx], "o", markersize=2, color="black")
            ax[2].set_xscale('log')
            ax[2].set_ylim(0, 1)
            ax[2].set_ylabel('g')
            ax[2].set_xlabel(r'Wavelength ($\mu$m)')
            ax[2].set_title(r'Asymmetry factor, g')

            for j in range(3):
                ax[j].legend(loc='best') 


        elif mode == "sizes":
            plot_idx = np.argmin(np.abs(wvls - value))
            plot_value = wvls[plot_idx]

            rmie_TAMU[plot_idx,:] = rmie_TAMU[plot_idx,:]*2*np.pi / plot_value #Transform rmie into size parameter

            ax[0].plot(rmie_TAMU[plot_idx,:], qext[plot_idx,:], linewidth=1, markersize=2, label=f"{i}, wvl={wvls[plot_idx]:.2f}um")
            ax[0].plot(rmie_TAMU[plot_idx,:], qext[plot_idx,:], "o", markersize=2, color="black")
            ax[0].set_xscale('log')
            ax[0].set_ylim(0, 3)
            ax[0].set_ylabel(r'$Q_{ext}$')
            ax[0].set_xlabel(r'Size parameter (2*$\pi$*$r_{mie}$/$\lambda$)')
            ax[0].set_title(r'Extinction efficiency factor, $Q_{ext}$')

            ax[1].plot(rmie_TAMU[plot_idx,:], ssa[plot_idx,:], linewidth=1, markersize=2, label=f"{i}, wvl={wvls[plot_idx]:.2f}um")
            ax[1].plot(rmie_TAMU[plot_idx,:], ssa[plot_idx,:], "o", markersize=2, color="black")
            ax[1].set_ylim(0, 1)
            ax[1].set_xscale('log')
            ax[1].set_ylabel(r'$\omega$')
            ax[1].set_xlabel(r'Size parameter (2*$\pi$*$r_{mie}$/$\lambda$)')
            ax[1].set_title(r'Single scattering albedo, $\omega$')

            ax[2].plot(rmie_TAMU[plot_idx,:], g[plot_idx,:], linewidth=1, markersize=2, label=f"{i}, wvl={wvls[plot_idx]:.2f}um")
            ax[2].plot(rmie_TAMU[plot_idx,:], g[plot_idx,:], "o", markersize=2, color="black")
            ax[2].set_xscale('log')
            ax[2].set_ylim(0, 1)
            ax[2].set_ylabel('g')
            ax[2].set_xlabel(r'Size parameter (2*$\pi$*$r_{mie}$/$\lambda$)')
            ax[2].set_title(r'Asymmetry factor, g')

            for j in range(3):
                ax[j].legend(loc='best') 

    plt.tight_layout()
    plt.show()

    return

def OptPropPlotGCM(*OptProp_path: str, mode: str = "wvls", value: float = 5.):

    fig, ax = plt.subplots(3, 1, figsize=(12,7))

    #From the GCM OptProp file we want to extract the optical properties in order to plot them
    #Loop to plot files
    for i in OptProp_path:

        #First we want to read wvls_num and sizes_num
        nwvl_found = False
        nwvl_assigned = False
        nsize_found = False
        nsize_assigned = False

        with open(i, "r", encoding="utf-8") as OptProp:
            for linea_num, linea in enumerate(OptProp, start=1):
                if nwvl_found == True:
                    split = linea.split()
                    wvls_num = int(split[0])
                    nwvl_found = False
                    nwvl_assigned = True
                if "(nwvl):" in linea:
                    nwvl_found = True
                if nsize_found == True:
                    split = linea.split()
                    sizes_num = int(split[0])
                    nsize_found = False
                    nsize_assigned = True
                if "(nsize):" in linea:
                    nsize_found = True
                if nwvl_assigned == True and nsize_assigned == True:
                    break

        #Then we want to obtain the wvls, sizes and properties arrays
        with open(i, "r", encoding="utf-8") as OptProp:
            lineas = OptProp.readlines()
            wvls = []
            sizes = []
            qext_temp = []
            ssa_temp = []
            g_temp = []
            case = None

            for linea in lineas:
                linea = linea.strip()
                if not linea or linea.startswith("#"):
                    if "Wavelength axis" in linea:
                        case = "wvls"
                    elif "Particle size axis" in linea:
                        case = "sizes"
                    elif "Extinction coef." in linea:
                        case = "qext"
                    elif "Single Scat Albedo" in linea:
                        case = "ssa"
                    elif "Assymetry Factor" in linea:
                        case = "g"
                    continue

                values = [float(x) for x in linea.split()]

                if case == "wvls":
                    wvls.extend(values)
                elif case == "sizes":
                    sizes.extend(values)
                elif case == "qext":
                    qext_temp.extend(values)
                elif case == "ssa":
                    ssa_temp.extend(values)
                elif case == "g":
                    g_temp.extend(values)
            
            wvls = np.array(wvls) * 1e6 #Covert to numpy array and change m to um
            sizes = np.array(sizes) *1e6
            qext_temp = np.array(qext_temp)
            ssa_temp = np.array(ssa_temp)
            g_temp = np.array(g_temp)

        #Now we can proceed exactly as with OptPropPlotTAMU
        #Initialize arrays
        qext = np.zeros((wvls_num, sizes_num)) #Size dependent optical properties
        ssa = np.zeros((wvls_num, sizes_num))
        g = np.zeros((wvls_num, sizes_num))

        #Creates 2-dim arrays of optical properties. First index indicates wavelength, second size bin
        k=0 #k ensures that when reading the TAMU output we copy only the unique wavelengths
        for j in range(sizes_num):
            for l in range(wvls_num):
                qext[l,j] = qext_temp[k]
                ssa[l,j] = ssa_temp[k]
                g[l,j] = g_temp[k]
                k=k+1

        #Choose wvl or size to plot
        if mode == "wvls":

            plot_idx = np.argmin(np.abs(sizes - value))

            ax[0].plot(wvls, qext[:,plot_idx], linewidth=1, markersize=2, label=f"{i}, rmie={sizes[plot_idx]:.2f}um")
            ax[0].plot(wvls, qext[:,plot_idx], "o", markersize=2, color="black")
            ax[0].set_xscale('log')
            ax[0].set_ylim(0, 4)
            ax[0].set_ylabel(r'$Q_{ext}$')
            ax[0].set_xlabel(r'Wavelength ($\mu$m)')
            ax[0].set_title(r'Extinction efficiency factor, $Q_{ext}$')

            ax[1].plot(wvls, ssa[:,plot_idx], linewidth=1, markersize=2, label=f"{i}, rmie={sizes[plot_idx]:.2f}um")
            ax[1].plot(wvls, ssa[:,plot_idx], "o", markersize=2, color="black")
            ax[1].set_ylim(0, 1)
            ax[1].set_xscale('log')
            ax[1].set_ylabel(r'$\omega$')
            ax[1].set_xlabel(r'Wavelength ($\mu$m)')
            ax[1].set_title(r'Single scattering albedo, $\omega$')

            ax[2].plot(wvls, g[:,plot_idx], linewidth=1, markersize=2, label=f"{i}, rmie={sizes[plot_idx]:.2f}um")
            ax[2].plot(wvls, g[:,plot_idx], "o", markersize=2, color="black")
            ax[2].set_xscale('log')
            ax[2].set_ylim(0, 1)
            ax[2].set_ylabel('g')
            ax[2].set_xlabel(r'Wavelength ($\mu$m)')
            ax[2].set_title(r'Asymmetry factor, g')

            for j in range(3):
                ax[j].legend(loc='best') 


        elif mode == "sizes":
            plot_idx = np.argmin(np.abs(wvls - value))
            plot_value = wvls[plot_idx]

            sizes = sizes*2*np.pi / plot_value #Transform rmie into size parameter

            ax[0].plot(sizes, qext[plot_idx,:], linewidth=1, markersize=2, label=f"{i}, wvl={wvls[plot_idx]:.2f}um")
            ax[0].plot(sizes, qext[plot_idx,:], "o", markersize=2, color="black")
            ax[0].set_xscale('log')
            ax[0].set_ylim(0, 4)
            ax[0].set_ylabel(r'$Q_{ext}$')
            ax[0].set_xlabel(r'Size parameter (2*$\pi$*$r_{mie}$/$\lambda$)')
            ax[0].set_title(r'Extinction efficiency factor, $Q_{ext}$')

            ax[1].plot(sizes, ssa[plot_idx,:], linewidth=1, markersize=2, label=f"{i}, wvl={wvls[plot_idx]:.2f}um")
            ax[1].plot(sizes, ssa[plot_idx,:], "o", markersize=2, color="black")
            ax[1].set_ylim(0, 1)
            ax[1].set_xscale('log')
            ax[1].set_ylabel(r'$\omega$')
            ax[1].set_xlabel(r'Size parameter (2*$\pi$*$r_{mie}$/$\lambda$)')
            ax[1].set_title(r'Single scattering albedo, $\omega$')

            ax[2].plot(sizes, g[plot_idx,:], linewidth=1, markersize=2, label=f"{i}, wvl={wvls[plot_idx]:.2f}um")
            ax[2].plot(sizes, g[plot_idx,:], "o", markersize=2, color="black")
            ax[2].set_xscale('log')
            ax[2].set_ylim(0, 1)
            ax[2].set_ylabel('g')
            ax[2].set_xlabel(r'Size parameter (2*$\pi$*$r_{mie}$/$\lambda$)')
            ax[2].set_title(r'Asymmetry factor, g')

            for j in range(3):
                ax[j].legend(loc='best') 

    plt.tight_layout()
    plt.show()

    return 

def OptPropExtract(OptProp_path: str, header: str):

    OptProp = pd.read_csv(OptProp_path, sep='\\s+', comment="#", header=None)
    output_path = "outputOptPropExtract.dat"

    #Read axis
    wvls, wvls_num, rmie_avg, rmie_num = OptPropAxis(OptProp)
    wvls = wvls * 1e-6
    rmie_avg = rmie_avg * 1e-6

    #Read essential arrays
    parea, qext, ssa, g = OptPropArrays(OptProp)

    #Makes sure ssa is not < 0 or > 1
    for i in range(wvls_num):
        for j in range(rmie_num):
            if ssa[i,j] <= 0.:
                ssa[i,j] = 0
            if g[i,j] <= 0.:
                g[i,j] = 0


    #Write header, nº of wvls and nº of radii in file
    with open(output_path, "w") as f:
        f.write(f"{header}\n")
        f.write(f"# Number of wavelengths (nwvl):\n  {wvls_num}\n")
        f.write(f"# Number of radius (nsize):\n  {rmie_num}\n")

    #Write wvls axis
    with open(output_path, "a") as f:
        f.write("# Wavelength axis (wvl):\n")
        k = 0
        groups = wvls_num//5 #Number of groups of 5 lines
        lastline = wvls_num - groups*5 #Number of elements in last line either 1,2,3 or 4
        for i in range(groups):
            f.write(f" {wvls[0+k]:.6E} {wvls[1+k]:.6E} {wvls[2+k]:.6E} {wvls[3+k]:.6E} {wvls[4+k]:.6E}\n")
            k = k + 5
        if lastline != 0:
            for i in range(lastline):
                f.write(f" {wvls[k+i]:.6E}")
            f.write("\n")

    #Write sizes axis
    with open(output_path, "a") as f:
        f.write("# Particle size axis (radius):\n")
        k = 0
        groups = rmie_num//5 #Number of groups of 5 lines
        lastline = rmie_num - groups*5 #Number of elements in last line either 1,2,3 or 4
        for i in range(groups):
            f.write(f" {rmie_avg[0+k]:.6E} {rmie_avg[1+k]:.6E} {rmie_avg[2+k]:.6E} {rmie_avg[3+k]:.6E} {rmie_avg[4+k]:.6E}\n")
            k = k + 5
        if lastline != 0:
            for i in range(lastline):
                f.write(f" {rmie_avg[k+i]:.6E}")
            f.write("\n")

    #Write extinction coefficient
    with open(output_path, "a") as f:
        f.write("# Extinction coef. Qext (ep):\n")
        groups = wvls_num//5
        lastline = wvls_num - groups*5        
        for i in range(rmie_num):
            k = 0
            f.write(f"# Radius number     {i+1}\n")
            for j in range(groups):
                f.write(f" {qext[0+k,i]:.6E} {qext[1+k,i]:.6E} {qext[2+k,i]:.6E} {qext[3+k,i]:.6E} {qext[4+k,i]:.6E}\n")
                k = k + 5
            if lastline != 0:
                for l in range(lastline):
                    f.write(f" {qext[l+k,i]:.6E}")
                f.write("\n")
    
    #Write single scattering albedo
    with open(output_path, "a") as f:
        f.write("# Single Scat Albedo (omeg):\n")
        groups = wvls_num//5
        lastline = wvls_num - groups*5        
        for i in range(rmie_num):
            k = 0
            f.write(f"# Radius number     {i+1}\n")
            for j in range(groups):
                f.write(f" {ssa[0+k,i]:.6E} {ssa[1+k,i]:.6E} {ssa[2+k,i]:.6E} {ssa[3+k,i]:.6E} {ssa[4+k,i]:.6E}\n")
                k = k + 5
            if lastline != 0:
                for l in range(lastline):
                    f.write(f" {ssa[l+k,i]:.6E}")
                f.write("\n")

    #Write asymmetry parameter
    with open(output_path, "a") as f:
        f.write("# Assymetry Factor (gfactor):\n")
        groups = wvls_num//5
        lastline = wvls_num - groups*5        
        for i in range(rmie_num):
            k = 0
            f.write(f"# Radius number     {i+1}\n")
            for j in range(groups):
                f.write(f" {g[0+k,i]:.6E} {g[1+k,i]:.6E} {g[2+k,i]:.6E} {g[3+k,i]:.6E} {g[4+k,i]:.6E}\n")
                k = k + 5
            if lastline != 0:
                for l in range(lastline):
                    f.write(f" {g[l+k,i]:.6E}")
                f.write("\n") 
    return