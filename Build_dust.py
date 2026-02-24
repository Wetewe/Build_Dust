import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sps
from scipy.interpolate import Akima1DInterpolator
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from scipy.constants import h, c, k
from pathlib import Path

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
    vol = np.zeros((wvls_num, sizes_num))
    parea = np.zeros((wvls_num, sizes_num))
    qext = np.zeros((wvls_num, sizes_num)) #Size dependent optical properties
    ssa = np.zeros((wvls_num, sizes_num))
    g = np.zeros((wvls_num, sizes_num))

    #Read values
    vol_temp = OptProp.iloc[:,2] #Volume
    parea_temp = OptProp.iloc[:,3] #Projected surface area
    qext_temp = OptProp.iloc[:,4] #Extinction efficiency
    ssa_temp = OptProp.iloc[:,5] #Single Scattering albedo
    g_temp = OptProp.iloc[:,6] #Asymmetry factor

    #Creates 2-dim arrays of optical properties. First index indicates wavelength, second size bin
    k=0 #k ensures that when reading the TAMU output we copy only the unique wavelengths
    for i in range(wvls_num):
        for j in range(sizes_num):
            vol[i,j] = vol_temp[k]
            parea[i,j] = parea_temp[k]
            qext[i,j] = qext_temp[k]
            ssa[i,j] = ssa_temp[k]
            g[i,j] = g_temp[k]
            k=k+1
    
    return vol, parea, qext, ssa, g

def NormPlanck(wvl: np.array, T: float, peak: float):
    #wvl is inputted in um but changed to m
    wvl_m = wvl * 1e-6

    a = 2*h*c**2 / wvl**5
    b = 1 / (np.exp((h*c)/(wvl_m*k*T)) - 1)

    B = a*b
    Bn = B / np.trapezoid(B,wvl)
    Bn = Bn * peak / Bn.max()

    return Bn

def FindLinePos(filename, key):
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if key in line:
                return i + 1  #Data reading starts AFTER this line
    raise ValueError("Key line not found")

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

    #Data that is going to be writen in files
    write_n = inter_n(wvls_int)
    write_k = inter_k(wvls_int)

    #Write interpolation
    inter_RI = pd.DataFrame({"# Wvl (um) visible": wvls_int, "  # n": write_n, "  # k": write_k})
    inter_RI.to_csv("outputInterpolateRI.txt", sep="\t", index=False, float_format="%.10f")

    #Plot if needed
    if plot == True:
        fig, ax = plt.subplots(2, 1, figsize=(5, 7))
        ax[0].plot(wvls_ori, n, 'o', markersize=2, color='red', label='n')
        ax[0].plot(wvls_ori, n, linewidth=1, color='red')
        ax[0].plot(wvls_int, inter_n(wvls_int), linewidth=1, color='orange', label='interpolation')
        ax[0].plot(wvls_int, inter_n(wvls_int), 'x', markersize=2, color='black')
        ax[0].set_xscale("log")
        ax[0].set_ylabel('Real part of RI')
        ax[0].set_title('Refractive index: m=n+ik')

        ax[1].plot(wvls_ori, k, 'o', markersize=1, color='purple', label='k')
        ax[1].plot(wvls_ori, k, linewidth=1, color='purple')
        ax[1].plot(wvls_int, inter_k(wvls_int), linewidth=1, color='blue', label='interpolation')
        ax[1].plot(wvls_int, inter_k(wvls_int), 'x', markersize=2, color='black')
        ax[1].set_xscale("log")
        ax[1].set_ylabel('Imaginary part of RI')
        ax[1].set_xlabel('Wavelength (um)')

        for j in range(2):
            ax[j].legend(loc='best')   

        plt.show()

    return

#### OPTICAL PROPERTIES UTILITIES ####

def OptPropWriteGCM(sizes: np.array, wvls: np.array, header: str, files: list):
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
    vol, parea, qext, ssa, g = OptPropArrays(OptProp)

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
    
    #Analyze volume
    #vol_sphere = 4./3. * np.pi * rmie_TAMU**3
    #ratio = vol / vol_sphere
    #ratio_avg = np.average(ratio,axis=0)
    #print(ratio_avg)

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
        x_axis = np.linspace(1e-6,50,num=10000)
        ax.plot(x_axis, SD(r_eff, v_eff, x_axis, SD_choice), linewidth=2, color='purple', label=SD_name)
        for i in range(sizes_num):
            ax.plot(rmie_TAMU[0,i],SD(r_eff, v_eff, rmie_TAMU[0,i], SD_choice),"x",color="black",)
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
    vol, parea, qext, ssa, g = OptPropArrays(OptProp)

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

def OptPropPlotTAMU(*OptProp_path: str, Labels: list = [], mode: str = "wvls", value: float):

    fig, ax = plt.subplots(3, 1, figsize=(12,7))
    m = 0 #Index for labels

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

        #Labels
        if Labels == []: label = i
        else: label = Labels[m]

        #Choose wvl or size to plot
        if mode == "wvls":
            plot_idx = np.argmin(np.abs(rmie_TAMU[0,:] - value))

            ax[0].plot(wvls, qext[:,plot_idx], linewidth=1, markersize=2, label=f"{label}," r" $r_{Mie}=$"f"{rmie_TAMU[0,plot_idx]:.2f}"r"$\mu$m")
            ax[0].plot(wvls, qext[:,plot_idx], "o", markersize=2, color="black")
            ax[0].set_xscale('log')
            ax[0].set_ylim(0, 3)
            ax[0].set_ylabel(r'$Q_{ext}$')
            ax[0].set_title(r'Extinction efficiency factor, $Q_{ext}$')

            ax[1].plot(wvls, ssa[:,plot_idx], linewidth=1, markersize=2, label=f"{label}," r" $r_{Mie}=$"f"{rmie_TAMU[0,plot_idx]:.2f}"r"$\mu$m")
            ax[1].plot(wvls, ssa[:,plot_idx], "o", markersize=2, color="black")
            ax[1].set_ylim(0, 1)
            ax[1].set_xscale('log')
            ax[1].set_ylabel(r'$\omega$')
            ax[1].set_title(r'Single scattering albedo, $\omega$')

            ax[2].plot(wvls, g[:,plot_idx], linewidth=1, markersize=2, label=f"{label}," r" $r_{Mie}=$"f"{rmie_TAMU[0,plot_idx]:.2f}"r"$\mu$m")
            ax[2].plot(wvls, g[:,plot_idx], "o", markersize=2, color="black")
            ax[2].set_xscale('log')
            ax[2].set_ylim(0, 1)
            ax[2].set_ylabel('g')
            ax[2].set_title(r'Asymmetry factor, g')

            for j in range(3):
                ax[j].legend(loc='best')
                ax[j].grid(True, which="both")
                ax[j].set_xlabel(r'Wavelength ($\mu$m)')
                ax[j].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
                ax[j].xaxis.set_minor_formatter(ScalarFormatter()) 


        elif mode == "sizes":
            plot_idx = np.argmin(np.abs(wvls - value))
            plot_value = wvls[plot_idx]

            rmie_TAMU[plot_idx,:] = rmie_TAMU[plot_idx,:]*2*np.pi / plot_value #Transform rmie into size parameter

            ax[0].plot(rmie_TAMU[plot_idx,:], qext[plot_idx,:], linewidth=1, markersize=2, label=f"{label}," r" $\lambda=$" f"{wvls[plot_idx]:.2f}"r"$\mu$m")
            ax[0].plot(rmie_TAMU[plot_idx,:], qext[plot_idx,:], "o", markersize=2, color="black")
            ax[0].set_xscale('log')
            ax[0].set_ylim(0, 3)
            ax[0].set_ylabel(r'$Q_{ext}$')
            ax[0].set_title(r'Extinction efficiency factor, $Q_{ext}$')

            ax[1].plot(rmie_TAMU[plot_idx,:], ssa[plot_idx,:], linewidth=1, markersize=2, label=f"{label}," r" $\lambda=$" f"{wvls[plot_idx]:.2f}"r"$\mu$m")
            ax[1].plot(rmie_TAMU[plot_idx,:], ssa[plot_idx,:], "o", markersize=2, color="black")
            ax[1].set_ylim(0, 1)
            ax[1].set_xscale('log')
            ax[1].set_ylabel(r'$\omega$')
            ax[1].set_title(r'Single scattering albedo, $\omega$')

            ax[2].plot(rmie_TAMU[plot_idx,:], g[plot_idx,:], linewidth=1, markersize=2, label=f"{label}," r" $\lambda=$" f"{wvls[plot_idx]:.2f}"r"$\mu$m")
            ax[2].plot(rmie_TAMU[plot_idx,:], g[plot_idx,:], "o", markersize=2, color="black")
            ax[2].set_xscale('log')
            ax[2].set_ylim(0, 1)
            ax[2].set_ylabel('g')
            ax[2].set_title(r'Asymmetry factor, g')

            for j in range(3):
                ax[j].legend(loc='best')
                ax[j].grid(True, which="both")
                ax[j].set_xlabel(r'Size parameter (2*$\pi$*$r_{mie}$/$\lambda$)')
                ax[j].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))  

        m = m + 1 #Label index
        
    plt.tight_layout()
    plt.show()

    return

def OptPropPlotGCM(*OptProp_path: str, Labels: list = [], mode: str = "wvls", value: float = 1.5):

    fig, ax = plt.subplots(3, 1, figsize=(12,7))
    m = 0 #Index for labels

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

        #Labels
        if Labels == []: label = i
        else: label = Labels[m]

        #ylims
        ylims=(4,1,1)

        #Choose wvl or size to plot
        if mode == "wvls":

            plot_idx = np.argmin(np.abs(sizes - value))

            ax[0].plot(wvls, qext[:,plot_idx], linewidth=1, markersize=2, label=f"{label}," r" $r_{Mie}=$"f"{sizes[plot_idx]:.2f}"r"$\mu$m")
            ax[0].plot(wvls, qext[:,plot_idx], "o", markersize=2, color="black")
            ax[0].set_xscale('log')
            ax[0].set_ylim(0, ylims[0])
            ax[0].set_ylabel(r'$Q_{ext}$')
            ax[0].set_title(r'Extinction efficiency factor, $Q_{ext}$')

            ax[1].plot(wvls, ssa[:,plot_idx], linewidth=1, markersize=2, label=f"{label}," r" $r_{Mie}=$"f"{sizes[plot_idx]:.2f}"r"$\mu$m")
            ax[1].plot(wvls, ssa[:,plot_idx], "o", markersize=2, color="black")
            ax[1].set_ylim(0, ylims[1])
            ax[1].set_xscale('log')
            ax[1].set_ylabel(r'$\omega$')
            ax[1].set_title(r'Single scattering albedo, $\omega$')

            ax[2].plot(wvls, g[:,plot_idx], linewidth=1, markersize=2, label=f"{label}," r" $r_{Mie}=$"f"{sizes[plot_idx]:.2f}"r"$\mu$m")
            ax[2].plot(wvls, g[:,plot_idx], "o", markersize=2, color="black")
            ax[2].set_xscale('log')
            ax[2].set_ylim(0, ylims[2])
            ax[2].set_ylabel('g')
            ax[2].set_title(r'Asymmetry factor, g')



            for j in range(3): #Common settings
                ax[j].legend(loc='best')

                ax[j].set_xlabel(r'Wavelength ($\mu$m)')
                ax[j].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
                ax[j].xaxis.set_minor_formatter(ScalarFormatter())

                # Crear eje superior para x_eff
                ax_top = ax[j].secondary_xaxis('top', functions=(
                lambda wvl: 2 * np.pi * sizes[plot_idx] / wvl,  # forward: wvl -> x_eff
                lambda xeff: 2 * np.pi * sizes[plot_idx] / xeff  # inverse: x_eff -> wvl
                ))
                ax_top.set_xlabel(r'$x_{eff} = 2\pi r_{eff}/\lambda$')
                ax_top.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

                if (np.max(wvls) <= 5):
                    ax[j].axvline(x=0.5, linestyle='--', color='red', linewidth=1.5)
                    ax[j].axvline(x=5, linestyle='--', color='red', linewidth=1.5)
                    bbaxis = np.logspace(-0.7,0.7,100)
                    ax[j].plot(bbaxis,NormPlanck(bbaxis,6000,ylims[j]),linewidth=1,linestyle="--",color="grey")
                if (np.max(wvls) >= 5):
                    ax[j].axvline(x=5, linestyle='--', color='red', linewidth=1.5)
                    ax[j].axvline(x=11.56, linestyle='--', color='red', linewidth=1.5)
                    ax[j].axvline(x=20, linestyle='--', color='red', linewidth=1.5)
                    bbaxis = np.logspace(0.7,2,300)
                    ax[j].plot(bbaxis,NormPlanck(bbaxis,215,ylims[j]),linewidth=1,linestyle="--",color="grey")                    

        elif mode == "sizes":
            plot_idx = np.argmin(np.abs(wvls - value))
            plot_value = wvls[plot_idx]

            sizes = sizes*2*np.pi / plot_value #Transform rmie into size parameter

            ax[0].plot(sizes, qext[plot_idx,:], linewidth=1, markersize=2, label=f"{label}," r" $\lambda=$" f"{wvls[plot_idx]:.2f}"r"$\mu$m")
            ax[0].plot(sizes, qext[plot_idx,:], "o", markersize=2, color="black")
            ax[0].set_xscale('log')
            ax[0].set_ylim(0, 4)
            ax[0].set_ylabel(r'$Q_{ext}$')
            ax[0].set_title(r'Extinction efficiency factor, $Q_{ext}$')

            ax[1].plot(sizes, ssa[plot_idx,:], linewidth=1, markersize=2, label=f"{label}," r" $\lambda=$" f"{wvls[plot_idx]:.2f}"r"$\mu$m")
            ax[1].plot(sizes, ssa[plot_idx,:], "o", markersize=2, color="black")
            ax[1].set_ylim(0, 1)
            ax[1].set_xscale('log')
            ax[1].set_ylabel(r'$\omega$')
            ax[1].set_title(r'Single scattering albedo, $\omega$')

            ax[2].plot(sizes, g[plot_idx,:], linewidth=1, markersize=2, label=f"{label}," r" $\lambda=$" f"{wvls[plot_idx]:.2f}"r"$\mu$m")
            ax[2].plot(sizes, g[plot_idx,:], "o", markersize=2, color="black")
            ax[2].set_xscale('log')
            ax[2].set_ylim(0, 1)
            ax[2].set_ylabel('g')
            ax[2].set_title(r'Asymmetry factor, g')

            for j in range(3):
                ax[j].legend(loc='best')
                ax[j].grid(True, which="both")
                ax[j].set_xlabel(r'Size parameter (2*$\pi$*$r_{mie}$/$\lambda$)')
                ax[j].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))                

        m = m + 1 #Label index

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
    vol, parea, qext, ssa, g = OptPropArrays(OptProp)

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

def OptPropInverse(OptProp_path: str, v_eff: float = 0.05):
    nwvl_found = False
    nwvl_assigned = False
    nsize_found = False
    nsize_assigned = False

    with open(OptProp_path, "r", encoding="utf-8") as OptProp:
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
    with open(OptProp_path, "r", encoding="utf-8") as OptProp:
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

    #With the arrays, we write the individual files
    Output_name = "OptPropInverse.dat"
    #Change size to projected area. Reff is the radius of the equivalent projected-surface-area
    areas = np.pi * sizes**2
    diams = 2*sizes

    with open(Output_name, 'w') as f:
        for i, wvl in enumerate(wvls):
            for j, area in enumerate(areas):
                f.write(f"   {wvl:.7E}   {diams[j]:.7E}   {0:.7E}   {area:.7E}   {qext[i,j]:.7E}   {ssa[i,j]:.7E}   {g[i,j]:.7E}\n")

    return

#### SCATTERING MATRIX UTILITIES ####

def FXXReadTAMU(FXX_path: str, OptProp_path: str):

    #Read files
    OptProp = pd.read_csv(OptProp_path, sep='\\s+', comment="#", header=None)
    FXX = pd.read_csv(FXX_path, sep='\\s+', comment="#", header=None)
    
    #Read wvls and sizes axis from isca.dat
    wvls, wvls_num, rmies, rmies_num = OptPropAxis(OptProp)

    #Read angles axis from FXX.dat
    angles = FXX.iloc[0,:]
    angles_num = len(angles)

    #Create 3dim array to store FXX
    FXX_array = np.zeros((wvls_num, rmies_num, angles_num))

    #Store information from file to array
    k = 1
    for i in range(wvls_num):
        for j in range(rmies_num):
            FXX_array[i,j,:] = FXX.iloc[k,:]
            k = k+1

    return FXX_array, wvls, rmies, angles

def FXXExtractTAMU(FXX_path: str, OptProp_path: str, rmie_set: float):

    #Get FXX, wvls and rmies arrays
    FXX_array, wvls, rmies, angles = FXXReadTAMU(FXX_path, OptProp_path)

    #Find the position of specific size
    rmie_idx = np.argmin(np.abs(rmies - rmie_set))

    #Trick that could probably be done more elegantly
    wvls_num = len(wvls)
    angles_num = len(angles)

    wvls_temp = np.zeros((wvls_num*angles_num))
    angles_temp = np.zeros((wvls_num*angles_num))
    FXX_temp = np.zeros((wvls_num*angles_num))

    k = 0
    for i in range(wvls_num):
        for j in range(angles_num):
            wvls_temp[k] = wvls[i]
            angles_temp[k] = angles[j]
            FXX_temp[k] = FXX_array[i, rmie_idx, j]
            k = k+1

    #output 
    output_path = "outputFXXExtractTAMU.dat"
    output = pd.DataFrame({"#Wavelength(um)": wvls_temp, "Scattering angle(°)": angles_temp, f"FXX (rmie={rmies[rmie_idx]})": FXX_temp})
    output.to_csv(output_path, sep="\t", index=False, float_format='%.7f')

    return

def FXXPlot(*FXXfile_path: str, wvl_set: float,
            Labels: list = [],ylabel: str = "", title: str = ""):

    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    m = 0 #Index for labels

    #Loop to plot files
    for i in FXXfile_path:
        FXXfile = pd.read_csv(i, sep='\\s+', comment="#", header=None)

        #Read wvls axis from FXXfile
        k = 0
        wvls = []
        for j in range(len(FXXfile[0]) - 1):
            if j == 0:
                wvls.append(FXXfile.iloc[0,0])
                wvl_temp = wvls[k]
            if wvl_temp != FXXfile.iloc[j+1,0]:
                wvls.append(FXXfile.iloc[j+1,0])
                k = k + 1
                wvl_temp = wvls[k]

        wvls = np.array(wvls)
        wvls_num = len(wvls)

        #Read scattering angles axis from FXXfile
        angles = []
        for j in range(len(FXXfile[1]) - 1):
            if j == 0:
                angles.append(FXXfile.iloc[0,1])
            if FXXfile.iloc[j+1,1] not in angles:
                angles.append(FXXfile.iloc[j+1,1])

        angles = np.array(angles)
        angles_num = len(angles)

        #Read values and create final array
        FXX = np.zeros((wvls_num, angles_num))
        FXX_temp = FXXfile[2]

        #Organize array
        k = 0
        for j in range(wvls_num):
            for l in range(angles_num):
                FXX[j,l] = FXX_temp[k]
                k = k+1
        
        #Fix wavelength to be plotted
        wvl_idx = np.argmin(np.abs(wvls - wvl_set))

        #Specific setting for every matrix element
        if title == "F11":
            ax.set_yscale("log")
            ax.set_ylim(0.01,1000)
            norm_idx = np.argmin(np.abs(angles - 30)) #normalization to 30deg
            for j in range(wvls_num):
                FXX[j,:] = FXX[j,:] / FXX[j,norm_idx]
        elif title == "F12":
            for j in range(wvls_num):
                FXX[j,:] = -1.0 * FXX[j,:]
                ax.set_ylim(-0.5,0.5)
        elif title == "F22":
            ax.set_ylim(0,1)
        elif title == "F33":
            ax.set_ylim(-1,1)
        elif title == "F43":
            ax.set_ylim(-0.5,0.5)
            #for j in range(wvls_num):
            #    FXX[j,:] = -1.0 * FXX[j,:]
        elif title == "F44":
            ax.set_ylim(-1,1)
        else:
            title = "WARNING: Unspecified matrix element"

        #Labels
        if Labels == []: label = i
        else: label = Labels[m]

        #Plot
        ax.set_title(f"{title}")
        ax.set_xlabel("Scattering angle (deg)")
        ax.set_xlim(0,180)
        ax.set_ylabel(f"{ylabel}")
        ax.grid(True)
        ax.plot(angles, FXX[wvl_idx, :], linewidth=1, label=f"{label}, wvl={wvls[wvl_idx]}um")
        #ax.plot(angles, FXX[wvl_idx, :], "o", markersize=2, color="black")
        ax.legend(loc="best")

        m = m + 1
        
    plt.show()

    return

def FXXAve(FXX_path: str, OptProp_path: str,
           r_eff: float, v_eff: float, SD_choice: int):

    #Read data from FXX file
    FXX, wvls, rmies, angles = FXXReadTAMU(FXX_path, OptProp_path)

    #Read data from OptProp file
    OptProp = pd.read_csv(OptProp_path, sep='\\s+', comment="#", header=None)
    vol, parea, qext, ssa, g = OptPropArrays(OptProp)

    #Initialize arrays
    angles_num = len(angles)
    wvls_num = len(wvls)
    sizes_num = len(rmies)

    rmie_TAMU = np.zeros((wvls_num, sizes_num))
    qsca = np.zeros((wvls_num,sizes_num))
    FXX_avg = np.zeros((wvls_num, angles_num))

    n_A_qsca_P = np.zeros((wvls_num, sizes_num, angles_num))
    n_A_qsca = np.zeros((wvls_num, sizes_num))

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
        n_A_qsca[i,:] = SD(r_eff, v_eff, rmie_TAMU[i,:], SD_choice)*parea[i,:]*qsca[i,:]
        denom_P = np.trapezoid(n_A_qsca[i,:], rmie_TAMU[i,:])

        for j in range(angles_num):
            n_A_qsca_P[i,:,j] = n_A_qsca[i,:]*FXX[i,:,j]
            numer_P = np.trapezoid(n_A_qsca_P[i,:,j], rmie_TAMU[i,:])

            FXX_avg[i,j] = numer_P / denom_P

    #Trick that could probably be done more elegantly
    wvls_temp = np.zeros((wvls_num*angles_num))
    angles_temp = np.zeros((wvls_num*angles_num))
    FXX_temp = np.zeros((wvls_num*angles_num))

    k = 0
    for i in range(wvls_num):
        for j in range(angles_num):
            wvls_temp[k] = wvls[i]
            angles_temp[k] = angles[j]
            FXX_temp[k] = FXX_avg[i, j]
            k = k+1

    #Save averaged FXX in file
    output_path = f"outputFXXAve_reff{r_eff}_veff{v_eff}.dat"
    output = pd.DataFrame({"#Wavelength(um)": wvls_temp, "Scattering angle(°)": angles_temp, f"FXX (r_eff={r_eff}, v_eff={v_eff})": FXX_temp})
    output.to_csv(output_path, sep="\t", index=False, float_format='%.7f')
        
    return

def FXXWriteGCM_MIE(sizes: np.array, wvls: np.array, folder: str, plot: bool = False):
    #Some necessary actions and definitions
    wvls_num = len(wvls)
    sizes_num = len(sizes)
    #angles = np.arange(0,181,1)
    angles = np.arange(0,361,1)
    angles_num = len(angles)
    sizes = sizes * 1e-6 #Change from um to m (LMD-GCM reads meters)
    wvls = wvls * 1e-6
    F11 = np.zeros((wvls_num,sizes_num,angles_num))
    F12 = np.zeros((wvls_num,sizes_num,angles_num))
    F22 = np.zeros((wvls_num,sizes_num,angles_num))
    F33 = np.zeros((wvls_num,sizes_num,angles_num))
    F43 = np.zeros((wvls_num,sizes_num,angles_num))
    F44 = np.zeros((wvls_num,sizes_num,angles_num))

    #Obtain file list from folder
    fpath = Path(folder)
    flist = sorted([f for f in fpath.iterdir() if f.is_file()])

    #Assuming that filenames show first WVL then SIZE:
    #Create 2dim list of file names
    farray = [[0 for _ in range(sizes_num)] for _ in range(wvls_num)]
    k = 0
    for i in range(wvls_num):
        for j in range(sizes_num):
            farray[i][j] = flist[k]
            k = k + 1

    #Start with data obtention and writing of output files
    for i in range(wvls_num):
        for j in range(sizes_num):
            StartLine = FindLinePos(farray[i][j],
            "   scat.angle     F11          F21/F11        F33/F11        F34/F11")

            #Read data from TMat file
            Mie = pd.read_csv(farray[i][j], sep='\\s+', comment="#", header=None, skiprows=StartLine)

            angles = Mie[0]
            F11[i,j,:] = Mie[1]
            F12[i,j,:] = Mie[2]
            F22[i,j,:] = Mie[1]/F11[i,j,:]
            F33[i,j,:] = Mie[3]
            F43[i,j,:] = -1.0*Mie[4]
            F44[i,j,:] = Mie[3]
            #SPHERES == MIE ==> F11=F22 and F33=F44

    #Elements
    matrix = [F11,F12,F22,F33,F43,F44]
    names = ["F11","F12","F22","F33","F43","F44"]

    #Writing section
    for i,ele in enumerate(matrix):
        tag = names[i]
        output_path = f"outputF{tag}WriteGCM_Mie.dat"
        header = "# ----------------------------------------------------------------- \n" + \
        f"# H2O ice scattering matrix element: {tag}. \n" + \
        "# Log-normal dist., 30 reffs, nueff=0.05, spheres \n" + \
        "# Computed by the Meerhoff code; \n" + \
        "# Based on the optical indices of Warren 2008; \n" + \
        "# -----------------------------------------------------------------"

        #Write header, nº of wvls and nº of radii in file
        with open(output_path, "w") as f:
            f.write(f"{header}\n")
            f.write(f"# Number of wavelengths (nwvl):\n  {wvls_num}\n")
            f.write(f"# Number of radius (nsize):\n  {sizes_num}\n")
            f.write(f"# Number of angles (nang):\n  {len(angles)}\n")

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

        #Write angles axis
        with open(output_path, "a") as f:
            f.write("# Scattering angle axis (ang):\n")
            k = 0
            groups = angles_num//5 #Number of groups of 5 lines
            lastline = angles_num - groups*5 #Number of elements in last line either 1,2,3 or 4
            for i in range(groups):
                f.write(f" {angles[0+k]:.6E} {angles[1+k]:.6E} {angles[2+k]:.6E} {angles[3+k]:.6E} {angles[4+k]:.6E}\n")
                k = k + 5
            if lastline != 0:
                for i in range(lastline):
                    f.write(f" {angles[k+i]:.6E}")
                f.write("\n") 

        # Write scattering matrix element
        with open(output_path,"a") as f:
            f.write(f"# Scattering matrix element ({tag}):\n")
            groups = wvls_num//5
            lastline = wvls_num - groups*5
            for i in range(angles_num):
                f.write(f"# Angle number     {i+1}\n")
                for j in range(sizes_num):
                    p1 = 0
                    f.write(f"# Radius number     {j+1}\n")                    
                    # Actual writing process
                    for k in range(groups):
                        f.write(f" {ele[0+p1,j,i]:.6E} {ele[1+p1,j,i]:.6E} {ele[2+p1,j,i]:.6E} {ele[3+p1,j,i]:.6E} {ele[4+p1,j,i]:.6E}\n")
                        p1 = p1 + 5
                    if lastline != 0:
                        for k in range(lastline):
                            f.write(f" {ele[k+p1,j,i]:.6E}")
                        f.write("\n")
    #Plotting if wanted


    if plot == True:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        index = [0,1,3,4]
        sel_wvl = 500e-09
        sel_rad = 1.5e-06
        iw = np.argmin(np.abs(wvls - sel_wvl))
        ir = np.argmin(np.abs(sizes - sel_rad))

        for i, pointer in enumerate(index):
            axes[i].plot(angles, matrix[pointer][iw,ir,:])
            axes[i].set_title(f"{names[pointer]}/{names[0]}\nλ={wvls[iw]:.2e}, r={sizes[ir]:.2e}")
            axes[i].set_xlabel("Ángulo (deg)")
            axes[i].set_ylabel(f"{names[pointer]}/{names[0]}")
            axes[i].grid(True)

            if i == 0: axes[i].set_yscale("log")

        plt.tight_layout()
        plt.show()

    return

def FXXWriteGCM(sizes: np.array, wvls: np.array, angles: np.array, header: str, files: list, orifile_path: str):
    #Some necessary actions and definitions
    wvls_num = len(wvls)
    sizes_num = len(sizes)
    angles_num = len(angles)
    files_num = len(files)
    sizes = sizes * 1e-6 #Change from um to m (LMD-GCM reads meters)
    wvls = wvls * 1e-6
    output_path = "outputFXXWriteGCM.dat"

    #Create and index array that signals the position of desired angles
    # over the original array
    orifile = pd.read_csv(orifile_path, sep='\\s+', comment="#", header=None)
    ori_angles = orifile.iloc[0,:]
    ori_angles_num = len(ori_angles)

    idx_array = np.zeros((angles_num))
    for i in range(angles_num):
        for j in range(ori_angles_num):
            if (angles[i] == ori_angles[j]):
                idx_array[i] = j

    idx_array = idx_array.astype(int)

    #Foolproofing
    if files_num != sizes_num:
        return print(f"Error: number of files given ({files_num}) is not equal to number of sizes ({sizes_num})")
    
    #Write header, nº of wvls and nº of radii in file
    with open(output_path, "w") as f:
        f.write(f"{header}\n")
        f.write(f"# Number of wavelengths (nwvl):\n  {wvls_num}\n")
        f.write(f"# Number of radius (nsize):\n  {sizes_num}\n")
        f.write(f"# Number of angles (nang):\n  {angles_num}\n")

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

    #Write angles axis
    with open(output_path, "a") as f:
        f.write("# Scattering angle axis (ang):\n")
        k = 0
        groups = angles_num//5 #Number of groups of 5 lines
        lastline = angles_num - groups*5 #Number of elements in last line either 1,2,3 or 4
        for i in range(groups):
            f.write(f" {angles[0+k]:.6E} {angles[1+k]:.6E} {angles[2+k]:.6E} {angles[3+k]:.6E} {angles[4+k]:.6E}\n")
            k = k + 5
        if lastline != 0:
            for i in range(lastline):
                f.write(f" {angles[k+i]:.6E}")
            f.write("\n")
    
    # Write scattering matrix element
    with open(output_path,"a") as f:
        f.write("# Scattering matrix element (FXX):\n")
        groups = wvls_num//5
        lastline = wvls_num - groups*5
        for i in range(angles_num):
            f.write(f"# Angle number     {i+1}\n")
            for j in range(sizes_num):
                p1 = 0
                f.write(f"# Radius number     {j+1}\n")

                #Store FXX in 2dim array: [wvls,angles]
                FXX = np.zeros((wvls_num, ori_angles_num))
                FXX_data = pd.read_csv(files[j], sep='\\s+', comment="#", header=None)
                FXX_temp = np.array(FXX_data.iloc[:,2])
                p2 = 0
                for k in range(wvls_num):
                    for l in range(ori_angles_num):
                        FXX[k,l] = FXX_temp[p2]
                        p2 = p2 + 1
                
                # Actual writing process
                a = idx_array[i]
                for k in range(groups):
                    f.write(f" {FXX[0+p1,a]:.6E} {FXX[1+p1,a]:.6E} {FXX[2+p1,a]:.6E} {FXX[3+p1,a]:.6E} {FXX[4+p1,a]:.6E}\n")
                    p1 = p1 + 5
                if lastline != 0:
                    for k in range(lastline):
                        f.write(f" {FXX[k+p1,a]:.6E}")
                    f.write("\n")

    return