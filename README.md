--------------------------------------------------------------------------------
These Python functions are meant to facilitate the implementation of scatterer
optical properties, calculated using TAMUdust2020, into the LMD-GCM. Some other
functions dedicated to work with the scattering matrix are also included; specifically
to compare the computations from TAMUdust2020 and T-Matrix code. Here is a
brief explanation of the main functions inputs, outputs and important notes.

################################# OPTICAL PROPERTIES UTILITIES #################################

#1. OptPropAve(OptProp_path: str, r_eff: float, v_eff: float, SD_choice: int, plot: bool = True)

    This function takes the "isca.dat" file outputed by TAMUdust2020 and returns a file with
    the resulting extinction efficiency factor (Qext), single scattering albedo (ssa) and assymetry
    parameter (g) averaged over a size distribution characterised by 2 parameters (r_eff and v_eff).
    
    OptProp_path -> Path to "isca.dat" file
    r_eff -> Effective radius of size distribution, in micrometers
    v_eff -> Effective variance of size distribution
    SD_choice -> Gamma distribution (=1) or Log-normal distribution (=2)
    plot -> Plots the results

    More size distributions can somewhat easily be added by defining them as a new function and including
    them into SD_choice as options number 3, 4, 5...

#2 OptPropAve_Disc(OptProp_path: str, SD_path: str, plot: bool = True)

    Alternatively, one can average "isca.dat" defining a size distribution by means of a set of discrete points.

    SD_path -> Path to the size distribution file, that should have exclusively two columns. First column is
                projected surface area equivalent sphere radius (r_mie, in um) and second column is n(r_mie)

#3 WriteGCMFormat(sizes: np.array, wvls: np.array, header: str, files: list):

    Takes the files outputed by OptPropAve/OptPropAve_Disc and merges them into a single file
    readable by the LMD-GCM

    sizes -> Array of sizes (r_eff, in micrometers) that were used to average the optical properties
    wvls -> Array of wavelengths (in micrometers) that were used in TAMUdust2020, and therefore are
            the same that appear in OptPropAve/OptPropAve_Disc output
    header -> Some comments that can be included at the beginning of the output file
    files -> List of file paths that are going to be merged

    As an important note, the sizes array should be ordered from smaller to larger sizes and should
    have a 1 to 1 correspondence to the files list. That is, is sizes[0] = 1.5um, list[0] = 1.5um-file

#4 OptPropExtract(OptProp_path: str, header: str):

    Alternatively, an "isca.dat" file can be rewritten into LMD-GCM format. Beware, this is not garanteed
    to be a usable input into LMD-GCM. Also, the LMD-GCM has a limit of 60 different sizes.

#5 InterpolateRI(RI_path: str, wvls_int: np.array, plot: bool = False)

    Takes a set of spectrally dependent refractive indeces and interpolates them.

    RI_path -> Path to the refractive indeces file, that should have exclusively, three columns.
                Col 1, wavelength (in micrometers)
                Col 2, real part of RI
                Col 3, imaginary part of RI
    wvls_int -> Array of wavelengths at which the RIs are to be interpolated
    plot -> Plots the results

#6 OptPropPlotTAMU(*OptProp_path: str, mode: str = "wvls", value: float):

    Plots an arbitrary number of "isca.dat" files.

    OptProp_path -> Path(s) to "isca.dat" file(s)
    mode -> If set to "wvls" the x-axis of the plot is wavelength in micrometers
            If set to "sizes" the x-axis of the plot is size parameter (2*pi*rmie/wvl)
    value -> If "mode" is set to "wvls", "value" is used to fix r_mie to some value in micrometers
            If "mode" is set to "sizes", "value" is used to fix wavelength to some value in micrometers

#7 OptPropPlotGCM(*OptProp_path: str, mode: str = "wvls", value: float = 5.)

    Plots an arbitrary number of LMD-GCM optical properties files, i.e. the ones included in their datadir
    or the ones outputed by WriteGCMFormat or OptPropExtract.
    
################################# SCATTERING MATRIX UTILITIES #################################

#1 FXXReadTamu(FXX_path: str, OptProp_path: str):
    
    This function takes the PXX.dat file and isca.dat file from the same TAMUdust2020 run in order
    to output a 3-dimensional (wvl,size,angle) FXX array and the wavelengths, r_mie and angles axis.

    FXX_path -> Path to "PXX.dat" file
    OptProp_path -> Path to "isca.dat" file

    Usage: FXX_array, wvls, rmies, angles = FXXReadTamu(FXX_path, OptProp_path)

    Note: the rmie axis is the projected surface-area-equivalent sphere radius that TAMUdust outputs
    but averaged over all wavelenghts

#2 FXXtoFile(FXX_path: str, OptProp_path: str, rmie_set: float):

    Takes a PXX.dat and isca.dat files in order to extract the data of the PXX.dat file for a single
    r_mie value, which needs to be specified. The output is a file whose format is designed to work with
    FXXPlot.

    rmie_set -> r_mie value in micrometers

#3 FXXRewrite(TMat_path: str, wvl: float ,element: str = "F11"):

    Takes the scattering matrix outputed by the T-Matrix code and converts it to a file readable
    by FXXPlot. The T-Matrix output format has to be changed slightly (an example is provided).

    TMat_path -> Path to the T-Matrix output

    wvl -> The wavelength value (in micrometers) that was inputed into the T-Matrix code
    element -> The scattering matrix element that wants to be extracted

#4 FXXAve(FXX_path: str, OptProp_path: str, r_eff: float, v_eff: float, SD_choice: int, plot: bool = True)

    Averages the PXX.dat given a specific size distribution defined by 2 parameters. Works exactly like
    OptPropAve. The output is readable by FXXPlot.

#5 FXXPlot(*FXXfile_path: str, wvl_set: float, ylabel: str = "", title: str = ""):

    Plots a scattering matrix element.

    FXXfile_path -> Path(s) to scattering matrix element file(s). File format are the outputs from
                    FXXAve, FXXtoFile and FXXRewrtie
    ylabel -> ylabel of the plot
    title -> Title of the plot. ALSO MUST BE EQUAL TO THE MATRIX ELEMENT THAT IS GOING TO BE PLOTTED.
                This is because F11 is normalized and the elements have different ylims in the plot.
                Options are: F11, F12, F22, F33, F43 and F44
    