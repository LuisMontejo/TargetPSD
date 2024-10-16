'''
DD_TargetPSD_Module
luis.montejo@upr.edu
Generation of Duration Dependent Target PSD functions.

===============================================================================
References:
    
Montejo, L.A. 2024. "Strong-Motion-Duration-Dependent Power Spectral Density 
Functions Compatible with Design Response Spectra" Geotechnics 4, no. 4: 1048-1064. 
https://doi.org/10.3390/geotechnics4040053

Montejo L.A.; Vidot-Vega, A.L.  2017. “An Empirical Relationship between Fourier 
and Response Spectra Using Spectrum-Compatible Times Series” Earthquake Spectra; 
33 (1): 179–199. doi: https://doi.org/10.1193/060316eqs089m

Chi-Miranda M.; Montejo, L.A. 2018. “FAS-Compatible Synthetic Signals for 
Equivalent-Linear Site Response Analyses” Earthquake Spectra; 34 (1): 377–396. 
doi: https://doi.org/10.1193/102116EQS177M

===============================================================================

This module contains the Python functions required to generate strong motion 
duration dependent target PSD function compatible with a design/target response 
spectrum. The functions included can also be used to compute SD5-75, 
record PSD (as specified in NRC SRP 3.7.1) and record response spectra (using 
frequency domain operations).


*DDTargetPSD:Generates a strong motion duration dependent target PSD compatible 
with a target response spectrum    

*DDTargetPSD_MP: Same as DDTargetPSD, but a more efficient version using multipl
processes (concurrent.features)
 
*PSDFFTEq: Calculates the power spectral density of earthquake acceleration 
time-series, FFT is normalized by dt, FFT/PSD is calculated over the strong 
motion duration returns the one-sided PSD and a "smoothed" version by taking 
the average over a frequency window width of user defined % of the subject 
frequency.

*DDTargetPSD:   Generates a strong motion duration dependent target PSD compatible 
                with a target response spectrum

* FASPSAratio: Target FAS based on empirical relationship between Fourier and 
response spectra (Montejo & Vidot-Vega, 2017)

*SignificantDuration: Estimates significant duration and Arias Intensity

*RSFD: Response spectra (operations in the frequency domain)

* log_interp: Performs logarithmic interpolation

* saragoni_hart_w: returns a Saragoni-Hart type of window

'''


def FASPSAcomps(nt,envelope,m,dt,TaFAS,T):
    import numpy as np
    so = np.random.randn(nt) # generates the synthetic signal
    so = envelope*so
    FSo = np.fft.fft(so)  # initial Fourier coefficients
    FASo = np.abs(FSo[m]) * dt  # initial FAS
    ff = TaFAS / FASo # modification 
    fsymm = np.concatenate((ff, ff[-2:0:-1]))
    FS = fsymm * FSo # modified Fourier coefficients
    sc = np.fft.ifft(FS).real # FAS compatible signal
    PSArecord, _, _, _, _= RSFD(T, sc, 0.05, dt) # calculates PSA 
    return PSArecord

def PSDFAScomps(nt,envelope,m,dt,TFASfin,fs):
    import numpy as np
    so = np.random.randn(nt) # generates the synthetic signal
    so = envelope*so
    FSo = np.fft.fft(so)  # initial Fourier coefficients
    FASo = np.abs(FSo[m]) * dt  # initial FAS
    ff = TFASfin / FASo # modification factors
    fsymm = np.concatenate((ff, ff[-2:0:-1]))
    FS = fsymm * FSo # modified Fourier coefficients
    s = np.fft.ifft(FS).real # FAS compatible signal
    _,PSDfinrecord,PSDavgfinrecord,_,sdfinrecord,_,_,_= PSDFFTEq(s,fs,alphaw=0.1,duration=(5,75),nFFT='same',basefornFFT = 0, overlap=20, detrend='linear')
    return PSDfinrecord,PSDavgfinrecord,sdfinrecord

def DDTargetPSD_MP(filename,sd575,TargetSpectrumName,F1=0.2,F2=50,
                allow_err=2.5,neqsPSD=1000,plots=1):
    '''
    Generates a strong motion duration dependent target PSD compatible with a
    target response spectrum 

    Parameters
    ----------
    filename :  string, name of the file that contains the desigd/target response
                spectrum
                two columsn: frequency [Hz] - PSA[g]
                define the spectrum as dense as possible in the range [0.01-100]Hz
                for large sd575 lower frequencies may be required
                
    sd575 :     float, target sd5-75 [s]
    
    TargetSpectrumName : string, name used to generate the output files
    
    F1 : float, lowest frequency to check PSA match [Hz]. The default is 0.2.
    
    F2 : float, highest frequency to check PSA match [Hz]. The default is 50.
    
    allow_err : float, aloowable error in PSA match [%]. The default is 2.5.
    
    neqsPSD :   int, number of synthetic motions to generate PSD from FAS.
                The default is 1000.
                
    plots: int, 1 generates plots. The default is 1.

    Returns
    -------
    freqs: np.array with the frequencies where FAS and PSD are reported 
    
    PSDrecordsavg_avgfin: np.array with the target PSD [m2/s3]
    
    TFASfin: np.array with the target FAS [m/s]

    '''
    import numpy as np
    import concurrent.futures
       
    g = 9.81
        
    neqsPSD = int(neqsPSD) # number of motions to generate target PSD
    
    dspec = np.loadtxt(filename) # load target spectrum file
    # two columsn: frequency [Hz] - PSA[g]
    # define the spectrum as dense as possible in the range [0.01-100]Hz
    # for large sd575 lower frequencies may be required
    
    f_or = dspec[:,0] # frequencies
    ds_or = dspec[:,1]*g # amplitudes
    
    # Create time envelope
    
    tf = 3.54*sd575  # total duration of the signal
    fs = 200; dt = 1/fs
    
    nt = int(tf/dt)+1
    if nt%2!=0: tf+=dt; nt+=1 # Adjust time vector for even length
    
    envelope = saragoni_hart_w(nt,eps=0.2,n=0.2,tn=0.6)
    
    sets = np.linspace(10,100,10, dtype=int)
    nsets = np.size(sets)
    
    m = np.arange(0, np.ceil(nt/2)+1, dtype=int)
    freqs = m * fs / nt  # Fourier frequencies
    
    # frequencies to estimate PSA and resample target spectrum
    f = np.hstack((np.array([0.01,0.02,0.04,0.06,0.08]),np.geomspace(0.1,50,100),np.array([55,60,70,80,90,100])))
    if f[0]>freqs[1]:
        f[0] = 0.99*freqs[1]
    
    # check frequency range where the target spectrum was defined:
    
    if f_or[0] > freqs[1]:
        print(f'the target response spectrum is currently defined from {f_or[0]:.4f} Hz but needs to be defined at least from {0.98*freqs[1]:.4f} Hz')
        return
    if f_or[-1] < freqs[-1]:
        print(f'the target response spectrum is currently defined until {f_or[-1]:.4f} Hz but needs to be defined at least until {freqs[-1]:.4f} Hz')
        return
    
    ds =  log_interp(f,f_or,ds_or)
    locs = np.where((f>=F1)&(f<=F2))[0]   # positions within frequency range for match check
    
    #initial FAS
    
    ratio = FASPSAratio(f,sd575)  # FAS/PSA ratio Montejo and Vidot 2017 
    
    TFAS = ds*ratio  # target FAS
    
    
    # Interpolate PSA and TFAS at required frequencies
    
    ds_freqs = log_interp(freqs[1:], f, ds)
    ds_freqs = np.concatenate(([0], ds_freqs))
    
    TFAS_freqs = log_interp(freqs[1:], f, TFAS)
    TFAS_freqs = np.concatenate(([0], TFAS_freqs))
    
    T = 1/f
    
    PSAavg = np.zeros((len(T),nsets))           # stores the average PSA per set
    FAStarget = np.zeros((len(freqs),nsets))    # stores the target FAS (initial and after each iteration)
    FAStarget[:,0] = TFAS_freqs
    calc_errs = np.zeros(nsets)
    
    print('*'*20)
    print('Now generating spectrum compatible FAS')
    print(f'Target error: {allow_err:.2f}%, max # of iters.: {nsets}')
    print('*'*20)
    for k in range(nsets):
        
        PSA  = np.zeros((len(T),sets[k]))        # stoeres individual record PSA, used to get the average
                                                  # after the loops are completed, reset each k loop
        
        TaFAS = FAStarget[:,k]
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            PSAallrecords = [executor.submit(FASPSAcomps,*[nt,envelope,m,dt,TaFAS,T]) for _ in range(sets[k])]
        
        q=0
        for PSAsinglerecord in PSAallrecords:
            PSA[:,q]=PSAsinglerecord.result()
            q=q+1
    
        PSAavg[:,k] = np.mean(PSA,axis=1) # takes average PSA per set
        diflimits = np.abs(ds[locs]-PSAavg[locs,k])/ds[locs]
    
        calc_errs[k] = np.mean(diflimits)*100
         
        print(f'iteration #: {k+1} - set with {sets[k]} records - error: {calc_errs[k]:.2f}%')
        
        if calc_errs[k]<allow_err:
            PSAavg = PSAavg[:,:k+1]
            FAStarget = FAStarget[:,:k+1]
            calc_errs = calc_errs[:k+1]
            print(f'error satisfied at iteration # {k+1} - error: {calc_errs[k]:.2f}%')
            break
        elif k!=nsets-1:
            PSAavg_interp = log_interp(freqs[1:], f, PSAavg[:,k]) # iterpolates to fourier frequencies 
                                                                  # to allow ratios calculation
            factor = ds_freqs[1:]/PSAavg_interp                   # take ratios between target and 
                                                                  # response spectra
            FAStarget[1:,k+1]=factor*FAStarget[1:,k]              # apply ratios to get updated target PSD
    
    else:
        
        print('max number of iterations was reached, error was not satisfied')
        print('the results from the iteration with the lowest error would be used')
    
    nsets = np.size(calc_errs)
    aux = np.arange(1,nsets+1)
    
    # find target PSD:
    
    print('*'*20)
    print(f'Now generating spectrum compatible PSD using {neqsPSD} records')
    
    minerrloc = np.argmin(calc_errs)
    TFASfin = FAStarget[:,minerrloc]
    
    PSDfin = np.zeros((len(freqs),neqsPSD))
    PSDavgfin = np.zeros((len(freqs),neqsPSD))
    
    sdfin = np.zeros(neqsPSD)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        allrecords = [executor.submit(PSDFAScomps,*[nt,envelope,m,dt,TFASfin,fs]) for _ in range(neqsPSD)]
    
    q=0
    for singlerecord in allrecords:
        PSDfin[:,q]=singlerecord.result()[0]
        PSDavgfin[:,q]=singlerecord.result()[1]
        sdfin[q]=singlerecord.result()[2]
        q=q+1
        
    PSDrecords_avgfin = np.mean(PSDfin,axis=1) # average per set of the "uaveraged" records
    PSDrecordsavg_avgfin = np.mean(PSDavgfin,axis=1) # average per set of the overlaped average PSD
    sdfinmean = np.mean(sdfin)
    print(f'Target SD5-75 was {sd575:.2f}s, actual records SD5-75:{sdfinmean:.2f}s')
    
    
    outputfile = np.vstack((freqs,PSDrecordsavg_avgfin,TFASfin)).T
    
    name = f'{TargetSpectrumName}_SD575_{sd575:.2f}s.txt'
    header = 'freq[s]  -  target PSD [m2/s3] - target FAS [m/s]' 
    np.savetxt(name,outputfile,header=header)
    
    if plots==1:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['font.size'] = 9 
        mpl.rcParams['legend.frameon'] = False
        
        plt.figure(figsize=(6.5,8))
        
        plt.suptitle(f'design spectrum: {TargetSpectrumName} \n target SD5-75: {sd575:.1f}s - average SD5-75: {sdfinmean:.1f}s')
        
        plt.subplot(221)
        plt.plot(aux,calc_errs,'--o',mfc='white')
        plt.hlines(allow_err,aux[0],aux[-1],colors='red')
        plt.text(1,allow_err+0.1,f'target: {allow_err:.2f}%')
        plt.xticks(aux)
        plt.xlabel('iteration #'); plt.ylabel('PSA error [%]')
        
        plt.subplot(222)
        plt.semilogx(f,ds/g,color='silver',lw=3,label='Target')
        plt.semilogx(f,PSAavg[:,0]/g,color='darkred',label='First it.')
        plt.semilogx(f,PSAavg[:,-1]/g,color='black',label='Last it.')
        plt.xlim(0.1,100)
        plt.xlabel('F [Hz]'); plt.ylabel('PSA [g]')
        plt.legend()
        
        plt.subplot(223)
        plt.loglog(freqs,FAStarget[:,0],color='darkred',label='Initial')
        plt.loglog(freqs,TFASfin,color='black',label='Last it.')
        plt.legend()
        plt.xlim(0.1,100)
        plt.xlabel('F [Hz]'); plt.ylabel('FAS [m/s]')
        
        plt.subplot(224)
        plt.loglog(freqs,PSDrecords_avgfin,color='darkgray',label='Unaveraged')
        plt.loglog(freqs,PSDrecordsavg_avgfin,color='black',label='Window avgd.')
        plt.legend()
        plt.xlim(0.1,100)
        plt.xlabel('F [Hz]'); plt.ylabel('PSD [m2/s3]')
        
        plt.tight_layout()
        
        plt.savefig(f'{TargetSpectrumName}_SD575_{sd575:.2f}s.jpg',dpi=300)
    
    return freqs,PSDrecordsavg_avgfin,TFASfin

def DDTargetPSD(filename,sd575,TargetSpectrumName,F1=0.2,F2=50,
                allow_err=2.5,neqsPSD=1000,plots=1):
    '''
    Generates a strong motion duration dependent target PSD compatible with a
    target response spectrum

    Parameters
    ----------
    filename :  string, name of the file that contains the desigd/target response
                spectrum
                two columsn: frequency [Hz] - PSA[g]
                define the spectrum as dense as possible in the range [0.01-100]Hz
                for large sd575 lower frequencies may be required
                
    sd575 :     float, target sd5-75 [s]
    
    TargetSpectrumName : string, name used to generate the output files
    
    F1 : float, lowest frequency to check PSA match [Hz]. The default is 0.2.
    
    F2 : float, highest frequency to check PSA match [Hz]. The default is 50.
    
    allow_err : float, aloowable error in PSA match [%]. The default is 2.5.
    
    neqsPSD :   int, number of synthetic motions to generate PSD from FAS.
                The default is 1000.
                
    plots: int, 1 generates plots. The default is 1.

    Returns
    -------
    freqs: np.array with the frequencies where FAS and PSD are reported 
    
    PSDrecordsavg_avgfin: np.array with the target PSD [m2/s3]
    
    TFASfin: np.array with the target FAS [m/s]

    '''
    import numpy as np
       
    g = 9.81
        
    neqsPSD = int(neqsPSD) # number of motions to generate target PSD
    
    dspec = np.loadtxt(filename) # load target spectrum file
    # two columsn: frequency [Hz] - PSA[g]
    # define the spectrum as dense as possible in the range [0.01-100]Hz
    # for large sd575 lower frequencies may be required
    
    f_or = dspec[:,0] # frequencies
    ds_or = dspec[:,1]*g # amplitudes
    
    # Create time envelope
    
    tf = 3.54*sd575  # total duration of the signal
    fs = 200; dt = 1/fs
    
    nt = int(tf/dt)+1
    if nt%2!=0: tf+=dt; nt+=1 # Adjust time vector for even length
    
    envelope = saragoni_hart_w(nt,eps=0.2,n=0.2,tn=0.6)
    
    sets = np.linspace(10,100,10, dtype=int)
    nsets = np.size(sets)
    
    m = np.arange(0, np.ceil(nt/2)+1, dtype=int)
    freqs = m * fs / nt  # Fourier frequencies
    
    # frequencies to estimate PSA and resample target spectrum
    f = np.hstack((np.array([0.01,0.02,0.04,0.06,0.08]),np.geomspace(0.1,50,100),np.array([55,60,70,80,90,100])))
    if f[0]>freqs[1]:
        f[0] = 0.99*freqs[1]
    
    # check frequency range where the target spectrum was defined:
    
    if f_or[0] > freqs[1]:
        print(f'the target response spectrum is currently defined from {f_or[0]:.4f} Hz but needs to be defined at least from {0.98*freqs[1]:.4f} Hz')
        return
    if f_or[-1] < freqs[-1]:
        print(f'the target response spectrum is currently defined until {f_or[-1]:.4f} Hz but needs to be defined at least until {freqs[-1]:.4f} Hz')
        return
    
    ds =  log_interp(f,f_or,ds_or)
    locs = np.where((f>=F1)&(f<=F2))[0]   # positions within frequency range for match check
    
    #initial FAS
    
    ratio = FASPSAratio(f,sd575)  # FAS/PSA ratio Montejo and Vidot 2017 
    
    TFAS = ds*ratio  # target FAS
    
    
    # Interpolate PSA and TFAS at required frequencies
    
    ds_freqs = log_interp(freqs[1:], f, ds)
    ds_freqs = np.concatenate(([0], ds_freqs))
    
    TFAS_freqs = log_interp(freqs[1:], f, TFAS)
    TFAS_freqs = np.concatenate(([0], TFAS_freqs))
    
    T = 1/f
    
    PSAavg = np.zeros((len(T),nsets))           # stores the average PSA per set
    FAStarget = np.zeros((len(freqs),nsets))    # stores the target FAS (initial and after each iteration)
    FAStarget[:,0] = TFAS_freqs
    calc_errs = np.zeros(nsets)
    
    print('*'*20)
    print('Now generating spectrum compatible FAS')
    print(f'Target error: {allow_err:.2f}%, max # of iters.: {nsets}')
    print('*'*20)
    for k in range(nsets):
        
        PSA  = np.zeros((len(T),sets[k]))        # stoeres individual record PSA, used to get the average
                                                  # after the loops are completed, reset each k loop
        
        for q in range(sets[k]):
            so = np.random.randn(nt) # generates the synthetic signal
            so = envelope*so
            FSo = np.fft.fft(so)  # initial Fourier coefficients
            FASo = np.abs(FSo[m]) * dt  # initial FAS
            ff = FAStarget[:,k] / FASo # modification 
            fsymm = np.concatenate((ff, ff[-2:0:-1]))
            FS = fsymm * FSo # modified Fourier coefficients
            sc = np.fft.ifft(FS).real # FAS compatible signal
            PSA[:,q], _, _, _, _= RSFD(T, sc, 0.05, dt) # calculates PSA 
    
        PSAavg[:,k] = np.mean(PSA,axis=1) # takes average PSA per set
        diflimits = np.abs(ds[locs]-PSAavg[locs,k])/ds[locs]
    
        calc_errs[k] = np.mean(diflimits)*100
         
        print(f'iteration #: {k+1} - set with {sets[k]} records - error: {calc_errs[k]:.2f}%')
        
        if calc_errs[k]<allow_err:
            PSAavg = PSAavg[:,:k+1]
            FAStarget = FAStarget[:,:k+1]
            calc_errs = calc_errs[:k+1]
            print(f'error satisfied at iteration # {k+1} - error: {calc_errs[k]:.2f}%')
            break
        elif k!=nsets-1:
            PSAavg_interp = log_interp(freqs[1:], f, PSAavg[:,k]) # iterpolates to fourier frequencies 
                                                                  # to allow ratios calculation
            factor = ds_freqs[1:]/PSAavg_interp                   # take ratios between target and 
                                                                  # response spectra
            FAStarget[1:,k+1]=factor*FAStarget[1:,k]              # apply ratios to get updated target PSD
    
    else:
        
        print('max number of iterations was reached, error was not satisfied')
        print('the results from the iteration with the lowest error would be used')
    
    nsets = np.size(calc_errs)
    aux = np.arange(1,nsets+1)
    
    # find target PSD:
    
    print('*'*20)
    print(f'Now generating spectrum compatible PSD using {neqsPSD} records')
    
    minerrloc = np.argmin(calc_errs)
    TFASfin = FAStarget[:,minerrloc]
    
    PSDfin = np.zeros((len(freqs),neqsPSD))
    PSDavgfin = np.zeros((len(freqs),neqsPSD))
    
    sdfin = np.zeros(neqsPSD)
    
    for q in range(neqsPSD):
        so = np.random.randn(nt) # generates the synthetic signal
        so = envelope*so
        FSo = np.fft.fft(so)  # initial Fourier coefficients
        FASo = np.abs(FSo[m]) * dt  # initial FAS
        ff = TFASfin / FASo # modification factors
        fsymm = np.concatenate((ff, ff[-2:0:-1]))
        FS = fsymm * FSo # modified Fourier coefficients
        s = np.fft.ifft(FS).real # FAS compatible signal
        _,PSDfin[:,q],PSDavgfin[:,q],_,sdfin[q],_,_,_= PSDFFTEq(s,fs,alphaw=0.1,duration=(5,75),nFFT='same',basefornFFT = 0, overlap=20, detrend='linear')
    
    
    PSDrecords_avgfin = np.mean(PSDfin,axis=1) # average per set of the "uaveraged" records
    PSDrecordsavg_avgfin = np.mean(PSDavgfin,axis=1) # average per set of the overlaped average PSD
    sdfinmean = np.mean(sdfin)
    print(f'Target SD5-75 was {sd575:.2f}s, actual records SD5-75:{sdfinmean:.2f}s')
    
    
    outputfile = np.vstack((freqs,PSDrecordsavg_avgfin,TFASfin)).T
    
    name = f'{TargetSpectrumName}_SD575_{sd575:.2f}s.txt'
    header = 'freq[s]  -  target PSD [m2/s3] - target FAS [m/s]' 
    np.savetxt(name,outputfile,header=header)
    
    if plots==1:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['font.size'] = 9 
        mpl.rcParams['legend.frameon'] = False
        
        plt.figure(figsize=(6.5,8))
        
        plt.suptitle(f'design spectrum: {TargetSpectrumName} \n target SD5-75: {sd575:.1f}s - average SD5-75: {sdfinmean:.1f}s')
        
        plt.subplot(221)
        plt.plot(aux,calc_errs,'--o',mfc='white')
        plt.hlines(allow_err,aux[0],aux[-1],colors='red')
        plt.text(1,allow_err+0.1,f'target: {allow_err:.2f}%')
        plt.xticks(aux)
        plt.xlabel('iteration #'); plt.ylabel('PSA error [%]')
        
        plt.subplot(222)
        plt.semilogx(f,ds/g,color='silver',lw=3,label='Target')
        plt.semilogx(f,PSAavg[:,0]/g,color='darkred',label='First it.')
        plt.semilogx(f,PSAavg[:,-1]/g,color='black',label='Last it.')
        plt.xlim(0.1,100)
        plt.xlabel('F [Hz]'); plt.ylabel('PSA [g]')
        plt.legend()
        
        plt.subplot(223)
        plt.loglog(freqs,FAStarget[:,0],color='darkred',label='Initial')
        plt.loglog(freqs,TFASfin,color='black',label='Last it.')
        plt.legend()
        plt.xlim(0.1,100)
        plt.xlabel('F [Hz]'); plt.ylabel('FAS [m/s]')
        
        plt.subplot(224)
        plt.loglog(freqs,PSDrecords_avgfin,color='darkgray',label='Unaveraged')
        plt.loglog(freqs,PSDrecordsavg_avgfin,color='black',label='Window avgd.')
        plt.legend()
        plt.xlim(0.1,100)
        plt.xlabel('F [Hz]'); plt.ylabel('PSD [m2/s3]')
        
        plt.tight_layout()
        
        plt.savefig(f'{TargetSpectrumName}_SD575_{sd575:.2f}s.jpg',dpi=300)
    
    return freqs,PSDrecordsavg_avgfin,TFASfin

def RSFD(T,s,z,dt):
    '''   
    Response spectra (operations in the frequency domain)
    
    Input:
        T: vector with periods (s)
        s: acceleration time series
        z: damping ratio
        dt: time steps for s
    
    Returns:
        PSA, PSV, SA, SV, SD
    
    '''
    import numpy as np
    from numpy.fft import fft, ifft
    
    pi = np.pi

    npo = np.size(s)
    nT  = np.size(T)
    SD  = np.zeros(nT)
    SV  = np.zeros(nT)
    SA  = np.zeros(nT)
    
    n = int(2**np.ceil(np.log2(npo+10*np.max(T)/dt)))  # add zeros to provide enough quiet time
    fs=1/dt;
    s = np.append(s,np.zeros(n-npo))
    
    fres  = fs/n                            # frequency resolution
    nfrs  = int(np.ceil(n/2))               # number of frequencies
    freqs = fres*np.arange(0,nfrs+1,1)      # vector with frequencies
    ww    = 2*pi*freqs                      # vector with frequencies [rad/s]
    ffts = fft(s);         
    
    m = 1
    for kk in range(nT):
        w = 2*pi/T[kk] ; k=m*w**2; c = 2*z*m*w
        
        H1 = 1       / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Receptance
        H2 = 1j*ww   / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Mobility
        H3 = -ww**2  / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Accelerance
        
        H1 = np.append(H1,np.conj(H1[n//2-1:0:-1]))
        H1[n//2] = np.real(H1[n//2])     # Transfer function (complete) - Receptance
        
        H2 = np.append(H2,np.conj(H2[n//2-1:0:-1]))
        H2[n//2] = np.real(H2[n//2])     # Transfer function (complete) - Mobility
        
        H3 = np.append(H3,np.conj(H3[n//2-1:0:-1]))
        H3[n//2] = np.real(H3[n//2])     # Transfer function (complete) - Accelerance
        
        CoF1 = H1*ffts   # frequency domain convolution
        d = ifft(CoF1)   # go back to the time domain (displacement)
        SD[kk] = np.max(np.abs(d))
            
        CoF2 = H2*ffts   # frequency domain convolution
        v = ifft(CoF2)   # go back to the time domain (velocity)
        SV[kk] = np.max(np.abs(v))
        
        CoF3 = H3*ffts   # frequency domain convolution
        a = ifft(CoF3)   # go back to the time domain (acceleration)
        a = a - s
        SA[kk] = np.max(np.abs(a))
    
    PSV = (2*pi/T)* SD
    PSA = (2*pi/T)**2 * SD
    
    return PSA, PSV, SA, SV, SD


def SignificantDuration(s,t,ival=5,fval=75):
    '''
    Estimates significant duration and Arias Intensity
    
    Parameters
    ----------
    s : 1d array
        acceleration time-history
    t : 1d array
        time vector
    ival : float, optional
        Initial % of Arias Intensity to estimate significant duration. 
        The default is 5.
    fval :float, optional
        Final % of Arias Intensity to estimate significant duration. 
        The default is 75.

    Returns
    -------
    sd : float
        significant duration
    AIcumnorm : 1d array
        normalized cummulative AI
    AI : float
        Arias Intensity (just the integral, 2*pi/g not included)
    t1 : float
        initial time for sd
    t2 : float
        final time for sd

    '''
    from scipy import integrate
    AIcum = integrate.cumtrapz(s**2, t, initial=0)
    AI = AIcum[-1]
    AIcumnorm = AIcum/AI
    t_strong = t[(AIcumnorm>=ival/100)&(AIcumnorm<=fval/100)]
    t1, t2 = t_strong[0], t_strong[-1]
    sd = t2-t1
    return sd,AIcumnorm,AI,t1,t2


def PSDFFTEq(so,fs,alphaw=0.1,duration=(5,75),nFFT='nextpow2',basefornFFT = 0, overlap=20, detrend='linear'):
    '''
    Calculates the power spectral density of earthquake acceleration time-series, FFT 
    is normalized by dt, FFT/PSD is calculated over the strong motion duration returns 
    the one-sided PSD and a "smoothed" version by taking the average over a frequency 
    window width of user defined % of the subject frequency.
    
    Parameters
    ----------
    so : 1D array
        acceleration time-series
        
    fs : integer
        sampling frequency
        
    alphaw : Optional, float, tukey window parameter [0 1], defaults to 0.1
             0 -> rectangular, 1 -> Hann
    
    duration: Optional, tuple or None
    
              (a,b) strong motion duration used to defined the portion of the signal 
              used to calculate FFT and PSD.Defined as the duration corresponding 
              to a a%-to-b% rise of the cumulative Arias energy
              
              None: the whole signal is used
              
              The default is (5,75).
        
    nFFT : Optional, number of points to claculate the FFT, options:
        
        
        'nextpow2': zero padding until the mext power of 2 is reached
        
        'same': keep the number of points equal to the number of poitns in
                the signal
                
        An integer:  
        If n is smaller than the length of the input, the input is cropped. 
        If it is larger, the input is padded with zeros. 
    
        Defaults to 'nextpow2'
        
    basefornFFT: Optional, interger 0 or 1, whether nFFT is determined based on
                 the original/total number of datapoints in the signal or based
                 on the strong motion part.
                 
                 0 -> total number, 1 -> strong motion part
                 
                 defaults to 0
        
    overlap : Optional, float
        
        ±% frequency window width to smooth PSD 
        The default is 20.
        
    detrend = None, 'linear' or 'constant' (defaults to linear)
        'linear' (default), the result of a linear least-squares fit to data is subtracted from data. 
        'constant', only the mean of data is subtracted

    Returns
    -------
    mags : One-sided Fourier amplitudes
    PSD  : One-sided power spectral density
    PSDavg : One-sided average power spectral density
    freqs :  Vector with the frequencies
    sd : duration used to calculated FFT/SD
    AI : Arias intensity of the signal (Just the integral, units depend on the
                                        initial signal units, pi/2g is not applied)
    '''
    import numpy as np
    from scipy import signal
    
    no = np.size(so)
    dt = 1/fs
    t = np.linspace(0,(no-1)*dt,no)   # time vector 
      
    if duration==None:
        duration = (0,100)
        
    if len(duration)==2 :
        sd,AIcum,AI,t1,t2 = SignificantDuration(so,t,ival=duration[0],fval=duration[1])
        locs = np.where((t>=t1-dt/2)&(t<=t2+dt/2))
        nlocs = np.size(locs)
        s = so[locs]
        window = signal.windows.tukey(nlocs,alphaw)
        if detrend=='linear':
            s = signal.detrend(s,type='linear')
        elif detrend=='constant':
            s = signal.detrend(s,type='constant')
        elif detrend!=None:
            print('*** error definig detrend in PSDFFTEq function ***')
            return
        s = window*s
        
    else:
        print('*** error definig duration in PSDFFTEq function ***')
        return
    
    if basefornFFT == 0:
        n = no
    else:
        n = nlocs
        
    if nFFT=='nextpow2':
        nFFT = int(2**np.ceil(np.log2(n)))
    elif nFFT=='same':
        nFFT = n
    elif not isinstance(nFFT, int):
        print('*** error definig nFFT in PSDFFTEq function ***')
        return
        
    fres = fs/nFFT; nfrs = int(np.ceil(nFFT/2))
    freqs = fres*np.arange(0,nfrs+1,1)   # vector with frequencies
    
    Fs = np.fft.fft(s,nFFT)
    mags = dt*np.abs(Fs[:nfrs+1])
    
    PSD = 2*mags**2/(2*np.pi*sd)
    
    PSDavg = np.copy(PSD)
    overl = overlap/100
    if overl>0:
        for k in range(1,nfrs-1):
            lim1 = (1-overl)*freqs[k]
            lim2 = (1+overl)*freqs[k]
            
            if freqs[0]>lim1:
                lim1 = freqs[0]
                lim2 = freqs[k]+(freqs[k]-freqs[0])
            if freqs[-1]<lim2:
                lim2 = freqs[-1]
                lim1 = freqs[k]-(freqs[-1]-freqs[k])
                
            locsf = np.where((freqs>=lim1)&(freqs<=lim2))
            PSDavg[k]=np.mean(PSD[locsf])

    
    return mags,PSD,PSDavg,freqs,sd,AI,t1,t2






def log_interp(x, xp, fp):
    import numpy as np
    logx = np.log10(x)
    logxp = np.log10(xp)
    logfp = np.log10(fp)
    return np.power(10.0, np.interp(logx, logxp, logfp))


def FASPSAratio(f,sd575):
    '''
    Target FAS based on empirical relationship between Fourier and 
    response spectra (Montejo & Vidot-Vega, 2017)
    '''

    aa75 = 0.0512
    ab75 = 0.4920
    ac75 = 0.1123
    ba75 = -0.5869
    bb75 = -0.2650
    bc75 = -0.4580
    ratio = (aa75*sd575**ab75+ac75)*f**(ba75*sd575**bb75+bc75)
    
    return ratio

def saragoni_hart_w(npoints,eps=0.25,n=0.4,tn=0.6):
    '''
    returns a Saragoni-Hart type of window 

    Parameters
    ----------
    npoints : integer
        DESCRIPTION.
    eps : float (0-1), optional
        relative distance/time at which the amplitude reach 1. The default is 0.25.
    n : float (0-1), optional
        relative amplitude at tn. The default is 0.4.
    tn : float (eps,1], optional
        relative distance/time at which the amplitude reach n. The default is 0.6.

    Returns
    -------
    w : Saragoni-Hart window (1D array)

    '''
    import numpy as np
    
    b = -(eps*np.log(n))/(1+eps*(np.log(eps)-1))

    c = b/eps

    a = (np.exp(1)/eps)**b
    
    t = np.linspace(0,1,npoints)

    w = a*(t/tn)**b*np.exp(-c*(t/tn))
    
    return w

