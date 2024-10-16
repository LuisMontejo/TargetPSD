'''
Montejo, L.A. 2024. "Strong-Motion-Duration-Dependent Power Spectral Density 
Functions Compatible with Design Response Spectra" Geotechnics 4, no. 4: 1048-1064. 
https://doi.org/10.3390/geotechnics4040053

luis.montejo@upr.edu
'''

from DD_TargetPSD_Module import DDTargetPSD_MP, DDTargetPSD
 

if __name__ == '__main__':
    
    TargetSpectrumName = 'CEUS_M7.5_R150' # used for the output file name
    
    filename = 'CEUS_M7.5_R150_Frequencies.txt' 
                # target spectrum file
                # two columsn: frequency [Hz] - PSA[g]
                # define the spectrum as dense as possible in the range [0.01-100]Hz
                # for large sd575 lower frequencies may be required
    
    sd575 = 9 # target SD5-75 [s]
    
    freqs,TPSD,TFAS=DDTargetPSD_MP(filename,sd575,TargetSpectrumName,F1=0.2,F2=50,
                                allow_err=2.5,neqsPSD=1000,plots=1)


