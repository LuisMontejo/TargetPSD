# TargetPSD

This module contains the Python functions required to generate strong motion duration dependent target PSD functions compatible with a design/target response spectrum. 

cite the article: Montejo, L.A. Strong-Motion-Duration-Dependent Power Spectral Density Functions Compatible with Design Response Spectra. Geotechnics 2024, 4, 1048-1064. https://doi.org/10.3390/geotechnics4040053

cite the code: [![DOI](https://zenodo.org/badge/873791474.svg)](https://doi.org/10.5281/zenodo.13942090)

# Other references:

Montejo L.A.; Vidot-Vega, A.L.  2017. “An Empirical Relationship between Fourier and Response Spectra Using Spectrum-Compatible Times Series” Earthquake Spectra; 33 (1): 179–199. doi: https://doi.org/10.1193/060316eqs089m

Chi-Miranda M.; Montejo, L.A. 2018. “FAS-Compatible Synthetic Signals for Equivalent-Linear Site Response Analyses” Earthquake Spectra; 34 (1): 377–396. doi: https://doi.org/10.1193/102116EQS177M

# List of functions included:

The functions included can also be used to compute SD5-75, record PSD (as specified in NRC SRP 3.7.1) and record response spectra (using frequency domain operations).

The following is a list of the functions included in the module:

*DDTargetPSD: Generates a strong motion duration dependent target PSD compatible with target response spectrum

*DDTargetPSD_MP: Same as DDTargetPSD, but a more efficient version using multiple processes (concurrent.features)

*PSDFFTEq: Calculates the power spectral density of earthquake acceleration time-series, FFT is normalized by dt, FFT/PSD is calculated over the strong motion duration returns the one-sided PSD and a "smoothed" version by taking the average over a frequency window width of user defined % of the subject frequency.

*FASPSAratio: Target FAS based on empirical relationship between Fourier and response spectra (Montejo & Vidot-Vega, 2017)
  
*SignificantDuration: Estimates significant duration and Arias Intensity

*RSFD: Response spectra (operations in the frequency domain)

*log_interp: Performs logarithmic interpolation

*saragoni_hart_w: returns a Saragoni-Hart type of window
