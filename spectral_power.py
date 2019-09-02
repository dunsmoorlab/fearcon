import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def filter_ts(y, Fs=.5, Wn=10., order=3, filt_type='lowpass'):
    Wn_rad = Wn/Fs*2
    b,a = signal.butter(order, Wn_rad, filt_type)
    y_filt = signal.filtfilt(b,a,y)
    return y_filt

def gen_hrf(tr=2, n_trs=15, c=1./6, a1=6, a2=16):
    # a1, a2: timepoints of peaks
    # c1: ratio between peak and trough
    t = tr*np.arange(n_trs) + tr*.5
    h = (np.exp(-t)*(t**(a1-1)/np.math.factorial(a1-1)
         - c*t**(a2-1)/np.math.factorial(a2-1)))
    return h/np.sum(h)

def psd_analysis(y, Fs=.5, window='hanning', nperseg=256):
    f, Pxx = signal.welch(y, Fs, window, nperseg)
    return f, Pxx
    # plt.semilogy(f, Pxx)

sig = np.append(np.ones(2),np.zeros(5)) # in TRs
sig = np.tile(sig, 33) # 33 or 99 trials
hrf = gen_hrf()
t = np.arange(0,2*len(sig),2) # time to plot
sig_conv = np.convolve(sig,hrf)[0:len(t)]

alt_sig = np.append(np.ones(2),np.zeros(5))
alt_sig = np.append(alt_sig,0.5*alt_sig)
alt_sig = np.tile(alt_sig, 33)[0:len(sig)]
alt_sig_conv = np.convolve(alt_sig,hrf)[0:len(t)]

# plot time series
fig, ax = plt.subplots()
ax.plot(t, sig_conv)
ax.plot(t, alt_sig_conv)
plt.legend(['no trial variability','alternating trial variability'])
plt.ylabel('convolved BOLD')
plt.xlabel('timepoints')

# plot power spectrum
f,wxx = psd_analysis(sig_conv)
f,wxx_alt = psd_analysis(alt_sig_conv)
fig, ax2 = plt.subplots()
# plt.semilogy(f,wxx) # looks much worse with semilog
ax2.plot(f,wxx)
ax2.plot(f,wxx_alt)
plt.legend(['no trial variability','alternating trial variability'])
plt.ylabel('amygdala power')
plt.xlabel('frequency (Hz)')