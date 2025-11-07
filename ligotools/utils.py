import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, filtfilt
from scipy.interpolate import interp1d
from scipy.io import wavfile
import matplotlib.mlab as mlab

# function to whiten data
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0, 2048, Nt // 2 + 1)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

# function to keep the data within integer limits, and write to wavfile:
def write_wavfile(filename,fs,data):
    d = np.int16(data/np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename,int(fs), d)

# function that shifts frequency of a band-passed signal
def reqshift(data,fshift=100,sample_rate=4096):
    """Frequency shift the signal by constant
    """
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real,nbins) + 1j*np.roll(x.imag,nbins)
    y[0:nbins]=0.
    z = np.fft.irfft(y)
    return z


def plot_matched_filter_psd(strain_H1, strain_L1, template, time, fs, template_offset=0, dt=1/4096,
                            bb=None, ab=None, normalization=1, tevent=0, eventname='event',
                            plottype='png', make_plots=True):
    """
    Calculate PSD, matched filter output, and plot results for both detectors H1 and L1.
    """
    NFFT = 4*fs
    psd_window = np.blackman(NFFT)
    NOVL = NFFT // 2

    template = template_p + template_c*1.j if 'template_p' in globals() else template
    etime = time + template_offset
    datafreq = np.fft.fftfreq(template.size) * fs
    df = np.abs(datafreq[1] - datafreq[0])

    try:
        dwindow = windows.tukey(template.size, alpha=1./8)
    except:
        dwindow = windows.blackman(template.size)

    template_fft = np.fft.fft(template*dwindow) / fs

    dets = ['H1', 'L1']
    results = {}

    for det in dets:
        data = strain_L1.copy() if det=='L1' else strain_H1.copy()
        data_psd, freqs = mlab.psd(data, Fs=fs, NFFT=NFFT, window=psd_window, noverlap=NOVL)
        data_fft = np.fft.fft(data*dwindow) / fs
        power_vec = np.interp(np.abs(datafreq), freqs, data_psd)

        optimal = data_fft * template_fft.conjugate() / power_vec
        optimal_time = 2*np.fft.ifft(optimal) * fs

        sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
        sigma = np.sqrt(np.abs(sigmasq))
        SNR_complex = optimal_time / sigma

        peaksample = int(data.size / 2)
        SNR_complex = np.roll(SNR_complex, peaksample)
        SNR = np.abs(SNR_complex)

        indmax = np.argmax(SNR)
        timemax = time[indmax]
        SNRmax = SNR[indmax]
        d_eff = sigma / SNRmax
        horizon = sigma / 8
        phase = np.angle(SNR_complex[indmax])
        offset = indmax - peaksample

        template_phaseshifted = np.real(template*np.exp(1j*phase))
        template_rolled = np.roll(template_phaseshifted, offset) / d_eff
        template_whitened = whiten(template_rolled, interp1d(freqs, data_psd), dt)
        template_match = filtfilt(bb, ab, template_whitened) / normalization if bb is not None else template_whitened

        print(f"For detector {det}, maximum at {timemax:.4f} with SNR = {SNRmax:.1f}, D_eff = {d_eff:.2f}, horizon = {horizon:.1f} Mpc")

        if make_plots:
            pcolor = 'g' if det=='L1' else 'r'
            strain_whitenbp = strain_L1.copy() if det=='L1' else strain_H1.copy()

            # Plot SNR vs time
            plt.figure(figsize=(10,8))
            plt.subplot(2,1,1)
            plt.plot(time-timemax, SNR, pcolor, label=f'{det} SNR(t)')
            plt.grid(True)
            plt.ylabel('SNR')
            plt.xlabel(f'Time since {timemax:.4f}')
            plt.legend()
            plt.title(f'{det} matched filter SNR around event')

            plt.subplot(2,1,2)
            plt.plot(time-timemax, SNR, pcolor, label=f'{det} SNR(t)')
            plt.xlim([-0.15,0.05])
            plt.grid(True)
            plt.ylabel('SNR')
            plt.xlabel(f'Time since {timemax:.4f}')
            plt.legend()
            plt.savefig(f'figures/{eventname}_{det}_SNR.{plottype}')

            # Plot whitened strain and template
            plt.figure(figsize=(10,8))
            plt.subplot(2,1,1)
            plt.plot(time-tevent, strain_whitenbp, pcolor, label=f'{det} whitened h(t)')
            plt.plot(time-tevent, template_match, 'k', label='Template(t)')
            plt.xlim([-0.15,0.05])
            plt.ylim([-10,10])
            plt.grid(True)
            plt.xlabel(f'Time since {timemax:.4f}')
            plt.ylabel('Whitened strain')
            plt.legend()
            plt.title(f'{det} whitened data around event')

            # Plot residual
            plt.subplot(2,1,2)
            plt.plot(time-tevent, strain_whitenbp-template_match, pcolor, label=f'{det} resid')
            plt.xlim([-0.15,0.05])
            plt.ylim([-10,10])
            plt.grid(True)
            plt.xlabel(f'Time since {timemax:.4f}')
            plt.ylabel('Residual strain')
            plt.legend()
            plt.title(f'{det} Residual whitened data after subtracting template around event')
            plt.savefig(f'figures/{eventname}_{det}_matchtime.{plottype}')

        results[det] = {'SNR': SNR, 'timemax': timemax, 'SNRmax': SNRmax, 'd_eff': d_eff}

    return results
