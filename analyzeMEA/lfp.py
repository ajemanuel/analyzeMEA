from scipy.signal import butter, lfilter, filtfilt, periodogram
import analyzeMEA.read_rhd_controller as read_rhd_controller
import numpy as np



def extractLFPs(rhdFile,filterFreq=250,sampleRate=20000,downSample=10,stimChannel=1):
    """
    Extract LFPs from RHDfiles
    Input:
        rhdFile - string, filename for the file from the intan recording
        filterFreq - sequence, len 2, - cutoffs for bandpass filter (Hz)
        sampleRate - int, sample rate of intan recording (Hz)
        downSample - int, factor by which to downsample the LFPs
        stimChannel - int, designate the analog input channel that you'd like to match to the LFPs
    Output:
        LFPs - ndarray with filtered traces for each channel
        stim - ndarray with downsampled stim trace
        sweepOnsets - ndarray with downsampled samples at which trigger signal turned on
        meanPower500_5000 - ndarray with mean power between 500 and 5000 Hz for each channel (in channel order)
    """

    def butter_lowpass(cutoffs, fs, order=8):
        nyq = 0.5 * fs
        normal_cutoffs = cutoffs / nyq
        b,a = butter(order,normal_cutoffs,btype='lowpass',analog=False)
        return b, a
    def butter_bandpass_filter(data, cutoffs, fs, order=8):
        b,a = butter_bandpass(cutoffs, fs, order=order)
        y = scipy.signal.lfilter(b,a,data)
        return y
    b,a = butter_lowpass(filterFreq, sampleRate) ## using default order=8

    rhdContents = read_rhd_controller.read_data(rhdFile)
    #output = np.zeros(rhdContents['amplifier_data'].shape)

    LFPs = filtfilt(b,a,rhdContents['amplifier_data'],padlen=150)
    LFPs = LFPs[:,::downSample]
    try:
        stim = filtfilt(b,a,rhdContents['board_adc_data'][stimChannel,:]) ## filtering the stimulus to to make it easier to find the starts
        stim = stim[::downSample]

    except KeyError:
        stim = [0]
    digIn0 = rhdContents['board_dig_in_data'][0,:]
    sweepOnsets = np.where(digIn0[1:] > digIn0[:-1])[0]/downSample
    meanPower500_5000 = []
    for channel in rhdContents['amplifier_data']:
        f, Pxx_den = periodogram(channel, sampleRate)
        meanPower500_5000.append(np.mean(Pxx_den[np.where((f > 500) & (f < 5000))[0]]))


    return LFPs, stim, sweepOnsets, meanPower500_5000
