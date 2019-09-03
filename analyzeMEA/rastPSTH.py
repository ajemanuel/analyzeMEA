import matplotlib.pyplot as plt
import numpy as np



def plotStimRasters(stimulus, samples, spikes, unit, ltime, rtime, save=False, saveString = '',
                    baseline=0, sample_rate=20000, fig_size=(10,4),
                    heightRatio=[1,4], markerSize=3,stimLabel='Force (mN)'):
    """
    Generate plots with stimulus displayed at top and rasters below for individual units.

    inputs:
        stimulus - list of ndarrays, stimulus waveform
        samples - list of ndarrays, spike times in samples
        spikes - list of ndarrays, cluster identity for each spike
        unit - int, unit to include in raster
        save - boolean, save plot to disk, default=False
        baseline - float, time before first stimulus in s, default=0.0
        sample_rate - int, sample rate in Hz, default=20000
        fig_size - tuple, ratio of width to length
        heightRatio - list, ratio of heights of stimulus and raster plots

    generates a plot; no outputs
    """

    # Plot stimulus waveform
    f, (a0, a1) = plt.subplots(2,1,gridspec_kw={'height_ratios':heightRatio},figsize=fig_size)
    xaxis = np.arange(ltime-baseline,rtime-baseline,1/sample_rate)
    for sweep in stimulus:
        a0.plot(xaxis,sweep[int(sample_rate*ltime):int(sample_rate*rtime)],linewidth=.5,color='gray') # add +5*i to the y axis to get separate traces
    topxlim = a0.get_xlim()
    a0.set_title('Unit '+str(unit))
    a0.set_xticks([])
    a0.set_ylabel(stimLabel)

    # Plot Rasters
    for sweep in range(len(samples)):
        sweepspikes = spikes[sweep][spikes[sweep]==unit]
        sweepsamples = samples[sweep][spikes[sweep]==unit]
        sweepspikes = sweepspikes[(sweepsamples > ltime*sample_rate) & (sweepsamples < rtime*sample_rate)]
        sweepsamples = sweepsamples[(sweepsamples > ltime*sample_rate) & (sweepsamples < rtime*sample_rate)]
        a1.plot(sweepsamples/sample_rate-baseline,(sweepspikes+sweep-unit),'|',color='k',markersize=markerSize,mew=.5)
    a1.set_xlim(topxlim)
    a1.set_ylim(-1,len(samples))
    a1.set_xlabel('Time (s)')
    a1.set_ylabel('Step #')
    plt.tight_layout()

    if save:
        plt.savefig('RasterUnit'+str(unit)+saveString+'.png',dpi=300)
    plt.show()
    plt.close()

def makeSweepPSTH(bin_size, samples, spikes,sample_rate=20000, units=None, duration=None, verbose=False, rate=True, bs_window=[0, 0.25]):
    """
    Use this to convert spike time rasters into PSTHs with user-defined bin

    inputs:
        bin_size - float, bin size in seconds
        samples - list of ndarrays, time of spikes in samples
        spikes- list of ndarrays, spike cluster identities
        sample_rate - int, Hz, default = 20000
        units - None or sequence, list of units to include in PSTH
        duration - None or float, duration of PSTH; if None, inferred from last spike
        verbose - boolean, print information about psth during calculation
        rate - boolean; Output rate (divide by bin_size and # of trials) or total spikes per trial (divide by # trials only)
        bs_window - sequence, len 2; window (in s) to use for baseline subtraction; default = [0, 0.25]
    output: dict with keys:
        psths - ndarray
        bin_size - float, same as input
        sample_rate - int, same as input
        xaxis - ndarray, gives the left side of the bins
        units - ndarray, units included in psth
    """

    bin_samples = bin_size * sample_rate

    if duration is None:
        maxBin = max(np.concatenate(samples))/sample_rate
    else:
        maxBin = duration

    if units is None:  # if user does not specify which units to use (usually done with np.unique(goodSpikes))
        units = np.unique(np.hstack(spikes))
    numUnits = len(units)

    psths = np.zeros([int(np.ceil(maxBin/bin_size)), numUnits])
    if verbose:
        print('psth size is',psths.shape)
    for i in range(len(samples)):
        for stepSample, stepSpike in zip(samples[i], spikes[i]):
            if stepSpike in units:
                if int(np.floor(stepSample/bin_samples)) == psths.shape[0]:
                    psths[int(np.floor(stepSample/bin_samples))-1, np.where(units == stepSpike)[0][0]] += 1 ## for the rare instance when a spike is detected at the last sample of a sweep
                else:
                    psths[int(np.floor(stepSample/bin_samples)), np.where(units == stepSpike)[0][0]] += 1
    psth_dict = {}
    if rate:
        psth_dict['psths'] = psths/bin_size/len(samples) # in units of Hz
    else:
        psth_dict['psths'] = psths/len(samples) # in units of spikes/trial in each bin

    psths_bs = np.copy(np.transpose(psth_dict['psths']))
    psths_z = np.copy(np.transpose(psth_dict['psths']))
    for i,psth in enumerate(psths_bs):
        tempMean = np.mean(psth[int(bs_window[0]/bin_size):int(bs_window[1]/bin_size)])
        tempStDev = np.std(psth[int(bs_window[0]/bin_size):int(bs_window[1]/bin_size)])
        #print(tempMean)
        psths_bs[i] = psth - tempMean
        psths_z[i] = (psth - tempMean)/tempStDev
    psth_dict['psths_bs'] = np.transpose(psths_bs)
    psth_dict['psths_z'] = np.transpose(psths_z)
    psth_dict['bin_size'] = bin_size # in s
    psth_dict['sample_rate'] = sample_rate # in Hz
    psth_dict['xaxis'] = np.arange(0,maxBin,bin_size)
    psth_dict['units'] = units
    psth_dict['num_sweeps'] = len(samples)
    return psth_dict

def spikeTriggeredAverage(spikes, stimulusTrace, window=(-.1,.1), sampleRate=20000):
    """
    Calculates the average of a stimulus trace around each spike

    Inputs:
        spikes - sequence - list of spike times to trigger against
        stimulusTrace - ndarray - stimulus to average
        window - sequence, len 2 - period over which to average in s, defaults to (-0.1, 0.1)
        sampleRate - int - sample rate
    Outputs:
        sta - sta from -window to window at rate of sampleRate
        xaxis - ndarray, sequence for xaxis of sta
    """
    sta = np.zeros(int((window[1]-window[0])*sampleRate))
    window_samples = [int(n*sampleRate) for n in window]
    numErrors = 0
    for spike in spikes:
        try:
            sta = sta + stimulusTrace[int(spike+window_samples[0]):int(spike+window_samples[1])]
        except ValueError:
            numErrors += 1
    sta /= len(spikes) - numErrors
    xaxis = np.arange(window[0],window[1],1/sampleRate)

    return sta, xaxis
