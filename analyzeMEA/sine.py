import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import analyzeMEA.rastPSTH

def importSineData(sineFile):
    """
    Import key features of the stimulus from the matlab file generated during experiment.

    Inputs:
        sineFile - str, path to matlab file generated during experiment
    Outputs:
        sineAmplitudes - ndarray, intensity (in mN) for each trial
        frequencies - ndarray, frequency (in Hz) for each trial
        baseline - int, time (in samples) prior to stimulus (trials are 1/4 no stim, 1/2 stim, 1/4 no stim)
        sampleRate - int, stimulus sample rate
    """
    try:
        stimInfo = scipy.io.loadmat(sineFile,variable_names=['sineAmplitude','sineFrequency','trigger','Fs'])
        stimType = 'step'
        sineAmplitudes = stimInfo['sineAmplitude'][:,0]

    except:
        stimInfo = scipy.io.loadmat(sineFile,variable_names=['sineFrequency','trigger','Fs','forceRange'])
        stimType = 'ramp'
        forceRange = stimInfo['forceRange'][0,:]
    frequencies = stimInfo['sineFrequency'][:,0]
    sweepDuration_samples = np.where(stimInfo['trigger'][1:] < stimInfo['trigger'][:-1])[0][0] - np.where(stimInfo['trigger'][1:] > stimInfo['trigger'][:-1])[0][0]+2
    baseline = sweepDuration_samples/4
    sampleRate = stimInfo['Fs']

    if stimType == 'step':
        return sineAmplitudes, frequencies, baseline, sampleRate
    elif stimType == 'ramp':
        return forceRange, frequencies, baseline, sampleRate

def calculateSpikesPerCycle(sineFile,samples,spikes,sampleRate=20000):
    outDict = {}
    sineAmplitudes, frequencies, baseline, Fs = importSineData(sineFile)
    if (len(sineAmplitudes) != len(samples)) or (len(sineAmplitudes) != len(spikes)):
        print('Number of trials does not match stim file.')
        return
    uniqueFrequencies = np.unique(frequencies)
    # calculating baseline spike rate
    units = np.unique(np.concatenate(spikes))
    tempPSTH = analyzeMEA.rastPSTH.makeSweepPSTH(0.02,samples,spikes,units=units,bs_window=[0,baseline/Fs],sample_rate=sampleRate)
    baselines = np.mean(tempPSTH['psths'][:int(baseline/Fs/0.02),:],axis=0)

    evokedSpikes = np.zeros((len(units),len(samples))) #
    for i, unit in enumerate(units):
        for j, (sample, spike) in enumerate(zip(samples,spikes)):
            evokedSpikes[i,j] = np.sum(spike[(sample > baseline) & (sample < baseline * 3)] == unit) - baselines[i] * ((baseline * 3 - baseline)/Fs) ## fix this later if ever using differently sampled stim and acquisition
        outDict[unit] = {}
        for frequency in uniqueFrequencies:
            ind = np.where(frequencies == frequency)[0]
            amplitudes = sineAmplitudes[ind]
            sortInd = np.argsort(amplitudes)
            overallInd = ind[sortInd] ## this index all sweeps == frequency, sorted by amplitude of sine wave
            sortedAmplitudes = amplitudes[sortInd]
            outDict[unit][frequency] = {}
            outDict[unit][frequency]['amplitudes'] = sortedAmplitudes
            outDict[unit][frequency]['spikesPerCycle'] = np.reshape(evokedSpikes[i,overallInd]/(((baseline*3-baseline)/Fs)*frequency),-1)
    outDict['evokedSpikes'] = evokedSpikes
    outDict['units'] = units
    outDict['baselines'] = baselines
    return outDict

def plotSineRasters(sineFile,samples,spikes,sampleRate=20000,binSize=0.005,save=False, saveString = ''):
    """
    Plot Raster and PSTH for each unit at each frequency.
    Inputs:
        sineFile - str, path to matlab file generated during experiment
        samples - list, spike times associated with sine stimulus
        spikes - list, units corresponding to spike times
        sampleRate - int, sample rate of intan acquisition
        binSize - float, bin size for PSTH
        save - boolean or str, whether to save plot, can specify 'png' or 'pdf'
    Output: displays and saves pyplot plots
    """
    sineAmplitudes, frequencies, baseline, Fs = importSineData(sineFile)
    if (len(frequencies) != len(samples)) or (len(frequencies) != len(spikes)):
        print('Number of trials does not match stim file.')
        return
    if len(sineAmplitudes) == len(frequencies): ## this won't work if the ramp stimulus is used with only two frequencies
        stimType = 'step'
    else:
        stimType = 'ramp'
    xlims = [-baseline/Fs,baseline*3/Fs] # used for all plots, so defining here
    uniqueFrequencies = np.unique(frequencies)

    if stimType == 'step':
        for frequency in uniqueFrequencies:
            ind = np.where(frequencies == frequency)[0]
            amplitudes = sineAmplitudes[ind]
            sortInd = np.argsort(amplitudes)
            overallInd = ind[sortInd] ## this index all sweeps == frequency, sorted by amplitude of sine wave

            ## generating PSTH -- an average of all trials at the current frequency
            units = np.unique(np.concatenate(spikes))
            tempPSTH = analyzeMEA.rastPSTH.makeSweepPSTH(binSize,[samples[n] for n in overallInd],[spikes[n] for n in overallInd],units=units,bs_window=[0,baseline/Fs])

            ## plotting raster and PSTH for each unit
            for i, unit in enumerate(units):
                f, ax = plt.subplots(2,1,figsize=[3.5,3],gridspec_kw={'height_ratios':[4,1]})
                for j, index in enumerate(overallInd):
                    samps = (np.array(samples[index][spikes[index] == unit]) - baseline)/sampleRate
                    sps = np.array(spikes[index][spikes[index] == unit]) - unit + j
                    ax[0].plot(samps,sps,'|',color='gray',markersize=4,mew=0.5)
                    ax[1].plot(tempPSTH['xaxis']-baseline/sampleRate,tempPSTH['psths_bs'][:,i],color='gray',linewidth=0.5)
                # indicating where the stimulus occurred
                forceBarY = j + 3
                ll = ax[0].plot((0,baseline*2/sampleRate),[forceBarY,forceBarY],color='k',linewidth=4,scalex=False,scaley=False)
                ll[0].set_clip_on(False)
                # labeling and formatting plot
                ax[0].set_xlim(xlims)
                ax[1].set_xlim(xlims)
                ax[0].set_ylim([-1,j+1])
                ax[0].set_xticks([])
                ax[1].set_xlabel('Time (s)')
                ax[1].set_ylabel('Rate (Hz)')
                ax[0].set_ylabel('Trial')
                ax[0].set_title('Unit {0:d}, {1:d} Hz'.format(unit,frequency),pad=8)
                plt.subplots_adjust(left=0.15,bottom=0.15,top=0.9,hspace=0.05,right=0.95)
                if save == True:
                    plt.savefig('Unit{0:d}_{1:d}Hz.png'.format(unit,frequency),dpi=300,transparent=True)
                elif save == 'png':
                    plt.savefig('Unit{0:d}_{1:d}Hz_{2}.png'.format(unit,frequency,saveString),dpi=300,transparent=True)
                elif save == 'pdf':
                    plt.savefig('Unit{0:d}_{1:d}Hz.pdf'.format(unit,frequency),transparent=True)
                plt.show()
                plt.close()
    else:
        for frequency in uniqueFrequencies:
            ind = np.where(frequencies == frequency)[0]

            ## generating PSTH -- an average of all trials at the current frequency
            units = np.unique(np.concatenate(spikes))
            tempPSTH = analyzeMEA.rastPSTH.makeSweepPSTH(binSize,[samples[n] for n in ind],[spikes[n] for n in ind],units=units,bs_window=[0,baseline/Fs],duration=baseline/Fs*4)

            ## plotting raster and PSTH for each unit
            for i, unit in enumerate(units):
                f, ax = plt.subplots(2,1,figsize=[5,3],gridspec_kw={'height_ratios':[5,1]})
                for j, index in enumerate(ind):
                    samps = (np.array(samples[index][spikes[index] == unit]) - baseline)/sampleRate
                    sps = np.array(spikes[index][spikes[index] == unit]) - unit + j +1
                    ax[0].plot(samps,sps,'|',color='gray',markersize=10,mew=0.5)
                    ax[1].plot(tempPSTH['xaxis']-baseline/sampleRate,tempPSTH['psths_bs'][:,i],color='gray',linewidth=0.5)
                # indicating where the stimulus occurred
                forceBarY = (j + 1)/50 + j+1.5
                ll = ax[0].plot((0,baseline*2/sampleRate),[forceBarY,forceBarY],color='k',linewidth=4,scalex=False,scaley=False)
                ll[0].set_clip_on(False)
                # labeling and formatting plot
                ax[0].set_xlim(xlims)
                ax[1].set_xlim(xlims)
                ax[0].set_ylim([0.5,j+1.5])
                ax[0].set_xticks([])
                ax[1].set_xlabel('Time (s)')
                ax[1].set_ylabel('Rate (Hz)')
                ax[0].set_ylabel('Trial')
                ax[0].set_title('Unit {0:d}, {1:d} Hz'.format(unit,frequency),pad=8)
                plt.subplots_adjust(left=0.15,bottom=0.15,top=0.9,hspace=0.05,right=0.95)
                if save == True:
                    plt.savefig('Unit{0:d}_{1:d}Hz.png'.format(unit,frequency),dpi=300,transparent=True)
                elif save == 'png':
                    plt.savefig('Unit{0:d}_{1:d}Hz_{2}.png'.format(unit,frequency,saveString),dpi=300,transparent=True)
                elif save == 'pdf':
                    plt.savefig('Unit{0:d}_{1:d}Hz.pdf'.format(unit,frequency),transparent=True)
                plt.show()
                plt.close()

def plotPhaseRaster(spikeSamples,frequency,stimTimes=[0.5,1.5],sampleRate=20000):
    phaseStarts = np.arange(stimTimes[0],stimTimes[1],1/frequency/(stimTimes[1]-stimTimes[0]))
    phaseEnds = phaseStarts + 1/frequency
    spikeTimes = spikeSamples/sampleRate
    xlims = [0, 1/frequency]
    f, ax = plt.subplots(2,1,figsize=[2.5,3],gridspec_kw={'height_ratios':[1,5]})
    for i, (start,end) in enumerate(zip(phaseStarts, phaseEnds)):
        tempTimes = spikeTimes[(spikeTimes > start) & (spikeTimes < end)]  - start
        ax[1].plot(tempTimes,np.ones(len(tempTimes))*i,'|',mew=0.5,markersize=4)
    sineWaveX = np.arange(0,1/frequency,1/frequency/100)
    sineWaveY = np.sin((np.pi*2*frequency) * sineWaveX - np.pi/2)
    ax[0].plot(sineWaveX,sineWaveY)
    ax[0].set_xlim(xlims)
    ax[1].set_xlim(xlims)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
