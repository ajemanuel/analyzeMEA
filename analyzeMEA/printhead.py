import scipy.io
import numpy as np
import analyzeMEA.rastPSTH

def randSingleAnalysis(matFile, samples, spikes, units,
                        window=[0,25], psthSize = 1000, psthBin = 1, sampleRate=20000,verbose=True):
    """
    Generates relative spike rates in analysis window to overall mean to map out receptive fields using single indentations.
    The window is defined relative to the start of the pin in question and defaults to 25 ms.

    matFile - string, path to file generated with printHead singlePin stimulus
    samples - list, samples during stimulus
    spikes - list, cluster identities during stimulus
    window - sequence, len 2 - window of analysis relative to start of pin (in ms)
    units - sequence - units to include
    psthSize - float, size of psth in ms
    psthBin - float, size of bin to use in psth, in ms

    output - dict containing various computations
        KEYS:
        positions - positions in mm of the 24 pins
        units - same as input
        positionRepsonse - 2-dimensional ndarray - change in rate for each position (dim 1) x unit (dim 2)
        psth - dict (same as makeSweepPSTH
    """

    a = scipy.io.loadmat(matFile)

    if a['protocol'] != 'randSingle':
        print('Protocol is {0}, not "randSingle"\n'.format(a['protocol']))
        return -1
    stim = a['stim'][np.where(a['trigger']==1)[0],:] ## excluding first and last samples so that intan & matlab samples match
    psthSize_samples = int(psthSize/1000*sampleRate)
    psthBin_seconds = psthBin/1000
    window_bins = np.int32(np.array(window)/psthBin)


    output = {}
    output['units'] = units

    ## calculating positions (position one is furthest from rig) (in mm)
    output['positions'] = np.zeros([24,2])
    for i in range(24):
        if i % 2 != 0:
            output['positions'][i,0] = 1
        output['positions'][i,1] = (i/24)*4

    positionResponse = np.zeros((24, len(units)))
    for i in range(24): # for each of 24 positions
        eventStarts = np.where(stim[1:,i] > stim[:-1,i])[0]
        temp_samples = []
        temp_spikes = []
        #temp_stim = []
        for start in eventStarts:
            #temp_stim.append(stim[int(start - psthSize_samples/2):int(start+psthSize_samples/2)])
            tempStart = int(start - psthSize_samples/2)
            tempEnd = int(start + psthSize_samples/2)
            temp_samples.append(samples[(samples > tempStart) & (samples < tempEnd)] - tempStart)
            temp_spikes.append(spikes[(samples > tempStart) & (samples < tempEnd)])
        psth = analyzeMEA.rastPSTH.makeSweepPSTH(psthBin_seconds, temp_samples, temp_spikes, units=units,duration = psthSize/1000, bs_window=[0,psthSize/1000]) # taking the mean of the overall mean of the psth window for baseline subtraction
        for j in range(len(units)):
            startBin = int(psthSize/psthBin/2)
            positionResponse[i, j] = np.nanmean(psth['psths_bs'][startBin + window_bins[0]:startBin + window_bins[1],j])

    output['psth'] = psth
    output['positionResponse'] = positionResponse
    return output
