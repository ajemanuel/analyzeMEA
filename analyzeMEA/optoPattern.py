import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import analyzeMEA.rastPSTH
import scipy.ndimage

def extractLaserPositions(matFile, voltageToDistance=3.843750000e+03):
    """
    Calculate the positions of the stimulus at each point.

    input:
    matFile - str, path to file generated from stimulus
    voltageToDistance - float, calibration for converting voltage; on DRG rig voltageToDistance=7000
    output:
    positions - list of tuples containing (x, y) coordinates at each position.
    """

    temp = scipy.io.loadmat(matFile, variable_names=['laser','lz1','x','y'])
    try:
        laser = temp['laser']
    except(KeyError):
        laser = temp['lz1'] ## old version
    x = temp['x']
    y = temp['y']
    positions = []
    laserSamples = np.where(laser[1:] > laser[:-1])[0]
    for sample in laserSamples:
        positions.append((float(x[sample]*voltageToDistance), float(y[sample]*voltageToDistance)))
    return positions

def extractLaserPSTH(matFile, samples, spikes, duration=None, sampleRate=20000, includeLaserList=True):
    """
    Make lists of samples and spikes at each laser pulse
    inputs:
        matFile - str, path to file made when stimulating
        samples - sequence of spike times
        spikes - sequence of cluster identities for each spike
        duration - period to include after each spike (in s), default is ISI
        includeLaserList - boolean, use False to not calculate laser list
    outputs:
        samplesList - list of lists of spike samples after each laser pulse
        spikesList - list of lists of cluster identity corresponding to samplesList
        laserList - list of ndarrays with waveform of laser pulse command, only returned if includeLaserList == True
    """


    temp = scipy.io.loadmat(matFile)
    try:
        laserOnsets = np.where(temp['laser'][1:] > temp['laser'][:-1])[0]
    except(KeyError):
        laserOnsets = np.where(temp['lz1'][1:] > temp['lz1'][:-1])[0] ### old version of stim file
    if duration is None:
        duration = temp['ISI']

    samplesList = []
    spikesList = []
    if includeLaserList:
        laserList = []

    for start in laserOnsets:
        adjStart = int(start * (sampleRate/temp['Fs'])) ## adjusting the start in case the sample rates differ between nidaq and intan
        end = int(adjStart + sampleRate * duration)
        samplesList.append(samples[(samples > adjStart) & (samples < end)] - adjStart)
        spikesList.append(spikes[(samples > adjStart) & (samples < end)])
        if includeLaserList:
            try:
                laserList.append(temp['laser'][start:int(start+temp['Fs']*duration)])
            except(KeyError):
                laserList.append(temp['lz1'][start:int(start+temp['Fs']*duration)])

    if includeLaserList:
        return samplesList, spikesList, laserList
    else:
        return samplesList, spikesList

def extractLaserPSTH_intan(laser_trigger, samples, spikes, duration=0.1, sampleRate=20000, includeLaserList=True):
    """
    Make lists of samples and spikes at each laser pulse (does not require the matlab stim file)
    inputs:
        laser_trigger - sequence, digital signal containing laser onsets
        samples - sequence of spike times
        spikes - sequence of cluster identities for each spike
        duration - period to include after each onset (in s), default is 0.1 s
        sampleRate  - int, sample rate of acquisition (in Hz), default = 20000
        includeLaserList - boolean, use False to not calculate laser list (saves time/memory)

    outputs:
        samplesList - list of lists of spike samples after each laser pulse
        spikesList - list of lists of cluster identity corresponding to samplesList
        laserList - list of ndarrays with waveform of laser pulse command
    """

    laserOnsets = np.where(laser_trigger[1:] > laser_trigger[:-1])[0]
    samplesList = []
    spikesList = []
    if includeLaserList:
        laserList = []

    for start in laserOnsets:
        end = start + sampleRate * duration
        samplesList.append(samples[(samples > start) & (samples < end)] - start)
        spikesList.append(spikes[(samples > start) & (samples < end)])
        if includeLaserList:
            laserList.append(laser_trigger[start:int(start+sampleRate*duration)])
    if includeLaserList:
        return samplesList, spikesList, laserList
    else:
        return samplesList, spikesList


def calcBinnedOpticalResponse(matFile, samples, spikes, binSize, window, bs_window, units, save=False, saveString='', smoothBin=0, voltageToDistance = 3.843750000e+03):
    """
    Inputs:
    matFile - string, path to file generated with randSquareOffset stimulus
    samples - list, samples during stimulus
    spikes - list, cluster identities during stimulus
    binSize - float, size of spatial bin
    window - sequence, len 2 - window of analysis (in ms)
    bs_window - sequence, len 2 - spikes in this window subtracted from those in window ( in ms)
    units - sequence - units to include
    save - boolean, whether to save plot or not
    saveString - string, string appended to filename when saving
    smoothBin - float, size of gaussian filter for smoothing (in bin units), default=0, no smoothing
    voltageToDistance - float, calibration for converting voltage into micron distance; DRG rig = 7000
    Output:
    ouput - ndarray, optical receptive fields with shape (numBins, numBins, numUnits)
    """

    samplesList, spikesList = extractLaserPSTH(matFile, samples, spikes, includeLaserList=False)
    parameters = scipy.io.loadmat(matFile, variable_names=['edgeLength','offsetX','offsetY','ISI'])
    laserPositions = np.transpose(extractLaserPositions(matFile,voltageToDistance=voltageToDistance))
    binSizeMicron = binSize * 1000
    halfEdgeLength = parameters['edgeLength']/2
    xmin = int(parameters['offsetX'] - halfEdgeLength)
    xmax = int(parameters['offsetX'] + halfEdgeLength)
    ymin = int(parameters['offsetY'] - halfEdgeLength)
    ymax = int(parameters['offsetY'] + halfEdgeLength)

    numBins = int(parameters['edgeLength']/binSizeMicron)
    numUnits = len(units)

    output = np.zeros([numBins, numBins, numUnits])

    for Bin in range(numBins * numBins):
        binxy = np.unravel_index(Bin,(numBins,numBins))
        tempPositions = np.where((laserPositions[0] > (xmin + binSizeMicron*binxy[0])) &
                             (laserPositions[0] < xmin + binSizeMicron*(binxy[0]+1)) &
                             (laserPositions[1] > (ymin + binSizeMicron*binxy[1])) &
                             (laserPositions[1] < ymin + binSizeMicron*(binxy[1]+1)))[0]
        if len(tempPositions > 0):
            tempPSTH = analyzeMEA.rastPSTH.makeSweepPSTH(0.001,[samplesList[a] for a in tempPositions],[spikesList[a] for a in tempPositions],
                units=units, duration=float(parameters['ISI']), rate=False)
            for unit in range(numUnits):
                output[binxy[0],binxy[1],unit] = np.mean(tempPSTH['psths'][window[0]:window[1],unit]) - np.mean(tempPSTH['psths'][bs_window[0]:bs_window[1],unit])
    for unit in range(numUnits):
        if smoothBin > 0:
            output[:,:,unit] = scipy.ndimage.gaussian_filter(output[:,:,unit],smoothBin)
        plt.figure(figsize=(4,4))
        a0 = plt.axes()
        absMax = np.amax(np.absolute(output[:,:,unit]))
        sc = a0.imshow(output[:,:,unit],extent=[ymin/1000, ymax/1000, xmin/1000, xmax/1000],origin='lower',
                        clim=[-absMax,absMax],cmap='bwr')
        a0.set_title('Unit {0}'.format(units[unit]))
        a0.set_xlabel('mm')
        a0.set_ylabel('mm')
        cb = plt.colorbar(sc,fraction=.03)
        cb.set_label(r'$\Delta$ Rate (Hz)')
        plt.tight_layout()
        if save:
            plt.savefig('lasRFunit{0}{1}.png'.format(units[unit],saveString),dpi=300,transparent=True)
        plt.show()
        plt.close()
    return output
