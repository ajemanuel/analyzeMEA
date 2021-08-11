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

def extractLaserPSTH(matFile, samples, spikes, baseline = 0.01, duration=None, sampleRate=20000, includeLaserList=True):
    """
    Make lists of samples and spikes at each laser pulse
    inputs:
        matFile - str, path to file made when stimulating
        samples - sequence of spike times
        spikes - sequence of cluster identities for each spike
        baseline - period to include prior to laser onset (in s), default is 10 ms
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
        duration = temp['ISI'] + baseline

    samplesList = []
    spikesList = []
    if includeLaserList:
        laserList = []

    for start in laserOnsets:
        adjStart = int(start * (sampleRate/temp['Fs'])) ## adjusting the start in case the sample rates differ between nidaq and intan
        adjStart = int(adjStart - baseline*sampleRate) ## adjusting the start to includea pre-laser baseline period
        end = int(adjStart + sampleRate * duration)
        samplesList.append(samples[(samples > adjStart) & (samples < end)] - adjStart)
        spikesList.append(spikes[(samples > adjStart) & (samples < end)])
        if includeLaserList:
            try:
                laserList.append(temp['laser'][adjStart:int(adjStart*(temp['Fs']/sampleRate)+temp['Fs']*duration)])
            except(KeyError):
                laserList.append(temp['lz1'][adjStart:int(adjStart*(temp['Fs']/sampleRate)+temp['Fs']*duration)])

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


def calcBinnedOpticalResponse(matFile, samples, spikes, binSize, window, bs_window, units, baseline=10, save=False, saveString='', smoothBin=0, voltageToDistance = 3.843750000e+03, showplots=True):
    """
    Inputs:
    matFile - string, path to file generated with randSquareOffset stimulus
    samples - list, samples during stimulus
    spikes - list, cluster identities during stimulus
    binSize - float, size of spatial bin
    window - sequence, len 2 - window of analysis (in ms)
    bs_window - sequence, len 2 - spikes in this window subtracted from those in window ( in ms)
    units - sequence - units to include
    baseline - int or float, size of baseline period for laser PSTH (in ms; default = 10)
    save - boolean, whether to save plot or not
    saveString - string, string appended to filename when saving
    smoothBin - float, size of gaussian filter for smoothing (in bin units), default=0, no smoothing
    voltageToDistance - float, calibration for converting voltage into micron distance; DRG rig = 7000
    Output:
    ouput - ndarray, optical receptive fields with shape (numBins, numBins, numUnits)
    """
    baseline = baseline/1000 ## converting to s
    samplesList, spikesList = extractLaserPSTH(matFile, samples, spikes, baseline=baseline, includeLaserList=False)
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
        if len(tempPositions) > 0:
            if len(np.concatenate([samplesList[a] for a in tempPositions])) > 0:
                tempPSTH = analyzeMEA.rastPSTH.makeSweepPSTH(0.001,[samplesList[a] for a in tempPositions],[spikesList[a] for a in tempPositions],
                    units=units, duration=float(parameters['ISI']+baseline), rate=False)
                for unit in range(numUnits):
                    output[binxy[0],binxy[1],unit] = np.mean(tempPSTH['psths'][window[0]:window[1],unit]) - np.mean(tempPSTH['psths'][bs_window[0]:bs_window[1],unit])
    if showplots:
        for unit in range(numUnits):
            if smoothBin > 0:
                output[:,:,unit] = scipy.ndimage.gaussian_filter(output[:,:,unit],smoothBin)
            plt.figure(figsize=(4,4))
            a0 = plt.axes()
            absMax = np.amax(np.absolute(output[:,:,unit]))
            sc = a0.imshow(output[:,:,unit],extent=[ymin/1000, ymax/1000, xmin/1000, xmax/1000],origin='lower',
                            clim=[-absMax,absMax],cmap='bwr',interpolation='none')
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

def calcOpticalResponseTS(matFile, samples, spikes, binSize, windowSize, bWindowStart, units, baseline=10, smoothBin=0, voltageToDistance = 3.843750000e+03):
    """
    Inputs:
    matFile - string, path to file generated with randSquareOffset stimulus
    samples - list, samples during stimulus
    spikes - list, cluster identities during stimulus
    binSize - float, size of spatial bin
    windowSize - int, size of window of analysis (in ms)
    bWindowStart - time for start of baseline window (in ms)
    units - sequence - units to include
    baseline - int or float, size of baseline period for laser PSTH (in ms; default = 10) e.g., stimulus onset at 10 ms after start of laser PSTH
    smoothBin - float, size of gaussian filter for smoothing (in bin units), default=0, no smoothing
    voltageToDistance - float, calibration for converting voltage into micron distance; DRG rig = 7000
    Output:
    ouput - ndarray, optical receptive fields with shape (numBins, numBins, numTimeBins, numUnits)
    """
    baseline = baseline/1000 ## converting to s
    samplesList, spikesList = extractLaserPSTH(matFile, samples, spikes, baseline=baseline, includeLaserList=False)
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
    numTimeBins = int((parameters['ISI'][0][0] + baseline)*1000/windowSize)
    output = np.zeros([numBins, numBins, numTimeBins, numUnits])
    bs_window = (bWindowStart, bWindowStart+windowSize)

    for Bin in range(numBins * numBins):
        binxy = np.unravel_index(Bin,(numBins,numBins))
        tempPositions = np.where((laserPositions[0] > (xmin + binSizeMicron*binxy[0])) &
                             (laserPositions[0] < xmin + binSizeMicron*(binxy[0]+1)) &
                             (laserPositions[1] > (ymin + binSizeMicron*binxy[1])) &
                             (laserPositions[1] < ymin + binSizeMicron*(binxy[1]+1)))[0]
        if len(tempPositions > 0):
            try:
                tempPSTH = analyzeMEA.rastPSTH.makeSweepPSTH(0.001,[samplesList[a] for a in tempPositions],[spikesList[a] for a in tempPositions],
                    units=units, duration=float(parameters['ISI']+baseline), rate=False)

                for timeBin in range(numTimeBins):
                    window = (windowSize * timeBin, windowSize*timeBin+windowSize)
                    for unit in range(numUnits):
                        output[binxy[0],binxy[1],timeBin,unit] = np.mean(tempPSTH['psths'][window[0]:window[1],unit]) - np.mean(tempPSTH['psths'][bs_window[0]:bs_window[1],unit])
            except ValueError: # thrown when empty sequence entered into the makeSweepPSTH
                pass ##output assigned as 0
    if smoothBin != 0:
        for unit in range(numUnits):
            output[:,:,:,unit] = scipy.ndimage.gaussian_filter(output[:,:,:,unit],smoothBin)
    return output


def calcAddedSpikes(file, samples, spikes, units, addedSpikeWindow=(0,75), compareWindow=(75,150),
radius=450, spatialBin=0.25, windowSize=8, baseline=10, bWindowStart=0, smoothBin=[1,1,0], useMeanBaselineRate=True):
    """
    Calculation of the number of action potentials evoked by each stimulus in an area surrounding the most responsive region for each specified unit.

    Inputs:
    file - string, path to file generated with randSquareOffset stimulus
    samples - list, samples during stimulus
    spikes - list, cluster IDs for each sample in samples
    units - list or ndarray, units to evaluate
    addedSpikeWindow - tuple len 2, beginning and end of window in which to consider added spikes
    compareWindow - tuple len 2, beginning and end of window in which to compare added spikes (should be same duration)
    radius - int or float, radius around peak value to consider for added spikes, in microns
    spatialBin - float, size of spatial bins, in mm
    windowSize - float, size of window for binned time series, in ms
    baseline - int, duration prior to laser pulse, in ms
    bWindowStart - int, time bin used to for subtraction of firing rate in binned time series, in ms
    smoothBin - float or list len 3, sigmas used for the gaussian filter, default smooths in spatial but not temporal domains
    useMeanBaselineRate - boolean, compare spikes against mean baseline firing rate (True,default) or trial-by-trial (False)

    Output:
    returns dictionary, keys are unitIDs
     - each value is a list of added spikes by trial
    """

    parameters = scipy.io.loadmat(file,variable_names=['edgeLength','offsetX','offsetY','ISI'])
    edgeLength = int(parameters['edgeLength'])
    offsetX = int(parameters['offsetX'])
    offsetY = int(parameters['offsetY'])

    positions = np.array(extractLaserPositions(file))


    laserSamples, laserSpikes = extractLaserPSTH(file,samples,spikes,baseline=baseline/1000,includeLaserList=False,duration=parameters['ISI']+baseline/1000)
    if useMeanBaselineRate:
        tempPSTH = analyzeMEA.rastPSTH.makeSweepPSTH(0.001,laserSamples,laserSpikes,units=units,bs_window=[0,baseline/1000])
        meanBaselines = tempPSTH['psths'][:baseline,:].mean(axis=0)
    addedSpikes = {}

    # calc binned time series for optopattern stimulus
    binnedTimeSeries = calcOpticalResponseTS(file,samples,spikes,spatialBin,windowSize,bWindowStart,units,baseline=baseline,smoothBin = [1,1,0])

    for i, unit in enumerate(units):

        # find location and time of maximum response
        temp = binnedTimeSeries[:,:,:,i].reshape(binnedTimeSeries.shape[:-1])
        maxX, maxY, maxT = np.array(np.where(np.abs(temp) == np.max(np.abs(temp))))[:,0]
        print('Unit {} max response location: {}'.format(unit,(maxX,maxY,maxT)))
        # convert location to actual micron position that was sent to galvos
        X = maxX/binnedTimeSeries.shape[0] * edgeLength + calcXYlims(file)[0][0]
        Y = maxY/binnedTimeSeries.shape[1] * edgeLength + calcXYlims(file)[1][0]

        dists = []
        for pos in positions:
            dists.append(np.linalg.norm(pos - (X,Y)))
        dists = np.array(dists)
        indices = dists < radius

        tempSamples = [laserSamples[n] for n in np.where(indices)[0]]
        tempSpikes = [laserSpikes[n] for n in np.where(indices)[0]]

        unitSamples = []
        unitSpikes = []

        for j, (sample, spike) in enumerate(zip(tempSamples,tempSpikes)):
            unitSamples.append(sample[spike==unit]/20-baseline) # in ms (assuming sample rate is 20kHz)
            unitSpikes.append(spike[spike==unit]-unit)

        addedSpikes[unit] = []
        if useMeanBaselineRate:
            numBaselineSpikes = meanBaselines[i] * (addedSpikeWindow[1]-addedSpikeWindow[0])/1000
            for k, (sample, spike) in enumerate(zip(unitSamples,unitSpikes)): # for each trial
                addedSpikes[unit].append(sum((sample > addedSpikeWindow[0]) & (sample <= addedSpikeWindow[1])) -
                   numBaselineSpikes)

        else:
            for k, (sample, spike) in enumerate(zip(unitSamples,unitSpikes)):
                addedSpikes[unit].append(sum((sample > addedSpikeWindow[0]) & (sample <= addedSpikeWindow[1])) -
                               sum((sample > compareWindow[0]) & (sample <= compareWindow[1])))
    return(addedSpikes)


def calcXYlims(file):
    temp = scipy.io.loadmat(file,variable_names=['offsetX','offsetY','edgeLength'])
    offsetX = temp['offsetX'][0][0]
    offsetY = temp['offsetY'][0][0]
    edgeLength = temp['edgeLength'][0][0]

    return (offsetX - edgeLength/2,offsetX+edgeLength/2),(offsetY-edgeLength/2,offsetY+edgeLength/2)
