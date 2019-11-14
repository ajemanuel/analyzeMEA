### functions regarding indentOnGrid

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.io
import analyzeMEA.rastPSTH

def plotActualPositions(filename, setup='alan', center=True, labelPositions=True, save=False):
    """
    Plot locations of grid indentation.

    inputs:
        filename - str, file containing indentOnGrid output
        setup - str, specifies which setup used, specified because some x and y stages are transposed
            current options: 'alan'
        center - boolean, specify whether to center grid on 0,0
        labelPositions - boolean, label order of the positions with text annotations
        save - boolean, whether to save figure as image to current path

    No output, generates plots.
    """
    gridIndent = scipy.io.loadmat(filename)
    try:
        gridPosActual = gridIndent['grid_positions_actual']
    except KeyError:
        print('File not from indentOnGrid')
        return -1
    gridPosActual = np.transpose(gridPosActual)
    # plotting
    if gridIndent['num_repetitions'][0][0] > 1:
        gridPosActual = gridPosActual[0] ## take only the first round for now

    if setup == 'alan':  # displays the points so that they match the orientation of the image.
        xmultiplier = 1  ## my stage is not transposed in x
        ymultiplier = -1  ## my stage is transposed in y
        if center:
            xOffset = -np.median(gridPosActual[0])
            yOffset = np.median(gridPosActual[1])
        else:
            xOffset = 0
            yOffset = 0
    else:
        xmultiplier = 1
        ymultiplier = 1
        if center:
            xOffset = -((np.median(gridPosActual[0])))
            yOffset = -((np.median(gridPosActual[1])))
        else:
            xOffset = 0
            yOffset = 0

    f0 = plt.figure(figsize=(8,8))
    a0 = plt.axes()
    if setup == 'alan':
        a0.scatter(gridPosActual[0]*xmultiplier+xOffset,gridPosActual[1]*ymultiplier+yOffset,s=1500,marker='.')
        if labelPositions:
            for i,pos in enumerate(np.transpose(gridPosActual)):
                #print(pos)
                a0.annotate(str(i+1),(pos[0]*xmultiplier+xOffset,pos[1]*ymultiplier+yOffset),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white',
                    weight='bold')
    else:
        a0.scatter(gridPosActual[0]*xmultiplier + xOffset, gridPosActual[1]*ymultiplier+yOffset,s=1500,marker='.')
        if labelPositions:
            for i,pos in enumerate(np.transpose(gridPosActual)):
                #print(pos)
                a0.annotate(str(i+1),(pos[0]*xmultiplier+xOffset,pos[1]*ymultiplier+yOffset),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white',
                    weight='bold')

    a0.set_ylabel('mm')
    a0.set_xlabel('mm')
    a0.set_aspect('equal')
    if save:
        f0.savefig('gridPositions.png')

def plotGridPSTHs(filename, bs_window, samples, spikes,
                    binSize=0.02,goodSteps=None, units=None, numRepeats=None, numSteps=1,
                    sampleRate=20000,plot=True, xscale = 1, yscale = 1, save=False, force=0, center=True,
                    saveString='',duration=1):
    """
    Plots multiunit mechanical spatial receptive field as a PSTH at each grid location
    and provides output allowing for plotting individual units.
    Inputs:
    filename - str; .mat filename produced by indentOnGrid
    bs_window - sequence, len 2; start and stop of baseline window for PSTH
    samples - sequence; list of samples at which spikes are detected for each sweep
    spikes - sequence; list of spike IDs corresponding to samples in goodsamples_sweeps
    binSize - float; size of PSTh bins (in s)
    goodSteps - None or sequence; list of steps to be included
    units - None or str; sequence of units to plot, if None, calculated by finding unique value in spikes
    sampleRate - int; sample rate in Hz, defaults to 20000
    saveString - string; string to add to filename, default ''

    Outputs:
    outdict -dictionary; ...
    """

    gridIndent = scipy.io.loadmat(filename)
    if numRepeats is None:
        numRepeats = int(gridIndent['num_repetitions'])
    try:
        gridPosActual = gridIndent['grid_positions_actual'] #
        gridPosActual = np.transpose(gridPosActual)
        numPositions = gridPosActual.shape[-1]
        if gridIndent['num_repetitions'] > 1:
            gridPosActual = gridPosActual[0] # taking the first grid positions here -- perhaps change this in the future
    except KeyError:
        print('File not from indentOnGrid')
        return -1
    if center:
        gridPosActual[0] = gridPosActual[0] - np.mean(gridPosActual[0])
        gridPosActual[1] = gridPosActual[1] - np.mean(gridPosActual[1])

    ### generating positionIndices, which will be used to index the samples/spikes for PSTH generation
    positionIndices = np.zeros((numPositions,numSteps*numRepeats))
    for pos in range(numPositions):
        for repeat in range(numRepeats):
            positionIndices[pos,int(numSteps*repeat):int(numSteps*(repeat+1))] = np.arange(pos*numSteps+(repeat*numSteps*numPositions),pos*numSteps+numSteps+(repeat*numSteps*numPositions))
    positionIndices = np.int32(positionIndices)


    outDict = gridIndent # save all variables from the grid file
    numX = int((gridIndent['max_x'] - gridIndent['min_x'])/gridIndent['grid_spacing'])+1
    numY = int((gridIndent['max_y'] - gridIndent['min_y'])/gridIndent['grid_spacing'])+1
    ind = calculatePositionIndex(filename) # ind is a an index that can reorder the position responses for making a matrix
    PSTHs = {}
    if units is None:
        units = np.unique(np.concatenate(spikes))
    outDict['units'] = units
    for i, position in enumerate(positionIndices):
        PSTHs[i] = analyzeMEA.rastPSTH.makeSweepPSTH(binSize,[samples[n] for n in position],[spikes[n] for n in position],
        units=units,bs_window=bs_window,duration=duration,sample_rate=sampleRate)
    outDict['PSTHs'] = PSTHs
    if plot:
        plt.figure(figsize=[6,6])
        ax = plt.axes()
        for i, position in enumerate(np.transpose(gridPosActual)):
            if duration == 1:
                ax.plot(np.array([-0.25,0.25])*xscale+position[0], np.array([0,0])+position[1],color='gray',linewidth=2)
            tempPSTH = np.sum(PSTHs[i]['psths_bs'],axis=1)
            #tempPSTH = tempPSTH - np.mean(tempPSTH[:int(0.25/binSize)])
            ax.plot((PSTHs[i]['xaxis']-duration/2)*xscale+position[0],tempPSTH*yscale+position[1],color='k',linewidth=1)
        scaleBarX = ax.plot(np.max(gridPosActual[0])+0.6+np.array([0,1])*0.25,np.array((np.mean(gridPosActual[1]),np.mean(gridPosActual[1]))),color='k',linewidth=4,scalex=False,scaley=False)
        scaleBarY = ax.plot(np.max(gridPosActual[0])+0.6+np.array([0,0]),np.array((np.mean(gridPosActual[1]),np.mean(gridPosActual[1]+1)*0.5)),color='k',linewidth=4,scalex=False,scaley=False)
        scaleBarTextX = ax.text(np.max(gridPosActual[0])+0.6+0.125,np.mean(gridPosActual[1])-.1,'{0:0.2f} s'.format(0.25/xscale), horizontalalignment='center',verticalalignment='top')
        scaleBarTextY = ax.text(np.max(gridPosActual[0])+0.6+0.125,np.mean(gridPosActual[1])+0.25,'{0:0.1f} spikes/step'.format(0.5/yscale), horizontalalignment='left',verticalalignment='center')
        scaleBarX[0].set_clip_on(False)
        scaleBarY[0].set_clip_on(False)
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        ax.set_title('Multiunit RF, {} mN'.format(force))
        ax.set_aspect(1)
        if save:
            plt.savefig('GridPSTH_{0}mN_{1}.pdf'.format(force, saveString),bbox_inches='tight',bbox_extra_artists=[scaleBarTextX,scaleBarTextY],transparent=True,dpi=300)
    return outDict


def plotGridResponses(filename, window, bs_window, samples, spikes,
                        goodSteps=None, units='all', numRepeats=None, numSteps=1, sampleRate=20000,
                        save=False, force=0, center=True, setup='alan',
                        doShuffle=False, numShuffles=10000, size=300, saveString=''):
    """
    Plots each unit's mechanical spatial receptive field.
    Inputs:
    filename - str; .mat filename produced by indentOnGrid
    window - sequence, len 2; start and stop of window of interest
    bs_window - sequence, len 2; start and stop of baseline window
    samples - sequence; list of samples at which spikes are detected for each sweep
    spikes - sequence; list of spike IDs corresponding to samples in goodsamples_sweeps
    goodSteps - None or sequence; list of steps to be included
    units - sequence or str; sequence of units to plot or str = 'all'
    sampleRate - int; sample rate in Hz, defaults to 20000
    size - value; argument for scatter plot size of point
    saveString - string; string to add to filename, default ''
    Output is a plot.
    """
    if abs((window[1]-window[0]) - (bs_window[1] - bs_window[0])) > 1e-8: # requires some tolerance for float encoding; could also use np.isclose()
        print('Warning: Window and baseline are not same size.')


    gridIndent = scipy.io.loadmat(filename)
    if numRepeats is None:
        numRepeats = int(gridIndent['num_repetitions'])
    try:
        gridPosActual = gridIndent['grid_positions_actual'] #
        gridPosActual = np.transpose(gridPosActual)
        if gridIndent['num_repetitions'] > 1:
            gridPosActual = gridPosActual[0] # taking the first grid positions here -- perhaps change this in the future
    except KeyError:
        print('File not from indentOnGrid')
        return -1


    gridSpikes = extractSpikesInWindow(window, samples, spikes, sampleRate=sampleRate)
    gridSpikesBS = extractSpikesInWindow(bs_window, samples, spikes, sampleRate=sampleRate)
    outDict = gridIndent # save all variables from the grid file
    numX = int((gridIndent['max_x'] - gridIndent['min_x'])/gridIndent['grid_spacing'])+1
    numY = int((gridIndent['max_y'] - gridIndent['min_y'])/gridIndent['grid_spacing'])+1
    ind = []
    for position in gridIndent['grid_positions']:
        for i, position2 in enumerate(gridIndent['grid_positions_rand']):
            if np.all(position == position2):
                ind.append(i) ## ind is a an index that can reorder the position responses for making a matrix
    if type(units) is not str: # units != 'all'
        outDict['units'] = units
        for unit in units:
            outDict[unit] = {}
            positionResponses, positionResponsesShuffle, numGoodPositions = generatePositionResponses(gridPosActual, gridSpikes,
                                                                                                        numRepeats=numRepeats, numSteps=numSteps,
                                                                                                        unit=unit, goodSteps=goodSteps,
                                                                                                        doShuffle=doShuffle, numShuffles=numShuffles)

            positionResponses_baseline, positionResponseShuffle_baseline = generatePositionResponses(gridPosActual, gridSpikesBS, numRepeats=numRepeats,
                                                                                                        numSteps=numSteps, unit=unit, goodSteps=goodSteps,
                                                                                                        doShuffle=doShuffle, numShuffles=numShuffles)[:2]

            positionResponsesBS = {}
            for index in positionResponses:
                positionResponsesBS[index] = positionResponses[index] - positionResponses_baseline[index] ## subtract spikes in baseline window from # spikes in response window

            plotPositionResponses(positionResponsesBS, gridPosActual, force=force, save=save, saveString=saveString, unit=unit, center=center, setup=setup, size=size) ## edit plotPositionResponses

            if doShuffle:
                positionResponsesBS_shuffles = {}
                for index in positionResponsesShuffle:
                    positionResponsesBS_shuffles[index] = positionResponsesShuffle[index] - positionResponseShuffle_baseline[index] ## subtract baselines
                pValues = {}
                for index in positionResponsesBS:
                    pValues[index] = (np.sum(np.abs(positionResponsesBS_shuffles[index]) >= np.abs(positionResponsesBS[index]))+1)/numShuffles
                plotPositionResponses(pValues, gridPosActual, force=force, save=save, saveString=saveString, unit=unit, center=center, setup=setup, pValues=True,size=size)
                outDict[unit]['pValue'] = pValues
            outDict[unit]['positionResponsesBS'] = positionResponsesBS
            posResp = []
            for index in positionResponsesBS:
                posResp.append([index, positionResponsesBS[index]])
            posResp.sort(key=lambda pos: pos[0]) ## sort by position
            matrix = np.reshape(np.transpose(posResp)[1][ind],[numX,numY])
            outDict[unit]['matrix'] = np.transpose(matrix) ## this empirically matches the plotted output for my grids
    else:
        positionResponses, positionResponsesShuffle, numGoodPositions = generatePositionResponses(gridPosActual, gridSpikes, numRepeats=numRepeats, numSteps=numSteps, goodSteps=goodSteps,
                                                                                                    doShuffle=doShuffle, numShuffles=numShuffles)
        positionResponses_baseline, positionResponseShuffle_baseline = generatePositionResponses(gridPosActual, gridSpikesBS, numRepeats=numRepeats, numSteps=numSteps, goodSteps=goodSteps,
                                                                                                    doShuffle=doShuffle, numShuffles=numShuffles)[:2]
        positionResponsesBS = {}
        for index in positionResponses:
            positionResponsesBS[index] = positionResponses[index] - positionResponses_baseline[index] ## subtract spikes in baseline window from spikes in response window

        plotPositionResponses(positionResponsesBS, gridPosActual, force=force, save=save, saveString=saveString, center=center, setup=setup, size=size)
        outDict['all'] = {}
        outDict['all']['positionResponsesBS'] = positionResponsesBS

        posResp = []
        for index in positionResponsesBS:
            posResp.append([index, positionResponsesBS[index]])
        posResp.sort(key=lambda pos: pos[0]) ## sort by position
        matrix = np.reshape(np.transpose(posResp)[1][ind],[numX,numY])

        outDict['all']['matrix'] = np.transpose(matrix) ## this empirically matches the plotted output for my grids



    return outDict

def extractSpikesInWindow(window, samples, spikes, sampleRate=20000):
    """
    Inputs:
    window = sequence, len 2; start and stop of window in s
    samples = sequence; list of samples at which spikes are detected for each sweep
    spikes = sequence; list of spike IDs corresponding to samples in goodsamples_sweeps
    sampleRate = int; sample rate in Hz, defaults to 20000

    Returns:
    spikesOut - list of spikes in that window for each sweep

    """
    windowOnsetinSamples = window[0]*sampleRate # in samples
    windowDurinSamples =  (window[1]-window[0])*sampleRate # in samples
    spikesOut = []
    i = 0
    for spikeSample, spike in zip(samples,spikes):
        i += 1
        spikesOut.append((spikeSample[(spikeSample > windowOnsetinSamples) & (spikeSample < windowOnsetinSamples + windowDurinSamples)],
                         spike[(spikeSample > windowOnsetinSamples) &  (spikeSample < windowOnsetinSamples + windowDurinSamples)]))
        #plt.plot(spikeSample[(spikeSample > windowOnsetinSamples) & (spikeSample < windowOnsetinSamples + windowDurinSamples)],
        #         spike[(spikeSample > windowOnsetinSamples) & (spikeSample < windowOnsetinSamples + windowDurinSamples)],'|')
    return spikesOut

def generatePositionResponses(gridPosActual, spikes, numRepeats=3, numSteps=1, unit=None, goodSteps=None, doShuffle=True, numShuffles=10000):
    """
    Calculate the number of spikes belonging to each unit (or all units) evoked at each position. Also generate shuffled versions of these responses (for statistical analysis).

    Inputs:
        gridPosActual - ndarray, sequence of positions from gridIndent*[0-9].mat file generated during experiment
        spikes - list of spikes at each position
        numRepeats - int,  # of times the whole grid was repeated
        numSteps - int, # of times a step was repeated at each position of the grid
        shuffle - bool, if True, will shuffle and return positionResponseShuffle
        numShuffles - int, # of times to shuffle the positions

    Outpus:
        positionResponse - dict, keys refer to position indices, int values are # of spikes per good step
        positionResponseShuffle - dict, keys refer to position indices, ndarray values are arrays, len(numShuffles) of # of spikes per good step at each shuffle position
            (none if doShuffle == False)
        numGoodPositions - dict, keys refer to position indices, int values are # of steps included in that position
    """



    gridPosActualAll = np.transpose(gridPosActual)
    if numRepeats > 1:
        gridPosActualAll = np.matlib.repmat(gridPosActualAll,numRepeats,1)
    else:
        gridPosActualAll = np.array(gridPosActualAll)

    positionIndex = np.arange(len(np.transpose(gridPosActual)))
    positionIndex = np.matlib.repmat(positionIndex,numSteps,numRepeats)

    if numSteps > 1:
        positionIndex = np.transpose(positionIndex)
        positionIndex = positionIndex.reshape(positionIndex.shape[0]*positionIndex.shape[1]) # linearize
    if goodSteps is None:
        goodSteps = np.ones(len(spikes)) ## all steps included
    if not len(spikes) == len(positionIndex):
        print('Incorrect # of steps')
        print('len(spikes) = '+str(len(spikes)))
        print('len(positionIndex) = '+str(len(positionIndex)))
    positionResponse = {}
    numGoodPositions = {}

    ## Extracting Actual Responses
    if unit:
        for sweep, index, good in zip(spikes,positionIndex,goodSteps):
            if good:
                positionResponse[index] = positionResponse.get(index,0) + len(sweep[1][sweep[1]==unit])
                if index in numGoodPositions:
                    numGoodPositions[index] += 1
                else:
                    numGoodPositions[index] = 1
    else:
        for sweep, index, good in zip(spikes, positionIndex, goodSteps):
            if good:
                positionResponse[index] = positionResponse.get(index,0) + len(sweep[1])
                if index in numGoodPositions:
                    numGoodPositions[index] += 1
                else:
                    numGoodPositions[index] = 1
    for index in positionResponse:
        positionResponse[index] = positionResponse[index]/numGoodPositions[index]


    ## Extracting Shuffled Responses

    positionResponseShuffle  = {}
    np.random.seed(20180407) # for replicability
    if doShuffle:
        if len(positionIndex) < 20:
            if numShuffles > np.math.factorial(len(positionIndex)): ## unlikely, but just in case used a small grid
                numShuffles = np.math.factorial(len(positionIndex))
                print('numShuffles > possible permutations; assigning numShuffles to '+str(numShuffles))

        for shuffle in range(numShuffles):
            positionIndexShuffle = np.random.permutation(positionIndex)

            tempResponse = {}
            tempGoodPositions = {}

            if unit:
                for sweep, index, good in zip(spikes, positionIndexShuffle, goodSteps):
                    if good:
                        tempResponse[index] = positionResponse.get(index,0) + len(sweep[1][sweep[1]==unit])
                        tempGoodPositions[index] = tempGoodPositions.get(index,1) + 1
            else:
                for sweep, index, good in zip(spikes, positionIndexShuffle, goodSteps):
                    if good:
                        tempResponse[index] = positionResponse.get(index,0) + len(sweep[1])
                        tempGoodPositions[index] = tempGoodPositions.get(index,1) + 1

            if shuffle == 0:
                for index in tempResponse:
                    positionResponseShuffle[index] = [tempResponse[index]/tempGoodPositions[index]] ## making lists so I can append when shuffle not 0
            else:
                for index in positionResponseShuffle:
                    positionResponseShuffle[index].append(tempResponse[index]/tempGoodPositions[index])

        for index in positionResponseShuffle:
            positionResponseShuffle[index] = np.array(positionResponseShuffle[index])  ## making into ndarrays
    else:
        positionResponseShuffle = None
    return positionResponse, positionResponseShuffle, numGoodPositions

def plotPositionResponses(positionResponses, gridPosActual, force=0, size=300, save=False, saveString='', unit=None, setup='alan', center=True, pValues=False):
    """
    plotting function for spatial receptive fields

    Inputs:
    positionResponses - dict, from generatePositionResponses
    force - int, in mN, for titling and savename of graph
    saveString - string, for filename saving, default ''
    Output: plot
    f0 is the plot handle
    """
    if setup == 'alan': # my axes are transposed
        xmultiplier = 1
        ymultiplier = -1
        if center:
            xOffset = -int(round(np.median(gridPosActual[0])))
            #print('xOffset = {0}'.format(xOffset))
            yOffset = int(round(np.median(gridPosActual[1])))
            #print('yOffset = {0}'.format(yOffset))
        else:
            xOffset, yOffset = (0, 0)
    else:
        xmultiplier = 1
        ymultiplier = 1
        if center:
            xOffset = -np.median(gridPosActual[0])
            #print('xOffset = {0}'.format(xOffset))
            yOffset = -np.median(gridPosActual[1])
            #print('yOffset = {0}'.format(yOffset))
        else:
            xOffset, yOffset = (0, 0)

    positionResponse = []
    for index in positionResponses:
        positionResponse.append([index, positionResponses[index]])
    positionResponse.sort(key=lambda pos: pos[0]) ## sort by position

    minSpikes = min(np.transpose(positionResponse)[1])
    maxSpikes = max(np.transpose(positionResponse)[1])
    if abs(minSpikes) > abs(maxSpikes):
        absMax = abs(minSpikes)
    else:
        absMax = abs(maxSpikes)

    f0 = plt.figure(figsize=(6,6))
    a0 = plt.axes()
    if pValues: # plotting pValues rather than actual response
        sc = a0.scatter(gridPosActual[0][:len(positionResponse)]*xmultiplier+xOffset,gridPosActual[1][:len(positionResponse)]*ymultiplier+yOffset,c=np.transpose(np.log10(positionResponse))[1], s=size, cmap='viridis_r')
    else:
        sc = a0.scatter(gridPosActual[0][:len(positionResponse)]*xmultiplier+xOffset,gridPosActual[1][:len(positionResponse)]*ymultiplier+yOffset,c=np.transpose(positionResponse)[1], s=size, cmap='bwr', vmin=-absMax,vmax=absMax)
    a0.set_aspect('equal')
    a0.set_xlabel('mm')
    a0.set_ylabel('mm')
    if unit:
        a0.set_title('Unit %d, %d mN'%(unit, force))
    else:
        a0.set_title('{0} mN'.format(force))
    cb = f0.colorbar(sc,fraction=0.1,shrink=.5)
    if pValues:
        cb.set_label('log(p)')
    else:
        cb.set_label(r'$\Delta$ spikes per step')
    f0.tight_layout()
    if save:
        if pValues:
            plt.savefig('positionPVALUE_unit{0}_{1}mN{2}.png'.format(unit, force, saveString),transparent=True)
        else:
            plt.savefig('positionResponse_unit{0}_{1}mN{2}.png'.format(unit, force, saveString),transparent=True)
    plt.show()
    plt.close()

def calculatePositionIndex(gridFile):
    """
    returns the index needed to reorder positions so that they're suitable for a matrix plot

    Input:
    gridFile - str; .mat filename produced by indentOnGrid
    Output:
    index - np.array, len = numLocations
    """
    gridIndent = scipy.io.loadmat(gridFile)
    index = []
    for position in gridIndent['grid_positions']:
        for i, position2 in enumerate(gridIndent['grid_positions_rand']):
            if np.all(position == position2):
                index.append(i)
    return  np.int32(index)

def generateSingleSweepPSTH(gridFile, samples, spikes, numSteps=8, numRepeats=2, units=None, bin_size=0.01):
    """
    generate a single-sweep PSTH for grid analysis
    inputs:
        gridFile - str; .mat filename produced by indentOnGrid
        samples - list, spike times for each step
        spikes - list, spike unit ID for each step
        numSteps - int, number of steps used to stimulate at each location
        numRepeats - int, number of times the entire grid is repeated
        units - None or np.array, units to include in PSTH output (usually use np.unique(goodSpikes)[spikeDict['depthIndices']]))
        bin_size - float, bin size of PSTH in s
    outputs:
        dictionary, with keys:
            PSTH - np.array - M locations x N trials x O bins x P units
            bin_size - float, bin size of PSTH in s
            locations - np.array - M locations x N coordinates for locations included in PSTH
            units - np.array - unit identities
    """
    if units is None:
        units = np.unique(np.concatenate(spikes))
    outDict = {}
    outDict['units'] = units
    outDict['bin_size'] = bin_size
    gridIndent = scipy.io.loadmat(gridFile)
    outDict['locations'] = gridIndent['grid_positions']
    ind = calculatePositionIndex(gridFile)
    tempPSTH = analyzeMEA.rastPSTH.singleSweepPSTH(bin_size,samples,spikes,units=units,rate=False)
    numUnits = len(units)
    PSTH = np.zeros([len(ind), numSteps*numRepeats, tempPSTH['psths'].shape[0],numUnits])
    for stimLocation in ind:
        for unit in range(numUnits):
            for repeat in range(numRepeats):
                initialStep = repeat * len(ind) * numSteps + numSteps * stimLocation
                endStep = initialStep + numSteps
                PSTH[stimLocation,int(repeat*numSteps):int((repeat+1)*numSteps),:,unit] = np.transpose(tempPSTH['psths'][:,initialStep:endStep,unit])
    outDict['PSTH'] = np.int16(PSTH)
    return outDict
