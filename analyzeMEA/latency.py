import numpy as np

def calculateLatencyParameters(eventSamples, baselinePeriod, samples, spikes, units=None, sampleRate=20000, numShuffles=100,
                                JSwindow=[0,10,0.25],resetRandSeed=True):
    """
    Calculating latencies with distribution of first spikes following onset of stimulus
    Inputs:
        eventSamples - sequence; time (in samples) at which events start
        baselinePeriod - sequence, len=2 of np.int64; beginning and end of baseline period (in samples)
            - alternatively a sequence of sequences, each sequence with a beginning and end for a baseline epoch
        samples - sequence; samples at which spike fires
        spikes - sequence; unit firing spike at time corresponding to the same item in the samples sequence
        units - sequence; units to include in analysis
        numShuffles - int; number of times to calculate baseline latencies
        JSwindow - sequence; first two items are the time window (ms)  to consider for calculating Jensen-Shannon divergences, the last is the size of the bins
        resetRandSeed - boolean; whether or not to reset the random seed prior to generating baseline samples
    Outputs:
        Dictionary (outDict) containing the following keys
        latencies - ndarray; M units x N events latency array in seconds
        latenciesBaseline - ndarray; M units x N shuffles x O baseline events latency array
        mean - ndarray; mean latency for each unit (M)
        meanBaseline - ndarray; mean baseline latency for each unit (M) for each shuffle (N)
        stdev - ndarray; stdev of latency distribution for each unit (M)
        stdevBaseline - ndarray;  stdev of baseline latencies for each unit (M) for each shuffle (N)
        median - ndarray; median latency for each unit (M)
        medianBaseline - ndarray; median baseline latency for each unit (M) for each shuffle (N)
        units - same as input, or if None, = np.unique(spikes)
    Written by AE 9/26/18
    updated to include baseline latencies 11/27/18
    updated to include Jensen-Shannon divergence calculations 11/28/18 (modified from Kepecs lab matlab code)
    updated to include shuffles of baseline latencies and p-value calculations from distance metric 12/4/18
    """
    if units is None:
        units = np.unique(spikes)
    outDict = {}
    outDict['units'] = units
    latencies = np.zeros([len(units),len(eventSamples)])
    JSwindow_s = np.array(JSwindow)/1000.0
    print('Calculating Event Latencies')
    for i, unit in enumerate(units):
        print('unit '+str(unit))
        unitSamples = samples[spikes == unit]
        for j, sample in enumerate(eventSamples):
            try:
                latencies[i,j] = unitSamples[np.searchsorted(unitSamples,sample)] - sample ## take first spike fired by unit after eventSample
            except IndexError: ## encounter IndexError if there is no spike after eventSample that matches
                latencies[i,j] = np.nan


    print('Generating Baseline Samples')
    if resetRandSeed:
        np.random.seed(20181204)  # set random seed for reproducibility
    baselineSamples = np.zeros((numShuffles,len(eventSamples))) ## pre-allocating matrix for baseline samples
    for shuffle in range(numShuffles):
        if isinstance(baselinePeriod[0],np.int64): ## if only one baseline epoch
            temp = np.random.rand(len(eventSamples)) # matching # of events for baseline and stimulus-evoked samples
            temp *= (baselinePeriod[1] - baselinePeriod[0])
            temp += baselinePeriod[0]
            baselineSamples[shuffle,:] = np.int32(temp)
        elif len(baselinePeriod[0]) == 2: ## if multiple baseline epochs
            temp2=[]
            for epoch in baselinePeriod:
                temp = np.random.rand(len(eventSamples)/len(baselinePeriod)) # matching # of events for baseline and stimulus-evoked samples
                temp *= (epoch[1] - epoch[0]) # scaling to epoch
                temp += epoch[0] # adjusting start
                temp = np.int32(temp) # integers that correspond to samples
                temp2.append(temp)
            baselineSamples[shuffle,:] = np.concatenate(temp2)
        else:
            print('Baseline period incorrectly formatted, try again.')
            return -1


    print('Calculating Baseline Latencies')
    latenciesBaseline = np.zeros([len(units),numShuffles,len(eventSamples)])

    for i, unit in enumerate(units):
        print('unit '+str(unit))
        unitSamples = samples[spikes == unit]
        for shuffle in range(numShuffles):
            for j, sample in enumerate(baselineSamples[shuffle,:]):
                try:
                    latenciesBaseline[i,shuffle,j] = unitSamples[np.searchsorted(unitSamples,sample)] - sample
                except IndexError:
                     latenciesBaseline[i,shuffle,j] = np.nan
    JSdivergences = np.zeros((len(units),numShuffles+1,numShuffles+1))
    JSdivergences.fill(np.nan)
    histBins = np.arange(JSwindow_s[0],JSwindow_s[1],JSwindow_s[2])

    for i in range(len(units)):

        test = latencies[i,:]
        testHist = np.histogram(test[~np.isnan(test)]/sampleRate,bins=histBins,density=False)[0]#/sum((test > 0.0005 ) & (test < 0.02))
        testHist = testHist / sum((test[~np.isnan(test)]/sampleRate >= JSwindow_s[0]) & (test[~np.isnan(test)]/sampleRate <= JSwindow_s[1]))

        allHists = np.zeros((len(histBins)-1,numShuffles+1))
        for shuffle in range(numShuffles):
            baseline = latenciesBaseline[i,shuffle,:]
            baselineHist = np.histogram(baseline[~np.isnan(baseline)]/sampleRate,bins=histBins,density=False)[0]#/sum((baseline > 0.0005) & (baseline < 0.02))
            baselineHist = baselineHist / sum((baseline[~np.isnan(baseline)]/sampleRate >= JSwindow_s[0]) & (baseline[~np.isnan(baseline)]/sampleRate <= JSwindow_s[1]))
            allHists[:,shuffle] = baselineHist
        allHists[:,-1] = testHist

        for k1 in range(numShuffles+1):
            D1 = allHists[:,k1]
            for k2 in np.arange(k1+1,numShuffles+1):
                D2 = allHists[:,k2]
                JSdivergences[i,k1,k2] = np.sqrt(JSdiv(D1,D2)) ##  Kepecs lab code was equivalent to  np.sqrt(JSdiv(D1,D2)*2) , unsure why *2 multiplier included

    pValues = np.zeros(len(units))
    Idiffs = np.zeros(len(units))

    for unit in range(len(units)):
        pValues[unit], Idiffs[unit] = makep(JSdivergences[unit,:,:],numShuffles+1)

    outDict['latencies'] = latencies/sampleRate ## in s
    outDict['latenciesBaseline'] = latenciesBaseline/sampleRate ## in s
    outDict['mean'] = np.nanmean(outDict['latencies'],axis=1)
    outDict['meanBaseline'] = np.nanmean(outDict['latenciesBaseline'],axis=2)
    outDict['median'] = np.nanmedian(outDict['latencies'],axis=1)
    outDict['medianBaseline'] = np.nanmedian(outDict['latenciesBaseline'],axis=2)
    outDict['stdev'] = np.nanstd(outDict['latencies'],axis=1)
    outDict['stdevBaseline'] = np.nanstd(outDict['latenciesBaseline'],axis=2)
    outDict['JSdivergences'] = JSdivergences
    outDict['pValues'] = pValues
    outDict['Idiffs'] = Idiffs
    return outDict

def makep(kld,kn):
    """
    Calculates a p-value from distance matrix
    modified from Kepecs lab matlab code
    """
    pnhk = kld[:kn-1,:kn-1]
    nullhypkld = pnhk[~np.isnan(pnhk)]
    testkld = np.nanmedian(kld[:,-1])
    sno = len(nullhypkld[:])
    p_value = sum(nullhypkld[:] >= testkld) / sno
    Idiff = testkld - np.median(nullhypkld)
    return p_value, Idiff


def JSdiv(P,Q):
    """
    JSDIV Jensen-Shannon divergence
    D = JSDIV(P,1) calculates the Jensen-Shannon divergence of the two input distributions.

    modified from Kepecs lab matlab code
    """

    if P.shape != Q.shape:
        print('P and Q have different shapes')
    ## normalizing P and Q:

    P = P/np.sum(P,axis=0)
    Q = Q/np.sum(Q,axis=0)


    M = (P + Q) /2.0
    D1 = KLdist(P,M)
    D2 = KLdist(Q,M)
    D = (D1+D2) / 2
    return D

def KLdist(P,Q):
    """
    KLDIST   Kullbach-Leibler distance.
    D = KLDIST(P,Q) calculates the Kullbach-Leibler distance (information
    divergence) of the two input distributions.
    """
    P2 = P[P*Q>0]
    Q2 = Q[P*Q>0]
    P2 = P2 / np.sum(P2)
    Q2 = Q2 / np.sum(Q2)

    D = np.sum(P2*np.log(P2/Q2))

    return D


def calculateLatencyParametersSweeps(eventSample, samples_sweeps, spikes_sweeps, units=None, sampleRate=20000):
    """
    Calculating latencies with distribution of first spikes following onset of stimulus
    Inputs:
        eventSample - int; time (in samples) at which event start
        samples_sweeps - sequence; lists of samples at which spikes fires
        spikes_sweeps - sequence; lists of  unit firing spike at time corresponding to the same item in the samples sequence
        units - sequence; units to include in analysis
    Outputs:
        Dictionary (outDict) containing the following keys
        latencies - sequence of sequences; lists of latencies for each unit
        mean - sequence; mean latency for each unit
        stdev - sequence; stdev of latency distribution for each unit
        median - sequence; median latency for each unit
        units - same as input, or if None, = np.unique(spikes)
    Written by AE 9/26/18
    """
    if units is None:
        units = np.unique(spikes)
    outDict = {}
    outDict['units'] = units
    latencies = np.zeros([len(units),len(samples_sweeps)])
    for i, unit in enumerate(units):
        for j, (samples, spikes) in enumerate(zip(samples_sweeps, spikes_sweeps)):
            try:
                latencies[i,j] = (samples[(samples > eventSample) & (spikes == unit)][0] - eventSample)/sampleRate ## take first spike fired by unit after eventSample
            except(IndexError): ## occurs if the unit doesn't spike between eventSample and end
                latencies[i,j] = np.nan
    outDict['latencies'] = latencies
    outDict['mean'] = np.nanmean(latencies,axis=1)
    outDict['median'] = np.nanmedian(latencies,axis=1)
    outDict['stdev'] = np.nanstd(latencies,axis=1)
    return outDict



def determineThresholdCrossings(latencyDict, alpha=0.001):
    """
    Calculate threshold crossings for all units in latencyDict

    Inputs:
        latencyDict - dictionary, output of calculateLatencyParameters
        alpha - float, significance level

    Outputs:
        latenciesAboveThreshold - ndarray, latencies of crossing upper CI, in order of latencyDict['units']
        latenciesBelowThreshold - ndarray, latencies of crossing lower CI
    """

    latenciesAboveThreshold = []
    latenciesBelowThreshold = []

    for unitInd, unit in enumerate(latencyDict['units']):
        temp = determineThresholdCrossing(latencyDict['latencies'][unitInd,:],latencyDict['latenciesBaseline'][unitInd,:,:], alpha=alpha)
        print('Unit {0}, {1:0.3f}; {2:0.3f}'.format(unit,temp[0],temp[1]))
        latenciesAboveThreshold.append(temp[0])
        latenciesBelowThreshold.append(temp[1])

    latenciesAboveThreshold = np.array(latenciesAboveThreshold)
    latenciesBelowThreshold = np.array(latenciesBelowThreshold)

    return latenciesAboveThreshold, latenciesBelowThreshold


def determineThresholdCrossing(latencies, baselineLatencies, alpha=0.001):
    """
    Determines when the latencies cross the significance level designated by alpha.
    Confidence intervales of the baseline latency distribution are determined by applying the Dvoretzky-Kiefer-Wolfowitz inequality.

    Inputs:
        latencies - np.array, actual latencies
        baselineLatencies - np.array, shuffled latencies (if multidimensional, will be reshaped)

    Outputs:
        latencyAboveThreshold - float, latency when the lower confidence interval is crossed (e.g., larger fraction than expected by baseline distribution)
        latencyBelowThreshold - float, latency when the upper confidence interval is crossed (e.g., smaller fraction than expected by baseline distribution)
            either output will be -1 when the given interval is not crossed

    Written by AE 4/16/2019
    """
    baselineLatencies = np.sort(np.reshape(baselineLatencies,-1)) ## making sure the array is sorted and in one dimension
    numSamples = len(baselineLatencies)
    lower = np.array(baselineLatencies - epsilon(numSamples, alpha))
    upper = np.array(baselineLatencies + epsilon(numSamples,alpha))

    significant = 0
    for i, latenc in enumerate(np.sort(latencies)):
        actualFraction = i/len(latencies)
        try:
            lowerFraction = np.where(lower > latenc)[0][0]/len(lower)
        except IndexError:
            if significant == 0:
                latencyAboveThreshold = -1
                significant = 1
                break
        if (actualFraction > lowerFraction) & (significant == 0):
            significant = 1
            latencyAboveThreshold = latenc
            break

    significant = 0 ## resetting significance for testing crossing in the negative direction
    upperFraction = 0
    for i, latenc in enumerate(np.sort(latencies)):
        actualFraction = i/len(latencies)
        try:
            upperFraction = np.where(upper > latenc)[0][0]/len(upper)
        except IndexError:

            if significant == 0:
                latencyBelowThreshold = -1
                significant = 1
                break
        if (actualFraction < upperFraction) & (significant == 0):
            significant = 1
            latencyBelowThreshold = latenc
            break
    if 'latencyAboveThreshold' not in locals():
        latencyAboveThreshold = -1
    if 'latencyBelowThreshold' not in locals():
        latencyBelowThreshold = -1
    return latencyAboveThreshold, latencyBelowThreshold


def epsilon(n, alpha=0.01):
    return np.sqrt(1. / (2. * n) * np.log(2. / alpha))
