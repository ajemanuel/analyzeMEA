import scipy.io
import numpy as np
import re
import glob
import os
import pandas as pd

def importJRCLUST(filepath, annotation='single', depth=250):
    """
    Imports the features of the JrClust output I use most.

    inputs:
        filepath - str with path to S0 filename
        annotation - str that indicates which spikes to include 'single' or 'multi'
            -- in the future, increase this functionality
        depth - int/float, depth of top electrode site for neuronexus_poly2.prb or depth of bottom electrode site for cnt_h4.prb,
                in microns (default 250 microns, my typical insertion depth of neuronexus_poly2 probe tip is 1100 microns)
    output: Dict with keys
        goodSpikes - ndarray of clusters (unit identities of spikes)
        goodSamples - ndarray of spike samples (time of spike)
        sampleRate - int sample rate in Hz
        goodTimes - ndarray of spike times (in s)
        unitPosXY - tuple of two ndarrays, (X center of mass, Y center of mass)
        depthIndices - index of good units in the order of their depth
        depths - depth of site (taking into account depth of probe)
        layers - the cortical layer to which the depth corresponds
        units - list of all units included in goodSpikes
    """
    outDict = {}



    try:
        loadedFile = scipy.io.loadmat(filepath,squeeze_me=True, struct_as_record=False)

        spikeAnnotations = loadedFile['S0'].S_clu.csNote_clu

        try:
            annotatedUnits = np.where(spikeAnnotations == annotation)[0]+1 # +1 to account for 1-indexing of jrclust output; jrc spikes that = 0 are not classified
        except(FutureWarning):
            print('Not all units are annotated (FutureWarning triggered).')
            pass
        goodSamples = loadedFile['S0'].viTime_spk
        goodSpikes = loadedFile['S0'].S_clu.viClu

        goodSamples = goodSamples[np.isin(goodSpikes,annotatedUnits)]
        goodSpikes = goodSpikes[np.isin(goodSpikes,annotatedUnits)]

        outDict['spikeSites'] = loadedFile['S0'].viSite_spk ## list of site identity for all spikes
        outDict['allSamples'] = loadedFile['S0'].viTime_spk ## list of samples for all spikes
        outDict['units'] = np.unique(goodSpikes)
        outDict['sampleRate'] = loadedFile['S0'].P.sRateHz
        outDict['goodSamples'] = goodSamples
        outDict['goodSpikes'] = goodSpikes
        outDict['goodTimes'] = goodSamples/loadedFile['S0'].P.sRateHz
        outDict['unitPosXY'] = (loadedFile['S0'].S_clu.vrPosX_clu[spikeAnnotations == annotation],loadedFile['S0'].S_clu.vrPosY_clu[spikeAnnotations == annotation])
        outDict['depthIndices'] = np.argsort(loadedFile['S0'].S_clu.vrPosY_clu[spikeAnnotations == annotation]) ## to get an index to use for sorting by depth
        outDict['tmrWav_raw_clu'] = np.transpose(loadedFile['S0'].S_clu.tmrWav_raw_clu[:,:,spikeAnnotations == annotation])
        outDict['tmrWav_spk_clu'] = np.transpose(loadedFile['S0'].S_clu.tmrWav_spk_clu[:,:,spikeAnnotations == annotation])
        outDict['Lratio'] = loadedFile['S0'].S_clu.vrLRatio_clu[spikeAnnotations == annotation]
        outDict['IsoDist'] = loadedFile['S0'].S_clu.vrIsoDist_clu[spikeAnnotations == annotation]
        outDict['ISIratio'] = loadedFile['S0'].S_clu.vrIsiRatio_clu[spikeAnnotations == annotation]
        outDict['viSite_clu'] = loadedFile['S0'].S_clu.viSite_clu[spikeAnnotations == annotation] - 1 # subtract 1 for python indexing


        ## calculating trough to peak time
        spikeTroughPeak = []
        for i in range(len(np.unique(goodSpikes))):
            waveform = outDict['tmrWav_raw_clu'][i,outDict['viSite_clu'][i],:] ## extracts the waveform from the best spike
            if loadedFile['S0'].dimm_raw[0] == 81:
                spikeTroughPeak.append(np.where(waveform[22:] == np.max(waveform[22:]))[0][0]) # trough occurs at sample 22 for raw waveforms with 81 samples
            elif loadedFile['S0'].dimm_raw[0] == 41:
                spikeTroughPeak.append(np.where(waveform[12:] == np.max(waveform[12:]))[0][0]) # for raw waveforms with 41 samples, trough occurs at sample 12, finding location of maximum post trough
            else:
                print('Raw waveform dimensions do not match those hard-coded into this function...')
                ## perhaps this is generalizable: np.int(np.ceil(81/4)+1)
                # need to more test cases to be sure
        spikeTroughPeak = np.array(spikeTroughPeak)/outDict['sampleRate'] # convert to s
        outDict['spikeTroughPeak'] = spikeTroughPeak
    except NotImplementedError:
        import h5py ### use the following to import from JRCLUST v4.0
        loadedFile = h5py.File(filepath,'r')
        outDict['spikeSites'] = loadedFile['spikeSites'][:].reshape(-1)
        outDict['allSamples'] = loadedFile['spikeTimes'][:].reshape(-1)

        spikeAnnotations = []
        for column in loadedFile['clusterNotes']:
            row_data = []
            for row_number in range(len(column)):
                row_data.append(''.join(map(chr,loadedFile[column[row_number]][:].reshape(-1))))
            spikeAnnotations.append(row_data[0])
        spikeAnnotations = np.transpose(spikeAnnotations)
        annotatedUnits = np.where(spikeAnnotations == annotation)[0]+1 # +1 to account for 1-indexing of jrclust output; jrc spikes that = 0 are not classified
        goodSamples = loadedFile['spikeTimes'][:].reshape(-1)
        goodSpikes = loadedFile['spikeClusters'][:].reshape(-1)

        goodSamples = goodSamples[np.isin(goodSpikes,annotatedUnits)]
        goodSpikes = goodSpikes[np.isin(goodSpikes,annotatedUnits)]

        outDict['units'] = np.unique(goodSpikes)
        outDict['sampleRate'] = 20000 ## hard coded for now because not specified in new results file
        outDict['goodSamples'] = goodSamples
        outDict['goodSpikes'] = goodSpikes
        outDict['goodTimes'] = goodSamples/20000
        outDict['unitPosXY'] = loadedFile['clusterCentroids'][:,spikeAnnotations=='single'] ## units (rows) by centroidPositions (X then Y)
        outDict['depthIndices'] = np.argsort(loadedFile['clusterCentroids'][:,spikeAnnotations=='single'][1,:]) ## to get an index to use for sorting by depth

        outDict['rawWaveforms'] = np.transpose(loadedFile['meanWfGlobalRaw'][spikeAnnotations == annotation,:,:])
        outDict['filteredWaveforms'] = np.transpose(loadedFile['meanWfGlobal'][spikeAnnotations == annotation,:,:])
        outDict['Lratio'] = loadedFile['unitLRatio'][0,spikeAnnotations == annotation]
        outDict['IsoDist'] = loadedFile['unitIsoDist'][0,spikeAnnotations == annotation]
        outDict['ISIratio'] = loadedFile['unitISIRatio'][0,spikeAnnotations == annotation]
        outDict['viSite_clu'] = np.int32(loadedFile['clusterSites'][spikeAnnotations == annotation,0] - 1) # subtract 1 for python indexing

        spikeTroughPeak = []
        for i in range(len(np.unique(goodSpikes))):
            waveform = outDict['rawWaveforms'][:,outDict['viSite_clu'][i],i] ## extracts the waveform from the best spike
            if len(waveform) == 81:
                spikeTroughPeak.append(np.where(waveform[22:] == np.max(waveform[22:]))[0][0]) # trough occurs at sample 22 for raw waveforms with 81 samples
            elif len(waveform) == 41:
                spikeTroughPeak.append(np.where(waveform[12:] == np.max(waveform[12:]))[0][0]) # for raw waveforms with 41 samples, trough occurs at sample 12, finding location of maximum post trough
            else:
                print('Raw waveform dimensions do not match those hard-coded into this function...')
                ## perhaps this is generalizable: np.int(np.ceil(81/4)+1)
                # need to more test cases to be sure
        spikeTroughPeak = np.array(spikeTroughPeak)/outDict['sampleRate'] # convert to s
        outDict['spikeTroughPeak'] = spikeTroughPeak

    ## calculating layer
    depths = outDict['unitPosXY'][1] - depth
    outDict['depths'] = depths
    layer_demarcations = -np.array([119,416.5,535.5,952,1300]) ## for S1 recordings; from post-hoc anatomy with DAPI/layer V labeled + DiI, appears to match well with depth of Layer IV optotagged units
    layers = []
    for d in depths:
        if d > layer_demarcations[0]:
            layers.append(1)
        elif (d > layer_demarcations[1]) & (d < layer_demarcations[0]):
            layers.append(2)
        elif (d > layer_demarcations[2]) & (d < layer_demarcations[1]):
            layers.append(4)
        elif (d > layer_demarcations[3]) & (d < layer_demarcations[2]):
            layers.append(5)
        elif (d > layer_demarcations[4]) & (d < layer_demarcations[3]):
            layers.append(6)
        else:
            layers.append(10) ## not cortical
    layers = np.array(layers)
    outDict['layers'] = layers



    return outDict

def importKS(folderpath,depth=250,sampleRate=20000, probe='poly2'):
    """
    Imports the features of the kilosort output I use most.

    inputs:
        folderpath - str with path to kilosort output
        depth - int/float, depth of top electrode site for neuronexus_poly2.prb or depth of bottom electrode site for cnt_h4.prb,
                in microns (default 250 microns, my typical insertion depth of neuronexus_poly2 probe tip is 1100 microns)
        sampleRate - int sample rate in Hz (find in params.py if unknown)
    output: Dict with keys
        goodSpikes - ndarray of clusters (unit identities of spikes)
        goodSamples - ndarray of spike samples (time of spike)
        goodTimes - ndarray of spike times (in s)
        sampleRate - int sample rate in Hz (same as input)
        depths - depth of site (taking into account depth of probe)
        depthIndices - index of good units in the order of their depth
        layers - the cortical layer to which the depth corresponds
        units - list of all units included in goodSpikes

        not yet implemented:
            unitPosXY - tuple of two ndarrays, (X center of mass, Y center of mass)
    """
    
    clusterInfo = pd.read_csv(folderpath+'\\cluster_info.tsv',sep='\t')
    spikeClusters = np.load(folderpath+'\\spike_clusters.npy')
    spikeTimes = np.load(folderpath+'\\spike_times.npy')


    try:
        good_ids = np.array(clusterInfo['id'][clusterInfo['group'] == 'good'])
    except KeyError:
        good_ids = np.array(clusterInfo['cluster_id'][clusterInfo['group'] == 'good'])


    outDict = {}
    outDict['goodSpikes'] = spikeClusters[np.array([n in good_ids for n in spikeClusters])]
    outDict['goodSamples'] = np.int64(spikeTimes[np.array([n in good_ids for n in spikeClusters])].reshape(-1))
    outDict['goodTimes'] = outDict['goodSamples']/sampleRate
    outDict['sampleRate'] = sampleRate
    outDict['units'] = np.unique(outDict['goodSpikes'])
    if probe == 'poly2':
        outDict['depths'] = np.array(clusterInfo['depth'][clusterInfo['group'] == 'good'] - 775 - depth)
    else:
        outDict['depths'] = np.array(clusterInfo['depth'][clusterInfo['group'] == 'good'] - depth)
    outDict['depthIndices'] = np.argsort(outDict['depths']) ## to get an index to use for sorting by depth


    ## calculating layer
    layer_demarcations = -np.array([119,416.5,535.5,952,1300]) ## for S1 recordings; from post-hoc anatomy with DAPI/layer V labeled + DiI, appears to match well with depth of Layer IV optotagged units
    layers = []
    for d in outDict['depths']:
        if d > layer_demarcations[0]:
            layers.append(1)
        elif (d > layer_demarcations[1]) & (d < layer_demarcations[0]):
            layers.append(2)
        elif (d > layer_demarcations[2]) & (d < layer_demarcations[1]):
            layers.append(4)
        elif (d > layer_demarcations[3]) & (d < layer_demarcations[2]):
            layers.append(5)
        elif (d > layer_demarcations[4]) & (d < layer_demarcations[3]):
            layers.append(6)
        else:
            layers.append(10) ## not cortical
    layers = np.array(layers)
    outDict['layers'] = layers

    return outDict

def importphy(folderpath,depth=1250,sampleRate=20000):
    """
    Imports the features of the kilosort output I use most.

    inputs:
        folderpath - str with path to kilosort output
        depth - int/float, depth of top electrode site for neuronexus_poly2.prb or depth of bottom electrode site for cnt_h4.prb,
                in microns (default 250 microns, my typical insertion depth of neuronexus_poly2 probe tip is 1100 microns)
        sampleRate - int sample rate in Hz (find in params.py if unknown)
    output: Dict with keys
        goodSpikes - ndarray of clusters (unit identities of spikes)
        goodSamples - ndarray of spike samples (time of spike)
        goodTimes - ndarray of spike times (in s)
        sampleRate - int sample rate in Hz (same as input)
        not yet implemented:
            unitPosXY - tuple of two ndarrays, (X center of mass, Y center of mass)
            depthIndices - index of good units in the order of their depth
            depths - depth of site (taking into account depth of probe)
            layers - the cortical layer to which the depth corresponds
            units - list of all units included in goodSpikes
    """

    
    spikeClusters = np.load(folderpath+'\\spike_clusters.npy')
    cluster_group = pd.read_csv(folderpath+'\\cluster_group.tsv',sep='\t')
    spikeTimes = np.load(folderpath+'\\spike_times.npy')
    good_ids = np.array(cluster_group['cluster_id'][cluster_group['group'] == 'good'])
    good_index =np.array(cluster_group['group'] == 'good',dtype='bool')
    clusterInfo = pd.read_csv(folderpath+'\\cluster_info.tsv',sep='\t')
    
    outDict = {}
    outDict['goodSpikes'] = spikeClusters[np.array([n in good_ids for n in spikeClusters ])]
    outDict['goodSamples'] = np.int64(spikeTimes[np.array([n in good_ids for n in spikeClusters])]).reshape(-1)

    outDict['goodTimes'] = outDict['goodSamples']/sampleRate
    outDict['sampleRate'] = sampleRate
    #outDict['depths'] = np.array(clusterInfo['depth'])[good_index]
    #outDict['depthIndices'] = np.argsort(np.array(clusterInfo['depth'])[good_index])
    

    return outDict



def importDImat(filepath, sortOption='mtime'):
    """
    Imports digital inputs saved as '*DigitalInputs.mat'

    input:
        filepath - str with directory containing files
        sortOption - str designating sorting method, options include 'mtime' or 'regexp'
    output:
        DI, ndarray with all digital channels
    """


    if sortOption == 'mtime':
        diFiles = glob.glob(filepath+'*DigitalInputs.mat')
        diFiles.sort(key=os.path.getmtime) # sorting by file creation time (may be problematic in mac or linux)
    elif sortOption == 'regexp':
        diFiles = glob.glob('*DigitalInputs.mat') # including full filepath results in regezp matches
        diFiles.sort(key=lambda l: grp('_[0-9]*D',l)) # regular expression finding string of numbers before D
    else:
        print('Invalid sortOption')
        return -1

    DI = []

    for file in diFiles:
        print(file)
        temp = scipy.io.loadmat(file)

        if(temp['board_dig_in_data'].shape[0] == 1):  ## haven't checked if this works yet -- made for Anda
            tempList = [temp['board_dig_in_data'][0], np.zeros(temp['board_dig_in_data'].shape[1])]
            tempArray = np.array(tempList)
            DI.append(tempArray)
        else:
            DI.append(temp['board_dig_in_data'])
    DI = np.concatenate(DI,axis=1)

    return DI

def importAImat(filepath, sortOption='mtime'):
    """
    Yurika wrote this part, modified by AE 3/8/18:
    Imports analog inputs saved as '*AnalogInputs.mat'

    input:
        filepath - str with directory containing files
        sortOption - str designating sorting method, options include 'mtime' or 'regexp'
            if you use 'regexp' your current working diretory must include the *AnalogInputs.mat files
    output:
        AI, ndarray with all analog channels
    """

    if sortOption == 'mtime':
        aiFiles = glob.glob(filepath+'*AnalogInputs.mat')
        aiFiles.sort(key=os.path.getmtime) # sorting by file creation time (may be problematic in mac or linux)
    elif sortOption == 'regexp':
        aiFiles = glob.glob('*AnalogInputs.mat') # including full filepath results in regezp matches
        aiFiles.sort(key=lambda l: grp('[0-9]*A',l)) # regular expression finding string of numbers before D
    else:
        print('Invalid sortOption')
        return -1


    AI = []

    for file in aiFiles:
        print(file)
        temp = scipy.io.loadmat(file)
        #print(temp['board_adc_data'].shape)
        AI.append(temp['board_adc_data'])
    AI = np.concatenate(AI,axis=1)
    return AI

###### helper functions below
def grp(pat, txt):
    r = re.search(pat, txt)
    return r.group(0) if r else '%'
