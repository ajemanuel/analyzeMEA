import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
import pandas as pd
import time
from scipy.signal import savgol_filter, find_peaks
import sys

import matplotlib

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'



def find_footfalls_DEG(predictions):
    """
    Determine the timing of footfalls based on the predicted contact events from deepethogram.

    Inputs:
    - predictions.csv files, sorted to match video order
    Output:
    - dictionary with footfalls and related variables


    Usage example:
    deg_files = glob.glob('*predictions.csv')
    deg_files.sort(key=lambda x: x[-35:])
    ffDict = analyzeMEA.wheel.find_footfalls_DEG(deg_files)
    """
    outDict = {}
    
    labels = []

    for file in predictions:
        labels.append(pd.read_csv(file))
    labels = pd.concat(labels,ignore_index=True)

    contact = np.bool8(labels['contact']+labels['partialContact'] > 0)
    footfalls = np.where(contact[1:] > contact[:-1])[0]
    footrises = np.where(contact[1:] < contact[:-1])[0]

    footfallMatrix  = np.zeros([len(footfalls),300])

    for i, plant in enumerate(footfalls):
        if (plant > 100) and (plant < labels.shape[0]-200):
            footfallMatrix[i,:] += 1* np.array(labels['partialContact'])[plant-100:plant+200]
            footfallMatrix[i,:] += 2* np.array(labels['contact'])[plant-100:plant+200]
            footfallMatrix[i, footfallMatrix[i,:] == 3] = 2

    if footfalls[0] > footrises[0]: # foot in contact at beginning of recording
        print('foot in contact at beginning of recording')
        try:
            stepLengths = footfalls - footrises
            print('foot in contact at end of recording')
        except ValueError:
            stepLengths = footfalls - footrises[:-1]
            print('foot not in contact at end of recording')
        try:
            durations = footrises[1:] - footfalls
        except ValueError:
            durations = footrises[1:] - footfalls[:-1]
    elif footfalls[0] < footrises[0]: # foot not in contact at beginning of recording
        print('foot not in contact at beginning of recording')
        try:
            stepLengths = footfalls[1:] - footrises
        except ValueError:
            stepLengths = footfalls[1:] - footrises[:-1]
        try:
            durations = footrises - footfalls
        except ValueError:
            durations = footrises - footfalls[:-1]

    outDict['labels'] = labels
    outDict['footfalls'] = footfalls
    outDict['footrises'] = footrises
    outDict['footfallMatrix'] = footfallMatrix
    outDict['stepLengths'] = stepLengths
    outDict['durations'] = durations
    try:
        outDict['footfalls_filtered'] = footfalls[:len(durations)][(durations > 15) & (stepLengths > 8)]
        outDict['footrises_filtered'] = footrises[:len(durations)][(durations > 15) & (stepLengths > 8)]
        outDict['durations_filtered'] = durations[(durations > 15) & (stepLengths > 8)]
        outDict['stepLengths_filtered'] = stepLengths[(durations > 15) & (stepLengths > 8)]
    except ValueError:
        outDict['footfalls_filtered'] = footfalls[:len(durations)][(durations > 15) & (stepLengths[:-1] > 8)]
        outDict['footrises_filtered'] = footrises[:len(durations)][(durations > 15) & (stepLengths[:-1] > 8)]
        outDict['durations_filtered'] = durations[(durations > 15) & (stepLengths[:-1] > 8)]
        outDict['stepLengths_filtered'] = stepLengths[:-1][(durations > 15) & (stepLengths[:-1] > 8)]
    return outDict

def plot_onset_offset(footfalls, footrises, durations,
                      frameSamples_sweeps, goodsamples_sweeps, goodspikes_sweeps,
                      minDur = 0.4, maxDur = 1.0, cameraRate=200, save=True, figsize=[6,3],
                      savePath = None,savePSTH=True, units=None, layers=None):
    """
    inputs:
    - footfalls, frames of footfalls
    - footrises, frames of footrises
    - durations, contact duration

    - frameSamples_sweeps, time of frames in neural recording samples
    - goodsamples_sweeps, spike times in neural recording samples
    - goodspikes_sweeps, unit identity corresponding to spike times

    - minDur, minimum duration contacts to include (in s)
    - maxDur, maximum duration contacts to include (in s)

    - cameraRate, frames per second

    Usage Example:
    
    goodsamples_sweeps = []
    goodspikes_sweeps = []
    ventralTrigger_sweeps = []
    frameSamples_sweeps = []
    for i in range(len(matlab_starts)):
        print(intan_starts[i],matlab_starts[i])
        spikeIndex = (goodSamples > intan_starts[i]) & (goodSamples < intan_ends[i])
        goodsamples_sweeps.append(goodSamples[spikeIndex] - intan_starts[i])
        goodspikes_sweeps.append(goodSpikes[spikeIndex])
        ventralTrigger_sweeps.append(matVentralTrigger[matlab_starts[i]:matlab_ends[i]])
        frameSamples_sweeps.append(np.where(ventralTrigger_sweeps[i][1:] > ventralTrigger_sweeps[i][:-1])[0])
        
    analyzeMEA.wheel.plot_onset_offset(footfalls,footrises,durations,frameSamples_sweeps,
                                   goodsamples_sweeps,goodspikes_sweeps, units=spikeDict['units'][spikeDict['depthIndices']],
                                   layers=spikeDict['layers'][spikeDict['depthIndices']])

    """
    
    if units is None:
        units = np.unique(np.concatenate(goodspikes_sweeps))
    
    footfalls_filtered = footfalls[(durations/cameraRate >= minDur) & (durations/cameraRate <= maxDur)]
    footrises_filtered = footrises[(durations/cameraRate >= minDur) & (durations/cameraRate <= maxDur)]
    durations_filtered = durations[(durations/cameraRate >= minDur) & (durations/cameraRate <= maxDur)]

    footfalls_sweeps = []
    footrises_sweeps = []

    cumulative_frames = 0
    for sweep in range(len(frameSamples_sweeps)):
        numFrames = len(frameSamples_sweeps[sweep])
        footfalls_sweeps.append(footfalls_filtered[footfalls_filtered > cumulative_frames] - cumulative_frames)
        footfalls_sweeps[-1] = footfalls_sweeps[-1][footfalls_sweeps[-1] < numFrames]
        footrises_sweeps.append(footrises_filtered[footrises_filtered > cumulative_frames] - cumulative_frames)
        footrises_sweeps[-1] = footrises_sweeps[-1][footrises_sweeps[-1] < numFrames]
        cumulative_frames += numFrames
    footfalls_samples = []
    footrises_samples = []
    for sweep in range(len(footfalls_sweeps)):
        footfalls_samples.append(frameSamples_sweeps[sweep][footfalls_sweeps[sweep]])
        footrises_samples.append(frameSamples_sweeps[sweep][footrises_sweeps[sweep]])

        
    ## for all sweeps
    goodspikes_footfalls = []
    goodsamples_footfalls = []

    goodspikes_footrises = []
    goodsamples_footrises = []

    for sweep in range(len(footfalls_sweeps)):
        for footfall in footfalls_samples[sweep]:
            gs = goodsamples_sweeps[sweep]
            gsp = goodspikes_sweeps[sweep]
            lastSample = frameSamples_sweeps[sweep][-1]
            if ((footfall > 10000) & (footfall < lastSample-20000)): ## only consider those at least 0.5 s after start of recording and 1s before end
                goodsamples_footfalls.append(gs[(gs > footfall - 10000) & (gs < footfall +20000)] - footfall)
                goodspikes_footfalls.append(gsp[(gs > footfall - 10000) & (gs < footfall +20000)])
        for footrise in footrises_samples[sweep]:
            gs = goodsamples_sweeps[sweep]
            gsp = goodspikes_sweeps[sweep]
            lastSample = frameSamples_sweeps[sweep][-1]
            if ((footrise > 20000) & (footrise < lastSample-10000)): ## only consider those at least 1 s after start of recording and 0.5s before end
                goodsamples_footrises.append(gs[(gs > footrise - 20000) & (gs < footrise +10000)] - footrise)
                goodspikes_footrises.append(gsp[(gs > footrise - 20000) & (gs < footrise +10000)])
    
    
    from . import rastPSTH
    footfall_psth = rastPSTH.makeSweepPSTH(0.005,goodsamples_footfalls,
                                                      goodspikes_footfalls,units=units,
                                                      bs_window=[-0.05,0])
    footrise_psth = rastPSTH.makeSweepPSTH(0.005,goodsamples_footrises,
                                                      goodspikes_footrises,units=units,
                                                      bs_window=[0.25,0.5])
    footfall_psth['footfalls'] = footfalls_filtered
    footfall_psth['durations'] = durations_filtered
    footrise_psth['footrises'] = footrises_filtered
    footrise_psth['durations'] = durations_filtered
    if layers is not None:
        footfall_psth['layers'] = layers
        footrise_psth['layers'] = layers

    if savePSTH:
        import pickle
        with open('onset_aligned_psth.pickle', 'wb') as handle:
            pickle.dump(footfall_psth, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('offset_aligned_psth.pickle', 'wb') as handle:
            pickle.dump(footrise_psth, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for i, unit in enumerate(units):
        f, ax = plt.subplots(1,2,figsize=figsize)

        ax[0].plot(footfall_psth['xaxis'],footfall_psth['psths_bs'][:,i])
        ax[0].axvline(0,ls='--',color='gray')
        ax[0].set_xlim([-0.05,0.3])
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Firing Rate (Hz)')
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        
        ax[1].plot(footrise_psth['xaxis'],footrise_psth['psths'][:,i]-footfall_psth['psths'][90:100,i].mean(axis=0))
        ax[1].axvline(0,ls='--',color='gray')
        ax[1].set_xlim([-0.3,0.05])
        ax[1].set_ylim(ax[0].set_ylim())
        ax[1].set_xlabel('Time (s)')
        ax[1].set_yticks([])
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ylim = [min(ax[0].set_ylim()[0],ax[1].set_ylim()[0]),
                max(ax[0].set_ylim()[1],ax[1].set_ylim()[1])]
        
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(ylim)
        
        plt.suptitle('Unit {}'.format(unit))
        if save:
            if savePath is None:
                import os
                savePath = os.getcwd()
            plt.savefig(savePath+'\\onset_offset_unit_{}.png'.format(unit),transparent=True,
                        dpi=600,bbox_inches='tight')
        plt.show()

        plt.close()

def find_footfalls(intensity,peak_height = None,peak_distance = 0.1, frame_rate=200, plot=False):
    """
    Determine the timing of footfalls based on the intensity of the keypoint. This function finds peaks in the

    Inputs:
    - intensity, ndarray
    - peak_height, range of acceptable peaks
    - peak_distance, minimum time between footfalls, in seconds
    - frame_rate, frame rate of video

    Outputs:
    - footfalls, ndarray containing frames with footfalls
    """
    peak_distance = int(peak_distance * frame_rate) ## converting to frames

    diff = np .diff(intensity)
    diff_smoothed = savgol_filter(diff, 17, 3) ## consider making these parameters accessible as inputs
    diff_smoothed = np.insert(diff_smoothed,0,0)
    if peak_height is None:
        peak_height = [2*np.std(diff_smoothed),np.max(diff_smoothed)+1]
    footfalls = find_peaks(diff_smoothed,height=peak_height,distance=peak_distance)[0]
    footrises = find_peaks(-diff_smoothed,height=peak_height,distance=peak_distance)[0]

    if plot:
        plt.figure(figsize=[15,3])
        plt.plot(diff_smoothed)
        #plt.plot(LFPintensity)
        plt.plot(footfalls,np.ones(len(footfalls))*peak_height[1],'.',label='Footfalls')
        plt.plot(footrises,-np.ones(len(footrises))*peak_height[1],'.',label='Footrises')
        #plt.xlim([150000,153000])
        plt.ylabel('Derivative of Pixel Intensity of Paw')
        plt.xlabel('Frame ({} Hz sampling)'.format(frame_rate))
        plt.xlim([0,len(intensity)])
        plt.legend()
        plt.show()
        plt.close()

    return footfalls, footrises



def classify_gratings_manually(footfallImages,savename = 'grating_classification.npy', pawPositions=None):
    """
    Prompt for manual classification of gratings. Loads file at savename to continue

    Inputs:
    - footfallImages, list: images to be classsified
    - savename, string: .npy filename for saving classifications
    """

    try:
        grating_classification = np.load(savename)
        grating_classification = [n for n in grating_classification]
    except:
        print('No gratings file found, starting new one')
        grating_classification = []

    numFrames = len(footfallImages[len(grating_classification):])
    prevFrames = len(grating_classification)
    for i, im in enumerate(footfallImages[len(grating_classification):]):
        plt.imshow(im)
        if pawPositions is not None:
            plt.plot(pawPositions[prevFrames+i,0],pawPositions[prevFrames+i,1],'.',color='r')
        plt.show()
        plt.close()
        time.sleep(0.1)
        temp = input('Enter gratings direction for frame {} of {}\n1 = left diag, 2 = horizontal, 3 = right diag, 4 = vertical, 0 = other\nrelative to scorer, not mouse'.format(i, numFrames)) ## these are relative to scorer (not mouse)
        grating_classification.append(temp)
        np.save(savename,np.int32(grating_classification))


def permuteOS(trialGratings, units, numShuffles=10000, plot=True, saveFig=True):
    

    ## set random seed and create generator
    
    rng = np.random.default_rng(20230424)
    numUnits = len(units)
    oris = []
    responses = []
    weights = []

    for ori in trialGratings:
        oris.append(np.ones(len(trialGratings[ori]))*ori)
        responses.append(trialGratings[ori])
        weights.append(np.ones(len(trialGratings[ori]))*(1/len(trialGratings[ori])))

    oris = np.concatenate(oris)
    responses = np.concatenate(responses)
    weights = np.concatenate(weights)
    vector = np.zeros([numUnits,len(responses),2]) ## Units x Footfalls x XY
    for i, response in enumerate(responses):
        if oris[i] == 1:
            vector[:,i,1] = response ## positive Y
        elif oris[i] == 2:
            vector[:,i,0] = response ## positive X
        elif oris[i] == 3:
            vector[:,i,1] = -response ## negative Y
        elif oris[i] == 4:
            vector[:,i,0] = -response ## negative X
    
    shuffleVectors = np.zeros([len(responses),numUnits,numShuffles,2])
    shuffleOris = np.zeros([len(responses),numShuffles])
    shuffleWeights = np.zeros([len(responses),numShuffles])
    for shuffle in range(numShuffles):
        permIndex = rng.permutation(np.arange(len(responses)))
        shuffleOris[:,shuffle] = oris[permIndex]
        shuffleWeights[:,shuffle] = weights[permIndex]

        for i, response in enumerate(responses):
            if shuffleOris[i,shuffle] == 1:
                shuffleVectors[i,:,shuffle,1] = response ## positive Y
            elif shuffleOris[i,shuffle] == 2:
                shuffleVectors[i,:,shuffle,0] = response ## positive X
            elif shuffleOris[i,shuffle] == 3:
                shuffleVectors[i,:,shuffle,1] = -response ## negative Y
            elif shuffleOris[i,shuffle] == 4:
                shuffleVectors[i,:,shuffle,0] = -response ## negative X

    meanVector = np.average(vector,axis=1,weights=weights)
    meanShuffleVectors = np.zeros([numUnits,numShuffles,2])

    for unit in range(numUnits):
        meanShuffleVectors[unit, :, 0] = np.average(shuffleVectors[:,unit,:,0],axis=0,weights=shuffleWeights)
        meanShuffleVectors[unit, :, 1] = np.average(shuffleVectors[:,unit,:,1],axis=0,weights=shuffleWeights)


    distance = (meanVector[:,0]**2 + meanVector[:,1]**2)**0.5
    shuffleDistances = (meanShuffleVectors[:,:,0]**2 + meanShuffleVectors[:,:,1]**2)**0.5

    pvalues = np.ones(len(units)) - np.sum(np.matlib.repmat(distance,numShuffles,1).T > shuffleDistances,axis=1)/numShuffles
    
    import statsmodels.stats.multitest
    OSunits = units[np.where(statsmodels.stats.multitest.multipletests(pvalues,method='hs')[0])[0]]
    OSunits_noCorrection = units[pvalues < 0.05]

    print('Without multiple comparison correction:')
    print('{} of {} units are OS'.format(len(OSunits_noCorrection),len(units)))
    print('Units {}'.format(OSunits_noCorrection))

    print('\nWith HS correction:')
    print('{} of {} units are OS'.format(len(OSunits),len(units)))
    print('Units {}'.format(OSunits))

    if plot:
        theta = np.arctan2(meanVector[:,1],meanVector[:,0])
        thetaShuffled = np.arctan2(meanShuffleVectors[:,:,1],meanShuffleVectors[:,:,0])
        
        
        
        for unit in range(len(units)):
            plt.figure()
            ax = plt.axes(projection='polar')
            ax.plot(thetaShuffled[unit,:],shuffleDistances[unit,:],'.',color='gray',label='shuffle',alpha=0.1)
            ax.plot([0,theta[unit]],[0,distance[unit]],color='r',label='actual',lw=1.5)
            dist95 = np.sort(shuffleDistances[unit,:])[int(numShuffles * 0.95)]
            ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100)*dist95, color='k', linestyle='--',lw=1.5,label='95% CI')
            ax.set_xticks(np.arange(0,2.1*np.pi,np.pi/2)) #radians
            ax.set_xticklabels([])
            ylim = ax.get_ylim()
            ax.set_yticks([ylim[1]/4,ylim[1]/2,ylim[1]*(3/4),ylim[1]])
            if units[unit] in OSunits:
                ax.set_title('Neuron {0}, p = {1: 0.4f}'.format(units[unit],pvalues[unit]),color='r')
            else:
                ax.set_title('Neuron {0}, p = {1: 0.4f}'.format(units[unit],pvalues[unit]),color='k')
            ax.legend()
            if saveFig:
                plt.savefig('PolarPlot_Neuron{0}.pdf'.format(units[unit]),dpi=600,bbox_inches='tight',transparent=True)    
            plt.show()
            plt.close()

    
    return pvalues, OSunits, OSunits_noCorrection, meanVector, meanShuffleVectors


def plotMeanOris(trialRates,units,OSunits, save=True, saveFile='Oris_Mean_Rate.pdf'):
    """
    Example function call:
    analyzeMEA.wheel.plotMeanOris(trialRates,np.unique(goodSpikes),OSunits,save=False)
    """
    
    oris = [-45, 0, 45, 90]
    f, ax = plt.subplots(1,4,figsize=[14,4])
    for unit in range(len(units)):
        if units[unit] in OSunits:
            resps = []
            for ori in [1,2,3,4]:
                resps.append(np.mean(trialRates[ori][:,unit]))
            ax[np.argmax(resps)].plot(oris,resps/np.max(resps),marker='o',color='gray',markerfacecolor='white')
    ax[0].set_ylabel('Norm. Mean Firing Rate')
    for a in range(4):
        ax[a].set_xlabel('Orientation')
        ax[a].spines['top'].set_visible(False)
        ax[a].spines['right'].set_visible(False)
        ax[a].set_ylim([0,1.05])
        ax[a].set_xticks([-45,0,45,90])
        if a > 0:
            ax[a].set_yticklabels([])
    if save:
        plt.savefig(saveFile, transparent=True,bbox_inches='tight')


### helper functions

def TwoSampleT2Test(X, Y):
    from scipy.stats import f
    nx , p = X.shape
    ny, _ = Y.shape
    delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
    Sx = np.cov(X, rowvar=False)
    Sy = np.cov(Y, rowvar=False)
    S_pooled = ((nx-1)*Sx + (ny-1)*Sy)/(nx+ny-2)
    t_squared = (nx*ny)/(nx+ny) * np.matmul(np.matmul(delta.transpose(), np.linalg.inv(S_pooled)), delta)
    statistic = t_squared * (nx+ny-p-1)/(p*(nx+ny-2))
    F = f(p, nx+ny-p-1)
    p_value = 1 - F.cdf(statistic)
    print(f"test statistic: {statistic}\nDegrees of freedom: {p} and {nx+ny-p-1}\np-value: {p_value}")
    return statistic, p_value

def OneSampleT2Test(X, nullHyp = None, verbose=False):
    from scipy.stats import f
    nx , k = X.shape
    if nullHyp is None:
        nullHyp = np.zeros([1,k]) ## null hypothesis is zero
    delta = np.mean(X, axis=0) - nullHyp
    Sx = np.cov(X, rowvar=False)
    t_squared = nx * np.matmul(np.matmul(delta, np.linalg.inv(Sx)), delta.T)
    statistic = t_squared * (nx-k)/(k*(nx-1))
    F = f(k, nx-k)
    p_value = float(1 - F.cdf(statistic))
    if verbose:
        print(f"test statistic: {statistic}\nDegrees of freedom: {k} and {nx-k}\np-value: {p_value}")
    return statistic, p_value


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
def rotate_point(center,angle,point):
    s = np.sin(angle/360*2*np.pi)
    c = np.cos(angle/360*2*np.pi)

    # translate point back to origin:
    point[0] -= center[0]
    point[1] -= center[1]

    # rotate point
    xnew = point[0] * c - point[1] * s
    ynew = point[0] * s + point[1] * c

    # translate point back:
    p = np.zeros(2)
    p[0] = xnew + center[0]
    p[1] = ynew + center[1]
    return p

def rotate_points(center,angle,points):

    theta = angle/360*2*np.pi # convert to radians
    A = [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta),  np.cos(theta)]]

    temp = points - center

    temp = np.matmul(temp,A)
    return temp + center


def loadDepthMap(file,cropY=[300,-300]):
    """
    load file used to make 3D-printed texture
    """
    depthMap = np.array(h5py.File(file,'r')['top_depth_map'])[cropY[0]:cropY[1],:]
    ## scale depth map values to image values
    #print(np.min(depthMap),np.max(depthMap))
    depthMap = depthMap * 255/0.75  ## future, make this dependent on depth map range
    return depthMap


def estimateRotFactor(image,depthMap,initialEstimate=0,resolution = 0.2, scaleFactor=0.210,
                     cropY=[80,250],cropX=[10,325]):
    
    rots = np.arange(initialEstimate-5,initialEstimate+5,resolution)
    corrCoefs = []
    
    for rotFactor in rots:
        
        cropped_im = rotate_image(image,rotFactor)[cropY[0]:cropY[1],cropX[0]:cropX[1]]
        croppedFrame = cv2.cvtColor(cropped_im, cv2.COLOR_BGR2GRAY)
        croppedFrame_thresh = cv2.adaptiveThreshold(croppedFrame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                                    cv2.THRESH_BINARY,35,-25)


        D_shrunk = cv2.flip(cv2.resize(np.uint8(depthMap),(0,0),fx=scaleFactor,fy=scaleFactor),1)
        D_shrunk_cat = np.concatenate([D_shrunk,D_shrunk[:,:croppedFrame.shape[1]]],axis=1)
        result = cv2.matchTemplate(D_shrunk_cat,croppedFrame_thresh,cv2.TM_CCORR_NORMED)
        
        corrCoefs.append(np.max(result))
    return np.array(corrCoefs)

def estimateRotFactors(images,plot=True,initialEstimate=0,resolution = 0.2, scaleFactor=0.210,
                     cropY=[80,250],cropX=[10,325],depthMap=None,
                     depthMapFile=r'Z:/HarveyLab/Tier1/Alan/Data/20220831/sparse_bumps_wheel_v2_220712_data.mat'):
    if depthMap is None:
        depthMap = loadDepthMap(depthMapFile)
    rots = np.arange(initialEstimate-5,initialEstimate+5,resolution)
    coeffs = np.zeros([len(images),len(rots)])
    depthMap = loadDepthMap(depthMapFile)
    for i, image in enumerate(images):
        print('on image {} of {}'.format(i+1,len(images)))
        corrCoefs = estimateRotFactor(image, depthMap, initialEstimate=initialEstimate,resolution=resolution,scaleFactor=scaleFactor,
                          cropY=cropY,cropX=cropX)
        coeffs[i,:] = corrCoefs
    bestRots = np.array([float(rots[np.argmax(coeffs[i,:])]) for i in range(len(coeffs))])
    if plot:
        plt.figure(figsize=[2,2])
        ax = plt.axes()
        ax.plot(rots,coeffs.T)
        ax.set_xlabel('Rotation (deg)')
        ax.set_ylabel('Max. Corr. Coefficient')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()
        plt.close()
    return rots, coeffs, bestRots

def findTemplateMatch(image,depthMap=None, rotFactor=-1,scaleFactor=0.210,
                      depthMapFile=r'Z:/HarveyLab/Tier1/Alan/Data/20220831/sparse_bumps_wheel_v2_220712_data.mat'):
    
    if depthMap is None:
        depthMap = loadDepthMap(depthMapFile)
    
    rotated_im = cv2.cvtColor(rotate_image(image,rotFactor),cv2.COLOR_BGR2GRAY)
    croppedFrame = rotated_im[80:250,10:326]
    croppedFrame_thresh = cv2.adaptiveThreshold(croppedFrame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                                    cv2.THRESH_BINARY,35,-25)

    D_shrunk = cv2.flip(cv2.resize(np.uint8(depthMap),(0,0),fx=scaleFactor,fy=scaleFactor),1)
    D_shrunk_cat = np.concatenate([D_shrunk,D_shrunk[:,:croppedFrame.shape[1]]],axis=1)
    result = cv2.matchTemplate(D_shrunk_cat,croppedFrame_thresh,cv2.TM_CCORR_NORMED)

    resultMax = np.where(result == np.max(result))
    #print(np.max(result))
    return resultMax[0][0],resultMax[1][0]

def findTemplateMatches(images,depthMap=None, rotFactor=[-1],scaleFactor=0.210,
                        depthMapFile=r'Z:/HarveyLab/Tier1/Alan/Data/20220831/sparse_bumps_wheel_v2_220712_data.mat'):
    if depthMap is None:
        depthMap = loadDepthMap(depthMapFile)
    maxLoc = []
    for i, image in enumerate(images):
        if i % 50 == 0:
            print('On image {} of {}'.format(i, len(images)))
        if len(rotFactor) == 1:
            maxLoc.append(findTemplateMatch(image,depthMap=depthMap,rotFactor=rotFactor[0],scaleFactor=scaleFactor))
        elif len(rotFactor) == len(images):
            maxLoc.append(findTemplateMatch(image,depthMap=depthMap,rotFactor=rotFactor[i],scaleFactor=scaleFactor))
    templateMaches = np.array(maxLoc)
    return templateMaches

def findMapPositions(templateMatches,pawPositions,image,rotFactor=[-1],scaleFactor=0.210,
                     cropY=[80,250],cropX=[10,325],
                     depthMap=None,plot=True,depthMapFile=r'Z:/HarveyLab/Tier1/Alan/Data/20220831/sparse_bumps_wheel_v2_220712_data.mat'):
    center = np.array(image.shape[:2])/2
    
    topLeftOfCroppedFrame = np.array([cropX[0],cropY[0]]) # X,Y position of top left of cropped frame
    if len(rotFactor) == 1:
        footfallPositions_rot = rotate_points(center,rotFactor[0],pawPositions) ## in the future, change rotation according to video
    elif len(rotFactor) == len(pawPositions):
        footfallPositions_rot = []
        for i, point in enumerate(pawPositions):
            footfallPositions_rot.append(rotate_points(center,rotFactor[i],pawPositions[i]))
        footfallPositions_rot = np.array(footfallPositions_rot)
                                             
    footfallPositions_rot_relCrop = footfallPositions_rot - topLeftOfCroppedFrame ## adjust for location of crop
    mapPositions = np.fliplr(templateMatches) + footfallPositions_rot_relCrop ## templateMatch is Y, X, so need to flip
    
    
    if plot:
        if depthMap is None:
            depthMap = loadDepthMap(depthMapFile)
        D_shrunk = cv2.flip(cv2.resize(np.uint8(depthMap),(0,0),fx=scaleFactor,fy=scaleFactor),1)
        plt.figure(figsize=[30,4])
        plt.imshow(D_shrunk,aspect=None,cmap='gray')
        for ff in range(len(mapPositions)):#range(len(footfalls)):
            if mapPositions[ff,0] < D_shrunk.shape[1]:
                plt.plot(mapPositions[ff,0],mapPositions[ff,1],'.',ms=6,color='r')
            else:
                x = mapPositions[ff,0] - D_shrunk.shape[1]
                plt.plot(x,mapPositions[ff,1],'.',ms=6,color='r')
        plt.xticks([])
        plt.yticks([])
        #plt.savefig('FootfallsOnMap_AKR2.png',dpi=600,bbox_inches='tight')
    return mapPositions

def alignToMapPositions(mapPositions,depthMap=None,scaleFactor=0.210,imageSize=[120,120],
                        depthMapFile=r'Z:/HarveyLab/Tier1/Alan/Data/20220831/sparse_bumps_wheel_v2_220712_data.mat'):
    if depthMap is None:
            depthMap = loadDepthMap(depthMapFile)
    D_shrunk = cv2.flip(cv2.resize(np.uint8(depthMap),(0,0),fx=scaleFactor,fy=scaleFactor),1)
    D_padded = np.copy(D_shrunk)
    zeroPadding = np.zeros(D_padded.shape)
    D_padded = np.concatenate([zeroPadding,D_padded,zeroPadding])
    D_padded = np.concatenate([D_padded,D_padded,D_padded],axis=1)
    map_positions_offset = mapPositions + np.flip(D_shrunk.shape)

    alignedImages = np.zeros([imageSize[1],imageSize[0],len(mapPositions)],dtype=int)

    halfSizeX = int(imageSize[0]/2)
    halfSizeY = int(imageSize[1]/2)
    for ff in range(len(mapPositions)):
        pos = np.int32(map_positions_offset[ff])
        alignedImages[:,:,ff] = D_padded[pos[1]-halfSizeY:pos[1]+halfSizeY,pos[0]-halfSizeX:pos[0]+halfSizeX]
        
    return alignedImages


def runAlignment(firstImages,videoIndex, images, positions,
                   depthMap=None,
                   depthMapFile=r'Z:/HarveyLab/Tier1/Alan/Data/20220831/sparse_bumps_wheel_v2_220712_data.mat'):
    """
    Run all alignment routines, starting with estimating rotation factors from first images of each video.
    Inputs:
    - firstImages, list of images generated with capture of first frame from each video with openCV
    - videoIndex, ndarray int16, video from which each event occurs
    - images, list of images taken at time of each event, generated with capture of appropraite frame with openCV
    - positions, ndarray,  XY points for keypoint of interest in frame
    - depthMap, ndarray, depthMap used to make 3D-printed texture
    - depthMapFile, path to .mat file containing 3D-printed texture information
    Outputs:
    - mapPositions, ndarray, XY location of keypoint at each event on scaled depthMap
    - alignedImages, ndarray, images of depthMap centered on keypoint at time of each event
    
    Usage Example:

    videos_ventral = glob.glob('v*.avi')
    videos_ventral.sort(key=os.path.getmtime) ## sort by modification time -- use regular expressions if mtime is changed
    dlcFiles_ventral = []
    for video in videos_ventral:
        dlcFiles_ventral.append(glob.glob(video[:-4]+'*filtered.h5'))
    dlcFiles_ventral = np.concatenate(dlcFiles_ventral)


    firstImages = []
    videoIndex = []
    for i, video in enumerate(videos_ventral):
        cap = cv2.VideoCapture(video)
        f, im = cap.read()
        firstImages.append(im)
        videoIndex.append(np.ones(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))*i)
        cap.release()
    videoIndex = np.concatenate(videoIndex)
    deg_files = glob.glob('*predictions.csv')
    deg_files.sort(key=lambda x: x[-35:])
    print(deg_files)
    ffDict = analyzeMEA.wheel.find_footfalls_DEG(deg_files)
    footfalls = ffDict['footfalls_filtered']
    footfallImages = []
    for i, video in enumerate(videos_ventral):
        print(i)
        ffs = footfalls[videoIndex[footfalls] == i]
        firstFrame = np.where(videoIndex == i)[0][0] ## find first frame of video
        ffs = ffs-firstFrame

        cap = cv2.VideoCapture(video)
        
        for ff in ffs:
            cap.set(cv2.CAP_PROP_POS_FRAMES,ff)
            f, im = cap.read()
            footfallImages.append(im)
    RFPposition = []
    for i, file in enumerate(dlcFiles_ventral):
        DLCmodel = pd.read_hdf(file).keys()[0][0]
        RFPposition.append(pd.read_hdf(file)[DLCmodel]['rightForepaw'])
    RFPposition = np.concatenate(RFPposition)

    ### here I am assuming the first 1945 footfalls are of interest -- change according to experiment
    videoInd = np.int16(videoIndex[footfalls[:1945]])
    images = footfallImages[:1945]
    positions = RFPposition[footfalls[:1945],:-1]

    mapPositions, alignedImages = runAlignment(firstImages, videoInd, images, positions)
    
    """
    if depthMap is None:
        depthMap = loadDepthMap(depthMapFile)
    rots, coeffs, bestRots = estimateRotFactors(firstImages)
    ffRots = bestRots[videoIndex] ## creating list of rotation factors for each image
    templateMatches = findTemplateMatches(images, depthMap=depthMap, rotFactor=ffRots)
    mapPositions = findMapPositions(templateMatches, positions, images[0], ffRots,depthMap=depthMap)
    alignedImages = alignToMapPositions(mapPositions,depthMap=depthMap)

    return mapPositions, alignedImages



def crop_aligned_images(alignedImages, dim1_length = 30, dim2_length=50, dsFactor = 5, scaleFactor=109.36):
    ## crop, downsample, and scale aligned images to only conider area around paw
    # assuming paw is centered in image (60,60)

    origCenter = np.array(alignedImages.shape[:-1])/2
    alignedImages_cropped = alignedImages[int(origCenter[0]-dim1_length/2):int(origCenter[0]+dim1_length/2),
                                          int(origCenter[1]-dim2_length/2):int(origCenter[1]+dim2_length/2),:]

    ## downsample so that we are not fitting too many parameters

    croppedShape = alignedImages_cropped.shape
    alignedImages_c_ds = np.reshape(alignedImages_cropped,(int(croppedShape[0]/dsFactor),dsFactor,int(croppedShape[1]/dsFactor),dsFactor,alignedImages_cropped.shape[-1])).mean(1).mean(2)
    
    
    
    # scale so that each value represents the number of bumps in pixel
    alignedImages_c_ds = alignedImages_c_ds/scaleFactor

    return alignedImages_c_ds

def locationGLM(dsImages,responses,plot=True, imageInd=None, units=None,save=True):
    """
    Use GLM to generate receptive fields.
    Predict number of spikes by weighting number of bumps in each pixel of the downsampled images, which contain the stimulus aligned to the center of the paw.

    Inputs:
    - dsImages, ndarray of downsamples images imDim1 x imDim2 x numEvents
    - responses, ndarray number of spikes in response window; numEvents X numUnits



    """

    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from sklearn.preprocessing import SplineTransformer, QuantileTransformer
    import tensorflow as tf
    from . import glm_class as glm
    
    if imageInd is not None:
        dsImages = dsImages[:,:,imageInd] ## subselect images included in responses

    sh = dsImages.shape

    X = dsImages.reshape([sh[0]*sh[1],-1]).T
    Y = responses

    n_samples = X.shape[0]
    group_id = np.arange(n_samples)
    gss = GroupShuffleSplit(n_splits=1,train_size=0.85, random_state=42)
    train_idx, test_idx = next(gss.split(X,Y,group_id))

    # Split data into train and test set
    X_train = X[train_idx,:]
    Y_train = Y[train_idx,:]
    X_test = X[test_idx,:]
    Y_test = Y[test_idx,:]
    trial_id_train = group_id[train_idx] # extract trial_id for training data, which is used in CV splits later during fitting  

    tf.keras.backend.clear_session()

    # Initialize GLM_CV (here we're only specifying key input arguments; others are left as default values; see documentation for details)
    model_cv = glm.GLM_CV(n_folds = 5, auto_split = True, 
                      activation = 'exp', loss_type = 'poisson', 
                      regularization = 'elastic_net',l1_ratio=0.8,lambda_series = 10.0 ** np.linspace(-1, -7, 11), 
                      optimizer = 'adam', learning_rate = 1e-3)
    
    model_cv.fit(X_train, Y_train, group_idx = trial_id_train)

    model_cv.select_model(se_fraction=0., min_lambda=0., make_fig=True)
    # Evaluate model performance on test data
    frac_dev_expl, dev_model, dev_null, dev_expl = model_cv.evaluate(X_test, Y_test, make_fig = True)

    if plot:
        clim = model_cv.selected_w.max()/2
        for neuron in range(responses.shape[1]):
            plt.figure(figsize=[3,2])
            weightImage = model_cv.selected_w[:int(sh[0]*sh[1]),neuron].reshape(sh[0],sh[1])
            imageMax = np.max(np.abs(weightImage))
            plt.imshow(cv2.rotate(weightImage,cv2.ROTATE_90_CLOCKWISE),cmap='RdBu_r',clim=[-imageMax,imageMax])
            if units is None:
                plt.title(r'Neuron {0}, $\beta_0$ = {1: 0.3f}'.format(neuron,model_cv.selected_w0[neuron][0]))
            else:
                plt.title(r'Neuron {0}, $\beta_0$ = {1: 0.3f}'.format(units[neuron],model_cv.selected_w0[neuron][0]))
            cb = plt.colorbar()
            cb.set_label(r'$\beta$')
            plt.xticks([])
            plt.yticks([])
            if save:
                plt.savefig('GLM_RF_Neuron{}.pdf'.format(neuron),bbox_inches='tight',dpi=600,transparent=True)
            plt.show()
            plt.close()
    return model_cv