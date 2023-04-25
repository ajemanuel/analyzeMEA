import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
import pandas as pd
import time
from scipy.signal import savgol_filter, find_peaks


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


def permuteOS(trialGratings, units, numShuffles=10000, plot=True):
    

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
            plt.polar([0,theta[unit]],[0,distance[unit]],color='r',label='actual')
            plt.polar(thetaShuffled[unit,:],shuffleDistances[unit,:],'.',color='gray',label='shuffle',alpha=0.2)
            if units[unit] in OSunits:
                plt.title('Unit {}, p = {}'.format(units[unit],pvalues[unit]),color='r')
            else:
                plt.title('Unit {}, p = {}'.format(units[unit],pvalues[unit]),color='k')
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
