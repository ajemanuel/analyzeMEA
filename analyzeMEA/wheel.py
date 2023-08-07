import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
import pandas as pd
import time
from scipy.signal import savgol_filter, find_peaks



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
    except ValueError:
        outDict['footfalls_filtered'] = footfalls[:len(durations)][(durations > 15) & (stepLengths[:-1] > 8)]
        outDict['footrises_filtered'] = footrises[:len(durations)][(durations > 15) & (stepLengths[:-1] > 8)]
        outDict['durations_filtered'] = durations[(durations > 15) & (stepLengths[:-1] > 8)]
    return outDict



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
            if mapPositions[ff,0] < D_shrunk.shape[0]:
                plt.plot(mapPositions[ff,0],mapPositions[ff,1],'.',ms=6,color='r')
            else:
                x = mapPositions[ff,0] - D_shrunk.shape[0]
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

    alignedImages = np.zeros([imageSize[1],imageSize[0],len(mapPositions)])

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
