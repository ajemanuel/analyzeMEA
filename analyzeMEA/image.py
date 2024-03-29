from skimage import filters
import numpy as np
from multiprocessing import Pool
import pims
import cv2

def createDiffLine(video, cropx1, cropx2, cropy1, cropy2):
    """
    Binarizes video and returns a frame to frame difference trace.
    Inputs:
    video - pims object
    cropx1 - int, crop pixel starting for first dimension
    cropx2 - int, crop pixel ending for first dimension
    cropy1 - int, crop pixel starting for second dimension
    cropy2 - int, crop pixel ending for second dimension

    Output:
    diffLine, ndarray; frame to frame differences

    as of 4/24/19, this code takes about 152 seconds for a 9050 frame video
    """
    threshold = filters.threshold_otsu(video[0][cropx1:cropx2,cropy1:cropy2])
    numFrames = len(video)
    binarized = np.zeros((numFrames,video.frame_shape[0],video.frame_shape[1]))

    for i, frame in enumerate(video):
        binarized[i,:,:] = frame > threshold
        if divmod(i,100)[1] == 0:
            print('on frame {0} of {1}'.format(i,numFrames))
    diffLine = np.sum(np.sum(binarized[1:,cropx1:cropx2,cropy1:cropy2] != binarized[:-1,cropx1:cropx2,cropy1:cropy2],axis=1),axis=1)
    diffLine = np.insert(diffLine,-1,diffLine[-1])


    # for i in range(numFrames-1):
    #     binary1 = video[i+1][cropx1:cropx2,cropy1:cropy2] > threshold
    #     binary2 = video[i][cropx1:cropx2,cropy1:cropy2] > threshold
    #     diffLine[i] = np.sum(binary1 != binary2)
    #     if divmod(i,100)[1] == 0:
    #         print('on frame {0} of {1}'.format(i,numFrames))
    # diffLine[-1] = diffLine[-2] ## duplicate the last value to make the array the right size
    # print('Took {0:0.3f} s'.format(time.time()-b))

    return diffLine

def createDiffLineCV(video, cropx1, cropx2, cropy1, cropy2):
    """
    Binarizes video and returns a frame to frame difference trace.
    Inputs:
    video - list of images
    cropx1 - int, crop pixel starting for first dimension
    cropx2 - int, crop pixel ending for first dimension
    cropy1 - int, crop pixel starting for second dimension
    cropy2 - int, crop pixel ending for second dimension

    Output:
    diffLine, ndarray; frame to frame differences

    as of 4/24/19, this code takes about 152 seconds for a 9050 frame video
    """
    initialFrame = cv2.imread(video[0],cv2.IMREAD_GRAYSCALE)[cropx1:cropx2,cropy1:cropy2]
    threshold = filters.threshold_otsu(initialFrame)
    numFrames = len(video)
    diffLine = np.zeros(numFrames)
    for i in np.arange(1,len(video)):
        temp0 = cv2.imread(video[i-1],cv2.IMREAD_GRAYSCALE)[cropx1:cropx2,cropy1:cropy2]
        temp1 = cv2.imread(video[i],cv2.IMREAD_GRAYSCALE)[cropx1:cropx2,cropy1:cropy2]
        temp0_bin = temp0 > threshold
        temp1_bin = temp1 > threshold
        diffLine[i-1] = np.sum(temp0_bin != temp1_bin)
        if divmod(i,100)[1] == 0:
            print('on frame {0} of {1}'.format(i,numFrames))
    diffLine[-1] = diffLine[-2]
    return diffLine

def cropImage(image, cropx1, cropx2, cropy1, cropy2):
    return image[cropx1:cropx2,cropy1:cropy2]

def calculateDifference(images): ## defining this as a nested function so that the threshold is defined based on the video
    threshold = filters.threshold_otsu(images[0])
    binary1 = images[0] > threshold
    binary2 = images[1] > threshold
    return np.sum(binary1 != binary2)

def createDiffLineParallel(video, cropx1, cropx2, cropy1, cropy2):
    """
    Binarizes video and returns a frame to frame difference trace.
    Inputs:
    video - pims object
    cropx1 - int, crop pixel starting for first dimension
    cropx2 - int, crop pixel ending for first dimension
    cropy1 - int, crop pixel starting for second dimension
    cropy2 - int, crop pixel ending for second dimension

    Output:
    diffLine, ndarray; frame to frame differences

    as of 4/24/19, this code takes about 172 s on a 9050 frame video
    the output is slightly different from the original function because we calculate the threshold for each image pair
    """

    diffLine = []

    cropPipeline = pims.pipeline(cropImage)
    print('Cropping Video')
    croppedVideo = cropPipeline(video,cropx1,cropx2,cropy1,cropy2)
    print('Finished Cropping')



    print('Calculating Differences')
    with Pool() as pool:
        diffLine = pool.map(calculateDifference, zip(croppedVideo[1:],croppedVideo[:-1]))

    diffLine = np.array(diffLine)
    diffLine = np.append(diffLine, diffLine[-1]) ## duplicate the last value to make the array the right size
    print('Finished')
    return diffLine
