from skimage import filters
import numpy as np
from multiprocessing import Pool
import pims
from numba import jit

@jit(parallel=True)
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
    diffLine = np.zeros(numFrames)
    for i in range(numFrames-1):
        binary1 = video[i+1][cropx1:cropx2,cropy1:cropy2] > threshold
        binary2 = video[i][cropx1:cropx2,cropy1:cropy2] > threshold
        diffLine[i] = np.sum(binary1 != binary2)
        if divmod(i,100)[1] == 0:
            print('on frame {0} of {1}'.format(i,numFrames))
    diffLine[-1] = diffLine[-2] ## duplicate the last value to make the array the right size
    return diffLine



def cropImage(image, cropx1, cropx2, cropy1, cropy2):
    return image[cropx1:cropx2,cropy1:cropy2]

@jit()
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
