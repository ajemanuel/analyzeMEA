from skimage import filters
import numpy as np

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
    """
    threshold = filters.threshold_otsu(video[0][cropx1:cropx2,cropy1:cropy2])
    diffLine = []
    numFrames = len(video)
    for i, image in enumerate(video):
        if i > 0:
            binary1 = image[cropx1:cropx2,cropy1:cropy2] > threshold
            binary2 = video[i-1][cropx1:cropx2,cropy1:cropy2] > threshold
            diffLine.append(np.sum(binary1 != binary2))
        if i % 100 == 0:
            print('on frame {0} of {1}'.format(i,numFrames))
    diffLine = np.array(diffLine)
    diffLine = np.append(diffLine, diffLine[-1]) ## duplicate the last value to make the array the right size
    return diffLine
