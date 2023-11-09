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






### helper functions
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
