import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle

def undistort_img():
    # Prepare object points 0,0,0 ... 8,5,0
    obj_pts = np.zeros((6*9, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Stores all object points & img points from all images
    objpoints = []
    imgpoints = []

    # Get directory for all calibration images
    images = glob.glob('camera_cal/*.jpg')

    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)
    # Test undistortion on img
    img_size = (img.shape[1], img.shape[0])

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Save camera calibration for later use
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump(dist_pickle, open('camera_cal/cal_pickle.p', 'wb'))


def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    #cv2.imwrite('camera_cal/test_cal.jpg', dst)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst


undistort_img()


def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = undistort(img)
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    h_channel = hls[:, :, 0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &
             (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    color_binary = np.dstack(
        (np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def perspective_warp(img,
                     dst_size=(1280, 720),
                     src=np.float32(
                         [(0.24, 0.65), (0.76, 0.65), (0, 0.85), (1, 0.85)]),
                     dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped


def inv_perspective_warp(img,
                         dst_size=(1280, 720),
                         src=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
                         dst=np.float32([(0.24, 0.65), (0.76, 0.65), (0, 0.87), (1, 0.87)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped


def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:, :], axis=0)
    return hist

print(os.listdir("./SNU_DATASET"))

if __name__ == "__main__":
    for img in os.listdir("./SNU_DATASET"):
        img = cv2.imread("./SNU_DATASET/" + img)
        img = cv2.resize(img, (1280, 720))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dst = pipeline(img)
        dst = perspective_warp(dst, dst_size=(1280, 720))
        inv = inv_perspective_warp(dst, dst_size=(1280, 720))
        pipe = pipeline(img)

        cv2.imshow("img", img)
        cv2.imshow("dst", dst)
        cv2.imshow("inv", inv)
        cv2.imshow("pipe", pipe)
        
        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()


# img_size = np.float32([(720, 1280)])

# print(img_size)

