from locale import currency
from re import L
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time


def linearFuntion(y, pt1, pt2):
    if pt1[0] - pt2[0] != 0:
        k = (pt1[1] - pt2[1] / pt1[0] - pt2[0])
    else:
        return 0
    return int((y - pt1[1]) / k + pt1[0])


def linearFuntion_y(x, pt1, pt2):
    if pt1[0] - pt2[0] != 0:
        k = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
    else:
        return 0
    return int((x - pt1[0]) * k + pt1[1])

def drawLine(img, pt1, pt2):
    h, w, _ = img.shape
    a = linearFuntion(0, pt1, pt2)
    b = linearFuntion(h, pt1, pt2)
    print("a :", a)
    center = (a + b) // 2
    print("center :", center)
    
    if center < w // 2:
        print("Move Right")
        cv2.putText(img, "Move Right", (int(w * 0.75), h // 8),\
            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
    else:
        print("Move Left")
        cv2.putText(img, "Move Left", (int(w * 0.75), h // 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
        
    print(pt2[0], pt2[1])
    cv2.circle(img, (center, h // 2), 5, (255, 0, 0), -1)
    # cv2.line(img, (a, 0), (b, h), (0, 255, 0), 1)

def estimate(img, binary_img):
    current = time.time()
    # img = cv2.imread("./input_21.jpg")
    # binary_img = cv2.imread("./binary_21.jpg", cv2.IMREAD_GRAYSCALE)
    # binary_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w, _ = img.shape

    line = np.array([])
    for i in range(h // 2, h):
        left = []
        right = []
        for j in range(w):
            if binary_img[i, j] == 255:  # and abs(w // 2 - j) < (w // 2)
                if (w // 2 - j) > 0:
                    left.append([j, i])
                else:
                    right.append([j, i])
                    
        if len(left) > 0 and len(right) > 0:
            l_mean = np.mean(left, axis=0)
            r_mean = np.mean(right, axis=0)
            
            # print("l-mean : {}".format(l_mean))
            # print("r-mean : {}".format(r_mean))
            # print("np.shape(l_mean) :", np.shape(l_mean))
            # print("np.shape(r_mean) :", np.shape(r_mean))
            
            new = np.append(l_mean, r_mean, axis=0).reshape(2, -1)
            
            # print("new :", new)
            # print("new.shape :", new.shape)
            # print("line.shape :", line.shape)
            # print("np.mean(new, axis=0) :", np.mean(new, axis=0))
            
            line = np.append(line, np.mean(new, axis=0), axis=0)
            
            # print("len(line) :", len(line))
            
            # print(line)
            
            # print()
            # print()
            # print()
            # print()
            
    line = line.reshape(-1, 2)
    print(line.shape)
    print(line)
    
    if line.shape[0] > 10:
        drawLine(img, line[0], line[-1])
        # for x in line:
        #     cv2.circle(img, np.uint32(x), 1, (0, 255, 0), 0)
    # print(line)
    # ling = np.asarray(line)
    # img = cv2.polylines(img, np.uint32([line]), 1, (0, 255, 0), 2)
        
    
        

    cv2.imshow("img", img)
    print("Time : {}".format(time.time() - current))

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
