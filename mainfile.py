import numpy as np
import cv2
import glob
import functions
import os
import matplotlib.pyplot as plt

ImageSymmetry=("Asymmetric","Symmetric")
ImageFlag=ImageSymmetry[1]
ImageDir="TestImages_" + ImageFlag
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)

# points
patternsize = (11, 8)
if ImageFlag == "Asymmetric":
    #Object points are randomly generated. Because We actually dont know about size of Dots.
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((4 * 11, 3), np.float32)
    objp[:, :2] = np.mgrid[0:4, 0:11].T.reshape(-1, 2)
else:
    objp = np.zeros((7* 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
ImageName=[]
ImageWidth=[]
ImageHeight=[]
if (ImageFlag== ImageSymmetry[0]):
    images = glob.glob(r'\Asymmetric/snapshot_640_480_' + "*.jpg")
else:
    images = glob.glob(r'\Symmetric/snapshot_640_480_' + "*.jpg")
print(images)
cv2.namedWindow("Keypoints")
for fname in images:
    img = cv2.imread(fname,0)
    fn = (fname.split('\\', 2))[2]
    gray = img.copy()
    ImageName.append(fn)
    w,h= img.shape[:2]
    ImageHeight.append(h)
    ImageWidth.append(w)
    param = functions.simpleParamSetup()
    detector = cv2.SimpleBlobDetector_create(parameters=param)

    # # Find the chess board corners
    if (ImageFlag== ImageSymmetry[0]):
        ret, corners = cv2.findCirclesGrid(gray, (4, 11), flags=(cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING), blobDetector=detector)
    else:
        ret, corners = cv2.findCirclesGrid(gray, (7, 6), flags=(cv2.CALIB_CB_SYMMETRIC_GRID +  cv2.CALIB_CB_CLUSTERING), blobDetector=detector)
    # # If found, add object points, image points (after refining them)
    print(str(ret) + " by " + fname)
    if ret == True:
        objpoints.append(objp)
        if (ImageFlag == ImageSymmetry[0]):
            corners2 = cv2.cornerSubPix(gray, corners, (4, 11), (-1, -1), criteria)
        else:
            corners2 = cv2.cornerSubPix(gray, corners, (7, 6), (-1, -1), criteria)

        imgpoints.append(corners2)
        #
        #     # Draw and display the corners
        img_color=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        if (ImageFlag == ImageSymmetry[0]):
            img_color = cv2.drawChessboardCorners(img_color, (4, 11), corners2, ret)
        else:
            img_color = cv2.drawChessboardCorners(img_color, (7, 6), corners2, ret)
        #cv2.imshow("Keypoints", img_color)

        print(fn)
        cv2.imwrite(ImageDir + "\\" + fn, img_color)
        # while (True):
        #     char = cv2.waitKey(0)
        #     if (chr(char & 255) == 'a'):
        #         break
#Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print ("Camera Matrix:" + str(mtx) )
print("Distortion Coefficient: " + str(dist))
print("Rotation Vector" + str(rvecs))
print("Translation Vector" + str(tvecs))

error_list=[]
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    error_list.append(error)
    mean_error += error
print ("Mean error: " + str( mean_error/len(objpoints) ))
plt.plot(ImageName,error_list)
plt.ylabel("Errors")
plt.xlabel("Images")

plt.show()

cv2.destroyAllWindows()
