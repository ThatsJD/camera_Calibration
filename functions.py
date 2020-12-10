import cv2

def simpleParamSetup():
    param=cv2.SimpleBlobDetector_Params();
    param.minThreshold = 8
    param.maxThreshold = 255
    # Filter by Area.
    param.filterByArea = True
    param.minArea = 50
    param.maxArea = 10e5
    # Filter by Circularity
    param.filterByCircularity = True
    param.minCircularity = 0.8
    # Filter by Convexity
    param.filterByConvexity = True
    param.minConvexity = 0.87
    # Filter by Inertia
    param.filterByInertia = True
    param.minInertiaRatio = 0.01
    return param