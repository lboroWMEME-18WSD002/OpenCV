import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red colour in HSV
    lower_red = np.array([110,100,200])
    upper_red = np.array([130,255,255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    # Blur and close holes to help filter out noise etc
    kernel=np.ones((15,15), np.uint8)
    close=cv2.morphologyEx(res,cv2.MORPH_CLOSE,kernel)
    
    # Detect any remaining blobs and filter them by how dark they are
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor=1
    params.blobColor=100
    blobs=cv2.SimpleBlobDetector_create()
    
    #draw blobs onto the frame
    keypoints=blobs.detect(close)
    im_with_keypoints=cv2.drawKeypoints(frame,keypoints, np.array([]),(255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # display red components of image and the final image
    cv2.imshow('res',close)
    cv2.imshow('Output',im_with_keypoints)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
