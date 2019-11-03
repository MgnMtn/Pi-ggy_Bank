import numpy as np
import cv2

from classification import predict

cap = cv2.VideoCapture(2)

import random

count = float('inf')
file = open('file.txt', 'w+')

def mode(list):

    twop = 0
    twopound = 0

    for item in list:
        if item == '2p':
            twop += 1
        else:
            twopound += 1
    
    total = twop + twopound

    if (twopound/total) > 0.35:
        return '2pound'
    else:
        return '2p'

    return max(set(list), key=list.count)

def save_keypoints(keypoints, frame):

    decisions = []

    for index, kp in enumerate(keypoints):
        x, y = kp.pt
        size = kp.size

        frame = frame[
            int(y)-int(size//2):int(y)+int(size//2),
            int(x)-int(size//2):int(x)+int(size//2),
        ]

        cv2.imwrite('images/%s.png' % (random.getrandbits(128)), frame)


        # if frame.shape[0]:
        #     decisions.append(predict(frame))
    print(decisions)

    return decisions
    # print(mode(decisions))

        # print(predict(frame))

while(True):
    # Capture frame-by-frame
    ret, im = cap.read()

    # lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    # lab_planes = cv2.split(lab)
    # gridsize = 5
    # clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    # lab_planes[0] = clahe.apply(lab_planes[0])
    # lab = cv2.merge(lab_planes)
    # im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # im = cv2.normalize(im,  im, 0, 255, cv2.NORM_MINMAX)

    color_im = im
    # cimg = cv2.medianBlur(img,5)
    frame = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im = frame

    x = 120
    y = 100
    w = 300
    h = 200
    im = im[y:y+h, x:x+w]

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 5000

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500
    params.maxArea = 6000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.1

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    show = color_im[y:y+h, x:x+w]

    im_with_keypoints = cv2.drawKeypoints(show, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('frame', im_with_keypoints)

    im = im_with_keypoints
    


    # Display the resulting frame
    key = cv2.waitKey(1)

    if key == 32:
        count = 0
        decisions = []



    if count < 20:
        color_im = color_im[y:y+h, x:x+w]
        decisions += save_keypoints(keypoints, color_im)
    
    # if count == 20:
    #     print('Final: ', mode(decisions))

        # color_im = color_im[y:y+h, x:x+w]
        # im_with_keypoints = cv2.drawKeypoints(color_im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # cv2.imwrite('frame%d.png' % count, im_with_keypoints)


    count += 1
    if key & 0xFF == ord('q'):
        break

file.close()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
