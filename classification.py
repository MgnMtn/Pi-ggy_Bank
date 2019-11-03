import cv2 as cv
import numpy as np
import argparse
import os

np.random.seed(2)

path_2p = 'images_bad/2p'
all_2ps = [ os.path.join(path_2p, f) for f in os.listdir(path_2p) ]
np.random.shuffle(all_2ps)
train_2ps = all_2ps[30:]
test_2p = all_2ps[:30]

path_2pounds = 'images_bad/2pound'
all_2pounds = [ os.path.join(path_2pounds, f) for f in os.listdir(path_2pounds) ]
np.random.shuffle(all_2pounds)
train_2pounds = all_2pounds[20:]
test_2pounds = all_2pounds[:20]

def get_features(image_src):
    if type(image_src) is str:
        src = cv.imread(image_src)
    else:
        src = image_src
    if src is None:
        print('Could not open or find the image:', image_src)
        exit(0)
    bgr_planes = cv.split(src)
    histSize = 30
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = True
    b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)


    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

    return np.array([r_hist, b_hist, r_hist]).flatten()

train_2ps = np.array([ np.append(get_features(f), 0) for f in train_2ps])
train_2pounds = np.array([ np.append(get_features(f), 1) for f in train_2pounds])

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

# training = shuffle_along_axis(np.vstack((train_2pounds, train_2ps)), axis=0)

training = np.vstack((train_2pounds, train_2ps))
np.random.shuffle(training)

X = training[:, :-1]
y = training[:, -1]

test = np.array([ get_features(f) for f in test_2p ])

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

clf = svm.SVC()
clf = RandomForestClassifier()
clf = MLPClassifier()
# clf = svm()
clf.fit(X, y)  

def predict(img):
    feats = get_features(img)
    label = clf.predict([feats])[0]

    if label == 0:
        return '2p'
    elif label == 1:
        return '2pound'



# print(clf.predict(test))
test = np.array([ get_features(f) for f in test_2pounds ])

print(clf.predict(test))

test = np.array([ get_features(f) for f in test_2p ])

print(clf.predict(test))


