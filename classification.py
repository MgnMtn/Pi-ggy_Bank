import cv2 as cv
import numpy as np
import argparse
import os

np.random.seed(2)

path_2p = 'images/2p'
all_2ps = [ os.path.join(path_2p, f) for f in os.listdir(path_2p) ]
np.random.shuffle(all_2ps)
train_2ps = all_2ps[5:]
test_2p = all_2ps[:5]

path_2pounds = 'images/2pound'
all_2pounds = [ os.path.join(path_2pounds, f) for f in os.listdir(path_2pounds) ]
np.random.shuffle(all_2pounds)
train_2pounds = all_2pounds[5:]
test_2pounds = all_2pounds[:5]

path_10p = 'images/10p'
all_10ps = [ os.path.join(path_10p, f) for f in os.listdir(path_10p) ]
np.random.shuffle(all_10ps)
train_10ps = all_10ps[5:]
test_10p = all_10ps[:5]

path_50p = 'images/50p'
all_50ps = [ os.path.join(path_50p, f) for f in os.listdir(path_50p) ]
np.random.shuffle(all_50ps)
train_50ps = all_50ps[5:]
test_50p = all_50ps[:5]


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

clf = svm.SVC(gamma='scale')
data = np.vstack((
    [ np.append(get_features(f), 0) for f in all_2ps ],
    [ np.append(get_features(f), 1) for f in all_10ps ],
    [ np.append(get_features(f), 2) for f in all_50ps ],
    [ np.append(get_features(f), 3) for f in all_2pounds ],
))

# clf = RandomForestClassifier()
# clf = MLPClassifier()
# clf = svm()
clf.fit(data[:, :-1], data[:, -1])  

def predict(img):
    feats = get_features(img)
    label = clf.predict([feats])[0]

    if label == 0:
        return '2p'
    elif label == 1:
        return '10p'
    elif label == 2:
        return '50p'
    elif label == 3:
        return '2 pounds'


def loo(data):
    # nj
    n_samples = data.shape[0]
    correct = 0

    for i in range(n_samples):
        test = data[i]
        train = np.delete(data, i, axis=0)
        np.random.shuffle(train)
        # clf = RandomForestClassifier(n_estimators=10)
        clf = svm.SVC(gamma='scale')
        clf.fit(train[:, :-1], train[:, -1])

        guess = clf.predict([test[:-1]])
        correct += test[-1] == guess
    
    return (correct / n_samples)[0]

# print(all_2ps[0], all_2pounds[0])

if __name__ == "__main__":
    

    np.random.shuffle(data)
    print(loo(data))

# # print(clf.predict(test))
# test = np.array([ get_features(f) for f in test_2pounds ])

# print(clf.predict(test))

# test = np.array([ get_features(f) for f in test_2p ])

# print(clf.predict(test))


