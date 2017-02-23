import os
import struct
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

# for multithreading
from multiprocessing import Pool
from functools import partial
from sklearn import model_selection

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise (ValueError, "dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def parseData(dataSet):
    retValue = read(dataset = dataSet)
    X_train = []
    Y_train = []
    for val in retValue:
        X_train.append(val[1].flatten())
        Y_train.append(val[0])

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    return (X_train, Y_train)

def getData():
    (X_train, Y_train) = parseData("training")
    (X_test, Y_test) = parseData("testing")
    #print('returning data')
    return (X_train, Y_train, X_test, Y_test)


def getBinaryClassifierWeights(X_train, Y_train, c=0.1):
    assert len(np.unique(Y_train)) == 2, "binary classifier can't do multiclassification"
    clf = svm.LinearSVC(C=c)
    clf.fit(X_train, Y_train)
    #print("returning classifier weights")
    return clf.coef_

def trainBinary(X_train, Y_train, c, clas):
    bin_Y_train = np.ma.masked_equal(Y_train, clas).recordmask
    return getBinaryClassifierWeights(X_train, bin_Y_train, c)
    
if __name__ == "__main__":

    # get data
    (X_train, Y_train, X_test, Y_test) =  getData()
    # get unique classes
    classes = np.unique(Y_train)

    # create multiprocess pool for training clasess in parallel
    # greatly speeds up training
    pool = Pool(8)

    # maps c values to resulting model accuracy
    c_acc = {}

    # build cross validation indices
    kf = model_selection.KFold(n_splits=3, shuffle=False)

    # top loop over hyperparameter C
    for c in [ pow(10, x) for x in range(-8, 2) ]:

        print("c", c)

        # list of accuracies
        acc = []

        for train_index, test_index in kf.split(X_train):

            # get the test/train sets
            X_train_kf, X_test_kf = X_train[train_index], X_train[test_index]
            Y_train_kf, Y_test_kf = Y_train[train_index], Y_train[test_index]

            # curried function to pass to pool.map
            map_func = partial(trainBinary, X_train_kf, Y_train_kf, c)

            # getBinaryClassifier weights
            # some hackery required to properly format the output 
            weights = np.array(pool.map(map_func, classes)) \
                .reshape(len(classes), X_train_kf.shape[1]).T

            # predict using weights vector
            prediction = np.argmax(np.dot(X_test_kf, weights), axis=1)
            acc.append( accuracy_score(Y_test_kf, prediction) )

        # update the c_acc dictionary
        c_acc[c] = sum(acc)/len(acc)

    print(c_acc)

    # get the c with highest accuracy
    best_c = max(c_acc, key=c_acc.get) 

    print("best c", best_c)

    # build models on all training data with best c
    map_func = partial(trainBinary, X_train, Y_train, best_c)
    weights = np.array(pool.map(map_func, classes)) \
        .reshape(len(classes), X_train.shape[1]).T

    # predict on test set
    prediction = np.argmax(np.dot(X_test, weights), axis=1)
    print("test accuracy", accuracy_score(Y_test, prediction) )

    # 10-fold cross validation from 10^-8 to 10^2 yielded c=10^-6 and accuracy 91.68%
    # took about an hour to run
