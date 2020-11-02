import numpy as np
from KNNTest import KNN_Classifier
from sklearn.neighbors import KNeighborsClassifier  # for model testing

kNeighbors = [1, 5, 9, 11]
distance = ['euclidean', 'manhattan']
weights = ['uniform', 'distance']


def getAccuacy_and_GenError(total_predicts, yes, no):
    accuracy = (yes + no) / total_predicts
    generror = 1 - accuracy
    return accuracy, generror


def createCFmatrix(actual, predictions):
    actualNoPredictedNo = 0
    actualNoPredictedYes = 0
    actualYesPredictedNo = 0
    actualYesPredictedYes = 0

    for x in range(len(predictions)):
        if actual[x] == 0 and predictions[x] == 0:
            actualNoPredictedNo += 1
        if actual[x] == 0 and predictions[x] == 1:
            actualNoPredictedYes += 1
        if actual[x] == 1 and predictions[x] == 0:
            actualYesPredictedNo += 1
        if actual[x] == 1 and predictions[x] == 1:
            actualYesPredictedYes += 1

    return actualNoPredictedNo, actualNoPredictedYes, actualYesPredictedNo, actualYesPredictedYes


def sFold(K, data, labels, model, error_fuction, **model_args):
    folds = 4
    # do cross-validation:
    # 1. divide the training set into # folds
    trainingSet = sPartition(data, folds)
    expectedOutLabelBinaryArray = np.array([])
    predictedLabelBinaryArray = np.array([])
    avgErrArray = np.array([])
    avgAccuracyArray = np.array([])

    # 2. for value of nearest neighbor (e.g. k=2), we train (s-1) folds
    for i in range(0, folds):
        for j in range(0, folds):
            # 3 choose a performance measure(e.g. accuracy)
            if j != i:
                model.fit(trainingSet[j], labels, K, 'uniform')
    # 4 the performance scores form s runs are then averaged.
            (expectedOutLabelBinary, predictedLabelBinary, avgErr, averageAccuracy) \
                = trainModel(data, folds, model, error_fuction, model_args)
            expectedOutLabelBinaryArray = np.append(expectedOutLabelBinaryArray, expectedOutLabelBinary)
            predictedLabelBinaryArray = np.append(predictedLabelBinaryArray, predictedLabelBinary)
            avgErrArray = np.append(avgErrArray, avgErr)
            avgAccuracyArray = np.append(avgAccuracyArray, averageAccuracy)
    modelDict = {
        "Expected labels": expectedOutLabelBinaryArray,
        "Predicted labels": predictedLabelBinaryArray,
        "Average error": avgErrArray
    }
    return modelDict


# ===============helper functions=========================== a. use a helper function to calculate an s-partition of
# the data (i.e., partition the data into s equally sized portions)
def sPartition(features, folds):
    try:
        print("current shape of feature: %s" % features.shape)
        cutted_data = np.reshape(features, (folds, -1))
        return cutted_data
    except:
        # the value isn't enough for dividing n folds
        print("couldn't perfectly partition features into %d folds, the last row will be fragments" % folds)


def partition(data, t):
    size = len(data)
    train = data[0:int((size - 1) * t)]
    test = data[int((size - 1) * t) + 1:size]
    return train, test


def trainModel(data, folds, model, error_fuction, arg):
    # data ==> cutted_data
    # get training and test subset
    global expectedOutLabelBinary, predictedLabelBinary
    t = 0.8
    (training, testing) = partition(data, t)
    cutted_data = sPartition(training, folds)
    avgErr = 0
    I = 0
    J = 0
    averageAccuracy = 0
    print("in train model")
    for i in range(0, len(cutted_data)):  # tarin model
        # isolated ith part
        expectedOutLabelBinary = []
        predictedLabelBinary = []
        for j in range(0, len(cutted_data)):
            # b. fit the data to all other partitions(1 - folds)
            nneighbors = 5
            if j != i:  # use other part to train
                # fit(self, X, Y, nNeighbors, weights, **kwargs
                model.fit(cutted_data[j], testing[i], nneighbors, 'uniform')
            # c. make prediction on current partition
            # d. store expected labels and predicted labels for current partition
            if j > 0:
                # Test model using *sklearn* knn model: #
                classifier = KNeighborsClassifier(n_neighbors=nneighbors)

                classifier.fit(cutted_data[j], testing[i])  # Expected 2D array, got scalar array instead:

                predictedLabel = model.predict(cutted_data)
                expectedOutLabel = classifier.predict(training[j])

                predictedLabelBinary.append(predictedLabel)
                expectedOutLabelBinary.append(expectedOutLabel)

                (actualNoPredictedNo, actualNoPredictedYes, actualYesPredictedNo,
                 actualYesPredictedYes) = createCFmatrix(
                    expectedOutLabelBinary, predictedLabelBinary)

                avgErr = avgErr + error_fuction(actualYesPredictedYes, actualNoPredictedNo, actualNoPredictedYes)
                averageAccuracy += getAccuacy_and_GenError(I * J, actualYesPredictedYes + actualNoPredictedNo,
                                                           actualNoPredictedYes + actualYesPredictedNo)
                # total_predicts,yes,no

            J += 1
        I += 1
    avgErr = avgErr / (I * J)
    averageAccuracy = averageAccuracy / (I * J)
    return expectedOutLabelBinary, predictedLabelBinary, avgErr, averageAccuracy


classifier = KNN_Classifier()
# sFold(folds, data, labels, model,  error_fuction, **model_args)
# return predictlabel , expectedlabel , average error

a = np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 8], [3, 4, 6, 2, 7], [2, 3, 7, 3, 6],
              [1, 3, 4, 7, 9], [2, 3, 6, 2, 10], [4, 4, 6, 2, 4], [8, 2, 7, 3, 6]])
b = np.array([[3, 4, 8, 3, 3], [2, 5, 9, 2, 8], [5, 9, 0, 3, 5], [2, 5, 9, 2, 8], [1, 3, 5, 7, 9], [2, 3, 8, 8, 10],
              [3, 4, 6, 2, 7], [2, 3, 7, 3, 6]])
c = np.array([[1, 4, 5, 7, 9], [2, 4, 2, 2, 10], [3, 4, 5, 2, 7], [5, 3, 7, 3, 6], [1, 6, 5, 7, 9], [2, 4, 6, 8, 10],
              [7, 4, 6, 2, 7], [4, 3, 7, 3, 6]])
g = np.array([[1, 3, 5, 7, 9], [2, 34, 6, 6, 10], [3, 4, 6, 2, 7], [2, 3, 7, 3, 6]])
h = np.array([[1, 3, 4, 8, 7], [5, 4, 3, 8, 10], [3, 4, 6, 2, 7], [2, 3, 7, 3, 6]])
# i = np.array([[1, 5, 5, 7, 9], [2, 4, 4, 8, 8], [3, 4, 6, 2, 7], [2, 3, 8, 3, 6]])
# j = np.array([[1, 9, 5, 7, 9], [2, 6, 6, 8, 10], [3, 2, 6, 2, 7], [2, 3, 4, 3, 6]])
kNeighbors = np.array([[1, 3, 6, 7, 9], [2, 3, 6, 3, 10], [5, 4, 2, 2, 7], [2, 3, 7, 3, 6]])
l = np.array([[1, 3, 5, 7, 9], [2, 4, 6, 3, 10], [3, 4, 6, 2, 7], [2, 3, 4, 3, 6]])
m = np.array([[1, 3, 5, 3, 9], [2, 4, 3, 8, 10], [3, 4, 6, 2, 7], [3, 3, 7, 3, 6]])
target = np.array([1, 2, 4, 6])


def getFoneScore(n_tp, n_fn, n_fp):
    F_1 = n_tp / (n_tp + (n_fn + n_fp) / 2)
    return F_1


def testsFold():
    for i in range(1, len(kNeighbors)):
        # use sfold function to evaluate the performance of your model over each combination of k and distance matrics
        # store the returned dictionary for each
        # determine the best model based on the overall performance(lowest average error). for the *error_function* use
        # the f1 score function
        # getFoneScore(n_tp,n_fn,n_fp)
        # sFold(folds, data, labels, model,  error_fuction, **model_args):
        testsample = sFold(kNeighbors[i], a, target, classifier, getFoneScore)
        print("sample %d" % i)
        print("k = %d, test result: %s \n" % (kNeighbors[i], testsample))
        print("===========================================\n\n")


if __name__ == '__main__':
    testsFold()
