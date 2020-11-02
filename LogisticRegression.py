import math
import matplotlib.pyplot as plt
'''
True positive(tp): correctly classified or detected.
False positive(fp): incorrectly classified or detected. (type I error)
False negative(fn): incorrectly rejected. (type II error)
True negative(tn): correctly rejected. 
'''


# no need to test since it's formula on ppt
def getPrecision(n_tp, n_fp):
    # accuacy of the positive predictions
    # percentage of true positives
    pr = n_tp / (n_tp + n_fp)
    return pr


def getRecall(n_tp, n_fn):
    # (+) instance correctly detected by classifier
    rc = n_tp / (n_tp + n_fn)
    return rc


def getFoneScore(n_tp, n_fn, n_fp):
    # performance score: harmonic mean(调和平均数) of precision and recall
    F_1 = n_tp / (n_tp + (n_fn + n_fp) / 2)
    return F_1


def getROC(TP_rate, FP_rate):
    plt.plot(FP_rate, TP_rate)
    plt.ylabel('True Positive rate')
    plt.xlabel('False Positive rate')
    for i_x, i_y in zip(TP_rate, FP_rate):
        plt.text(i_x, i_y, '({:.3f},{:.3f})'.format(i_x, i_y))
    plt.show()


def createCFmatrix(actual, predictions):
    actualNoPredictedNo = 0
    actualNoPredictedYes = 0
    actualYesPredictedNo = 0
    actualYesPredictedYes = 0

    if len(predictions) != len(actual):
        print("missing pair outputs: # predictions not corresponding # actual output!\n")
    for i in range(len(predictions)):
        if actual[i] == 0 and predictions[i] == 0:
            actualNoPredictedNo += 1
        if actual[i] == 0 and predictions[i] == 1:
            actualNoPredictedYes += 1
        if actual[i] == 1 and predictions[i] == 0:
            actualYesPredictedNo += 1
        if actual[i] == 1 and predictions[i] == 1:
            actualYesPredictedYes += 1

    return (actualNoPredictedNo, actualNoPredictedYes, actualYesPredictedNo, actualYesPredictedYes)


def getAccuacy_and_GenError(total_predicts, yes, no):
    accuracy = (yes + no) / total_predicts
    generror = 1 - accuracy
    return (accuracy, generror)


def integralAUC_ROC(FP_rate1, FP_rate2, TP_rate):
    area = 0
    for i in range(FP_rate1, FP_rate2 + 1):
        # rectangle area
        area += TP_rate[i]
    return area


def getPrecisionRecall_curve(precision, recall):
    plt.plot(precision, recall)
    plt.xlabel('precision')
    plt.ylabel('recall')
    for i_x, i_y in zip(precision, recall):
        plt.text(i_x, i_y, '({:.3f},{:.3f})'.format(i_x, i_y))
    plt.show()


def partition(features, target, t):
    trainingSize = math.floor(len(features) * t)
    testSize = len(features) - trainingSize

    trainingFeature = np.array([])
    trainingTarget = np.array([])
    testFeature = np.array([])
    testTarget = np.array([])

    index = 0
    for row in features:
        if index < trainingSize:
            feature = np.array(row)
            TraintargetValue = target[index]
            if trainingFeature.size == 0:
                trainingFeature = np.append(trainingFeature, feature)
            else:
                trainingFeature = np.vstack((trainingFeature, feature))
            trainingTarget = np.append(trainingTarget, TraintargetValue)
            index += 1
        else:
            feature = np.array(row)
            TesttargetValue = target[index]
            if testFeature.size == 0:
                testFeature = np.append(testFeature, feature)
            else:
                testFeature = np.vstack((testFeature, feature))
            testTarget = np.append(testTarget, TesttargetValue)
            index += 1
    return trainingFeature, trainingTarget, testFeature, testTarget


def sPartition(k, data, labels):
    remainder = len(data) % k
    if remainder == 0:
        kfoldsData = np.vsplit(data, k)
        kfoldsLabel = np.split(labels, k)
    else:
        dataReadysplit = data[:-remainder, :]  # remove last n rows
        labelReadysplit = labels[:-remainder]
        kfoldsData = np.vsplit(dataReadysplit, k)
        kfoldsLabel = np.split(labelReadysplit, k)
        remainData = data[-remainder:]
        remainLabel = labels[-remainder:]
        kfoldsData[-1] = np.vstack((kfoldsData[-1], remainData))
        kfoldsLabel[-1] = np.append(kfoldsLabel[-1], remainLabel)
    return kfoldsData, kfoldsLabel


from sklearn.datasets import load_iris
import numpy as np

data = load_iris()  # ndarray
# print(data.target[[10, 25, 50]])
# print(list(data.target_names))

data_names = data.feature_names
X = data['data'][:, (0, 1, 2, 3)]  # sepal length, sepal width, petal length, petal width
Y = (data['target']).astype(np.int)  # 0,1,2
# shuffle data
data = load_iris(as_frame=True).frame
data = data.sample(frac=1)
import pandas as pd

data = pd.get_dummies(data)
most_correlated = data.corr().abs()['target'].sort_values(ascending=False)
most_correlated = most_correlated[:6]

feature = data.loc[:, most_correlated[3:5].index]
feature = feature.to_numpy()
target = Y
X_train, X_test, y_train, y_test = partition(feature, target, 0.8)
k = 5
(kfoldsData, kfoldsLabel) = sPartition(k, X_train, X_test)


class Softmax_Regression:
    # convert vector of class indices ==> matrix contain a one-hot vector for each instance
    # Y: 1d array , 'int' , represents class indices/labels
    # https://unl.zoom.us/rec/play/JSWuvdAUl4wugi3uoUZGZiPeHk0xMldEiN4A0Yt6jVC7-94ivP_jpUqeSz4yUXlHnYIL7NSpGNcmVDFF.n0WGpwDLAX1Ycowq?startTime=1602781012000&_x_zm_rtaid=bG1a-BqvRKucpqkCrsUanQ.1603303389056.33e1e95920df5ac78eb899651f52ede9&_x_zm_rhtaid=535
    # 13:50
    # Y=> value are [0,1,2] 三种花
    # iris_data classes: 3
    def one_hot_label(self, Y):
        label_matrix = np.zeros((len(Y), 3))
        flower_index = 0
        for flower in Y:
            if flower == 0:
                label_matrix[flower_index][0] = 1
            elif flower == 1:
                label_matrix[flower_index][1] = 1
            elif flower == 2:
                label_matrix[flower_index][2] = 1
            flower_index += 1
        return label_matrix

    # score: wx*b
    def softmax(self, score):  # 1.score: [number_of_feature x classes=3]==>score for each class 2. normalize the score
        # pass each score through exponential function and divide by the sum of exponential of all scores
        expMatrix = np.exp(score)
        total = np.sum(expMatrix)
        probMatrix = np.true_divide(expMatrix, total)
        return probMatrix

    def cross_entropy_loss(self, Y_one_hot, Y_proba):  # negative log likelihood
        # Y_one_hot: ndarray of each class labels for each instance:[[0 0 0 1][0 1 0 0][1 0 0 0]...]
        # Y_proba: ndarray of probability of a sample belonging to various classes
        # return cost(float)
        loss = -1 * np.sum(np.multiply(np.log(Y_proba), Y_one_hot))  # multiply: elementwise
        return loss

    # first order derivative of the loss in the gradient descent
    def fit(self, X, Y, learning_rate=0.01, epochs=1000, tol=None, regularizer=None, lambd=0.0, early_stopping=False,
            validation_fraction=0.1, **kwargs):  # batch gradient descent

        cost_history = np.ones(epochs) * np.inf
        self.W = np.ones((3, len(X[0])))  # 3 classes * 2 features

        for i in range(epochs):
            y_hat = X.dot(self.W.transpose())  # score: predict probability
            y_hat_normalized = self.softmax(y_hat)  # (24,3)
            y_one_hot = self.one_hot_label(Y)  # (24,3)
            loss = self.cross_entropy_loss(y_one_hot, y_hat_normalized)

            if regularizer == 'l1':
                loss += lambd * np.sum(np.abs(self.W))

            if regularizer == 'l2':
                loss += lambd * np.sum(np.square(self.W))

            cost_history[i] = loss
            if tol is not None and cost_history[i - 1] < cost_history[i] - tol:
                break
            if cost_history[i - 1] < cost_history[i] or tol is not None and cost_history[i - 1] < cost_history[i] - tol:
                break
            if early_stopping:
                dataFraction = len(X) * validation_fraction
                X = X[0:dataFraction - 1]
                Y = Y[0:dataFraction - 1]
            else:
                # W: (d, c)
                eta = learning_rate
                b = np.transpose(X)
                c = y_hat_normalized - y_one_hot  # y_hat: miu
                b = np.append(b, np.ones((1, len(b[0]))), axis=0)  # bias?
                e = np.dot(b, c)  # gradient of weight
                d = (eta / len(X)) * e  # gradient of weight
                if self.W.shape != d.shape:
                    d = d[:, 1:]  # get rid of the negative first col
                self.W = self.W - d

    def predict(self, X):
        y = np.dot(X, self.W.transpose())
        y_soft = self.softmax(y)
        return y_soft

    def __init__(self):
        pass

    def getW(self):
        return self.W

    def setW(self, w):
        self.W = w


lambd = [0.1, 0.01, 0.001, 0.0001]
tol = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
learning_rate = [0.1, 0.01, 0.001]
regularizer = ['l1', 'l2']

precision = []
accuracy = []
recall = []
fone = []
TPR = []
FPR = []
model = Softmax_Regression()
W = []

for lamd_val in lambd:
    for tol_val in tol:
        for rate in learning_rate:
            for regular in regularizer:
                for i in range(0, len(kfoldsData) - 1):
                    total_predicts = 0
                    yes = 0
                    no = 0
                    actual = []
                    predictions = []
                    for j in range(0, len(kfoldsData) - 1):
                        if j == i:
                            model.fit(kfoldsData[j], kfoldsLabel[j], lambd=lamd_val, tol=tol_val, learning_rate=rate,
                                      regularizer=regular)
                        else:
                            spamProb = model.predict(kfoldsData[j])  # 1spam0not
                            for predicts in spamProb:
                                predictions.append(predicts)
                                total_predicts += 1
                            index = 0
                            for actuals in kfoldsLabel[j]:
                                actual.append(actuals)
                                probSummary = np.sum(spamProb, axis=0)
                                predict_classIndex = np.argmax(probSummary)
                                if predict_classIndex == actuals:
                                    yes += 1
                                else:
                                    no += 1
                                index += 1
                            # d. confusion matrix
                            predic = []
                            for data in predictions:
                                clasPredict = np.argmax(data)
                                predic.append(clasPredict)
                            # (actualNoPredictedNo, actualNoPredictedYes, actualYesPredictedNo, actualYesPredictedYes)
                            (tn, fp, fn, tp) = createCFmatrix(actual, predic)
                            # a. precision
                            numpyactual = np.array(actual)
                            if tp + fp != 0:
                                precision_val = getPrecision(tp, fp)
                            else:
                                precision_val = 0  # in case of division 0
                            precision.append(precision_val)
                            # b. recall
                            if tp + fn != 0:
                                recall_val = getRecall(tp, fn)
                            else:
                                recall_val = 0
                            recall.append(recall_val)
                            # c. f1 score
                            if tp + (fn + fp) / 2 == 0:
                                fone_val = 0
                            else:
                                fone_val = getFoneScore(tp, fn, fp)
                            fone.append(fone_val)
                            # e. accuracy
                            accuracy_val, generror = getAccuacy_and_GenError(len(predictions), yes,
                                                                             no)  # total_predicts,yes,no
                            print("error = %.3f" % generror)
                            print("accuracy = %.3f" % accuracy_val)
                            print("w: ")
                            print(model.getW())
                            W.append(model.getW())
                            accuracy.append(accuracy_val)
                            # TPR and FPR
                            if tp + fn == 0:
                                tpr = 0
                            else:
                                tpr = tp / (tp + fn)
                            if fp + tn == 0:
                                fpr = 0
                            else:
                                fpr = fp / (fp + tn)
                            TPR.append(tpr)
                            FPR.append(fpr)

                        # precision-recall curve
                        getPrecisionRecall_curve(precision, recall)

                    # roc curve
                    getROC(TPR, FPR)

print(W[4])
