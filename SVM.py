from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


def getPrecision(n_tp, n_fp):
    # accuacy of the positive predictions
    # percentage of true positives
    if n_tp + n_fp == 0:
        return 0
    pr = n_tp / (n_tp + n_fp)
    return pr


def getRecall(n_tp, n_fn):
    # (+) instance correctly detected by classifier
    if n_tp + n_fn == 0:
        return 0
    rc = n_tp / (n_tp + n_fn)
    return rc


def getFoneScore(n_tp, n_fn, n_fp):
    # performance score: harmonic mean(调和平均数) of precision and recall
    if n_tp + (n_fn + n_fp) / 2 == 0:
        return 0
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
        if actual[i] == -1 and predictions[i] == -1:
            actualNoPredictedNo += 1
        if actual[i] == -1 and predictions[i] == 1:
            actualNoPredictedYes += 1
        if actual[i] == 1 and predictions[i] == -1:
            actualYesPredictedNo += 1
        if actual[i] == 1 and predictions[i] == 1:
            actualYesPredictedYes += 1

    return actualNoPredictedNo, actualNoPredictedYes, actualYesPredictedNo, actualYesPredictedYes


def getAccuacy_and_GenError(total_predicts, yes, no):
    accuracy = (yes + no) / total_predicts
    generror = 1 - accuracy
    return accuracy, generror


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


# model code
# Batch GD find w, b
class Linear_SVC:

    def __init__(self, C=1, max_iter=100, tol=None, learning_rate='constant', learning_rate_init=0.001, t_0=1, t_1=1000,
                 early_stopping=False, validation_fraction=0.1, **kwargs):
        self.C = C  # regularization/penalty coefficient (float)
        self.max_iter = max_iter  # max iterations                     (int)
        self.tol = tol  # tolerance for optimization         (float)
        self.learning_rate_type = learning_rate  # 'constant' learning rate / 'adaptive' : function
        self.learning_rate = learning_rate_init
        self.t_0 = t_0
        self.t_1 = t_1
        self.early_stopping = early_stopping  # set aside a fraction of training data as validation and terminate training when not improving (boolean)
        self.validation_fraction = validation_fraction  # btw (0,1) proportion of training data to set aside as validation, use when early_stopping = True (float)
        self.kwargs = kwargs

    '''
    batch GD
    weight vector => w (1d array for each attribute)
    intercept/bias => b (1d array for each W * X)
    self.intercept_ = np.array([b])
    self.coef_ = np.array([w])
    self.support_vectors_ = ...
    store the cost values for each iteration so later can use to create a learning curve

    X: nd array
    Y: nd array

    print total iterations in the end
    '''

    def fit(self, X, Y):  # DO NOT TOUCH, WORKS
        self.w = np.zeros((X.shape[1],))  # zeros
        self.b = 0  # intercept/bias: (n,)
        self.X = X  # (n,m) | n samples (24,2)
        self.Y = Y  # (n,1) (24,) # 2 type of iris: Iris-Virginica(1) and others(0)
        self.sv = {
            'X': np.array([[math.inf], [math.inf]]),  # X (m,2)
            'Y': np.array([math.inf])
        }

        if self.early_stopping:
            if self.validation_fraction < 1:
                self.X = self.X[0:len(X) * self.validation_fraction]
                self.Y = self.Y[0:len(Y) * self.validation_fraction]
            else:
                self.X = self.X[0:self.validation_fraction]
                self.Y = self.Y[0:self.validation_fraction]

        costarray = []

        self.findSV()  # (xs,ys)
        costarray.append(math.inf)

        for i in range(0, self.max_iter):  # for loop works, do not touch
            self.findSV()
            cost = np.sum(self.cost(self.sv))
            dw_J = np.subtract(self.w, self.C * np.sum(self.sv['X'], axis=0))
            db_J = -self.C * np.sum(self.sv['Y'])
            self.w -= self.learning_rate * dw_J
            self.b -= self.learning_rate * db_J

            costarray.append(cost)

    def findSV(self):  # DO NOT TOUCH, WORKS
        idx = (((self.X.dot(self.w) + self.b) * self.Y) < 1).ravel()
        self.sv['X'] = self.X[idx]
        self.sv['Y'] = self.Y[idx]

    def predict(self, X):  # only predict class -1(others) or class 1(setosa) TODO: Debug predict
        p = self.w.transpose() * X
        # scalar vector confused
        y_hat = p + self.b
        predict_array = []
        for y in y_hat:
            one = np.ones(y.shape)
            if np.all(np.greater_equal(y, one)):
                predict_array.append(1)  # iris
            else:
                predict_array.append(-1)  # others
        return predict_array

    def cost(self, sv):  # works , do not touch

        return 0.5 * self.w.T.dot(self.w) + self.C * (
                np.sum(1 - sv['X'].dot(self.w)) - np.multiply(self.b, np.sum(sv['Y'])))


from sklearn.datasets import load_iris
import numpy as np

data = load_iris()  # ndarray
# print(data.target[[10, 25, 50]])
# print(list(data.target_names))
data_names = data.feature_names[2:4]
X = data['data'][:, (2, 3)]  # petal length, petal width
Y = (data['target'] == 2).astype(np.int)  # 1 virginica ! , -1 not!

for i in range(0, len(Y)):
    if Y[i] == 0:
        Y[i] = -1
# print(X)
# print(Y)

# scale
X_scale = scaler.fit_transform(X)
X = X_scale


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


X_train, y_train, X_test, y_test = partition(X, Y, 0.8)

k = 5
kfoldsData, kfoldsLabel = sPartition(k, X_train, y_train)

learning_rate = [0.1, 0.01, 0.001]
max_iter = 1000
tol = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
C = [0, 1, 2, 3, 4, 5]  # around 5 pts are close and diff type in graph base on looking
training_score = []
validation_score = []

print("result of model fit and predict with fold 0:\n\n")
for i in range(0, len(learning_rate)):
    for j in range(0, len(tol)):
        for k in range(0, len(C)):
            model = Linear_SVC(C=C[k], learning_rate_init=learning_rate[i], max_iter=max_iter, tol=tol[j])
            # kfolds
            # for i in range(0, len(kfoldsData)):
            #     for j in range(0, len(kfoldsData)):
            #         if i == j:
            print("learning rate = %d, tolerance = %d, c=%d " % (learning_rate[i], tol[j], C[k]))
            model.fit(kfoldsData[0], kfoldsLabel[0])
            print("model finish fitting, ready for predicting:")
            # else:
            prediction = model.predict(kfoldsData[0])  # -1 class0 1 class1 or -1 class1 1 class2
            actual = kfoldsLabel[0]  # 2 type of iris: Iris-Virginica(1) and others(0)

            print("input X:")
            print(kfoldsData[0])
            print("prediction:")
            print(prediction)
            print("actual:")
            print(actual)
            print("======================================\n")
            # (tn, fp, fn, tp) = createCFmatrix(actual, prediction)
            # precision_val = getPrecision(tp, fp)
            # recall_val = getRecall(tp, fn)
            # fone_val = getFoneScore(tp, fn, fp)
            # accuracy_val, generror = getAccuacy_and_GenError(len(prediction), tp + tn,
            #                                                  fp + fn)  # total_predicts,yes,no
