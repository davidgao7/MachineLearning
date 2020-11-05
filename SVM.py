from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import math


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

    def fit(self, X, Y):
        self.w = np.zeros((X.shape[1],))
        self.b = 0  # intercept/bias: (n,)
        self.X = X  # (n,m) | n samples (24,2)
        self.Y = Y  # (n,1) (24,) # 2 type of iris: Iris-Virginica(1) and others(0)
        self.sv = {
            'X': np.empty((1, len(self.X[1]))),  # X (m,2)
            'label': np.empty(1)
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
        J_0 = self.cost(self.sv)
        costarray.append(J_0)

        for i in range(0, self.max_iter):
            if self.learning_rate_type == 'adaptive':
                self.learning_rate = self.t_0 / (self.max_iter + self.t_1)

            if i > 0 and costarray[i] < costarray[i - 1] - self.tol:
                break

            dw_J = self.C * np.sum(self.sv['X'])  # float
            sumX = np.ones((len(self.w),))
            sumX.fill(dw_J)
            delta_w = self.w - sumX
            db_J = np.ones(X[0].shape)
            db_J.fill(-1 * self.C * np.sum(
                self.sv['label']))  # TODO: J * label=0 will be 0, I think label has to be +1/-1 not +1/0
            self.w = self.w - self.learning_rate * delta_w
            self.b = self.b - self.learning_rate * db_J
            self.findSV()
            cost = self.cost(self.sv)
            costarray.append(cost)

    def isSV(self, Y, X):
        val = (X.dot(self.w) + self.b) * Y

        # one = np.ones(val.shape)
        # zero = np.zeros(val.shape)
        # greaterOne = np.greater_equal(val, one)
        # lessZero = np.less_equal(val, zero)
        # if np.all(greaterOne) or np.all(lessZero):  # is the support vectors
        #     return True
        # else:
        #     return False

    def findSV(self):
        val = (((self.X.dot(self.w) + self.b) * self.Y) < 1).ravel()  # find each val <1 and create index for those
        self.sv['X'] = self.X[val]  # get the index with true
        self.sv['Y'] = self.Y[val]
    # def findSV(self):
    #     for i in range(0, len(self.X)):
    #         if self.isSV(self.Y[i], self.X[i]):
    #             if i == 0:
    #                 self.sv['X'][0] = self.X[i]
    #                 self.sv['label'][0] = self.Y[i]
    #             else:
    #                 self.sv['X'] = np.vstack((self.sv['X'], self.X[i]))
    #                 self.sv['label'] = np.append(self.sv['label'], self.Y[i])

    '''
    X: (nd array) samples for prediction
    returns 1d array predicted labels

    use self.coef_[0], self.intercept_[0] for prediction   
    '''

    def predict(self, X):  # only predict class -1(others) or class 1(setosa)
        p = self.w.transpose() * X
        # scalar vector confused
        y_hat = p + self.b
        predict_array = []
        for y in y_hat:
            one = np.ones(y.shape)
            if np.all(np.greater_equal(y, one)):
                predict_array.append(0)
            else:
                predict_array.append(1)
        return predict_array

    def cost(self, sv):

        return 0.5 * self.w.T.dot(self.w)  + self.C * (
                np.sum(1 - sv['X'].dot(self.w)) - self.b * np.sum(sv['label']))


data = load_iris()  # ndarray

data_names = data.feature_names[2:4]
X = data['data'][:, (2, 3)]  # petal length, petal width (150,2)
# 2 type of iris: setosa(1) and others(-1)
Y = (data['target'] == 2).astype(np.int)  # 0,1
for i in range(0, len(Y)):
    if Y[i] == 0:
        Y[i] = -1
target_names = data.target_names


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

learning_rate = []
learning_rate_init = 0.01
max_iter = 1000
tol = 0.001
training_score = []
validation_score = []

# kfolds
model = Linear_SVC(learning_rate_init=learning_rate_init, max_iter=max_iter, tol=tol, C=50)
for i in range(0, len(kfoldsData)):
    for j in range(0, len(kfoldsData)):
        if i == j:
            model.fit(kfoldsData[i], kfoldsLabel[i])
        else:
            prediction = model.predict(kfoldsData[i])  # -1 class0 1 class1 or -1 class1 1 class2
            actual = kfoldsLabel[i]  # 2 type of iris: Iris-Virginica(1) and others(0)
            print("i = %d" % i)
            print("input X:")
            print(kfoldsData[i])
            print("prediction:")
            print(prediction)
            print("actual:")
            print(actual)
            print("\n\n")
