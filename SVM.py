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

        if self.learning_rate_type == 'adaptive':
            self.learning_rate = self.t_0 / (self.max_iter + self.t_1)

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
        self.b = np.zeros(X[0].shape)  # intercept/bias: (n,)
        self.X = X  # (n,m) | n samples (24,2)
        self.Y = Y  # (n,1) (24,)
        self.costarray = []

        self.w = np.ones(X[0].shape)  # w: (m,)
        sv = self.findSV()  # (xs,ys)
        J_0 = self.cost(sv)
        self.costarray.append(J_0)
        ephocs = 1000
        i = 0
        while i < ephocs:
            if i > 0 and np.sum(self.costarray[i]) < self.costarray[i - 1] - self.tol:
                break
            dw = self.C * np.sum(sv['X'])  # float
            sumX = np.ones((len(self.w),))
            sumX.fill(dw)
            delta_w = self.w - sumX
            delta_b = np.ones(X[0].shape)
            delta_b.fill(-1 * self.C * np.sum(sv['label']))
            self.w = self.w - self.learning_rate * delta_w
            self.b = self.b - self.learning_rate * delta_b

            cost = self.cost(sv)
            self.costarray.append(cost)
            sv = self.findSV()
            i += 1

    '''
    X: (nd array) samples for prediction
    returns 1d array predicted labels
    
    use self.coef_[0], self.intercept_[0] for prediction   
    '''

    def predict(self, X):  # only predict class 0 or class 1
        one = np.ones((2,))
        p = self.w.transpose() * X
        # scalar vector confused
        y_hat = p + np.ones(p.shape).fill(self.b)
        if np.greater_equal(y_hat, one):
            return 1
        else:
            return -1

    def cost(self, sv):

        return np.sum(0.5 * self.w.transpose() * self.w + self.C * (
                np.sum(np.ones((1, 1)) - sv['X'] * self.w) - self.b * np.sum(sv['label'])))

    def isSV(self, Y, X):
        val = np.multiply(Y, X) * self.w
        one = np.ones(val.shape)
        miusOne = -1 * one
        isEqualOne = np.equal(val, one)
        isEqualMinusOne = np.equal(val, miusOne)
        if np.all(isEqualOne):
            return True
        elif np.all(isEqualMinusOne):
            return True
        else:
            return False

    def findSV(self):
        sv = {
            'X': np.ones((len(self.X[0]), 1)),  # X (m,1)
            'label': np.ones(1)
        }
        for i in range(0, len(self.X)):
            if self.isSV(self.Y[i], self.X[i]):
                sv['X'] = np.vstack((sv, self.X[i]))
                sv['label'] = np.append(sv['label'], self.Y[i])
        return sv


data = load_iris()  # ndarray

data_names = data.feature_names[2:4]
X = data['data'][:, (2, 3)]  # petal length, petal width (150,2)
Y = data['target'].astype(np.int)  # 0 1 2 three classes


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
# print(X_train.shape)  # (120, 2)
# print(X_test.shape)  # (30, 2)
# print(y_train.shape)  # (100,)
# print(y_test.shape)  # (30,)

k = 5
kfoldsData, kfoldsLabel = sPartition(k, X_train, y_train)
# print(len(kfoldsData))  # 5
# print(kfoldsData[0].shape)  # (24,2)
# print(len(kfoldsLabel))  # 5
# print(len(kfoldsLabel[0]))  # (24,)
# Train X_train, y_train and test with X_test, y_test

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
            prediction = model.predict(kfoldsData[i])
            actual = kfoldsData[i]  # 0,1,2
