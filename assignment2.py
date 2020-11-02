import itertools
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plot
from numpy.linalg import det

whitewineData = pd.read_csv("~/Downloads/winequality-red.csv", sep=';')
whitewineData = pd.get_dummies(whitewineData)
most_correlated = whitewineData.corr().abs()['quality'].sort_values(ascending=False)
most_correlated = most_correlated[:8]
feature = whitewineData.loc[:, most_correlated[1:8].index]
target = whitewineData['quality']
feature = abs(feature - feature.mean()) / feature.std()
target = abs(target - target.mean()) / target.std()


def partition(features, target, t):
    trainingSize = math.floor(len(features) * t)
    testSize = len(features) - trainingSize

    trainingFeature = np.array([])
    trainingTarget = np.array([])
    testFeature = np.array([])
    testTarget = np.array([])

    index = 0
    for row in features.itertuples():
        if index < trainingSize:
            feature = np.array(row)[0:]
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
    return trainingFeature, testFeature, trainingTarget, testTarget


(trainingFeature, testFeature, trainingTarget, testTarget) = partition(feature, target, 0.75)


# divide data into k folds
def sPartition(k, data, labels):
    remainder = len(data) % k
    if remainder == 0:
        kfoldsData = np.vsplit(data, k)
        kfoldsLabel = np.split(labels, k)
    else:
        dataReadysplit = data[:len(data) - remainder]
        labelReadysplit = labels[:len(labels) - remainder]
        kfoldsData = np.vsplit(dataReadysplit, k)
        kfoldsLabel = np.split(labelReadysplit, k)
        remainData = data[remainder * (-1):]
        remainLabel = labels[remainder * (-1):]
        kfoldsData[-1] = np.vstack((kfoldsData[-1], remainData))
        kfoldsLabel[-1] = np.append(kfoldsLabel[-1], remainLabel)
    return kfoldsData, kfoldsLabel


# 4 folds:
(trainingFeature_kfoldsData, trainingFeature_kfoldsLabel) = sPartition(4, trainingFeature, trainingTarget)


# ========code above works, check code under==============================================================

# def polynomialFeatures(X, degree):
#     if degree == 1:
#         return X
#     combinations = []
#     col = len(X)
#     for i in range(col+1):
#         combinations_object = itertools.combinations(X,i)
#         combinations_list = list(combinations_object)
#         combinations += combinations_list
#     print(combinations)
#     return combinations
def polynomialFeatures(X, degree):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree, include_bias=False)
    return poly.fit_transform(X)  # columnwise


class PolynomialRegression:

    def __init__(self):
        pass
    # closed from == OLS
    def fit(self, X, Y, degrees, labdas, **kwargs):
        self.__weights = np.ones(len(X))
        self.__degrees = degrees
        self.__Y = Y
        self.__X = X
        self.__lambdas = labdas
        self.__kwargs = kwargs
        self.theta_hat = np.c_[np.ones((X.shape[0], 1)), X]  # theta(bias) done

    def predict(self, X):  # X: (239,8)
        X = polynomialFeatures(X, self.__degrees)  # X done
        try:
            
            return predictVal
        except RuntimeError as e:
            print(e)
            # X_transpose * X not invertible: some col colinear, but data preprocessing suppose to eliminate this problem
            print("X_T * X invertible, features in X not independent!")

    def getX(self):
        return self.__X

    def getY(self):
        return self.__Y

    def setX(self, X):
        self.__X = X

    def setY(self, Y):
        self.__Y = Y


def mse(Y_true, Y_pred):  # mean square erro of 2 vector
    Y_pred = Y_pred.flatten()
    sum = 0
    if len(Y_true) > len(Y_pred):
        Y_true = Y_true[0:len(Y_true) - 1]
    elif len(Y_true) < len(Y_pred):
        Y_pred = Y_pred[0:len(Y_pred) - 1]

    for i in range(len(Y_true) - 1):
        trueVal = Y_true[i]
        predVal = Y_pred[i]
        result = np.square(np.subtract(trueVal, predVal))
        sum += result
    return sum / (2 * len(Y_true))


def updateTheta_unregularized(theta, model, eta, m):  # update rule for regularized cost function
    theta = np.subtract(theta, eta / m * model.getX().to_numpy().transpose().dot(
        np.subtract(model.getX().to_numpy().dot(theta), model.getY().to_numpy())))
    return theta


def updateTheta_l1(theta, model, lamda, eta, m):
    theta = np.subtract(
        np.subtract(theta, eta / m * np.dot(model.X.to_numpy().transpose(), np.subtract(np.dot(model.X.to_numpy()
                                                                                               , theta),
                                                                                        model.Y.to_numpy()))),
        np.dot(np.dot(np.dot(eta, lamda), theta) / m, theta))
    return theta


def updateTheta_l2(theta, model, lamda, eta, m):
    theta = np.subtract(
        np.subtract(theta, eta / m * np.dot(model.X.to_numpy().transpose(), np.subtract(np.dot(model.X.to_numpy()
                                                                                               , theta),
                                                                                        model.Y.to_numpy()))),
        np.dot(np.dot(np.dot(eta, lamda), theta) / m, np.sign(theta)))
    return theta


# plot learning curves
# model: object type with fit & predict
# X: feature Y: labels
# cv: #of folds
# use cross-validation compute the average mse for training fold & validation fold
# e.g. 50 samples rows in X, train_size=10: func start from first 10 samples(according to learning rate)
# successively add 10(train_size) samples in each iteration.
# for each iteration: use k-fold to compute <avg_mse> for <training fold & validation fold>
# train_model(using the 'fit' method): put data in function, function calculate using
# parameter
# epochs: max# of passes over training data for updating weight vector
# regularizer: l1,l2,None: use cost function without regularization term
# lambd: float, regularization coefficient, used only when the "regularizer" set to l1 or l2

# return: 2 arrays containing <training> & <validation> [root-mean-square error(mse)] value:
# train_scores(ndarray):root-mean-square error(rmse) values on training sets
# val_scores(ndarray):root-mean-square error(rmse) values on validation sets
def learning_curve(model, X, Y, cv, train_size=1, learning_rate=0.01, epochs=1000, tol=None, regularizer=None,
                   lambd=0.0, **kwargs):
    train_scores = [500] * len(X)
    val_scores = [500] * len(X)
    (trainingFeature, testFeature, trainingTarget, testTarget) = partition(X, Y, 0.75)

    # Training ###################################################
    (kfoldsData, kfoldsLabel) = sPartition(cv, trainingFeature, trainingTarget)
    previous_error = 30.0
    error = 500.0
    i = 0
    degree = 3
    # batch gradient decent
    theta = np.ones((feature.shape[1], 1))  # random initialization for theta(weights)
    optimalTheta = np.array([])
    model.setX(X)
    model.setY(Y)
    iterTimes = 0
    while (i < epochs) or (error <= previous_error - tol):  # for some number of epochs:
        y_hat = X.to_numpy().dot(theta)  # y_hat = wx
        error = mse(Y, y_hat)
        alpha = learning_rate
        m = len(X)
        a = X.to_numpy().transpose().dot(y_hat)
        b = alpha / m * a
        theta = np.subtract(theta, b)
        i += 1
        previous_error = error

        # adding regularization
        if tol is None:
            tol = 0
        if regularizer == 'l1':
            eta = learning_rate
            m = int(len(X) / 2)
            theta = updateTheta_l1(theta, model, lambd, eta, m)
        elif regularizer == 'l2':
            eta = learning_rate
            m = int(len(X) / 2)
            theta = updateTheta_l2(theta, model, lambd, eta, m)
        else:  # no regularized
            eta = learning_rate
            m = int(len(X) / 2)
            theta = updateTheta_unregularized(theta, model, eta, m)  # model is empty
        if error <= previous_error - tol and iterTimes > 50:
            iterTimes = 0
            degree += 1  # increase degree of polynomial for better precision
        iterTimes += 1
        # running k fold for training set,get the best theta
        currentTheta = np.array([])
        optimalTheta = np.array([])
        scoreIndex = 0
        for j in range(len(kfoldsLabel)):
            # train jth fold, test other fold
            for k in range(len(kfoldsData)):
                if k == j:  # train
                    model.fit(kfoldsData[k], kfoldsLabel[j], degree, lambd)
                    currentTheta = model.theta_hat
                    # add bias to this(recitaion 6: augmentX , w )
                    if det(model.theta_hat.transpose().dot(model.theta_hat)) != 0:
                        z = model.theta_hat.transpose().dot(model.theta_hat)
                else:
                    # there's one Y set that have one more value, when operating k fold, error will occur when compare with other data chunk
                    predictVal = model.predict(kfoldsData[k]) # TODO: predict error , too large
                    actualVal = kfoldsLabel[j]
                    if len(actualVal) < len(predictVal):
                        predictVal = predictVal[0:len(actualVal)]
                    else:
                        actualVal = actualVal[0:len(predictVal)]
                    error = mse(actualVal, predictVal)
                    error = math.sqrt(error)  # for making number smaller
                    train_scores[scoreIndex] = error
                    if train_scores[scoreIndex] == min(train_scores):  # best theta
                        optimalTheta = currentTheta
                scoreIndex += 1

        #  sfold validate in the testing set by using the optimal theta to get val_scores[]:
        scoreIndex = 0
        (kfoldsData, kfoldsLabel) = sPartition(cv, testFeature, testTarget)
        model.theta_hat = optimalTheta
        for labelIndex in range(len(kfoldsLabel)):
            # train jth fold, test other fold
            for dataIndex in range(len(kfoldsData)):
                predictVal = model.predict(kfoldsData[dataIndex])
                actualVal = kfoldsLabel[labelIndex]
                error = mse(actualVal, predictVal)
                error = math.sqrt(error)
                val_scores[scoreIndex] = error
            scoreIndex += 1

    print("best theta(weight)")
    print(optimalTheta)
    # for j in range(len(kfoldsLabel)):
    #     # train jth fold, test other fold
    #     for k in range(len(kfoldsData)):
    #         if k == j:  # train
    #             model.fit(kfoldsData[k], kfoldsLabel[j], degree, lambd)
    #         else:
    #             predictVal = model.predict(kfoldsData[k])
    #             actualVal = kfoldsLabel[j]
    #             error = mse(actualVal, predictVal)
    #             error = math.sqrt(error)  # for making number smaller
    #             if tol is None:
    #                 tol = 0
    #             if error <= previous_error - tol:
    #                 degree += 1  # increase degree of polynomial for better precision
    #                 previous_error = error
    #             if regularizer == 'l1':
    #                 eta = learning_rate
    #                 m = len(X) / 2
    #                 theta = updateTheta_l1(theta, X, Y, lambd, eta, m)
    #             elif regularizer == 'l2':
    #                 eta = learning_rate
    #                 m = len(X) / 2
    #                 theta = updateTheta_l2(theta, X, Y, lambd, eta, m)
    #             else:
    #                 eta = learning_rate
    #                 m = len(X) / 2
    #                 theta = updateTheta_unregularized(theta, X, Y, eta, m)
    #             train_scores[i] = error
    # Testing ###################################################
    # (kfoldsData, kfoldsLabel) = sPartition(cv, testFeature, testTarget)
    # stepSize = learning_rate
    # previous_error = 30.0
    # error = 500.0
    # i = 0
    # # for each iteration: use k-fold to compute <avg_mse> for <training fold & validation fold> train_model(using the 'fit' method): put data in function, function calculate using  parameter
    # degree = 2
    # while (i < epochs) or (error <= previous_error - tol):
    #     for j in range(len(kfoldsLabel)):
    #         # train jth fold, test other fold
    #         for k in range(len(kfoldsData)):
    #             if k == j:  # train
    #                 model.fit(kfoldsData[k], kfoldsLabel[j], degree, lambd)
    #             else:
    #                 predictVal = model.predict(kfoldsData[k])
    #                 actualVal = kfoldsLabel[j]
    #                 error = mse(actualVal, predictVal)
    #                 if error <= previous_error - tol:
    #                     degree += 1  # increase degree of polynomial for better precision
    #                 if regularizer == 'l1' or regularizer == 'l2':
    #                     lambd += stepSize  # change lambda for diagonal matrix
    #
    #                 val_scores[i] = error
    #     i += 1

    return train_scores, val_scores


# get 2 arrays of  root mean square error of training and validation
# train_scores: [rmse1,rmse2,...]
# val_scores: [rmse1,rmse2,...]
model = PolynomialRegression()
cv = 5
(train_scores, val_scores) = learning_curve(model, feature, target, cv, kwargs="")
plot.plot(train_scores, val_scores)
plot.show()
