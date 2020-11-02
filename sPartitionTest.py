# divide data into k folds
import math

import numpy as np


def sPartition(k, data, labels):
    kfoldsData = np.array_split(data, k)
    kfoldsLabel = np.array_split(labels, k)
    for i in range(len(kfoldsData)):
        kfoldsData[i] = kfoldsData[i][0]
    for j in range(len(kfoldsLabel)):
        kfoldsLabel[j] = kfoldsLabel[j][0]
    return kfoldsData, kfoldsLabel


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


feature = np.array([
    # f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14
    [1, 2, 3, 4, 6, 3, 5, 7, 4, 2, 7, 2, 4, 7],
    [2, 4, 6, 9, 3, 7, 9, 4, 7, 0, 5, 3, 6, 3],
    [2, 5, 3, 7, 9, 5, 9, 9, 4, 4, 6, 8, 4, 8],
    [3, 6, 7, 9, 0, 4, 3, 6, 3, 6, 7, 8, 3, 5],
    [2, 5, 8, 4, 2, 6, 4, 7, 5, 4, 7, 9, 3, 5]
])
target = np.array(  # last feature
    # f15
    [3,
     6,
     4,
     3,
     7]
)
partition_rate = 0.6
(trainingFeature, testFeature, trainingTarget, testTarget) = partition(feature, target, partition_rate)
print("feature:")
print(feature)
print("target")
print(target)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
print("train test split: %f\n" % partition_rate)
print("training split:")
print("training feature:")
print(trainingFeature)
print("==========================")
print("testing feature:")
print(testFeature)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("target split:")
print("training target:")
print(trainingTarget)
print("==========================")
print("testing target:")
print(testTarget)
print("==========================\n\n")

k = 3
print("evenly divide training feature: into %d folds" % k)
(kfoldsData, kfoldsLabel) = sPartition(k, trainingFeature, trainingTarget)
print("training feature:")
print(trainingFeature)
print("==========================")
print("kfold data:")
print(kfoldsData)
print("==========================")
print("kfold label:")
print(kfoldsLabel)
print("==========================\n\n\n")

