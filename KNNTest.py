from numpy.lib.scimath import sqrt
import numpy as np


def eucliDistan(p1, p2):
    if p1.shape != p2.shape:
        print("unable to compute, vectors are in different dimensions")

    return np.linalg.norm(p2 - p1)


class KNN_Classifier:
    def __init__(self):
        pass

    def fit(self, X, Y, nNeighbors, weights, **kwargs):
        self.__kwargs = kwargs
        self.__weights = weights
        self.__nNeighbors = nNeighbors
        self.__Y = Y  # row number of feature(nd_array)
        self.__X = X  # features matrix(nd_array)

    def getX(self):
        return self.__X

    def getY(self):
        return self.__Y

    # return 1D array of predictions for each row in X
    # The 1D array should be designed as a column vector
    # get the nearest n neighbor
    # each point P: (sugar, total sulfur dioxide, wine quality)
    # X : wine data
    def predict(self, X):  # X numpy 1 dimensional array
        samplept = X[0]
        distance = np.zeros((len(X), 2))
        neighbors = np.array([])

        for i in range(len(X)):
            # calculate distance
            eu_distance = eucliDistan(samplept, X[i])
            # get neighbors
            print(eu_distance)
            distance[i] = [eu_distance, i]
        sorted_distance = np.sort(distance, axis=0)

        print('get %d neighbors:' % self.__nNeighbors)
        return sorted_distance[1:self.__nNeighbors+1, :]


def testKNN():
    a = np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [3, 4, 6, 2, 7], [2, 3, 7, 3, 6],
                  [1, 3, 4, 7, 9], [2, 3, 6, 2, 10], [4, 4, 6, 2, 4], [8, 2, 7, 3, 6]])
    b = np.array([[3, 4, 8, 3, 9], [2, 5, 9, 2, 8], [5, 9, 0, 3, 5], [2, 5, 9, 2, 8], [1, 3, 5, 7, 9], [2, 3, 8, 8, 10],
                  [3, 4, 6, 2, 7], [2, 3, 7, 3, 6]])
    c = np.array(
        [[1, 4, 5, 7, 9], [2, 4, 6, 2, 10], [3, 4, 5, 2, 7], [5, 3, 7, 3, 6], [1, 6, 5, 7, 9], [2, 4, 6, 8, 10],
         [7, 4, 6, 2, 7], [4, 3, 7, 3, 6]])

    g = np.array([[1, 3, 5, 7, 9], [2, 34, 6, 6, 10], [3, 4, 6, 2, 7], [2, 3, 7, 3, 6]])
    h = np.array([[1, 3, 4, 8, 9], [5, 4, 3, 8, 10], [3, 4, 6, 2, 7], [2, 3, 7, 3, 6]])
    i = np.array([[1, 5, 5, 7, 9], [2, 4, 4, 8, 8], [3, 4, 6, 2, 7], [2, 3, 6, 3, 6]])
    j = np.array([[1, 9, 5, 7, 9], [2, 6, 6, 8, 10], [3, 7, 6, 2, 7], [2, 3, 4, 3, 6]])
    k = np.array([[1, 3, 6, 7, 9], [2, 3, 6, 3, 10], [5, 4, 2, 2, 7], [2, 3, 7, 3, 6]])
    l = np.array([[1, 3, 5, 7, 9], [2, 4, 6, 4, 10], [3, 4, 6, 2, 7], [2, 3, 4, 3, 6]])
    m = np.array([[1, 3, 5, 3, 9], [2, 4, 3, 8, 10], [3, 4, 6, 2, 7], [3, 3, 7, 3, 6]])

    Knn = KNN_Classifier()
    # train:
    Knn.fit(a, b, 3, 'uniform')
    neighbors = Knn.predict(c)
    print(neighbors)


if __name__ == '__main__':
    testKNN()
