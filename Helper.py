from numpy import random, zeros, repeat, sqrt, median, newaxis, square, \
        transpose, asarray, asmatrix, minimum, multiply, c_
from scipy.spatial.distance import cdist

from datetime import datetime

class Helper:
    """
        X: samples
        N: subset size

        returns N random rows of X
    """
    @staticmethod
    def getRandomSubset(X, N):
        N = min(N, X.shape[0])

        idx = random.choice(X.shape[0], size=N, replace=False)
        return X[idx, :]

    """
        X: samples
        N: number of rows to pick

        returns N rows of X which represent the samples in X well
        (for each sample in X there is a row with "small" distance)
    """
    @staticmethod
    def getRepresentativeRows(X, N, normalize):
        if normalize:
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0)
            X = (X - X_mean) / X_std;

        N = min(N, X.shape[0])
        Y = zeros((N, X.shape[1]))

        Y[0, :] = X[random.randint(X.shape[0]), :]
        D = cdist(X, asmatrix(Y[0, :]))

        for i in range(1, N):
            Y[i, :] = X[D.argmax(), :]
            D = minimum(D, cdist(X, asmatrix(Y[i, :])))

        if normalize:
            Y = multiply(Y, X_std) + X_mean;

        return Y

    """
        X: samples
        N: number of samples to use to estimate the bandwidth
        bwFactor: factor to multiply the estimated bandwidth with

        returns the estimated bandwidth for the samples in X
    """
    @staticmethod
    def getBandwidth(X, N, bwFactor):
        if X.shape[0] > N:
            Xsub = asarray(Helper.getRandomSubset(X, N))
        else:
            Xsub = asarray(X)

        bw = zeros((1, X.shape[1]))

        for i in range(X.shape[1]):
            dist = repeat(Xsub[:, i][:, newaxis], Xsub.shape[0], axis=1)
            dist2 = square(dist - transpose(dist))

            bw[0, i] = sqrt(median(dist2)) * bwFactor

        bw[bw == 0] = 1
        return bw

    """
        S: states
        N: number of actions to sample for each state
        policy: policy to sample the actions from
        featureFunc: function to compute the state-action feature matrix

        returns the expected state-action feature matrix
    """
    @staticmethod
    def getFeatureExpectation(S, N, policy, kernelFunc, stateActionCombFunc, MuSA):
        SA = stateActionCombFunc(S, policy.sampleActions(S))

        PHI = kernelFunc.getGramMatrix(SA, MuSA)

        for i in range(N - 1):
            SA = stateActionCombFunc(S, policy.sampleActions(S))
            PHI += kernelFunc.getGramMatrix(SA, MuSA)

        return PHI / N

    """
        generates a folder name based on the current date
    """
    @staticmethod
    def getSaveName():
        t = datetime.now()
        return '{}_{}_{}_{}_{}_{}_{}'.format(t.year, t.month, t.day,
                t.hour, t.minute, t.second, t.microsecond)
