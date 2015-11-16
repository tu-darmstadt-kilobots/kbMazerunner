from numpy import random, zeros, repeat, sqrt, median, newaxis, square, \
        transpose, asarray
from scipy.spatial.distance import cdist


class Helper:
    """
        X: samples
        N: subset size

        returns N random rows of X
    """
    @staticmethod
    def getRandomSubset(X, N):
        idx = random.choice(X.shape[0], size=N, replace=False)
        return X[idx, :]

    """
        X: samples
        N: number of rows to pick

        returns N rows of X which represent the samples in X well
        (for each sample in X there is a row with "small" distance)
    """
    @staticmethod
    def getRepresentativeRows(X, N):
        N = min(N, X.shape[0])
        Y = zeros((N, X.shape[1]))

        for i in range(N):
            Y[i, :] = X[cdist(X, Y).min(axis=1).argmax(), :]

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
    def getFeatureExpectation(S, N, policy, featureFunc):
        Srep, Arep = policy.sampleActions(S, N)
        PHI_SA_rep = featureFunc.getStateActionFeatureMatrix(Srep, Arep)

        # mean over each N rows
        return asarray(PHI_SA_rep).reshape(-1, N, PHI_SA_rep.shape[1]).mean(1)


