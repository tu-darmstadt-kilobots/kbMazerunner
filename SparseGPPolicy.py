from numpy import eye, random, dot, sqrt, repeat, array, \
        square, newaxis, tile, empty, r_, matrix, abs, multiply as mul
from numpy.linalg import lstsq, solve, pinv
from scipy.linalg import LinAlgError, cholesky as chol

import Kernel
import pickle

from Helper import Helper


"""
    The Simulator can use this method to create the policy without knowing
    its class name.
"""
def fromSerializableDict(d):
    return SparseGPPolicy.fromSerializableDict(d)

class SparseGPPolicy:
    """
        aRange: d x 2, d: number of action dimensions
            aRange[i, :] = [l, h] -> A[:, i] in [l, h] when not trained
    """
    def __init__(self, kernel, aRange):
        self.numSamplesSubset = 500
        self.bwFactor = 2.0

        self.GPPriorVariance = 0.1
        self.GPRegularizer = 1e-6 # TODO toolbox / NLopt
        self.SparseGPInducingOutputRegularization = 1e-6

        self.GPMinVariance = 0.0
        self.UseGPBug = False

        self.trained = False

        self.kernel = kernel
        self.aRange = aRange

    """
        returns a dict which can be serialized with pickle.dumps
    """
    def getSerializableDict(self):
        d = self.__dict__.copy()
        d['kernel'] = self.kernel.getSerializableDict()
        d['aRange'] = pickle.dumps(self.aRange, protocol=2)
        if self.trained:
            d['Ssub'] = pickle.dumps(self.Ssub, protocol=2)
            d['alpha'] = pickle.dumps(self.alpha, protocol=2)
            d['cholKy'] = pickle.dumps(self.cholKy, protocol=2)
        return d

    @staticmethod
    def fromSerializableDict(d):
        d['kernel'] = Kernel.Kernel.fromSerializableDict(d['kernel'])
        d['aRange'] = pickle.loads(d['aRange'])
        if d['trained']:
            d['Ssub'] = pickle.loads(d['Ssub'])
            d['alpha'] = pickle.loads(d['alpha'])
            d['cholKy'] = pickle.loads(d['cholKy'])
        obj = SparseGPPolicy(d['kernel'], d['aRange'])
        obj.__dict__ = d
        return obj

    def _getRandomActions(self, numSamples, numRepetitions):
        ar = self.aRange

        A = empty((numSamples * numRepetitions, ar.shape[0]))
        for i in range(ar.shape[0]):
            A[:, i] = random.uniform(ar[i, 0], ar[i, 1], (A.shape[0],))

        return A

    def train(self, S, A, w):
        # choose random subset of state samples
        Nsubset = min(self.numSamplesSubset, S.shape[0])
        #self.Ssub = Helper.getRepresentativeRows(S, Nsubset)

        self.Ssub = Helper.getRandomSubset(S, Nsubset)

        # set kernel bandwidth
        self.kernel.setBandwidth(Helper.getBandwidth(self.Ssub,
            self.numSamplesSubset, self.bwFactor))

        # kernel matrix on subset of samples
        K = self.GPPriorVariance * \
            self.kernel.getGramMatrix(self.Ssub, self.Ssub)

        w /= w.max()

        GPRegularizerEffective = self.GPRegularizer
        counter = 1
        while True:
            Ky = K + eye(K.shape[0]) * GPRegularizerEffective
            try:
                self.cholKy = chol(Ky)
                break
            except LinAlgError:
                GPRegularizerEffective *= 2

            counter += 1
            assert counter < 100, 'SparseGPPolicy: chol failed'

        kernelVectors = self.GPPriorVariance * \
            self.kernel.getGramMatrix(self.Ssub, S).T
        featureVectors = dot(dot(kernelVectors, pinv(self.cholKy)),
                pinv(self.cholKy.T))

        featureVectorsW = mul(featureVectors, w)

        X = dot(featureVectorsW.T, featureVectors)
        X += eye(featureVectors.shape[1]) * self.SparseGPInducingOutputRegularization
        y = dot(solve(X, featureVectorsW.T), A)

        self.alpha = solve(self.cholKy, solve(self.cholKy.T, y))

        self.trained = True

    def getRandomAction(self):
        return self._getRandomActions(1, 1)

    def getMeanAction(self, S):
        if not self.trained:
            if len(S.shape) == 1:
                return self.getRandomAction()
            else:
                return self._getRandomActions(S.shape[0], 1)

        kVec = self.GPPriorVariance * self.kernel.getGramMatrix(self.Ssub, S).T
        return dot(kVec, self.alpha)

    def sampleActions(self, S, N):
        if not self.trained:
            return repeat(S, N, axis=0), self._getRandomActions(S.shape[0], N)

        actionDim = self.alpha.shape[1]

        kVec = self.GPPriorVariance * self.kernel.getGramMatrix(self.Ssub, S).T
        meanGP = dot(kVec, self.alpha)

        temp = solve(self.cholKy.T, kVec.T)
        temp = square(temp.T)
        sigmaGP = temp.sum(1)

        kernelSelf = self.GPPriorVariance * self.kernel.getGramDiag(S)
        sigmaGP = kernelSelf.squeeze() - sigmaGP.squeeze()

        if sigmaGP.shape == (): # single number
            sigmaGP = matrix([sigmaGP])

        sigmaGP[sigmaGP < 0] = 0
        sigmaGP = tile(sqrt(sigmaGP)[:, newaxis], (1, actionDim))

        if self.UseGPBug:
            sigmaGP += sqrt(self.GPRegularizer)
        else:
            sigmaGP = sqrt(square(sigmaGP) + self.GPRegularizer)
        sigmaGP[sigmaGP < self.GPMinVariance] = self.GPMinVariance

        # generate N action samples for each state in S
        Srep = repeat(S, N, axis=0)
        meanGPrep = repeat(meanGP, N, axis=0)
        sigmaGPrep = repeat(sigmaGP, N, axis=0)

        N = random.normal(0.0, 1.0, (Srep.shape[0], actionDim))
        A = mul(N, sigmaGPrep) + meanGPrep

        return Srep, A
