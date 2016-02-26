from numpy import asmatrix, square, diagflat, sqrt, pi, ones, array, \
        multiply as mul
from numpy import *
from numexpr import evaluate as ev
import pickle

import time
import matplotlib.pyplot as plt

class Kernel:
    """
        returns a dict, which can be serialized with json.dumps
    """
    def getSerializableDict(self):
        raise NotImplementedError("getSerializableDict() not implemented")

    """
        loads data from dict created using getSerializableDict
    """
    @staticmethod
    def fromSerializableDict(d):
        if d['type'] == 'ExpQuad':
            return ExponentialQuadraticKernel.fromSerializableDict(d)
        elif d['type'] == 'KilobotKernel':
            return KilobotKernel.fromSerializableDict(d)

    def getNumDimensions(self):
        raise NotImplementedError('getNumDimensions() not implemented.')

    """
        A: m1 x d
        B: m2 x d

        returns m1 x m2 gram matrix
    """
    def getGramMatrix(self, A, B):
        raise NotImplementedError('getGramMatrix() not implemented.')

    def getGramDiag(self, A):
        raise NotImplementedError('getGramDiag() not implemented')


class ExponentialQuadraticKernel(Kernel):
    def __init__(self, dim):
        self.setBandwidth(ones((1, dim)))

    def setBandwidth(self, bw):
        self.bw2 = square(bw)

    def getSerializableDict(self):
        d = self.__dict__.copy()
        d['type'] = 'ExpQuad'
        d['bw2'] = pickle.dumps(self.bw2, protocol=2)
        return d

    @staticmethod
    def fromSerializableDict(d):
        obj = ExponentialQuadraticKernel(1)
        d['bw2'] = pickle.loads(d['bw2'])
        obj.__dict__ = d
        return obj

    def getNumDimensions(self):
        return self.bw2.shape[1]

    def getGramMatrix(self, A, B, K2=None, w=None):
        Q = asmatrix(diagflat(1.0 / self.bw2))

        AQ = A * Q
        K = mul(AQ, A).sum(1) + mul(B * Q, B).sum(1).T
        K -= 2.0 * AQ * B.T

        if K2 is not None:
            K = w * K + (1 - w) * K2

        K = ev('exp(-0.5 * K)')

        return asmatrix(K)

    def getGramDiag(self, A):
        return ones((A.shape[0], 1))


class KilobotKernel(Kernel):
    def __init__(self, numNonKilobotDimensions):
        self.kernelNonKb = ExponentialQuadraticKernel(numNonKilobotDimensions)
        self.kernelKb = ExponentialQuadraticKernel(2)
        self.weightNonKb = 0.5

    def setBandwidth(self, bwNonKb, bwKb):
        self.kernelNonKb.setBandwidth(bwNonKb)
        self.kernelKb.setBandwidth(bwKb)

    def setWeighting(self, weightNonKb):
        self.weightNonKb = weightNonKb

    def getSerializableDict(self):
        return {'type': 'KilobotKernel',
                'kernelNonKb': self.kernelNonKb.getSerializableDict(),
                'kernelKb': self.kernelKb.getSerializableDict(),
                'weightNonKb': self.weightNonKb}

    @staticmethod
    def fromSerializableDict(d):
        kernelNonKb = Kernel.fromSerializableDict(d['kernelNonKb'])
        kernelKb = Kernel.fromSerializableDict(d['kernelKb'])
        weightNonKb = d['weightNonKb']

        k = KilobotKernel(1)
        k.kernelNonKb = kernelNonKb
        k.kernelKb = kernelKb
        k.weightNonKb = weightNonKb
        return k

    def getGramMatrix(self, A, B):
        dim1 = self.kernelNonKb.getNumDimensions()

        A1 = A[:, 0:dim1]
        B1 = B[:, 0:dim1]

        A2 = A[:, dim1:]
        B2 = B[:, dim1:]

        NA = A2.shape[1] / 2 # number of kilobots in A
        NB = B2.shape[1] / 2 # number of kilobots in B

        oneA = ones((1, NA))
        oneB = ones((1, NB))

        Kn = empty((A2.shape[0], 1))
        for i in range(A2.shape[0]):
            Are = reshape(A2[i, :], ((NA, 2)))
            Kn[i, :] = oneA * self.kernelKb.getGramMatrix(Are, Are) * oneA.T

        Km = empty((B2.shape[0], 1))
        for i in range(B2.shape[0]):
            Bre = reshape(B2[i, :], ((NB, 2)))
            Km[i, :] = oneB * self.kernelKb.getGramMatrix(Bre, Bre) * oneB.T

        # reshape kilobot positions to * x 2
        Are = c_[A2.flat[0::2].T, A2.flat[1::2].T]
        Bre = c_[B2.flat[0::2].T, B2.flat[1::2].T]
        Knm = self.kernelKb.getGramMatrix(Are, Bre)

        # pad left and top with zeros
        Knm = c_[zeros((Knm.shape[0] + 1, 1)), r_[zeros((1, Knm.shape[1])), Knm]]

        # sum over all NAxNB submatrices using the summed area table S
        S = Knm.cumsum(axis=0).cumsum(axis=1)
        K = Kn / (NA * NA) + Km.T / (NB * NB) - \
                2.0 * (S[NA::NA, NB::NB] + S[0:-1:NA, 0:-1:NB] -\
                       S[0:-1:NA, NB::NB] - S[NA::NA, 0:-1:NB]) / (NA * NB)

        return self.kernelNonKb.getGramMatrix(A1, B1, K, self.weightNonKb)

    def getGramDiag(self, A):
        return ones((A.shape[0], 1))
