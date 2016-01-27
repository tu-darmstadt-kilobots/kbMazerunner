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
            #K = (K - K.min()) / (K.max() - K.min())
            #K2 = (K2 - K2.min()) / (K2.max() - K2.min())

            K = w * K + (1 - w) * K2 # TODO

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

        N = A2.shape[1] / 2 # number of kilobots
        one = ones((1, N))
        N2 = N * N

        Kn = empty((A2.shape[0], 1))
        for i in range(A2.shape[0]):
            Are = reshape(A2[i, :], ((N, 2)))
            Kn[i, :] = one * self.kernelKb.getGramMatrix(Are, Are) * one.T

        Km = empty((B2.shape[0], 1))
        for i in range(B2.shape[0]):
            Bre = reshape(B2[i, :], ((N, 2)))
            Km[i, :] = one * self.kernelKb.getGramMatrix(Bre, Bre) * one.T

        # reshape kilobot positions to * x 2
        Are = c_[A2.flat[0::2].T, A2.flat[1::2].T]
        Bre = c_[B2.flat[0::2].T, B2.flat[1::2].T]
        Knm = self.kernelKb.getGramMatrix(Are, Bre)

        # pad left and top with zeros
        Knm = c_[zeros((Knm.shape[0] + 1, 1)), r_[zeros((1, Knm.shape[1])), Knm]]

        # sum over all NxN submatrices using a summed area table (S)
        S = Knm.cumsum(axis=0).cumsum(axis=1)
        K = (-2 * (S[N::N, N::N] + S[0:-1:N, 0:-1:N] -\
                   S[0:-1:N, N::N] - S[N::N, 0:-1:N]) + Kn + Km.T) / N2

        if False: #A2.shape[0] != 1 and B2.shape[0] != 1:
            plt.imshow(K)
            plt.title('K({}, {})'.format(A2.shape, B2.shape))
            plt.colorbar()
            plt.show()
            input('press key...')

        return self.kernelNonKb.getGramMatrix(A1, B1, K, self.weightNonKb)

    def getGramDiag(self, A):
        return ones((A.shape[0], 1))
