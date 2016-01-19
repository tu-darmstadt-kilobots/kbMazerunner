from numpy import asmatrix, square, diagflat, sqrt, pi, ones, array, \
        multiply as mul
from numpy import *
from numexpr import evaluate as ev
import pickle

import time

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
        elif d['type'] == 'KernelOverKernel':
            return KernelOverKernel.fromSerializableDict(d)

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

    def getGramMatrix(self, A, B, K2=None):
        Q = asmatrix(diagflat(1.0 / self.bw2))

        AQ = A * Q
        K = mul(AQ, A).sum(1) + mul(B * Q, B).sum(1).T
        K -= 2.0 * AQ * B.T

        if K2 is not None:
            K = (K - K.min()) / (K.max() - K.min())
            K2 = (K2 - K2.min()) / (K2.max() - K2.min())

            K = 0.5 * K + 0.5 * K2 # TODO

        K = ev('exp(-0.5 * K)')

        return asmatrix(K)

    def getGramDiag(self, A):
        return ones((A.shape[0], 1))


class KernelOverKernel(Kernel):
    """
        uses outerKernel for the final kernel matrix and
        innerKernel to get similarities between the "last" dimensions
    """
    def __init__(self, outerKernel, innerKernel):
        self.outerKernel = outerKernel
        self.innerKernel = innerKernel

    """
        bw1: bandwidth for outerKernel
        bw2: bandwidth for innerKernel
    """
    def setBandwidth(self, outerBw, innerBw):
        self.outerKernel.setBandwidth(outerBw)
        self.innerKernel.setBandwidth(innerBw)

    def getSerializableDict(self):
        return {'type': 'KernelOverKernel',
                'outerKernel': self.outerKernel.getSerializableDict(),
                'innerKernel': self.innerKernel.getSerializableDict()}

    @staticmethod
    def fromSerializableDict(d):
        outerKernel = Kernel.fromSerializableDict(d['outerKernel'])
        innerKernel = Kernel.fromSerializableDict(d['innerKernel'])

        return KernelOverKernel(outerKernel, innerKernel)

    def getGramMatrix(self, A, B):
        dim1 = self.outerKernel.getNumDimensions()

        A1 = A[:, 0:dim1]
        B1 = B[:, 0:dim1]

        A2 = A[:, dim1:]
        B2 = B[:, dim1:]

        N = A2.shape[1] / 2 # number of kilobots
        one = ones((1, N))
        N2 = N * N

        t = time.time()
        Kn = empty((A2.shape[0], 1))
        for i in range(A2.shape[0]):
            Are = reshape(A2[i, :], ((N, 2)))
            Kn[i, :] = one * self.innerKernel.getGramMatrix(Are, Are) * one.T
        print('Kn:', time.time() - t)

        t = time.time()
        Km = empty((B2.shape[0], 1))
        for i in range(B2.shape[0]):
            Bre = reshape(B2[i, :], ((N, 2)))
            Km[i, :] = one * self.innerKernel.getGramMatrix(Bre, Bre) * one.T
        print('Km:', time.time() - t)

        t = time.time()
        Are = c_[A2.flat[0::2].T, A2.flat[1::2].T]
        Bre = c_[B2.flat[0::2].T, B2.flat[1::2].T]
        Knm = self.innerKernel.getGramMatrix(Are, Bre)
        print('Kmn:', time.time() - t)

        t = time.time()
        # pad left and top with zeros
        Knm = c_[zeros((Knm.shape[0] + 1, 1)), r_[zeros((1, Knm.shape[1])), Knm]]

        # sum over all NxN submatrices using a summed area table (S)
        S = Knm.cumsum(axis=0).cumsum(axis=1)
        K = (-2 * (S[N::N, N::N] + S[0:-1:N, 0:-1:N] -\
                   S[0:-1:N, N::N] - S[N::N, 0:-1:N]) + Kn + Km.T) / N2
        print('K:', time.time() - t)

        return self.outerKernel.getGramMatrix(A1, B1, K)

    def getGramDiag(self, A):
        return ones((A.shape[0], 1))
