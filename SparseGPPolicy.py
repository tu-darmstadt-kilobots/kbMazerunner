from numpy import eye, zeros_like, random, dot, zeros, sqrt, c_, repeat, \
        square, diag, multiply as mul
from numpy.linalg import lstsq, solve
from scipy.linalg import LinAlgError, cholesky as chol

class SparseGPPolicy:
    numSamplesSubset = 100

    GPPriorVariance = 0.1
    GPRegularizer = 1e-6
    SparseGPInducingOutputRegularization = 1e-6

    GPMinVariance = 0.0
    UseGPBug = False

    trained = False

    def __init__(self, kernel):
        self.kernel = kernel

    def _getRandomStateSubset(self, S):
        N = min(self.numSamplesSubset, S.shape[0])

        idx = random.choice(S.shape[0], size=N, replace=False)
        return S[idx, :]

    def train(self, S, A, w):
        # choose random subset of state samples
        self.Ssub = self._getRandomStateSubset(S)

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
            assert counter < 100, "SparseGPPolicy: chol failed"

        kernelVectors = self.GPPriorVariance * \
                self.kernel.getGramMatrix(self.Ssub, S).T
        featureVectors = lstsq(self.cholKy,
                lstsq(self.cholKy.T, kernelVectors.T)[0])[0].T
        featureVectorsW = mul(featureVectors, w)

        X = dot(featureVectorsW.T, featureVectors)
        X += eye(featureVectors.shape[1]) * self.SparseGPInducingOutputRegularization
        y = dot(solve(X, featureVectorsW.T), A)

        self.alpha = solve(self.cholKy, solve(self.cholKy.T, y))

        self.trained = True

    def evaluate(self, S):
        kVec = self.GPPriorVariance * self.kernel.getGramMatrix(self.Ssub, S).T
        return dot(kVec, self.alpha)

    def sampleActions(self, S, N):
        if not self.trained:
            return repeat(S, N, axis=0), zeros((N * S.shape[0], 1))

        kVec = self.GPPriorVariance * self.kernel.getGramMatrix(self.Ssub, S).T
        meanGP = dot(kVec, self.alpha)

        temp = solve(self.cholKy.T, kVec.T)
        temp = square(temp.T)
        sigmaGP = temp.sum(1)

        kernelSelf = self.GPPriorVariance * self.kernel.getGramDiag(S)
        sigmaGP = kernelSelf.squeeze() - sigmaGP.squeeze()
        sigmaGP[sigmaGP < 0] = 0
        sigmaGP = sqrt(sigmaGP)

        if self.UseGPBug:
            sigmaGP += sqrt(self.GPRegularizer)
        else:
            sigmaGP = sqrt(square(sigmaGP) + self.GPRegularizer)
        sigmaGP[sigmaGP < self.GPMinVariance] = self.GPMinVariance

        # generate N action samples for each state in S
        Srep = repeat(S, N, axis=0)
        meanGPrep = repeat(meanGP, N, axis=0)
        sigma2GPrep = square(repeat(sigmaGP, N, axis=0))

        # TODO memory error
        A = random.multivariate_normal(meanGPrep.squeeze(), diag(sigma2GPrep))

        return Srep, A
