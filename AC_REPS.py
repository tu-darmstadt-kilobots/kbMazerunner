from numpy import asmatrix, zeros, std, ones_like, ones, log, asarray, Inf, \
        isnan, isinf, abs, random, r_, c_, nansum, zeros_like, finfo, double, \
        exp, matrix, minimum, maximum, squeeze, multiply as mul
from numexpr import evaluate as ev
from scipy.optimize import minimize

class AC_REPS:
    epsilonAction = 0.5
    alphaL2ThetaPunishment = 0.0

    maxIter = 300
    toleranceG = 1e-8
    toleranceF = 1e-12

    def __init__(self):
        pass

    def _dualFunction(self, params):
        theta = asmatrix(params[0:self.numFeatures]).T
        eta = params[-1]
        epsilon = self.epsilonAction

        V = self.PHI_S * theta
        VHat = self.PHI_HAT * theta

        advantage = self.Q - V
        maxAdvantage = advantage.max()
        QNorm = self.Q - maxAdvantage
        advantage = (QNorm - V) / eta

        g = 0
        gD = zeros((self.numFeatures + 1,))

        if advantage.max() > 500:
            g = 1e30 - eta
            gD[-2] = -1
            return g, gD

        expAdvantage = ev("exp(advantage)")
        sumExpAdvantage = expAdvantage.sum()

        realmin = finfo(double).tiny
        if sumExpAdvantage < realmin:
            sumExpAdvantage = realmin

        gLogPart = (1.0 / self.numSamples) * sumExpAdvantage

        g += eta * log(gLogPart) + VHat + maxAdvantage
        g += eta * epsilon + self.alphaL2ThetaPunishment * (theta.T * theta)

        # gradient
        gDEta = epsilon + log(gLogPart) - \
                mul(expAdvantage, QNorm - V).sum() / (eta * sumExpAdvantage)
        if (-eta * sumExpAdvantage) == 0:
            gDEta = 1e100
        gD[-1] = gDEta

        gDTheta = self.PHI_HAT + mul(-self.PHI_S, expAdvantage).sum(0) / \
                sumExpAdvantage + 2 * self.alphaL2ThetaPunishment * theta.T
        gD[0:self.numFeatures] = gDTheta

        return g, 0.5 * gD

    def _numericDualFunctionGradient(self, params):
        params = asarray(params).squeeze()
        gDNumeric = zeros((params.size,))

        g, gD = self._dualFunction(params)

        stepSize = maximum(minimum(abs(params) * 1e-4, 1e-6), 1e-6)
        for i in range(params.size):
            paramsTemp = params
            paramsTemp[i] = params[i] - stepSize[i]

            g1, tmp = self._dualFunction(paramsTemp)

            paramsTemp = params
            paramsTemp[i] = params[i] + stepSize[i]

            g2, tmp = self._dualFunction(paramsTemp)
            gDNumeric[i] = (g2 - g1) / (stepSize[i] * 2)

        return gD, gDNumeric


    def _computeWeightingFromThetaAndEta(self, theta, eta):
        advantage = self.Q - self.PHI_S * theta
        maxAdvantage = max(advantage)

        w = ev("exp((advantage - maxAdvantage) / eta)")
        return w / w.sum()

    def _getKLDivergence(self, weighting):
        p = asarray(weighting / weighting.sum())

        return nansum(p * log(p * self.numSamples))

    def _optimizeDualFunction(self, theta, eta):
        lowerBound = r_[-1e10 * ones((self.numFeatures, 1)), matrix([1e-20])]
        upperBound = r_[+1e10 * ones((self.numFeatures, 1)), matrix([1e+10])]
        bounds = tuple(map(tuple, asarray(c_[lowerBound, upperBound])))

        startParams = r_[theta, matrix([eta])]

        # test gradient
        if False:
            gD, gDNumeric = self._numericDualFunctionGradient(startParams)
            print("Gradient error: {:f}".format(abs(gD - gDNumeric).max()))

        res = minimize(self._dualFunction, startParams, method='L-BFGS-B',
                bounds=bounds, jac=True,
                options={'maxiter': self.maxIter,
                         'gtol': self.toleranceG,
                         'ftol': self.toleranceF,
                         'disp': True})

        return asmatrix(res.x[0:self.numFeatures]).T, res.x[-1]

    def computeWeighting(self, Q, PHI_S):
        self.numSamples, self.numFeatures = PHI_S.shape

        self.Q = Q
        self.PHI_S = PHI_S
        self.PHI_HAT = PHI_S.mean(0)

        # initial params
        theta = asmatrix(zeros((self.numFeatures, 1)))
        eta = max(1.0, std(Q) * 0.1)

        bestDiv = Inf
        lastDiv = Inf
        withoutImprovement = 0

        returnWeighting = ones((self.numSamples, 1)) / self.numSamples

        for i in range(40):
            theta, eta = self._optimizeDualFunction(theta, eta)

            weighting = self._computeWeightingFromThetaAndEta(theta, eta)
            divKL = self._getKLDivergence(weighting)

            if divKL > 3 or isnan(divKL):
                print('diVKL warning')

            stateFeatureDifference = self.PHI_HAT - mul(PHI_S, weighting).sum(0)

            featureError = abs(stateFeatureDifference).max()
            print('Feature Error: {:f}, KL: {:f}'.format(featureError, divKL))

            if not isinf(bestDiv) and i >= 10 and featureError >= bestDiv:
                withoutImprovement = withoutImprovement + 1
            if withoutImprovement >= 3:
                print('No improvement within the last 3 iterations.')
                break

            if abs(divKL - self.epsilonAction) < 0.05 \
                and featureError < 0.01 \
                and featureError < bestDiv:

                print('Accepted solution.')
                withoutImprovement = 0
                returnWeighting = weighting

                bestDiv = featureError

                if abs(divKL - self.epsilonAction) < 0.05 \
                    and featureError < 0.001:
                    print('Found sufficient solution.')
                    break

            if (abs(stateFeatureDifference) - lastDiv).max() > -0.000001:
                print('Solution unchanged or degrading, restart from new point')
                theta = random.random(theta.shape) * 2.0 - 1.0
                lastDiv = Inf
            else:
                lastDiv = featureError

        return returnWeighting
