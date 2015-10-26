from numpy import asmatrix, zeros, std, ones_like, ones, log, asarray, Inf, \
        isnan, isinf, abs, random, r_, multiply as mul
from numexpr import evaluate as ev
from scipy.optimize import minimize

class AC_REPS:
    epsilonAction = 0.5

    def __init__(self):
        pass

    def _dualFunction(self, params):
        theta = params[0:self.numFeatures]
        eta = params[-1, 0]



    def _dualFunctionGrad(self, params):
        pass

    def computeWeightingFromEtaAndTheta(self, q, Q, PHI_S, eta, theta):
        advantage = Q - PHI_S * theta

        return ev("q * exp((advantage - max(advantage)) / eta)")

    def getKLDivergence(self, sampleWeighting, weighting):
        p = asarray(weighting / weighting.sum())
        q = asarray(sampleWeighting / sampleWeighting.sum())

        return nansum(p * log(p / q))

    def optimizeDualFunction(self, theta, eta):
        lowerBound = r_[-1e10 * ones((self.numFeatures, 1)), matrix([1e-20])]
        upperBound = r_[+1e10 * ones((self.numFeatures, 1)), matrix([1e+10])]

        # TODO test gradient

        startParams = r_[theta, eta]
        res = minimize(self._dualFunction, startParams, method='BFGS',
                jac=self._dualFunctionGrad, options={'disp': True})
        return res.x

    def computeWeighting(self, Q, PHI_S):
        numSamples, self.numFeatures = PHI_S.shape
        self.PHI_S = PHI_S

        sampleWeighting = ones((numSamples, 1)) / numSamples
        eta = 1.0
        theta = asmatrix(zeros((numFeatures, 1)))

        # TODO
        if eta < std(Q) * 0.1:
            eta = std(Q) * 0.1;

        PHI_HAT = mul(PHI_S, sampleWeighting).sum(0)

        returnWeighting = self.computeWeightingFromEtaAndTheta(sampleWeighting,
                Q, PHI_S, eta, theta)
        returnweighting = returnweighting / returnweighting.sum()

        divKL = getKLDivergence(sampleWeighting, returnWeighting)
        stateFeatureDifference = PHI_HAT - mul(PHI_S, returnWeighting).sum(0)

        bestDiv = Inf
        lastDiv = Inf;
        withoutImprovement = 0

        for i in range(40):
            params = self.optimizeDualFunction(theta, eta) # TODO
            theta = params[0:numFeatures]
            eta = params[-1, 0]

            weighting = self.computeWeightingFromEtaAndTheta(sampleWeighting,
                    Q, PHI_S, eta, theta)
            weighting = weighting / weighting.sum()
            divKL = self.getKLDivergence(sampleWeighting, weighting)


            if divKL > 3 or isnan(divKL):
                print('diVKL warning')

            stateFeatureDifference = PHI_HAT - mul(PHI_S, weighting).sum(0)

            featureError = abs(stateFeatureDifference).max()
            print('Feature Error: {:f}, KL: {:f}'.format(featureError, divKL))

            if not isinf(bestDiv) and i >= 10 and featureError >= bestDiv:
                withoutImprovement = withoutImprovement + 1
            if withoutImprovement >= 3:
                print('No improvement within the last 3 iterations.')
                break

            if abs(divKL - epsilonAction) < 0.05 \
                and featureError < 0.01 \
                and featureError < bestDiv:

                print('Accepted solution.')
                withoutImprovement = 0
                returnweighting = weighting

                bestDiv = featureError

                if abs(divKL - epsilonAction) < 0.05 \
                    and featureError < 0.001:
                    disp('Found sufficient solution.')
                    break

            if max(abs(stateFeatureDifference) - lastDiv) > -0.000001:
                print('Solution unchanged or degrading, restart from new point')
                theta = random.random(theta.shape) * 2.0 - 1.0
                lastDiv = Inf
            else:
                lastDiv = featureError
