'''Add custom models to this module.'''
import numpy as np
from scipy import stats
from .base import BaseModel


class HighThreshold(BaseModel):

    def __init__(self, signal, noise):
        self.uses_criterion_values = False
        self.parameters = {
            'R': {'value': 1 - 1e-3, 'upper_bound': 1, 'lower_bound': 0}}
        BaseModel.__init__(self, signal, noise)

    def compute_expected(self, R=None):
        
        if self.opt:
            # If it is optimized, make the line extend from 0 to 1
            self.model_noise = np.array([0, 1])
        else:
            self.model_noise = self.p_noise
        
        self.model_signal = (1 - R) * self.model_noise + R

        return self.model_noise, self.model_signal


class SignalDetection(BaseModel):

    def __init__(self, signal, noise, equal_variance=True):
        self.equal_variance = equal_variance
        self.uses_criterion_values = True

        self.parameters = {
            'd': {'value': 1, 'upper_bound': None, 'lower_bound': None}}

        if not self.equal_variance:
            # Add the extra parameter to the params dictionary
            self.parameters['scale'] = {
                'value': 1, 'upper_bound': None, 'lower_bound': 0}

        BaseModel.__init__(self, signal, noise)

    def compute_expected(self, d=None, c=None, scale=None):
        
        if self.opt:
            # If optimized, make the curve extend from x=0 through x=1
            c = np.linspace(-5, 5, 501)

        if scale is None:
            scale = 1  # constrain it here.

        self.model_signal = stats.norm.cdf(d / 2 - c, scale=scale)
        self.model_noise = stats.norm.cdf(-d / 2 - c)

        return self.model_noise, self.model_signal


class DualProcess(BaseModel):

    def __init__(self, signal, noise):
        self.uses_criterion_values = True

        self.parameters = {
            'd': {'value': 1, 'lower_bound': None, 'upper_bound': None},
            'R': {'value': 0.99, 'lower_bound': 0, 'upper_bound': 1}}

        BaseModel.__init__(self, signal, noise)

    def compute_expected(self, R=None, d=None, c=None):
        
        if self.opt:
            # If optimized, make the curve extend from x=0 through x=1
            c = np.linspace(-5, 5, 501)
        
        self.model_noise = stats.norm.cdf(-d / 2 - c)
        self.model_signal = R + (1 - R) * stats.norm.cdf(d / 2 - c)
        
        if self.opt:
            self.recollection = R
            fitted_c = list(self.get_criterion_values().values())
            self.compute_familiarity(fitted_c, d)
            self.parameter_estimates = (self.familiarity, self.recollection)

        return self.model_noise, self.model_signal
    
    def compute_familiarity(self, c, d):
        n = len(c)
        # breakpoint()
        if n % 2 == 1:
            # E.g. [1,2,3,4,5], we want index 2 (i.e. 3)
            # Can get the middle element directly
            middle = int(np.floor(n/2))
            self.familiarity = stats.norm.cdf(d / 2 - c[middle])
            
        else:
            # Need to calculate familiarity twice and get mean
            # E.g. [1,2,3,4,5,6], we want indexes 2,3 (i.e. 3&4)
            mid_up = int(np.floor(n/2)) + 1
            mid_dwn = int(np.floor(n/2))

            f1 = stats.norm.cdf(d / 2 - c[mid_up])
            f2 = stats.norm.cdf(d / 2 - c[mid_dwn])
            self.familiarity = np.mean([f1, f2])


class DoubleDualProcess(BaseModel):

    def __init__(self, signal, noise):
        self.uses_criterion_values = True
        self.parameters = {
            'd': {'value': 1, 'lower_bound': None, 'upper_bound': None},
            'R': {'value': 0.99, 'lower_bound': 0, 'upper_bound': 1},
            'Rx': {'value': 0, 'lower_bound': -1, 'upper_bound': 1}}
        BaseModel.__init__(self, signal, noise)

    def compute_expected(self, R=None, Rx=None, d=None, c=None):
        
        if self.opt:
            # If optimized, make the curve extend from x=0 through x=1
            c = np.linspace(-5, 5, 501)

        self.model_noise = (1 - Rx) * stats.norm.cdf(-d / 2 - c)
        self.model_signal = R + (1 - R) * stats.norm.cdf(d / 2 - c)
        return self.model_noise, self.model_signal


if __name__ == '__main__':
    signal = [508, 224, 172, 135, 119, 63]
    noise = [102, 161, 288, 472, 492, 308]

    ht = HighThreshold(signal, noise)
    ht.fit()
    print("\nResults from High Threshold:\n", ht.result)

    sdt = SignalDetection(signal, noise)
    sdt.fit()
    print("\nResults from Signal Detection:\n", sdt.result)

    uvsdt = SignalDetection(signal, noise, equal_variance=False)
    uvsdt.fit()
    print("\nResults from Unequal Var. Signal Detection:\n", uvsdt.result)

    dp = DualProcess(signal, noise)
    dp.fit()
    print(f"Recollection: {dp.recollection}; Familiarity: {dp.familiarity}")
    print("\nResults from Dual Process:\n", dp.result)

    ddp = DoubleDualProcess(signal, noise)
    ddp.fit()
    print("\nResults from Double Dual Process:\n", ddp.result)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.axis('square')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.plot([0, 1], [0, 1], c='k', lw=1)

    # Plot the data
    ax.scatter(ht.p_noise, ht.p_signal, c='k', s=75, label='data')
    # High Threshold
    ax.plot(ht.model_noise, ht.model_signal, label='HT')
    # EVSDT
    ax.plot(sdt.model_noise, sdt.model_signal, label='EVSDT')
    # UVSDT
    ax.plot(uvsdt.model_noise, uvsdt.model_signal, label='UVSDT')
    # Dual Process
    ax.plot(dp.model_noise, dp.model_signal, label='Dual')
    # Dual Process + 2 Unequal Thresholds
    ax.plot(ddp.model_noise, ddp.model_signal, label='2xDual')

    ax.set(xlabel='1-specificity', ylabel='sensitivity')
    ax.legend(loc='lower right')

    plt.show()
