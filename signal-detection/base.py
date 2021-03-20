import numpy as np
from scipy import stats
from scipy.optimize import minimize

class BaseModel:

    def __init__(self, signal, noise):
        self.signal = signal
        self.noise = noise
        self.p_signal = self.__compute_proportions(self.signal)
        self.p_noise = self.__compute_proportions(self.noise)
        self.n_response_categories = len(list(zip(self.p_signal,
                                                  self.p_noise)))

        self.__n_signal = sum(signal)  # count resps signal trials
        self.__n_noise = sum(noise)
        self.__acc_signal = self.__accumulate(self.signal)[:-1]
        self.__acc_noise = self.__accumulate(self.noise)[:-1]

        self.non_criterion_parameters = self.parameters
        if self.uses_criterion_values:
            self.__set_criterion_parameters()

        self.parameter_labels = self.parameters.keys()
        self.parameter_values = {p: v['value'] for p, v
                                 in self.parameters.items()}

        self.parameter_bounds = {p: (v['lower_bound'], v['upper_bound'])
                                 for p, v in self.parameters.items()}
        self.opt = False

    def __accumulate(self, arr):
        return np.cumsum(arr)

    def __compute_proportions(self, arr, truncate=True):
        # Accumulates an array `arr` and converts to proportion of itself
        a = self.__accumulate(arr)
        freq = [(x + i / len(a)) / (max(a) + 1)
                for i, x in enumerate(a, start=1)]

        if truncate:
            assert freq[-1] == 1
            freq.pop()

        return np.array(freq)

    def __compute_z_proportions(self, arr):
        return stats.norm.ppf(arr)

    def __set_criterion_parameters(self):
        c_values = np.random.rand(self.n_response_categories)

        self.criterion_parameters = {
            f'c{i+1}': {'value': c,
                        'lower_bound': None,
                        'upper_bound': None} for i, c in enumerate(c_values)}

        self.parameters = {**self.parameters, **self.criterion_parameters}

    def get_non_criterion_values(self, fitted=False):
        if fitted:
            key = 'fitted'
        else:
            key = 'value'
        
        non_c = {m: p[key] for m, p
                 in self.non_criterion_parameters.items()}
        return non_c
    
    def get_criterion_values(self, fitted=False):
        if fitted:
            key = 'fitted'
        else:
            key = 'value'

        c = {m: p[key] for m, p in self.criterion_parameters.items()}
        return c

    def __sum_g2(self, x0):
        # This function is the target to minimize.
        # First it calls the self.compute_expected() method, 
        # Then it returns the sum of g values across signal & noise.

        # Because this is the target function to minimize, it takes the x0.
        # We define x0 as self.parameter_values, but because scipy is changing x0...
        # This means that x0 MUST match and we must use the index.
        assert len(x0) == len(self.parameter_values.values())
        # print(x0, self.parameter_values.values())
        # input("Ok?>")

        non_c_labels = list(self.get_non_criterion_values().keys())
        # non_c_values = list(self.get_non_criterion_values().values())
        to_fit = dict(zip(non_c_labels, x0[:len(non_c_labels)]))

        if self.uses_criterion_values:
            criterion_start_idx = len(self.get_non_criterion_values())
            criterion_values = x0[criterion_start_idx:]

            to_fit['c'] = criterion_values

        # Need to ensure that the length of x0 is the same as len(parameters)
        assert len(x0) == (len(self.parameters))

        expected_noise, expected_signal = self.compute_expected(**to_fit)

        # Sum of g-tests
        signal_gval = self.__gtest(N=self.__n_signal,
                                   n_c=self.__acc_signal,
                                   obs_c=self.p_signal,
                                   expct_c=expected_signal)

        noise_gval = self.__gtest(N=self.__n_noise,
                                  n_c=self.__acc_noise,
                                  obs_c=self.p_noise,
                                  expct_c=expected_noise)

        return np.sum([signal_gval, noise_gval])

    def __gtest(self, N, n_c, obs_c, expct_c):

        with np.errstate(divide='ignore'):
            # Ignore the infinite value warning & return inf anyway.
            g = 2 * n_c * np.log(obs_c / expct_c) + \
                2 * (N - n_c) * np.log((1 - obs_c) / (1 - expct_c))
        return g

    def __get_parameters_and_boundaries(self):
        # Gets all values for each parameter in order
        # Useful for x0 input to the minimize function
        all_parameters = [self.parameters[param]['value']
                          for param in self.parameters]

        all_boundaries = [(self.parameters[param]['lower_bound'],
                           self.parameters[param]['upper_bound'])
                          for param in self.parameters]

        return all_parameters, all_boundaries

    def fit(self):
        params, bounds = self.__get_parameters_and_boundaries()

        self.result = minimize(x0=params, fun=self.__sum_g2, bounds=bounds,
                               tol=1e-6)

        for i, param in enumerate(self.parameters.keys()):
            self.parameters[param]['fitted'] = self.result.x[i]

        self.sum_g2 = self.result.fun
        self.fitted_parameters = dict(zip(self.parameter_labels,
                                          self.result.x))
        self.opt = self.result.success
        
        # As long as it is successful, then re-fit it one last time
        if self.opt:            
            # Get the optimized parameters
            to_fit = self.get_non_criterion_values(fitted=True)

            if self.uses_criterion_values:
                to_fit = {**to_fit, **{'c': None}}
            
            self.compute_expected(**to_fit)

        return