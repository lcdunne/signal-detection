import numpy as np
from scipy import stats
from scipy.optimize import minimize
from .utils import *

class ResponseData:
    def __init__(self, freqs=None, props_acc=None, n=None, corrected=True):
        if freqs is not None:
            # Create derivatives based on the observed frequencies
            self.freqs = np.array(freqs)
            self.n = sum(self.freqs)
            self.corrected = corrected
            self.props = self.freqs / self.n 
            self.freqs_acc = accumulate(self.freqs)
            self.props_acc = compute_proportions(self.freqs, truncate=False, corrected=self.corrected)

        elif (props_acc is not None) and (n is not None):
            # Create all derived vars from the accumulated proportions. Useful 
            # for deriving expected frequencies based on model predicted 
            # accumulated propotions.
            # Note that this is not the reverse of defining ResponseData with 
            # freqs, unless the `corrected` argument to `compute_proportions` 
            # is set to True.
            self.n = n
            self.corrected = False
            self.props_acc = np.append(props_acc, 1)
            self.freqs_acc = self.props_acc * self.n
            self.freqs = deaccumulate(self.freqs_acc)
            self.props = deaccumulate(self.props_acc)
        else:
            raise ValueError("Either `freqs` or both of `props_acc` and `n` are required.")
        self.z = stats.norm.ppf(self.roc)
    
    def __repr__(self):
        return self.table.get_string()
    
    @property
    def table(self):
        return keyval_table(**self.as_dict)
    
    @property
    def as_dict(self):
        return {
            'N': self.n,
            'Freqs.': self.freqs,
            'Freqs. (Accum.)': self.freqs_acc,
            'Props.': self.props,
            'Props. (Accum)': self.props_acc,
            'z-score': self.z,
        }
    
    @property
    def roc(self):
        # Ok to call it ROC? I think so.
        # Test: create ResponseData from props_acc with and without the 1.0. This should behave well.
        return self.props_acc[:-1]

class GenericDataContainer:
    def __init__(self, **kwargs):
        self._inputs = kwargs
        for k, v in self._inputs.items():
            setattr(self, k, v)

class _BaseModel:
    """Base model class be inherited by all specific model classes. 
    
    Contains functionality and attributes that are used by 
    all models. Not to be instantiated directly.

    Parameters
    ----------
    signal : array_like
        An array of observed response counts to signal-present trials.
    noise : array_like
        An array of observed response counts to noise trials.

    Attributes
    ----------

    """
    # Defaults
    __modelname__ = 'none'
    _has_criteria = False
    _named_parameters = {}

    def __init__(self, signal, noise):
        self.obs_signal = ResponseData(signal)
        self.obs_noise = ResponseData(noise)
        self.auc = auc(self.obs_noise.props_acc, self.obs_signal.props_acc)
        
        # Dummy parameters in case no model is specified. This is the fully saturated model (not intended for use).
        if not self._named_parameters:
            _s = {f's{i+1}': {'initial': 0, 'bounds': (None, None)} for i, s in enumerate(self.obs_signal.props)}
            _n = {f'n{i+1}': {'initial': 0, 'bounds': (None, None)} for i, s in enumerate(self.obs_noise.props)}
            self._named_parameters = _s | _n
        
        # Set the criteria for the model if required
        if self._has_criteria: 
            self.n_criteria = len(self.obs_signal.roc)
            # Maybe allow the user to optionally pass in a set of "reasonable" starting values.
            # Reasonable starting parameters seem to be equally-spaced between +/-1.5.
            self._criteria = {}
            for i, c in enumerate(np.linspace(1.5, -1.5, self.n_criteria)):
                self._criteria[f"c{i}"] = {'initial': c, 'bounds': (None, None)}

            self._parameters = self._named_parameters | self._criteria
        else:
            # Only the case for high threshold model (currently)
            self.n_criteria = 0
            self._parameters = self._named_parameters.copy()
        
        if not hasattr(self, '_n_named_parameters'):
            self._n_named_parameters = len(self._named_parameters)
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.__modelname__}>"
    
    @property
    def initial_parameters(self):
        """dict: Starting parameters before applying the fitting procedure."""
        return {k: v['initial'] for k, v in self._parameters.items()}
    
    @property
    def parameter_estimates(self):
        """dict: All parameters and values after fitting."""
        # TODO: Prevent error when calling this before fitting.
        return self._parameter_estimates
    
    @property
    def parameter_labels(self):
        """list: The labels for all parameters in the model."""
        return list(self._parameters.keys())
    
    @property
    def parameter_boundaries(self):
        """list: The boundary conditions for each parameter during fitting."""
        return list({k: v['bounds'] for k, v in self._parameters.items()}.values())
    
    @property
    def signal_boundary(self):
        """int: The index in the criterion array that corresponds to the 
        boundary between signal and noise (the lowest signal criterion)."""
        if not self._has_criteria:
            return
        c = list(self._criteria.keys())
        return c.index( c[:int(np.ceil(len(c)/2))][-1] )
    
    @property
    def n_param(self):
        """int: The number of model parameters."""
        return self._n_named_parameters + self.n_criteria

    @property
    def dof(self):
        return len(self.obs_signal.roc) + len(self.obs_noise.roc) - self.n_param

    def define_model_inputs(self, labels: list, values: list, n_criteria: int=0):
        """Maps from flat list of labels and x0 values to dict accepted by the
        `<model>.compute_expected(...)` function.

        Parameters
        ----------
        labels : list
            A list of labels defining the parameter names.
        values : list
            A list of parameter values corresponding to the list of labels. 
            These must be in the same order.
        n_criteria : int, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        dict
            A dictionary that can be unpacked into the modelling function. All 
            named parameters are defined by unique key-value pairs; any 
            criterion parameters are stored in the 'criterion' key, as this 
            parameter in the modelling function requires a list.
        
        Example
        -------
        >>> evsd.define_model_inputs(
                labels=['d', 'c0', 'c1', 'c2', 'c3', 'c4'],
                values=[1.0201, 0.9459, 0.4768, 0.012, -0.5621, -1.2872],
                n_criteria=5
            )
        {'d': 1.0201, 'criteria': [0.9459, 0.4768, 0.012, -0.5621, -1.2872]}

        """
        if n_criteria == 0:
            return dict(zip(labels, values))        
        
        n_named = len(labels) - n_criteria
        
        named_params = dict(zip(labels[:n_named], values[:n_named]))
        criterion_parameters = {'criteria': values[n_named:]}
        
        return named_params | criterion_parameters
    
    def compute_expected(self, *args, **kwargs):
        """Placeholder function for an undefined model.

        Returns
        -------
        model_noise : array_like
            The expected values for the noise array, according to the model.
        model_signal : array_like
            The expected values for the signal array, according to the model.

        """
        return self.obs_noise.roc.copy(), self.obs_signal.roc.copy()

    def _arrange_fit_inputs(self, alt=True):
        """Convenience function to arrange observed & expected signal & noise 
        inputs.

        Parameters
        ----------
        alt : bool, optional
            Whether or not to use the alternative input approach. If `True`, 
            each input is a 2d array of cumulative inclusion (O, E; first 
            dimension) and exclusion (N-O, N-E; second dimension) frequencies.
            This method yields similar fits to the standard approach, but also 
            provides better fits for the high threshold model. If `False`, then
            the inputs are simply the observed and expected frequencies forced 
            to be non-zero (1e-100) to support calculation of goodness of fit 
            statistics. The default is True.

        Returns
        -------
        observed_signal : array_like
            DESCRIPTION.
        observed_noise : array_like
            DESCRIPTION.
        expected_signal : array_like
            DESCRIPTION.
        expected_noise : array_like
            DESCRIPTION.

        """
        if alt:
            observed_signal = np.array([self.obs_signal.freqs_acc[:-1], self.obs_signal.n - self.obs_signal.freqs_acc[:-1]])
            observed_noise = np.array([self.obs_noise.freqs_acc[:-1], self.obs_noise.n - self.obs_noise.freqs_acc[:-1]])
            expected_signal = np.array([self.exp_signal.freqs_acc[:-1], self.obs_signal.n - self.exp_signal.freqs_acc[:-1]])
            expected_noise = np.array([self.exp_noise.freqs_acc[:-1], self.obs_noise.n - self.exp_noise.freqs_acc[:-1]])
        else:
            observed_signal = self.obs_signal.freqs
            observed_noise = self.obs_noise.freqs
            # Correct expected freqs. of 0 to prevent division errors
            expected_signal = np.where(self.exp_signal.freqs == 0, 1e-100, self.exp_signal.freqs)
            expected_noise = np.where(self.exp_noise.freqs == 0, 1e-100, self.exp_noise.freqs)
        return {
            'observed_signal': observed_signal,
            'observed_noise': observed_noise,
            'expected_signal': expected_signal,
            'expected_noise': expected_noise
        }

    def _objective(self, x0: array_like, method: Optional[str]='G', alt: Optional[bool]=True) -> float:
        """The objective function to minimise. Not intended to be manually 
        called.
        
        During minimization, this function will be called on each iteration 
        with different values of x0 passed in by scipy.stats.optimize.minimize. 
        The values are then passed to the specific theoretical model being 
        fitted, resulting in a set of model-expected values. These expected 
        values are then compared (according to the objective function) with the 
        observed data to produce the resulting value of the objective function.
        
        When calling the <model>.fit() function, the method parameter defines 
        the method of this objective function (e.g. log-likelihood). This 
        argument is passed as a parameter during the call to optimize.minimize.

        Parameters
        ----------
        x0 : array_like
            List of parameters to be optimised. The first call uses the initial 
            guess, corresponding to list(self.initial_parameters). These values 
            are passed to the theoretical model being fitted, and the value of 
            the objective function is then calculated.
        method : str, optional
            See the .fit method for details. The default is 'G'.
        alt : bool, optional
            Use alternative inputs to goodness-of-fit function. If False, the 
            inputs are simply the observed and expected non-cumulative 
            frequencies for each rating category. If True, the inputs are the 
            observed and expected cumulative frequencies for each rating 
            category, in addition to the complement for that category, which 
            is calculated as N - f where N is the total number of responses for 
            that stimulus class and f is the observed frequency. This method 
            yields comparable fit statistics to the standard approach in most 
            cases, but it additionally fits the high threshold model 
            successfully (unlike the standard approach). Note that the 
            fitted statistics will be slightly different from those obtained 
            with the ROC toolbox. The default is True.

        Returns
        -------
        float
            The value, for this iteration, of the objective function to be 
            minimized (i.e. a χ^2, G^2, or sum of squared errors).

        """
        # Define the model inputs as kwargs for the models' compute_expected method.
        # See the specific model's `compute_expected` for more information.
        model_input = self.define_model_inputs(
            labels=self.parameter_labels,
            values=x0,
            n_criteria=self.n_criteria
        )
        
        # Compute the expected probabilities using the model function
        expected_p_noise, expected_p_signal = self.compute_expected(**model_input)
        self.exp_signal = ResponseData(props_acc=expected_p_signal, n=self.obs_signal.n)
        self.exp_noise = ResponseData(props_acc=expected_p_noise, n=self.obs_noise.n)    
            
        # Compute the goodness-of-fit
        if method.upper() == 'SSE':
            fit_value = self.sum_of_squared_errors()
        elif method.upper() == 'G':     
            fit_value = self.g_statistic(alt)
        elif method.upper() == 'X2':     
            fit_value = self.chi_squared_statistic(alt)
        elif method.upper() == 'LL':
            fit_value = -self.log_likelihood(alt) # Flip the sign for minimisation
        else:
            raise ValueError(f"Method must be one of SSE, G, X2, or LL, but got {method}.")
        if hasattr(self, 'convergence'):
            # Avoid errors if testing outside of the call to .fit()
            self.convergence.append(fit_value)
        return fit_value
    
    def sum_of_squared_errors(self):
        # SSE is 
        sse_signal = squared_errors(self.obs_signal.props, self.exp_signal.props).sum()
        sse_noise = squared_errors(self.obs_noise.props, self.exp_noise.props).sum()
        return sse_signal + sse_noise
    
    def g_statistic(self, alt=True):
        inputs = self._arrange_fit_inputs(alt=alt)
        g_signal = stats.power_divergence(inputs['observed_signal'], inputs['expected_signal'], lambda_='log-likelihood')
        g_noise = stats.power_divergence(inputs['observed_noise'], inputs['expected_noise'], lambda_='log-likelihood')    
        return np.sum(g_signal.statistic) + np.sum(g_noise.statistic)
    
    def chi_squared_statistic(self, alt=True):
        inputs = self._arrange_fit_inputs(alt=alt)
        chi_signal = stats.power_divergence(inputs['observed_signal'], inputs['expected_signal'], lambda_='pearson')
        chi_noise = stats.power_divergence(inputs['observed_noise'], inputs['expected_noise'], lambda_='pearson')    
        return np.sum(chi_signal.statistic) + np.sum(chi_noise.statistic)
    
    def log_likelihood(self, alt=True):
        inputs = self._arrange_fit_inputs(alt=alt)
        ll_signal = log_likelihood(inputs['observed_signal'], inputs['expected_signal'] / self.obs_signal.n)
        ll_noise = log_likelihood(inputs['observed_noise'], inputs['expected_noise'] / self.obs_noise.n)
        return ll_signal + ll_noise

    def fit(self, method: Optional[str]='G', alt: Optional[bool]=True):
        """Fits the theoretical model to the observed data.
        
        Runs the optimisation function according to the chosen method and 
        computes various statistics from the result.
        
        1) Fits the model using scipy.optimize.minimize
        2) Computes the expected signal and noise values after fitting
        3) Computes the squared errors for signal & noise fits
        4) Computes the AIC
        5) Computes euclidean distance between observed and expected values

        Parameters
        ----------
        method : str
            The name of the objective function. Currently accepted values are 
            'G', and 'sse'. The default is 'G'.

        Returns
        -------
        dict
            The fitted parameters.

        """
        self.fit_method = method
        self.convergence = []
        
        # Run the fit function
        self.optimisation_output = minimize(
            fun=self._objective,
            x0=list(self.initial_parameters.values()),
            args=(self.fit_method, alt),
            bounds=self.parameter_boundaries,
            method='nelder-mead',
            tol=1e-4,
            options={'maxiter': 100000, 'xatol': 1e-8 ,'fatol': 1e-4}
        )
        
        # Take the results
        self.fitted_values = self.optimisation_output.x
        self.statistic = self.optimisation_output.fun
        self.fit_success = self.optimisation_output.success
        if not self.fit_success:
            print(f"Fit failed for {self.__modelname__} model.")

        # Define the model inputs as kwargs for the model's `compute_expected` method        
        self._parameter_estimates = self.define_model_inputs(
            labels=self.parameter_labels,
            values=self.fitted_values,
            n_criteria=self.n_criteria
        )
        
        # TODO: After the above, would be nice to have a method to make all stats
        #   like ._make_results()
        
        # Fit statistics
        self.sse = self.sum_of_squared_errors()
        self.gstat = self.g_statistic(alt)
        self.chistat = self.chi_squared_statistic(alt)
        self.loglik = self.log_likelihood(alt)
        self.aic = 2 * self.n_param - 2 * self.loglik
        self.bic = self.n_param * np.log(self.obs_signal.n + self.obs_noise.n) - 2 * self.loglik
        
        # TODO: Define nice results output
        self.results = {
            'model': self.__modelname__,
            'fit-success': self.fit_success,
            'fit-method': method,
            'statistic': self.statistic,
            'log_likelihood': self.loglik,
            'AIC': self.aic,
            'BIC': self.bic,
            'SSE': self.sse,
        }
        
        return self.parameter_estimates

if __name__ == '__main__':
    signal = [505,248,226,172,144,93]
    noise = [115,185,304,523,551,397]
    
    x = _BaseModel(signal, noise)