import numpy as np
import scipy.stats
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS
import statsmodels.base.model as base
from statsmodels.tools.decorators import cached_value

class TobitRegression(base.GenericLikelihoodModel):
    
    __doc__ = """
    Tobit Regression

    %(params)s
    cens : tuple of (left_cens, right_cens)
        A tuple of left censoring and right censoring limits. 
        If None, automatically convert to (-np.inf, np.inf)
        If left_cens/right_cens is None, automatically convert to -np.inf/np.inf
    **kwargs
        Extra arguments that are used to set model properties when using the
        formula interface.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from tobit_model import TobitRegression
    >>> import numpy as np
    >>> duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
    >>> Y = duncan_prestige.data['income']
    >>> X = duncan_prestige.data['education']
    >>> X = sm.add_constant(X)
    >>> cens = (-np.inf, 80)
    >>> model = TobitRegression(Y,X, cens)
    >>> results = model.fit()
    >>> results.params
    const        10.358106
    education     0.605748
    dtype: float64

    >>> results.tvalues
    const        1.965388
    education    6.891241
    dtype: float64

    >>> print(results.t_test([1, 0]))
                                 Test for Constraints
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    c0            10.3581      5.270      1.965      0.056      -0.270      20.987
    ==============================================================================

    >>> print(results.f_test(np.identity(2)))
    <F test: F=array([[157.05408783]]), p=1.71544280870331e-20,
     df_denom=43, df_num=2>
    """ % {'params': base._model_params_doc,
           'extra_params': base._extra_param_doc}
    
    def __init__(self, endog, exog, cens = None, **kwargs):
        super().__init__(endog, exog, **kwargs)
        if cens is None:
            self.cens = (-np.inf, np.inf)
        else:
            left = -np.inf if cens[0] is None else cens[0]
            right = np.inf if cens[1] is None else cens[1]
            self.cens = (left, right)
            
    def whiten():
        
        """
        Tobit Regression whitener does nothing.

        Parameters
        ----------
        x : array_like
            Data to be whitened.

        Returns
        -------
        array_like
            The input array unmodified.

        See Also
        --------
        TobitRegression : Fit a tobit model
        """
        
        return x
        
        
    def loglikeobs(self, params, scale):
        
        """
        Log-likelihood of the model for all observations at params and scale.

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        scale : float
            The scale of the model

        Returns
        -------
        loglike : array_like
            The log likelihood of the model evaluated at `params` and `scale`.
        """
        
        llf = np.zeros(self.exog.shape[0])
        
        # Left censoring
        cond = self.endog <= self.cens[0]
        if cond.sum() > 0:
            exog = self.exog[cond]
            resid = (self.cens[0]-np.dot(exog, params))/np.sqrt(scale)
            llf[cond] += scipy.stats.norm.logcdf(resid)
        
        # Right censoring
        cond = self.endog >= self.cens[1]
        if cond.sum() > 0:
            exog = self.exog[cond]
            resid = (self.cens[1]-np.dot(exog, params))/np.sqrt(scale)
            llf[cond] += scipy.stats.norm.logsf(resid)
        
        # Uncensoring
        cond = (self.endog > self.cens[0]) &\
               (self.endog < self.cens[1])
        if cond.sum() > 0:
            endog = self.endog[cond]
            exog = self.exog[cond]
            resid = (endog-np.dot(exog, params))/np.sqrt(scale)
            llf[cond] += scipy.stats.norm.logpdf(resid)
            llf[cond] -= 1/2*np.log(scale)
        return llf
    
    def loglike(self, params, scale):
        
        """
        Compute the value of the log-likelihood function at params and scale.

        Given the whitened design matrix, the log-likelihood is evaluated
        at the parameter vector `params` for the dependent variable `Y`.

        Parameters
        ----------
        params : array_like
            The parameter estimates.
        scale : float
            The scale estimate

        Returns
        -------
        float
            The value of the log-likelihood function for a Tobit Model.
        """
        
        llf = self.loglikeobs(params, scale)
        llf = llf.sum()
        return llf
        
    def score_obs(self, params, scale):
        
        """score first derivative of the loglikelihood for each observation.

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.
        scale : float
            Scale at which score is evaluated

        Returns
        -------
        score_obs : ndarray, 2d
            The first derivative of the loglikelihood function evaluated at
            params for each observation.
        """
        
        s_params = np.zeros(self.exog.shape[0])
        s_scale = np.zeros(self.exog.shape[0])
        
        # Left censoring
        cond = self.endog <= self.cens[0]
        if cond.sum() > 0:
            exog = self.exog[cond]
            resid = (self.cens[0]-np.dot(exog, params))/np.sqrt(scale)
            w = scipy.stats.norm.pdf(resid)/scipy.stats.norm.cdf(resid)
            s_params[cond] -= w
            s_scale[cond] -= w*resid
        
        # Right censoring
        cond = self.endog >= self.cens[1]
        if cond.sum() > 0:
            exog = self.exog[cond]
            resid = (self.cens[1]-np.dot(exog, params))/np.sqrt(scale)
            w = scipy.stats.norm.pdf(resid)/scipy.stats.norm.sf(resid)
            s_params[cond] += w
            s_scale[cond] += w*resid
        
        # Uncensoring
        cond = (self.endog > self.cens[0]) &\
               (self.endog < self.cens[1])
        if cond.sum() > 0:
            endog = self.endog[cond]
            exog = self.exog[cond]
            resid = (endog-np.dot(exog, params))/np.sqrt(scale)
            s_params[cond] += resid
            s_scale[cond] += (resid**2-1)
        
        s_params = s_params.reshape(-1, 1)*self.exog/np.sqrt(scale)
        s_scale = s_scale.reshape(-1, 1)/(2*scale)
        s = np.append(s_params, s_scale, axis = 1)
        return s
    
    def score(self, params, scale):
        
        """
        Evaluate the score function at a given point.

        The score corresponds to the profile (concentrated)
        log-likelihood in which the scale parameter has been profiled
        out.

        Parameters
        ----------
        params : array_like
            The parameter vector at which the score function is
            computed.
        scale : float
            The scale at which the score function is

        Returns
        -------
        ndarray
            The score vector.
        """
        
        s = self.score_obs(params, scale)
        s = s.sum(axis = 0)
        return s
    
    def hessian_factor(self, params, scale):
        
        """
        Compute the weights for calculating the Hessian.

        Parameters
        ----------
        params : ndarray
            The parameter at which Hessian is evaluated.
        scale : float
            The scale at which Hessian is evaluated

        Returns
        -------
        ndarray
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`.
        """
        
        h_beta = np.zeros(self.exog.shape[0])
        
        # Left censoring
        cond = self.endog <= self.cens[0]
        if cond.sum() > 0:
            exog = self.exog[cond]
            resid = (self.cens[0]-np.dot(exog, params))/np.sqrt(scale)
            pdf = scipy.stats.norm.pdf(resid)
            cdf = scipy.stats.norm.cdf(resid)
            w = pdf/(cdf**2)
            h_beta[cond] += w*(cdf*resid+pdf)
            
        # Right censoring
        cond = self.endog >= self.cens[1]
        if cond.sum() > 0:
            exog = self.exog[cond]
            resid = (self.cens[1]-np.dot(exog, params))/np.sqrt(scale)
            pdf = scipy.stats.norm.pdf(resid)
            sf = scipy.stats.norm.sf(resid)
            w = pdf/(sf**2)
            h_beta[cond] -= w*(sf*resid-pdf)
        
        # Uncensoring
        cond = (self.endog > self.cens[0]) &\
               (self.endog < self.cens[1])
        if cond.sum() > 0:
            h_beta[cond] += 1
        
        h_beta = h_beta.reshape(1, -1)
        return h_beta
    
    def hessian(self, params, scale):
        
        """
        Evaluate the Hessian function at a given point.

        Parameters
        ----------
        params : array_like
            The parameter vector at which the Hessian is computed.
        scale : float
            The scale vector at which the Hessian is computed

        Returns
        -------
        ndarray
            The Hessian matrix.
        """

        
        h_beta = self.hessian_factor(params, scale)
        h_beta = np.dot(self.exog.T*h_beta, self.exog)
        return h_beta
        
    
    def fit(self, method = 'bfgs', use_t = True, **kwargs):
        
        """
        Full fit of the model.

        The results include an estimate of covariance matrix, (whitened)
        residuals and an estimate of scale.

        Parameters
        ----------
        method : str, optional
            Can only be "bfgs" to solve the maximum likelihood estimation problem. 
        use_t : bool
            Flag indicating to use the Student's t in inference.
        **kwargs
            Additional keyword arguments that contain information used when
            constructing a model using the formula interface.

        Returns
        -------
        RegressionResults
            The model estimation results.

        See Also
        --------
        TobitRegressionResults
            The results container.
        TobitRegressionResults.get_robustcov_results
            A method to change the covariance estimator used when fitting the
            model.

        Notes
        -----
        The fit method uses the BFGS to solve the maximum likelihood estimation.
        """
        
        lm = OLS(self.endog, self.exog).fit()
        if method == 'bfgs':
            start_params = np.append(lm.params, [lm.scale])
            loglike = lambda params: -self.loglike(params[:-1], params[-1])
            score = lambda params: -self.score(params[:-1], params[-1])
            res = minimize(loglike, start_params, method = 'bfgs', options={'maxiter': 2000, 'xrtol': 1e-15})
            xopt = res.x[:-1]
            self.scale = res.x[-1]
            Hinv = np.linalg.inv(self.hessian(xopt, self.scale))
        else:
            raise ValueError('method has to be "bfgs"')
            
        mlefit = base.LikelihoodModelResults(self, xopt, Hinv, scale=self.scale, use_t = use_t)
        genericmlefit = TobitRegressionResults(self, mlefit)
        return genericmlefit
    
    def predict(self, params, exog=None):
        
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        params : array_like
            Parameters of a linear model.
        exog : array_like, optional
            Design / exogenous data. Model exog is used if None.

        Returns
        -------
        array_like
            An array of fitted values.

        Notes
        -----
        If the model has not yet been fit, params is not optional.
        """
        
        if exog is None:
            exog = self.exog
        
        linpred = np.dot(exog, params)
        pred = np.zeros(exog.shape[0])
        if self.cens[0] > -np.inf:
            pred += scipy.stats.norm.cdf((self.cens[0]-linpred)/np.sqrt(self.scale))*self.cens[0]
        if self.cens[1] < np.inf:
            pred += scipy.stats.norm.sf((self.cens[1]-linpred)/np.sqrt(self.scale))*self.cens[1]
        pred += (scipy.stats.norm.cdf((self.cens[1]-linpred)/np.sqrt(self.scale)) - scipy.stats.norm.cdf((self.cens[0]-linpred)/np.sqrt(self.scale)))*linpred-np.sqrt(self.scale)*(scipy.stats.norm.pdf((self.cens[1]-linpred)/np.sqrt(self.scale)) - scipy.stats.norm.pdf((self.cens[0]-linpred)/np.sqrt(self.scale)))
        return pred
    
    def get_distribution(self, params, scale, exog=None, dist_class=None):
        
        """
        Construct a random number generator for the predictive distribution.

        Parameters
        ----------
        params : array_like
            The model parameters (regression coefficients).
        scale : scalar
            The variance parameter.
        exog : array_like
            The predictor variable matrix.
        dist_class : class
            A random number generator class.  Must take 'loc' and 'scale'
            as arguments and return a random number generator implementing
            an ``rvs`` method for simulating random values. Defaults to normal.

        Returns
        -------
        gen
            Frozen random number generator object with mean and variance
            determined by the fitted linear model.  Use the ``rvs`` method
            to generate random values.

        Notes
        -----
        Due to the behavior of ``scipy.stats.distributions objects``,
        the returned random number generator must be called with
        ``gen.rvs(n)`` where ``n`` is the number of observations in
        the data set used to fit the model.  If any other value is
        used for ``n``, misleading results will be produced.
        """
        
        fit = self.predict(params, exog)
        if dist_class is None:
            from scipy.stats.distributions import truncnorm
            dist_class = truncnorm
        gen = dist_class((self.cens[0]-fit)/np.sqrt(scale), (self.cens[1]-fit)/np.sqrt(scale), loc=fit, scale=np.sqrt(scale))
        return gen
    
class TobitRegressionResults(base.GenericLikelihoodModelResults):
    @cached_value
    def llf(self):
        """Log-likelihood of model"""
        return self.model.loglike(self.params, self.scale)