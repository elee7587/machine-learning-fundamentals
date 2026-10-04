"""
Logistic regression from scracth using NumPy

Develop the three components, first our linear chain denoting the regression component. Then we can develop our sigmoid function
where we will use the base sigmoid function that takes in our linear chain as an input, and finally we'll develop our Maximum Likelihood
Estimator to model our data as likely as possible

"""

import numpy as np

class LogisticRegression:
    """
    Binary logistic regression trained with batch gradient descent.

    Parameters

    -----------------
    learning_rate : float
    Step size alpha for gradient decesnt.

    n_iters : int
    Max number of gradient descent iterations

    l2 : float
    Regularization strength lambda (0 means no regularization)

    tol : float
    Tolerance, stop early if the loss improves by less than our tolerance

    verbose : bool
    Binary to print the loss for every iteration
    
    print_every : int
    Frequency of printing when verbose=True

    """

    def __init__(self, learning_rate=0.1, n_iters=1000, l2=0.0,
                 tol=1e-6, verbose=False, print_every=100):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l2 = l2
        self.tol = tol
        self.verbose = verbose
        self.print_every = print_every
 
        # Learned attributes (sklearn convention: trailing underscore,
        # only set after fit is called)
        self.weights_ = None        # shape (d,)
        self.bias_ = None           # scalar
        self.loss_history_ = []     # loss at each iteration
        self.n_iter_ = 0            # iterations actually run

    # ------------------------------------------
    # Input Handling
    # ------------------------------------------

    @staticmethod
    def _to_numpy(X, y=None):
        """Convert list / pandas / NumPy input to float arrays and validate."""
        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim == 1: x_arr = x_arr.reshape(-1, 1)
        if np.isnan(x_arr).any(): raise ValueError("X contains NaN, impute or drop missing values first")

        if y is None:
            return x_arr
        y_arr = np.asarray(y, dtype=float).ravel()
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X has {x_arr.shape[0]} rows but y has {y_arr.shape[0]}.")
        return x_arr, y_arr

    # -----------------------------------------
    # Math functions
    # -----------------------------------------

    @staticmethod
    def _sigmoid(z):
        """Section 2: sigma(z) = 1 / (1 + exp(-z)), clipped to avoid overflow."""
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def _log_likelihood(self, z, y):
        """Average log-likelihood (higher is better), minus the L2 penalty.
        log L / n = mean( y*z - log(1 + e^z) ), computed stably with logaddexp.
        """
        ll = np.mean(y * z - np.logaddexp(0, z))
        if self.l2 > 0:
            ll = ll - (( self.l2 / (2 * len(y))) * np.sum(self.weights_ ** 2))
        return ll