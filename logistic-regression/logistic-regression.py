"""
Logistic regression from scracth using NumPy

Develop the three components, first our linear chain denoting the regression component. Then we can develop our sigmoid function
where we will use the base sigmoid function that takes in our linear chain as an input, and finally we'll develop our Maximum Likelihood
Estimator to model our data as likely as possible

"""

import numpy as np
import warnings
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
        self.ll_history_ = []     # loss at each iteration
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

    def _gradients(self, X, y, p):
        """Compute gradients of the average log-likelihood (direction of steepest ascent)
        d(log L) / (dw_j) = sum_i (y_i - p_i) * x_ijdb
        
        """
        error = y - p
        dw = X.T @ error / X.shape[0]

        if self.l2 > 0:
            dw -= (self.l2 / X.shape[0]) * self.weights_
        db = np.mean(error)

        return dw, db


    def fit(self, X, y):
        """Maximize the log-likelihood with batch gradient ascent
        
        """
        # Initialization
        x_arr, y_arr = self._to_numpy(X, y)
        n, d = x_arr.shape
        self.weights_ = np.zeros(d)
        self.bias_ = 0.0
        self.ll_history_ = []
        prev_ll = -np.inf

        for i in range(self.n_iters):
            z = x_arr @ self.weights_ + self.bias_
            p = self._sigmoid(z)
            ll = self._log_likelihood(z, y_arr)
            self.ll_history_.append(ll)
            improvement = ll - prev_ll

            if improvement < 0:
                warnings.warn("Learning rate is most likely too high, decrease the learning rate and try again")
            if improvement < self.tol:
                break
            prev_ll = ll
            dw, db = self._gradients(x_arr, y_arr, p)
            self.weights_ += self.learning_rate * dw
            self.bias_ += self.learning_rate * db
            
            
        self.n_iter_ = i + 1
        return self

        


    def _check_fitted(self):
        """Raise an error if predict is called before fit"""
        if self.weights_ is None:
            raise RuntimeError("call fit() before predicting")

# ------------------------------------------
# Prediction functions
# ------------------------------------------

    def decision_function(self, X):
        """Raw logits z = Xw + b. Shape (n,).

        TODO:
        - self._check_fitted()
        - X = self._to_numpy(X)            (no y, so it returns just the array)
        - (optional) if X.shape[1] != len(self.weights_), raise ValueError
          with a message saying how many features were expected vs. received
        - return X @ self.weights_ + self.bias_
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """Probabilities, shape (n, 2): column 0 = P(y=0), column 1 = P(y=1).

        Matches sklearn's output format so metrics like log_loss work directly.

        TODO:
        - p1 = self._sigmoid(self.decision_function(X))     shape (n,)
        - return np.column_stack([1 - p1, p1])              shape (n, 2)
        """
        raise NotImplementedError

    def predict(self, X, threshold=0.5):
        """Class labels 0/1. Shape (n,).

        TODO:
        - p1 = self.predict_proba(X)[:, 1]        column 1 = P(y=1)
        - return (p1 >= threshold).astype(int)    booleans -> 0/1
        """
        raise NotImplementedError

    def score(self, X, y):
        """Accuracy: fraction of predictions that match y.

        TODO:
        - X, y = self._to_numpy(X, y)       so lists / pandas compare correctly
        - return np.mean(self.predict(X) == y)
        """
        raise NotImplementedError