import numpy as np
from scipy.optimize import differential_evolution

def solve_chol(L, b):
    """
    Solves L @ L.T @ x = b using forward and back substitution.
    """
    return np.linalg.solve(L.T, np.linalg.solve(L, b))

def likelihood(log_theta_vec, X_train, y_train, jitter=1e-12):
    """
    Computes the negative log-likelihood for the given hyperparameters.

    Parameters:
        log_theta_vec : array-like
            Log10-transformed theta hyperparameters.
        X_train : ndarray
            n x k matrix of training input locations.
        y_train : ndarray
            n x 1 vector of observed training data.
        jitter : float
            Small constant added to the diagonal of the correlation matrix for stability.

    Returns:
        NegLnLike : float
            The negative log likelihood to be minimized.
        Psi : ndarray
            The correlation matrix.
        L : ndarray or None
            The Cholesky factorization of Psi (or None if ill-conditioned).
    """
    theta_vec = 10 ** log_theta_vec
    n = X_train.shape[0]
    one_vec = np.ones((n, 1))

    # Build correlation matrix using a power exponential kernel
    diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]  # shape (n, n, k)
    Psi = np.exp(-np.sum(theta_vec * diff**2, axis=2))
    Psi += np.eye(n) * jitter

    try:
        L = np.linalg.cholesky(Psi)
        ill_conditioned = False
    except np.linalg.LinAlgError:
        ill_conditioned = True

    if ill_conditioned:
        NegLnLike = 1e4  # Large penalty for ill-conditioned cases
        L = None
    else:
        LnDetPsi = 2.0 * np.sum(np.log(np.diag(L)))
        numerator = one_vec.T @ solve_chol(L, y_train)
        denominator = one_vec.T @ solve_chol(L, one_vec)
        mu = numerator / denominator
        SigmaSqr = (y_train - one_vec * mu).T @ solve_chol(L, (y_train - one_vec * mu)) / n
        NegLnLike = ((n / 2.0) * np.log(SigmaSqr) - 0.5 * LnDetPsi)

    return NegLnLike, Psi, L

def objective(log_theta_vec, X_train, y_train):
    """
    Wrapper for the likelihood function for optimization.
    """
    NegLnLike, _, _ = likelihood(log_theta_vec, X_train, y_train)
    return NegLnLike

def pred(X_test, X_train, y_train, log_theta_vec, L):
    """
    Computes the kriging prediction and associated RMSE at test points.

    Parameters:
        X_test : ndarray
            m x k matrix of test locations.
        X_train : ndarray
            n x k training locations.
        y_train : ndarray
            n x 1 training data.
        log_theta_vec : array-like
            Log10-transformed optimal theta hyperparameters.
        L : ndarray
            Cholesky factor of the correlation matrix from training.

    Returns:
        y_pred : ndarray
            Predicted mean at test points.
        s : ndarray
            RMSE (standard deviation) at test points.
    """
    theta_vec = 10.0 ** log_theta_vec
    n = X_train.shape[0]
    one_vec = np.ones((n, 1))

    numerator = one_vec.T @ solve_chol(L, y_train)
    denominator = one_vec.T @ solve_chol(L, one_vec)
    mu = numerator / denominator

    SigmaSqr = (y_train - one_vec * mu).T @ solve_chol(L, (y_train - one_vec * mu)) / n

    # Calculate correlation between training and test points
    diff = X_train[:, np.newaxis, :] - X_test[np.newaxis, :, :]
    psi = np.exp(-np.sum(theta_vec * diff**2, axis=2))

    y_pred = mu + psi.T @ solve_chol(L, (y_train - one_vec * mu))

    psi_solved = solve_chol(L, psi)
    one_solved = solve_chol(L, one_vec)
    A = np.sum(psi * psi_solved, axis=0)
    B = (one_vec.T @ psi_solved).ravel()
    C = (one_vec.T @ one_solved).squeeze()

    SSqr = SigmaSqr * (1.0 - A + (1.0 - B)**2 / C)
    s = np.sqrt(np.maximum(SSqr, 0.0))
    return y_pred.ravel(), s.ravel()

class KrigingModel:
    """
    A reusable kriging model for interpolation and uncertainty quantification.
    """
    def __init__(self, bounds=[(-3, 2)], jitter=1e-12):
        """
        Parameters:
            bounds : list of tuples
                Bounds for the log10-transformed theta hyperparameters. By default,
                the same bound is used for each dimension.
            jitter : float
                Small constant added to the diagonal of the correlation matrix for numerical stability.
        """
        self.bounds = bounds
        self.jitter = jitter
        self.X_train = None
        self.y_train = None
        self.log_theta_optimal = None
        self.Psi = None
        self.L = None

    def fit(self, X_train, y_train):
        """
        Fit the kriging model to the training data by optimizing hyperparameters.

        Parameters:
            X_train : ndarray
                n x k matrix of training input locations.
            y_train : ndarray
                n x 1 vector of observed training data.

        Returns:
            result : OptimizeResult
                The result from the differential evolution optimizer.
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        k = self.X_train.shape[1]

        # Ensure the bounds list matches the number of features
        if len(self.bounds) != k:
            self.bounds = [self.bounds[0]] * k

        result = differential_evolution(
            objective,
            self.bounds,
            args=(self.X_train, self.y_train)
        )
        self.log_theta_optimal = result.x
        _, self.Psi, self.L = likelihood(self.log_theta_optimal, self.X_train, self.y_train, jitter=self.jitter)
        return result

    def predict(self, X_test):
        """
        Predict using the fitted kriging model.

        Parameters:
            X_test : ndarray
                m x k matrix of test locations.

        Returns:
            y_pred : ndarray
                Predicted mean at the test points.
            rmse : ndarray
                Root mean square error (uncertainty estimate) at the test points.
        """
        if self.X_train is None or self.L is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        X_test = np.array(X_test)
        y_pred, rmse = pred(X_test, self.X_train, self.y_train, self.log_theta_optimal, self.L)
        return y_pred, rmse

if __name__ == '__main__':
    # Example usage of the module

    import matplotlib.pyplot as plt

    def forrester(x):
        """
        Forrester function used as a test function.
        """
        return (6 * x - 2)**2 * np.sin(12 * x - 4)

    # Generate training data
    X_train = np.linspace(0, 1, 5).reshape(-1, 1)
    y_train = forrester(X_train)

    # Generate test data
    X_test = np.linspace(0, 1, 201).reshape(-1, 1)

    # Create and fit the kriging model
    model = KrigingModel(bounds=[(-3, 2)])
    model.fit(X_train, y_train)

    # Obtain predictions and uncertainty estimates
    y_pred, rmse = model.predict(X_test)

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(X_train, y_train, 'ro', label='Training Data')
    plt.plot(X_test, y_pred, 'b-', label='Prediction')
    plt.fill_between(X_test.ravel(), y_pred - 2 * rmse, y_pred + 2 * rmse,
                     color='gray', alpha=0.5, label='Confidence Interval')
    plt.legend()
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Kriging Prediction')
    plt.show()
