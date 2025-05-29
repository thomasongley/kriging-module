import numpy as np
from scipy.linalg import cho_factor, LinAlgError, cho_solve
from scipy.optimize import minimize


class OrdinaryKriging:
    def __init__(self, nugget_eps=1e-8, n_restarts=40, random_state=123):
        """
        nugget_eps: nugget for numerical stability
        n_restarts: number of random restarts for hyperparameter optimisation
        random_state: seed for reproducibility of restarts
        """
        self.nugget_eps = nugget_eps
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.is_fitted = False

    def _power_exp(self, X1, X2, params, cc=False):
        # extract p and theta from params
        k = X1.shape[1]
        theta_vec = 10.0 ** params[:k]
        p_vec = params[k : 2 * k]

        # difference matrix (n1 x n2 x k)
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]

        # generate covariance matrix by exponentiating and multiplying
        # the matrix elements by the relevant parameter vectors
        # then summing over last axis
        Psi = np.exp(-np.sum(theta_vec * diff**p_vec, axis=2))

        # if NOT cross-covariance, add nugget term for stability
        if not cc:
            # get square matrix size
            n = X1.shape[0]
            Psi += np.eye(n) * self.nugget_eps

        return Psi

    def _neg_ln_likelihood(self, params):
        # generate cheap covariance matrix
        Psi = self._power_exp(self.X_train, self.X_train, params, cc=False)

        # Cholesky factorisation
        try:
            U, lower = cho_factor(Psi, lower=False, check_finite=False)
        except LinAlgError:
            # non-positive definite, so apply penalty
            negL = 1e8

            return negL
        else:
            # find ln(det(Psi))
            LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(U))))

            # Number of training points
            n = self.y_train.shape[0]

            # vector of ones
            ones = np.ones_like(self.y_train)

            # find mu
            mu_numerator = ones.T @ cho_solve((U, lower), self.y_train)
            mu_denominator = ones.T @ cho_solve((U, lower), ones)
            mu = mu_numerator / mu_denominator

            # find sigmasq (!can be optimised by re-using cho_solve((U, lower), ones)!)
            sigmasq = (
                (self.y_train - mu).T @ cho_solve((U, lower), (self.y_train - mu)) / n
            )

            # concentrated ln(likelihood)
            L = -(n / 2) * np.log(sigmasq) - (1 / 2) * LnDetPsi

            # negative ln likelihood (for minimisation)
            negL = -L

        return negL

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train).reshape(-1, 1)
        k = self.X_train.shape[1]

        # initial params
        init = np.concatenate([np.ones(k), np.full(k, 2.0)])
        # params bounds
        bounds = [(-1, 2)] * k + [(2, 2)] * k

        # prepare restarts
        rng = np.random.default_rng(self.random_state)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        inits = rng.uniform(lower, upper, size=(self.n_restarts, 2 * k))
        inits = np.vstack([init, inits])

        # multi-start optimization
        best = None
        for x0 in inits:
            sol = minimize(
                self._neg_ln_likelihood, x0, bounds=bounds, method="L-BFGS-B"
            )
            if best is None or sol.fun < best.fun:
                best = sol

        self.params_ = best.x

        # precompute factorization for prediction
        Psi = self._power_exp(self.X_train, self.X_train, self.params_, cc=False)
        self.L_, self.lower_ = cho_factor(Psi, check_finite=False)

        # compute the “alpha” vector for fast predictions
        one = np.ones((self.X_train.shape[0], 1))
        mu_den = one.T @ cho_solve((self.L_, self.lower_), one)
        mu_num = one.T @ cho_solve((self.L_, self.lower_), self.y_train)
        self.mu_ = float(mu_num / mu_den)
        self.alpha_ = cho_solve((self.L_, self.lower_), (self.y_train - self.mu_))

        self.is_fitted = True
        return self

    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("Call fit before predict.")
        X_test = np.asarray(X_test)
        psi = self._power_exp(self.X_train, X_test, self.params_, cc=True)
        return self.mu_ + psi.T @ self.alpha_


class CoKriging:
    def __init__(self, nugget_eps=1e-8, n_restarts=40, random_state=123):
        self.nugget_eps = nugget_eps
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.is_fitted = False

    def _power_exp(self, X1, X2, params, cc=False):
        k = X1.shape[1]
        theta_vec = 10.0 ** params[:k]
        p_vec = params[k : 2 * k]
        diff = X1[:, None, :] - X2[None, :, :]
        Psi = np.exp(-np.sum(theta_vec * diff**p_vec, axis=2))
        if not cc:
            n = X1.shape[0]
            Psi += np.eye(n) * self.nugget_eps
        return Psi

    def _neg_ln_likelihood_c(self, params_c):
        Psi = self._power_exp(self.X_train_c, self.X_train_c, params_c, cc=False)
        try:
            U_c, lower_c = cho_factor(Psi, lower=False, check_finite=False)
        except LinAlgError:
            return 1e8
        LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(U_c))))
        n = self.y_train_c.shape[0]
        ones = np.ones_like(self.y_train_c)
        mu_num = ones.T @ cho_solve((U_c, lower_c), self.y_train_c)
        mu_den = ones.T @ cho_solve((U_c, lower_c), ones)
        mu = mu_num / mu_den
        resid = self.y_train_c - mu
        sigmasq = (resid.T @ cho_solve((U_c, lower_c), resid)) / n
        L = -(n / 2) * np.log(sigmasq) - 0.5 * LnDetPsi
        return -L

    def _neg_ln_likelihood_d(self, params_d):
        Psi = self._power_exp(self.X_train_e, self.X_train_e, params_d, cc=False)
        try:
            U_d, lower_d = cho_factor(Psi, lower=False, check_finite=False)
        except LinAlgError:
            return 1e8
        LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(U_d))))
        n = self.y_train_e.shape[0]
        ones = np.ones_like(self.y_train_e)
        rho = params_d[-1]
        y_c_exp = self._y_c_at_X_e()
        d = self.y_train_e - rho * y_c_exp
        mu_num = ones.T @ cho_solve((U_d, lower_d), d)
        mu_den = ones.T @ cho_solve((U_d, lower_d), ones)
        mu = mu_num / mu_den
        resid = d - mu
        sigmasq = (resid.T @ cho_solve((U_d, lower_d), resid)) / n
        L = -(n / 2) * np.log(sigmasq) - 0.5 * LnDetPsi
        return -L

    def _y_c_at_X_e(self):
        Xc = np.ascontiguousarray(self.X_train_c)
        Xe = np.ascontiguousarray(self.X_train_e)
        void_dt = np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))
        c_view = Xc.view(void_dt).ravel()
        e_view = Xe.view(void_dt).ravel()
        idx_map = {c.tobytes(): i for i, c in enumerate(c_view)}
        indices = [idx_map[e.tobytes()] for e in e_view]
        return self.y_train_c[indices]

    def fit(self, X_train_c, y_train_c, X_train_e, y_train_e):
        # store data
        self.X_train_c = np.asarray(X_train_c)
        self.y_train_c = np.asarray(y_train_c).reshape(-1, 1)
        self.X_train_e = np.asarray(X_train_e)
        self.y_train_e = np.asarray(y_train_e).reshape(-1, 1)
        k = self.X_train_c.shape[1]

        # cheap hyperparameters
        init_c = np.concatenate([np.ones(k), np.full(k, 2.0)])
        bounds_c = [(-2, 2)] * k + [(2, 2)] * k
        rng = np.random.default_rng(self.random_state)
        lower_c = np.array([b[0] for b in bounds_c])
        upper_c = np.array([b[1] for b in bounds_c])
        inits_c = rng.uniform(lower_c, upper_c, size=(self.n_restarts, 2 * k))
        inits_c = np.vstack([init_c, inits_c])
        best_c = None
        for x0 in inits_c:
            sol = minimize(
                self._neg_ln_likelihood_c, x0, bounds=bounds_c, method="L-BFGS-B"
            )
            if best_c is None or sol.fun < best_c.fun:
                best_c = sol
        self.params_c = best_c.x

        # expensive hyperparameters (+ rho)
        init_d = np.concatenate([np.ones(k), np.full(k, 2.0), [1.0]])
        bounds_d = [(-2, 2)] * k + [(2, 2)] * k + [(-1, 1)]
        lower_d = np.array([b[0] for b in bounds_d])
        upper_d = np.array([b[1] for b in bounds_d])
        inits_d = rng.uniform(lower_d, upper_d, size=(self.n_restarts, 2 * k + 1))
        inits_d = np.vstack([init_d, inits_d])
        best_d = None
        for x0 in inits_d:
            sol = minimize(
                self._neg_ln_likelihood_d, x0, bounds=bounds_d, method="L-BFGS-B"
            )
            if best_d is None or sol.fun < best_d.fun:
                best_d = sol
        self.params_d = best_d.x

        # build model
        self.build_cokriging(
            self.X_train_c,
            self.y_train_c,
            self.X_train_e,
            self.y_train_e,
            self.params_c,
            self.params_d,
        )
        return self

    def build_cokriging(
        self, X_train_c, y_train_c, X_train_e, y_train_e, params_c, params_d
    ):
        # store all inputs
        self.X_train_c = X_train_c
        self.y_train_c = y_train_c.reshape(-1, 1)
        self.X_train_e = X_train_e
        self.y_train_e = y_train_e.reshape(-1, 1)
        self.params_c = params_c
        self.params_d = params_d

        # cheap-to-cheap
        Psi_c_X_c = self._power_exp(self.X_train_c, self.X_train_c, params_c, cc=False)
        self.U_c, self.lower_c = cho_factor(Psi_c_X_c, lower=False, check_finite=False)

        # cheap-to-expensive & its transpose
        Psi_c_X_e = self._power_exp(self.X_train_e, self.X_train_e, params_c, cc=False)
        Psi_c_X_c_X_e = self._power_exp(
            self.X_train_c, self.X_train_e, params_c, cc=True
        )
        Psi_c_X_e_X_c = Psi_c_X_c_X_e.T

        # expensive-to-expensive
        Psi_d_X_e = self._power_exp(self.X_train_e, self.X_train_e, params_d, cc=False)
        self.U_d, self.lower_d = cho_factor(Psi_d_X_e, lower=False, check_finite=False)

        # cheap mu & sigmasq
        n_c = self.y_train_c.shape[0]
        ones_c = np.ones_like(self.y_train_c)
        mu_c_num = ones_c.T @ cho_solve((self.U_c, self.lower_c), self.y_train_c)
        mu_c_den = ones_c.T @ cho_solve((self.U_c, self.lower_c), ones_c)
        self.mu_c = mu_c_num / mu_c_den
        resid_c = self.y_train_c - self.mu_c
        self.sigmasq_c = (
            resid_c.T @ cho_solve((self.U_c, self.lower_c), resid_c)
        ) / n_c

        # difference mu & sigmasq
        self.rho = params_d[-1]
        y_c_exp = self._y_c_at_X_e()
        d = self.y_train_e - self.rho * y_c_exp
        n_d = d.shape[0]
        ones_d = np.ones_like(d)
        mu_d_num = ones_d.T @ cho_solve((self.U_d, self.lower_d), d)
        mu_d_den = ones_d.T @ cho_solve((self.U_d, self.lower_d), ones_d)
        self.mu_d = mu_d_num / mu_d_den
        resid_d = d - self.mu_d
        self.sigmasq_d = (
            resid_d.T @ cho_solve((self.U_d, self.lower_d), resid_d)
        ) / n_d

        # assemble full C
        C11 = self.sigmasq_c * Psi_c_X_c
        C12 = self.rho * self.sigmasq_c * Psi_c_X_c_X_e
        C21 = self.rho * self.sigmasq_c * Psi_c_X_e_X_c
        C22 = (self.rho**2) * self.sigmasq_c * Psi_c_X_e + self.sigmasq_d * Psi_d_X_e
        C = np.block([[C11, C12], [C21, C22]])

        self.U_C, self.lower_C = cho_factor(C, lower=False, check_finite=False)

        # combined mean
        y_combined = np.vstack((self.y_train_c, self.y_train_e))
        ones_combined = np.ones_like(y_combined)
        num = ones_combined.T @ cho_solve((self.U_C, self.lower_C), y_combined)
        den = ones_combined.T @ cho_solve((self.U_C, self.lower_C), ones_combined)
        self.mu_combined = num / den

        self.is_fitted = True
        return self.U_C, self.lower_C, self.mu_combined

    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("You must call build_cokriging first!")

        # unpack
        rho = self.rho
        sig_c = self.sigmasq_c
        sig_d = self.sigmasq_d
        U_C, lower_C = self.U_C, self.lower_C

        # cross-covariances
        cc_c = (
            rho
            * sig_c
            * self._power_exp(self.X_train_c, X_test, self.params_c, cc=True)
        )
        cc_d = (rho**2) * sig_c * self._power_exp(
            self.X_train_e, X_test, self.params_c, cc=True
        ) + sig_d * self._power_exp(self.X_train_e, X_test, self.params_d, cc=True)

        c = np.block([[cc_c], [cc_d]])
        y_combined = np.vstack((self.y_train_c, self.y_train_e))

        # predictor
        y_pred = self.mu_combined + (
            c.T @ cho_solve((U_C, lower_C), (y_combined - self.mu_combined))
        )

        # MSE / RMSE
        var1 = rho**2 * sig_c + sig_d
        tmp = cho_solve((U_C, lower_C), c)
        diag = np.sum(c * tmp, axis=0)[:, None]
        rmse_pred = np.sqrt(var1 - diag)

        return y_pred, rmse_pred


class HierarchicalKriging:
    def __init__(self, nugget_eps=1e-8, n_restarts=40, random_state=123):
        """
        Hierarchical Kriging (two-stage: cheap then residual)
        nugget_eps: nugget for numerical stability
        n_restarts: number of random restarts for hyperparameter optimisation
        random_state: seed for reproducibility of restarts
        """
        self.nugget_eps = nugget_eps
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.is_fitted = False

    def _power_exp(self, X1, X2, params, cc=False):
        k = X1.shape[1]
        theta = 10.0 ** params[:k]
        p = params[k : 2 * k]
        diff = X1[:, None, :] - X2[None, :, :]
        K = np.exp(-np.sum(theta * diff**p, axis=2))
        if not cc:
            n = X1.shape[0]
            K += np.eye(n) * self.nugget_eps
        return K

    def _neg_ln_likelihood_c(self, params):
        # cheap likelihood
        R = self._power_exp(self.X_c, self.X_c, params, cc=False)
        try:
            U, lower = cho_factor(R, lower=False, check_finite=False)
        except LinAlgError:
            return 1e8
        # log-det
        ld = 2 * np.sum(np.log(np.abs(np.diag(U))))
        n = self.y_c.shape[0]
        one = np.ones_like(self.y_c)
        mu_num = one.T @ cho_solve((U, lower), self.y_c)
        mu_den = one.T @ cho_solve((U, lower), one)
        mu = mu_num / mu_den
        res = self.y_c - mu
        sig2 = (res.T @ cho_solve((U, lower), res)) / n
        logL = -(n / 2) * np.log(sig2) - 0.5 * ld
        return -logL

    def _neg_ln_likelihood_e(self, params):
        # expensive likelihood on residual
        R = self._power_exp(self.X_e, self.X_e, params, cc=False)
        try:
            U, lower = cho_factor(R, lower=False, check_finite=False)
        except LinAlgError:
            return 1e8
        ld = 2 * np.sum(np.log(np.abs(np.diag(U))))
        n = self.y_e.shape[0]
        F = self.F  # design vector from cheap predictions
        y = self.y_e
        mu_num = F.T @ cho_solve((U, lower), y)
        mu_den = F.T @ cho_solve((U, lower), F)
        mu = mu_num / mu_den
        res = y - F * mu
        sig2 = (res.T @ cho_solve((U, lower), res)) / n
        logL = -(n / 2) * np.log(sig2) - 0.5 * ld
        return -logL

    def fit(self, X_c, y_c, X_e, y_e):
        # store data
        self.X_c = np.asarray(X_c)
        self.y_c = np.asarray(y_c).reshape(-1, 1)
        self.X_e = np.asarray(X_e)
        self.y_e = np.asarray(y_e).reshape(-1, 1)
        k = self.X_c.shape[1]

        # multi-start optimize cheap hyperparameters
        init_c = np.concatenate([np.ones(k), np.full(k, 2.0)])
        bounds_c = [(-2, 2)] * k + [(2, 2)] * k
        rng = np.random.default_rng(self.random_state)
        lows = np.array([b[0] for b in bounds_c])
        highs = np.array([b[1] for b in bounds_c])
        inits = rng.uniform(lows, highs, size=(self.n_restarts, 2 * k))
        inits = np.vstack([init_c, inits])
        best = None
        for x0 in inits:
            sol = minimize(
                self._neg_ln_likelihood_c, x0, bounds=bounds_c, method="L-BFGS-B"
            )
            if best is None or sol.fun < best.fun:
                best = sol
        self.params_c = best.x

        # factor and store cheap model
        R_c = self._power_exp(self.X_c, self.X_c, self.params_c, cc=False)
        self.U_c, self.lower_c = cho_factor(R_c, lower=False, check_finite=False)
        one_c = np.ones((self.X_c.shape[0], 1))
        num_c = one_c.T @ cho_solve((self.U_c, self.lower_c), self.y_c)
        den_c = one_c.T @ cho_solve((self.U_c, self.lower_c), one_c)
        self.mu_c = num_c / den_c
        res_c = self.y_c - self.mu_c
        self.sig2_c = (
            res_c.T @ cho_solve((self.U_c, self.lower_c), res_c)
        ) / self.X_c.shape[0]
        # cheap alpha for predictor
        self.alpha_c = cho_solve((self.U_c, self.lower_c), res_c)

        # compute design vector F at expensive locations
        self.F = self.predict_low(self.X_e)

        # multi-start optimize expensive hyperparameters
        init_e = np.concatenate([np.ones(k), np.full(k, 2.0)])
        bounds_e = [(-2, 2)] * k + [(2, 2)] * k
        lows = np.array([b[0] for b in bounds_e])
        highs = np.array([b[1] for b in bounds_e])
        inits = rng.uniform(lows, highs, size=(self.n_restarts, 2 * k))
        inits = np.vstack([init_e, inits])
        best = None
        for x0 in inits:
            sol = minimize(
                self._neg_ln_likelihood_e, x0, bounds=bounds_e, method="L-BFGS-B"
            )
            if best is None or sol.fun < best.fun:
                best = sol
        self.params_e = best.x

        # factor and store expensive model
        R_e = self._power_exp(self.X_e, self.X_e, self.params_e, cc=False)
        self.U_e, self.lower_e = cho_factor(R_e, lower=False, check_finite=False)
        # compute expensive mu and sig2
        num_e = self.F.T @ cho_solve((self.U_e, self.lower_e), self.y_e)
        den_e = self.F.T @ cho_solve((self.U_e, self.lower_e), self.F)
        self.mu_e = num_e / den_e
        res_e = self.y_e - self.F * self.mu_e
        self.sig2_e = (
            res_e.T @ cho_solve((self.U_e, self.lower_e), res_e)
        ) / self.X_e.shape[0]

        self.is_fitted = True
        return self

    def predict_low(self, X_test):
        if not hasattr(self, "U_c"):
            raise ValueError("Fit cheap model first.")
        r = self._power_exp(self.X_c, X_test, self.params_c, cc=True)
        one_c = np.ones((self.X_c.shape[0], 1))
        beta = (one_c.T @ cho_solve((self.U_c, self.lower_c), self.y_c)) / (
            one_c.T @ cho_solve((self.U_c, self.lower_c), one_c)
        )
        return beta + r.T @ cho_solve((self.U_c, self.lower_c), (self.y_c - beta))

    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("Call fit before predict.")
        # low-fidelity prediction
        y_lf = self.predict_low(X_test)
        # high-fidelity correction
        r_e = self._power_exp(self.X_e, X_test, self.params_e, cc=True)
        corr = cho_solve((self.U_e, self.lower_e), (self.y_e - self.F * self.mu_e))
        return self.mu_e * y_lf + r_e.T @ corr
