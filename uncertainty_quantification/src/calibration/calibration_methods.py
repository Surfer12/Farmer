"""
Calibration Methods for Uncertainty Quantification

Includes temperature scaling, isotonic regression, Platt scaling,
and other post-hoc calibration techniques to ensure predicted
probabilities match observed frequencies.
"""

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    IsotonicRegression = None
    LogisticRegression = None
    SKLEARN_AVAILABLE = False

from typing import List, Dict, Any

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    minimize = None
    SCIPY_AVAILABLE = False
import warnings


class TemperatureScaling:
    """
    Temperature Scaling for calibrating neural network predictions.

    Learns a single scalar temperature parameter T to calibrate
    logits: softmax(z/T)
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize Temperature Scaling.

        Args:
            temperature: Initial temperature value
        """
        self.temperature = temperature
        self.fitted = False

    def fit(
        self,
        logits: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        lr: float = 0.01,
        max_iter: int = 50
    ) -> float:
        """
        Fit temperature parameter using validation data.

        Args:
            logits: Model logits (before softmax)
            labels: True labels
            lr: Learning rate for optimization
            max_iter: Maximum iterations

        Returns:
            Optimal temperature value
        """
        # Convert to numpy if needed
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        def nll_loss(T):
            """Negative log-likelihood loss."""
            # Apply temperature scaling
            scaled_logits = logits / T

            # Compute softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Compute NLL
            n_samples = len(labels)
            nll = -np.sum(np.log(probs[np.arange(n_samples), labels] + 1e-8))

            return nll / n_samples

        # Optimize temperature
        result = minimize(
            nll_loss,
            x0=[self.temperature],
            method='L-BFGS-B',
            bounds=[(0.01, 10.0)],
            options={'maxiter': max_iter}
        )

        self.temperature = result.x[0]
        self.fitted = True

        return self.temperature

    def calibrate(
        self,
        logits: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Model logits

        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            warnings.warn("Temperature scaling not fitted. Using default T=1.0")

        if isinstance(logits, torch.Tensor):
            scaled_logits = logits / self.temperature
            return F.softmax(scaled_logits, dim=-1)
        else:
            scaled_logits = logits / self.temperature
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class VectorScaling:
    """
    Vector Scaling (Dirichlet Calibration).

    Learns a diagonal transformation matrix and bias vector
    for each class: softmax(W*z + b)
    """

    def __init__(self, n_classes: int):
        """
        Initialize Vector Scaling.

        Args:
            n_classes: Number of classes
        """
        self.n_classes = n_classes
        self.W = np.ones(n_classes)  # Diagonal weights
        self.b = np.zeros(n_classes)  # Bias
        self.fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
        reg_lambda: float = 1e-4
    ):
        """
        Fit vector scaling parameters.

        Args:
            logits: Model logits
            labels: True labels
            lr: Learning rate
            max_iter: Maximum iterations
            reg_lambda: L2 regularization strength
        """
        def objective(params):
            """Objective function with L2 regularization."""
            W = params[:self.n_classes]
            b = params[self.n_classes:]

            # Apply transformation
            scaled_logits = logits * W + b

            # Compute softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # NLL loss
            n_samples = len(labels)
            nll = -np.sum(np.log(probs[np.arange(n_samples), labels] + 1e-8))

            # L2 regularization
            reg = reg_lambda * (np.sum(W**2) + np.sum(b**2))

            return nll / n_samples + reg

        # Initial parameters
        init_params = np.concatenate([self.W, self.b])

        # Optimize
        result = minimize(
            objective,
            x0=init_params,
            method='L-BFGS-B',
            options={'maxiter': max_iter}
        )

        # Extract parameters
        self.W = result.x[:self.n_classes]
        self.b = result.x[self.n_classes:]
        self.fitted = True

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply vector scaling calibration."""
        if not self.fitted:
            warnings.warn("Vector scaling not fitted.")
            return self._softmax(logits)

        scaled_logits = logits * self.W + self.b
        return self._softmax(scaled_logits)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class IsotonicCalibration:
    """
    Isotonic Regression Calibration.

    Non-parametric calibration method that learns a monotonic
    mapping from predicted probabilities to calibrated probabilities.
    """

    def __init__(self):
        """Initialize Isotonic Calibration."""
        self.calibrators = {}
        self.fitted = False

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        multiclass: str = 'one-vs-rest'
    ):
        """
        Fit isotonic regression calibrators.

        Args:
            probs: Predicted probabilities
            labels: True labels
            multiclass: Strategy for multiclass ('one-vs-rest' or 'binary')
        """
        if len(probs.shape) == 1 or probs.shape[1] == 1:
            # Binary classification
            self.calibrators[0] = IsotonicRegression(out_of_bounds='clip')
            self.calibrators[0].fit(probs.flatten(), labels)
        else:
            # Multiclass
            n_classes = probs.shape[1]

            if multiclass == 'one-vs-rest':
                # Fit one calibrator per class
                for i in range(n_classes):
                    binary_labels = (labels == i).astype(int)
                    self.calibrators[i] = IsotonicRegression(out_of_bounds='clip')
                    self.calibrators[i].fit(probs[:, i], binary_labels)
            else:
                # Use max probability for binary calibration
                max_probs = np.max(probs, axis=1)
                max_class = np.argmax(probs, axis=1)
                binary_labels = (max_class == labels).astype(int)

                self.calibrators[0] = IsotonicRegression(out_of_bounds='clip')
                self.calibrators[0].fit(max_probs, binary_labels)

        self.fitted = True

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.

        Args:
            probs: Predicted probabilities

        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            warnings.warn("Isotonic calibration not fitted.")
            return probs

        if len(probs.shape) == 1 or probs.shape[1] == 1:
            # Binary
            return self.calibrators[0].transform(probs.flatten()).reshape(-1, 1)
        else:
            # Multiclass
            calibrated = np.zeros_like(probs)

            if len(self.calibrators) == 1:
                # Binary calibration of max probability
                max_probs = np.max(probs, axis=1)
                scale_factor = self.calibrators[0].transform(max_probs) / (max_probs + 1e-8)
                calibrated = probs * scale_factor.reshape(-1, 1)
            else:
                # One-vs-rest calibration
                for i in range(probs.shape[1]):
                    calibrated[:, i] = self.calibrators[i].transform(probs[:, i])

            # Normalize to ensure probabilities sum to 1
            calibrated = calibrated / np.sum(calibrated, axis=1, keepdims=True)

            return calibrated


class PlattScaling:
    """
    Platt Scaling (Sigmoid Calibration).

    Fits a sigmoid function to calibrate binary probabilities:
    P(y=1|f) = 1 / (1 + exp(A*f + B))
    """

    def __init__(self):
        """Initialize Platt Scaling."""
        self.A = -1.0
        self.B = 0.0
        self.fitted = False

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ):
        """
        Fit Platt scaling parameters.

        Args:
            scores: Model scores or probabilities
            labels: Binary labels (0 or 1)
        """
        # Convert probabilities to logits if needed
        if np.all((scores >= 0) & (scores <= 1)):
            # Clip to avoid log(0)
            scores = np.clip(scores, 1e-7, 1 - 1e-7)
            scores = np.log(scores / (1 - scores))

        # Fit logistic regression
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(scores.reshape(-1, 1), labels)

        self.A = lr.coef_[0, 0]
        self.B = lr.intercept_[0]
        self.fitted = True

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling.

        Args:
            scores: Model scores or probabilities

        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            warnings.warn("Platt scaling not fitted.")
            return scores

        # Convert to logits if needed
        if np.all((scores >= 0) & (scores <= 1)):
            scores = np.clip(scores, 1e-7, 1 - 1e-7)
            scores = np.log(scores / (1 - scores))

        # Apply sigmoid calibration
        calibrated = 1.0 / (1.0 + np.exp(-(self.A * scores + self.B)))

        return calibrated


class BetaCalibration:
    """
    Beta Calibration.

    Fits a Beta distribution to calibrate probabilities,
    allowing for more flexible calibration curves.
    """

    def __init__(self):
        """Initialize Beta Calibration."""
        self.a = 1.0
        self.b = 1.0
        self.fitted = False

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray
    ):
        """
        Fit Beta calibration parameters.

        Args:
            probs: Predicted probabilities
            labels: Binary labels
        """
        from scipy.stats import beta
        from scipy.optimize import minimize

        def negative_log_likelihood(params):
            a, b = params

            # Clip parameters to avoid numerical issues
            a = np.clip(a, 0.01, 100)
            b = np.clip(b, 0.01, 100)

            # Transform probabilities using Beta CDF
            calibrated = beta.cdf(probs, a, b)

            # Compute NLL
            eps = 1e-8
            nll = -np.sum(
                labels * np.log(calibrated + eps) +
                (1 - labels) * np.log(1 - calibrated + eps)
            )

            return nll

        # Optimize Beta parameters
        result = minimize(
            negative_log_likelihood,
            x0=[self.a, self.b],
            method='L-BFGS-B',
            bounds=[(0.01, 100), (0.01, 100)]
        )

        self.a, self.b = result.x
        self.fitted = True

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply Beta calibration."""
        if not self.fitted:
            warnings.warn("Beta calibration not fitted.")
            return probs

        from scipy.stats import beta
        return beta.cdf(probs, self.a, self.b)


class EnsembleTemperatureScaling:
    """
    Ensemble Temperature Scaling.

    Learns separate temperature parameters for each model
    in an ensemble.
    """

    def __init__(self, n_models: int):
        """
        Initialize Ensemble Temperature Scaling.

        Args:
            n_models: Number of models in ensemble
        """
        self.n_models = n_models
        self.temperatures = np.ones(n_models)
        self.fitted = False

    def fit(
        self,
        ensemble_logits: np.ndarray,
        labels: np.ndarray
    ):
        """
        Fit temperature parameters for ensemble.

        Args:
            ensemble_logits: Logits from each model (n_models, n_samples, n_classes)
            labels: True labels
        """
        def objective(temps):
            """Joint NLL objective."""
            total_nll = 0

            for i, T in enumerate(temps):
                # Apply temperature to model i
                scaled_logits = ensemble_logits[i] / T

                # Compute softmax
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

                # Add to total NLL
                n_samples = len(labels)
                nll = -np.sum(np.log(probs[np.arange(n_samples), labels] + 1e-8))
                total_nll += nll

            return total_nll / (self.n_models * len(labels))

        # Optimize temperatures jointly
        result = minimize(
            objective,
            x0=self.temperatures,
            method='L-BFGS-B',
            bounds=[(0.01, 10.0)] * self.n_models
        )

        self.temperatures = result.x
        self.fitted = True

    def calibrate(self, ensemble_logits: np.ndarray) -> np.ndarray:
        """
        Apply ensemble temperature scaling.

        Args:
            ensemble_logits: Logits from ensemble

        Returns:
            Calibrated ensemble probabilities
        """
        if not self.fitted:
            warnings.warn("Ensemble temperature scaling not fitted.")

        calibrated_probs = []

        for i in range(self.n_models):
            scaled_logits = ensemble_logits[i] / self.temperatures[i]
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            calibrated_probs.append(probs)

        # Average calibrated probabilities
        return np.mean(calibrated_probs, axis=0)
