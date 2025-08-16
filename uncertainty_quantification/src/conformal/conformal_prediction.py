"""
Conformal Prediction for Distribution-Free Uncertainty Quantification

Provides prediction intervals and sets with guaranteed coverage
without distributional assumptions.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
try:
    from sklearn.base import BaseEstimator
except ImportError:
    BaseEstimator = object
try:
    from scipy import stats
except ImportError:
    stats = None


class ConformalPredictor:
    """
    Base class for conformal prediction.

    Provides distribution-free prediction intervals/sets with
    guaranteed coverage probability.
    """

    def __init__(
        self,
        model,
        alpha: float = 0.1,
        method: str = 'split'
    ):
        """
        Initialize Conformal Predictor.

        Args:
            model: Base predictor (sklearn-compatible)
            alpha: Miscoverage level (1-alpha is the coverage)
            method: 'split', 'jackknife', or 'jackknife+'
        """
        self.model = model
        self.alpha = alpha
        self.method = method
        self.calibration_scores = None
        self.quantile = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None
    ):
        """
        Fit the conformal predictor.

        Args:
            X_train: Training features
            y_train: Training labels
            X_cal: Calibration features (for split conformal)
            y_cal: Calibration labels (for split conformal)
        """
        if self.method == 'split':
            if X_cal is None or y_cal is None:
                # Split training data
                n = len(X_train)
                n_train = n // 2

                X_cal = X_train[n_train:]
                y_cal = y_train[n_train:]
                X_train = X_train[:n_train]
                y_train = y_train[:n_train]

            # Train model on training set
            self.model.fit(X_train, y_train)

            # Compute calibration scores
            self.calibration_scores = self._compute_scores(X_cal, y_cal)

        elif self.method in ['jackknife', 'jackknife+']:
            # Full conformal or jackknife+
            self.model.fit(X_train, y_train)
            self.calibration_scores = self._compute_scores(X_train, y_train)

        # Compute quantile
        if self.calibration_scores is not None:
            n_cal = len(self.calibration_scores)
            q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
            self.quantile = np.quantile(self.calibration_scores, q_level)
        else:
            self.quantile = 0.0

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores."""
        raise NotImplementedError("Subclasses must implement _compute_scores")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Standard prediction."""
        return self.model.predict(X)


class ConformalRegressor(ConformalPredictor):
    """
    Conformal prediction for regression.

    Provides prediction intervals with guaranteed coverage.
    """

    def __init__(
        self,
        model,
        alpha: float = 0.1,
        method: str = 'split',
        score_function: str = 'absolute'
    ):
        """
        Initialize Conformal Regressor.

        Args:
            model: Base regression model
            alpha: Miscoverage level
            method: 'split', 'jackknife', or 'jackknife+'
            score_function: 'absolute' or 'scaled'
        """
        super().__init__(model, alpha, method)
        self.score_function = score_function
        self.scale_model = None

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores for regression."""
        predictions = self.model.predict(X)

        if self.score_function == 'absolute':
            # Absolute residuals
            scores = np.abs(y - predictions)

        elif self.score_function == 'scaled':
            # Scaled residuals (normalized by predicted variance)
            if hasattr(self.model, 'predict_std'):
                std_pred = self.model.predict_std(X)
            else:
                # Estimate variance using MAD
                residuals = y - predictions
                std_pred = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))

            scores = np.abs(y - predictions) / (std_pred + 1e-8)
        else:
            scores = np.abs(y - predictions)

        return scores

    def predict_interval(
        self,
        X: np.ndarray,
        confidence: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict intervals with coverage guarantee.

        Args:
            X: Input features
            confidence: Confidence level (overrides alpha if provided)

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if confidence is not None:
            alpha = 1 - confidence
            if self.calibration_scores is not None:
                n_cal = len(self.calibration_scores)
                q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
                quantile = np.quantile(self.calibration_scores, q_level)
            else:
                quantile = 0.0
        else:
            quantile = self.quantile if self.quantile is not None else 0.0

        predictions = self.model.predict(X)

        if self.method == 'jackknife+':
            # Jackknife+ intervals
            lower, upper = self._jackknife_plus_intervals(X, predictions)
        else:
            # Standard intervals
            if self.score_function == 'scaled' and hasattr(self.model, 'predict_std'):
                std_pred = self.model.predict_std(X)
                margin = quantile * std_pred
            else:
                margin = quantile

            lower = predictions - margin
            upper = predictions + margin

        return lower, upper

    def _jackknife_plus_intervals(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Jackknife+ prediction intervals."""
        if self.calibration_scores is None:
            return predictions, predictions
        n = len(self.calibration_scores)
        n_test = len(X)

        # Store training data if not already stored
        if not hasattr(self, 'X_train'):
            raise ValueError("Training data not stored for jackknife+")

        self.X_train = getattr(self, '_X_train', X[:n])
        self.y_train = getattr(self, '_y_train', np.zeros(n))

        # Leave-one-out predictions
        loo_preds = np.zeros((n, n_test))

        for i in range(n):
            # Train without sample i
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            if len(self.X_train) > i:
                X_loo = self.X_train[mask]
                y_loo = self.y_train[mask]

                model_loo = self.model.__class__(**self.model.get_params())
                model_loo.fit(X_loo, y_loo)

                loo_preds[i] = model_loo.predict(X)
            else:
                loo_preds[i] = predictions

        # Compute intervals
        residuals = self.calibration_scores
        if residuals is None:
            return predictions, predictions

        lower = np.zeros(n_test)
        upper = np.zeros(n_test)

        for j in range(n_test):
            lower[j] = np.quantile(loo_preds[:, j] - residuals, self.alpha / 2)
            upper[j] = np.quantile(loo_preds[:, j] + residuals, 1 - self.alpha / 2)

        return lower, upper


class ConformalClassifier(ConformalPredictor):
    """
    Conformal prediction for classification.

    Provides prediction sets with guaranteed coverage.
    """

    def __init__(
        self,
        model,
        alpha: float = 0.1,
        method: str = 'split',
        score_function: str = 'lac'
    ):
        """
        Initialize Conformal Classifier.

        Args:
            model: Base classification model
            alpha: Miscoverage level
            method: 'split' or 'full'
            score_function: 'lac' (least ambiguous), 'aps' (adaptive)
        """
        super().__init__(model, alpha, method)
        self.score_function = score_function
        self.n_classes = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None
    ):
        """Fit conformal classifier."""
        # Determine number of classes
        self.n_classes = len(np.unique(y_train))

        # Call parent fit
        super().fit(X_train, y_train, X_cal, y_cal)

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores for classification."""
        # Get predicted probabilities
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
        else:
            # Use decision function if available
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X)
                # Convert to probabilities using softmax
                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            else:
                raise ValueError("Model must have predict_proba or decision_function")

        n_samples = len(y)
        scores = np.zeros(n_samples)

        if self.score_function == 'lac':
            # Least Ambiguous Set-valued Classifier
            for i in range(n_samples):
                scores[i] = 1 - probs[i, y[i]]

        elif self.score_function == 'aps':
            # Adaptive Prediction Sets
            sorted_probs = np.sort(probs, axis=1)[:, ::-1]
            # cumsum_probs = np.cumsum(sorted_probs, axis=1)  # Unused variable

            for i in range(n_samples):
                # Find smallest set containing true label
                true_prob = probs[i, y[i]]
                scores[i] = np.sum(probs[i] >= true_prob) / self.n_classes

        return scores

    def predict_set(
        self,
        X: np.ndarray,
        confidence: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Predict sets with coverage guarantee.

        Args:
            X: Input features
            confidence: Confidence level (overrides alpha if provided)

        Returns:
            List of prediction sets (arrays of class indices)
        """
        if confidence is not None:
            alpha = 1 - confidence
            if self.calibration_scores is not None:
                n_cal = len(self.calibration_scores)
                q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
                quantile = np.quantile(self.calibration_scores, q_level)
            else:
                quantile = 0.0
        else:
            quantile = self.quantile if self.quantile is not None else 0.0

        # Get predicted probabilities
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
        else:
            scores = self.model.decision_function(X)
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        n_samples = len(X)
        prediction_sets = []

        for i in range(n_samples):
            if self.score_function == 'lac':
                # Include classes with score <= quantile
                class_scores = 1 - probs[i]
                pred_set = np.where(class_scores <= quantile)[0]

            elif self.score_function == 'aps':
                # Adaptive prediction sets
                sorted_idx = np.argsort(probs[i])[::-1]
                cumsum = 0
                pred_set = []

                for idx in sorted_idx:
                    pred_set.append(idx)
                    cumsum += probs[i, idx]
                    if cumsum >= 1 - quantile:
                        break

                pred_set = np.array(pred_set)
            else:
                pred_set = np.array([0])  # Default prediction set

            prediction_sets.append(pred_set)

        return prediction_sets

    def set_sizes(self, X: np.ndarray) -> np.ndarray:
        """Get sizes of prediction sets."""
        pred_sets = self.predict_set(X)
        return np.array([len(pred_set) for pred_set in pred_sets])


class AdaptiveConformalPredictor:
    """
    Adaptive Conformal Prediction with time-varying coverage.

    Adjusts coverage dynamically based on recent performance.
    """

    def __init__(
        self,
        base_predictor: ConformalPredictor,
        target_coverage: float = 0.9,
        gamma: float = 0.01,
    ):
        """
        Initialize Adaptive Conformal Predictor.

        Args:
            base_predictor: Base conformal predictor
            target_coverage: Target coverage level
            gamma: Learning rate for adaptation
        """
        self.base_predictor = base_predictor
        self.target_coverage = target_coverage
        self.gamma = gamma
        self.alpha_t = 1 - target_coverage
        self.coverage_history = []

    def update(self, y_true: np.ndarray, intervals: Tuple[np.ndarray, np.ndarray]):
        """
        Update alpha based on observed coverage.

        Args:
            y_true: True values
            intervals: Predicted intervals (lower, upper)
        """
        lower, upper = intervals
        covered = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(covered)

        self.coverage_history.append(coverage)

        # Update alpha using gradient descent
        error = coverage - self.target_coverage
        self.alpha_t = float(self.alpha_t + self.gamma * error)

        # Clip to valid range
        self.alpha_t = float(np.clip(self.alpha_t, 0.001, 0.999))

        # Update base predictor
        self.base_predictor.alpha = self.alpha_t

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with adaptive alpha."""
        if hasattr(self.base_predictor, 'predict_interval'):
            return self.base_predictor.predict_interval(X)
        else:
            raise AttributeError("Base predictor does not have predict_interval method")


class ConditionalConformalPredictor:
    """
    Conditional Conformal Prediction.

    Provides locally adaptive coverage based on feature similarity.
    """

    def __init__(
        self,
        model,
        alpha: float = 0.1,
        kernel: str = 'gaussian',
        bandwidth: float = 1.0
    ):
        """
        Initialize Conditional Conformal Predictor.

        Args:
            model: Base model
            alpha: Miscoverage level
            kernel: Kernel type ('gaussian', 'exponential')
            bandwidth: Kernel bandwidth
        """
        self.model = model
        self.alpha = alpha
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.X_cal = None
        self.y_cal = None
        self.scores_cal = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ):
        """Fit conditional conformal predictor."""
        # Train model
        self.model.fit(X_train, y_train)

        # Store calibration data
        self.X_cal = X_cal
        self.y_cal = y_cal

        # Compute calibration scores
        predictions = self.model.predict(X_cal)
        self.scores_cal = np.abs(y_cal - predictions)

    def _compute_weights(self, x: np.ndarray) -> np.ndarray:
        """Compute weights based on feature similarity."""
        if self.X_cal is None:
            return np.array([])

        n_cal = len(self.X_cal)
        weights = np.zeros(n_cal)

        for i in range(n_cal):
            dist = np.linalg.norm(x - self.X_cal[i])

            if self.kernel == 'gaussian':
                weights[i] = np.exp(-dist**2 / (2 * self.bandwidth**2))
            elif self.kernel == 'exponential':
                weights[i] = np.exp(-dist / self.bandwidth)

        # Normalize weights
        weights = weights / np.sum(weights)

        return weights

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict locally adaptive intervals.

        Args:
            X: Input features

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        n_test = len(X)
        predictions = self.model.predict(X)

        lower = np.zeros(n_test)
        upper = np.zeros(n_test)

        for i in range(n_test):
            # Compute weights for test point
            weights = self._compute_weights(X[i])

            if len(weights) == 0 or self.scores_cal is None:
                lower[i] = predictions[i]
                upper[i] = predictions[i]
                continue

            # Weighted quantile of scores
            sorted_idx = np.argsort(self.scores_cal)
            sorted_scores = self.scores_cal[sorted_idx]
            sorted_weights = weights[sorted_idx]

            cumsum = np.cumsum(sorted_weights)
            quantile_idx = np.searchsorted(cumsum, 1 - self.alpha)

            if quantile_idx < len(sorted_scores):
                quantile = sorted_scores[quantile_idx]
            else:
                quantile = sorted_scores[-1]

            # Construct interval
            lower[i] = predictions[i] - quantile
            upper[i] = predictions[i] + quantile

        return lower, upper


class QuantileRegressor:
    """
    Quantile Regression for uncertainty quantification.

    Directly predicts quantiles without distributional assumptions.
    """

    def __init__(
        self,
        model_class: type,
        quantiles: List[float] = [0.025, 0.5, 0.975],
        **model_kwargs
    ):
        """
        Initialize Quantile Regressor.

        Args:
            model_class: Class of base model (must support quantile loss)
            quantiles: List of quantiles to predict
            **model_kwargs: Arguments for model initialization
        """
        self.quantiles = quantiles
        self.models = {}

        for q in quantiles:
            self.models[q] = model_class(**model_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit quantile models."""
        for q in self.quantiles:
            # Set quantile-specific parameters if supported
            if hasattr(self.models[q], 'set_params'):
                self.models[q].set_params(alpha=q)

            self.models[q].fit(X, y)

    def predict_quantiles(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """Predict all quantiles."""
        predictions = {}

        for q in self.quantiles:
            predictions[q] = self.models[q].predict(X)

        return predictions

    def predict_interval(
        self,
        X: np.ndarray,
        coverage: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict intervals with specified coverage.

        Args:
            X: Input features
            coverage: Coverage level

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        alpha = 1 - coverage
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        # Find closest quantiles
        lower_idx = np.argmin(np.abs(np.array(self.quantiles) - lower_q))
        upper_idx = np.argmin(np.abs(np.array(self.quantiles) - upper_q))

        lower_quantile = self.quantiles[lower_idx]
        upper_quantile = self.quantiles[upper_idx]

        lower = self.models[lower_quantile].predict(X)
        upper = self.models[upper_quantile].predict(X)

        return lower, upper
