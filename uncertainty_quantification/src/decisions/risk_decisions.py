"""
Risk-Aware Decision Making Framework

Implements decision rules based on uncertainty quantification for
optimal risk management including expected cost, VaR, CVaR, and
selective prediction strategies.
"""

import numpy as np
from typing import Callable, Optional, Dict, Tuple, List, Union
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.optimize import minimize


@dataclass
class DecisionConfig:
    """Configuration for risk-aware decision making."""
    abstention_threshold: float = 0.3  # Uncertainty threshold for abstention
    escalation_threshold: float = 0.5  # Threshold for escalation
    max_set_size: int = 3  # Maximum conformal set size
    cost_matrix: Optional[np.ndarray] = None  # Cost matrix for decisions
    tail_risk_alpha: float = 0.95  # Alpha for VaR/CVaR
    risk_aversion: float = 1.0  # Risk aversion parameter


class RiskAwareDecisionMaker:
    """
    Framework for making decisions under uncertainty.
    
    Integrates uncertainty estimates with cost functions to make
    optimal risk-aware decisions.
    """
    
    def __init__(self, config: Optional[DecisionConfig] = None):
        """
        Initialize decision maker.
        
        Args:
            config: Decision configuration
        """
        self.config = config or DecisionConfig()
    
    def expected_cost_decision(
        self,
        probabilities: np.ndarray,
        cost_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make decisions minimizing expected cost.
        
        Args:
            probabilities: Predicted probabilities (n_samples, n_classes)
            cost_matrix: Cost matrix C[a, y] = cost of action a when true class is y
            
        Returns:
            Optimal actions minimizing expected cost
        """
        if cost_matrix is None:
            cost_matrix = self.config.cost_matrix
            
        if cost_matrix is None:
            # Default: 0-1 loss (misclassification cost)
            n_classes = probabilities.shape[1]
            cost_matrix = 1 - np.eye(n_classes)
        
        # Compute expected cost for each action
        # E[C(a, Y) | X] = sum_y P(Y=y|X) * C(a, y)
        expected_costs = probabilities @ cost_matrix.T
        
        # Choose action with minimum expected cost
        optimal_actions = np.argmin(expected_costs, axis=1)
        
        return optimal_actions
    
    def selective_prediction(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selective prediction with abstention based on uncertainty.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates (e.g., entropy, variance)
            threshold: Abstention threshold (uses config if None)
            
        Returns:
            Tuple of (decisions, abstain_mask)
        """
        if threshold is None:
            threshold = self.config.abstention_threshold
        
        # Abstain when uncertainty exceeds threshold
        abstain_mask = uncertainties > threshold
        
        # Create decision array (-1 for abstention)
        decisions = predictions.copy()
        decisions[abstain_mask] = -1
        
        return decisions, abstain_mask
    
    def hierarchical_decision(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        conformal_sets: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Hierarchical decision making with multiple thresholds.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            conformal_sets: Conformal prediction sets
            
        Returns:
            Decision codes: 0=predict, 1=abstain, 2=escalate
        """
        n_samples = len(predictions)
        decisions = np.zeros(n_samples, dtype=int)
        
        # Level 1: High confidence - make prediction
        confident_mask = uncertainties <= self.config.abstention_threshold
        decisions[confident_mask] = 0
        
        # Level 2: Medium confidence - abstain
        abstain_mask = (
            (uncertainties > self.config.abstention_threshold) &
            (uncertainties <= self.config.escalation_threshold)
        )
        decisions[abstain_mask] = 1
        
        # Level 3: Low confidence - escalate
        escalate_mask = uncertainties > self.config.escalation_threshold
        decisions[escalate_mask] = 2
        
        # Additional check: large conformal sets trigger escalation
        if conformal_sets is not None:
            for i, conf_set in enumerate(conformal_sets):
                if len(conf_set) > self.config.max_set_size:
                    decisions[i] = max(decisions[i], 2)
        
        return decisions
    
    def value_at_risk(
        self,
        loss_samples: np.ndarray,
        alpha: Optional[float] = None
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        VaR_α = inf{t : F_Y(t) ≥ α}
        
        Args:
            loss_samples: Samples from loss distribution
            alpha: Confidence level (uses config if None)
            
        Returns:
            VaR at specified confidence level
        """
        if alpha is None:
            alpha = self.config.tail_risk_alpha
        
        return np.quantile(loss_samples, alpha)
    
    def conditional_value_at_risk(
        self,
        loss_samples: np.ndarray,
        alpha: Optional[float] = None
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        CVaR_α = E[Y | Y ≥ VaR_α]
        
        Args:
            loss_samples: Samples from loss distribution
            alpha: Confidence level
            
        Returns:
            CVaR at specified confidence level
        """
        if alpha is None:
            alpha = self.config.tail_risk_alpha
        
        var = self.value_at_risk(loss_samples, alpha)
        return np.mean(loss_samples[loss_samples >= var])
    
    def tail_probability(
        self,
        samples: np.ndarray,
        threshold: float
    ) -> float:
        """
        Calculate tail probability P(Y ≥ t | X).
        
        Args:
            samples: Samples from predictive distribution
            threshold: Threshold value
            
        Returns:
            Probability of exceeding threshold
        """
        return np.mean(samples >= threshold)
    
    def risk_adjusted_decision(
        self,
        mean_outcomes: np.ndarray,
        std_outcomes: np.ndarray,
        risk_aversion: Optional[float] = None
    ) -> np.ndarray:
        """
        Make decisions with mean-variance risk adjustment.
        
        Utility = mean - (risk_aversion * variance)
        
        Args:
            mean_outcomes: Expected outcomes for each action
            std_outcomes: Standard deviation of outcomes
            risk_aversion: Risk aversion parameter (uses config if None)
            
        Returns:
            Risk-adjusted optimal actions
        """
        if risk_aversion is None:
            risk_aversion = self.config.risk_aversion
        
        # Calculate risk-adjusted utility
        variance = std_outcomes ** 2
        utility = mean_outcomes - risk_aversion * variance
        
        # Choose action with maximum utility
        return np.argmax(utility, axis=-1)
    
    def robust_decision(
        self,
        probabilities: np.ndarray,
        epsilon: float = 0.1
    ) -> np.ndarray:
        """
        Robust decision making under distributional uncertainty.
        
        Considers worst-case distribution within epsilon-ball.
        
        Args:
            probabilities: Predicted probabilities
            epsilon: Uncertainty radius
            
        Returns:
            Robust optimal actions
        """
        n_samples, n_classes = probabilities.shape
        decisions = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            p = probabilities[i]
            
            # Define worst-case optimization for each action
            best_worst_case = -np.inf
            best_action = 0
            
            for action in range(n_classes):
                # Worst-case: minimize probability of correct decision
                # subject to ||q - p||_1 <= epsilon
                worst_case_prob = max(0, p[action] - epsilon/2)
                
                if worst_case_prob > best_worst_case:
                    best_worst_case = worst_case_prob
                    best_action = action
            
            decisions[i] = best_action
        
        return decisions


class CostSensitiveDecision:
    """
    Cost-sensitive decision making with various cost structures.
    """
    
    def __init__(self):
        """Initialize cost-sensitive decision maker."""
        pass
    
    @staticmethod
    def asymmetric_cost_matrix(
        false_positive_cost: float,
        false_negative_cost: float
    ) -> np.ndarray:
        """
        Create asymmetric cost matrix for binary classification.
        
        Args:
            false_positive_cost: Cost of false positive
            false_negative_cost: Cost of false negative
            
        Returns:
            2x2 cost matrix
        """
        return np.array([
            [0, false_positive_cost],
            [false_negative_cost, 0]
        ])
    
    @staticmethod
    def rejection_cost_matrix(
        n_classes: int,
        misclass_cost: float = 1.0,
        rejection_cost: float = 0.3
    ) -> np.ndarray:
        """
        Create cost matrix with rejection option.
        
        Args:
            n_classes: Number of classes
            misclass_cost: Cost of misclassification
            rejection_cost: Cost of rejection/abstention
            
        Returns:
            Cost matrix including rejection action
        """
        # Standard misclassification costs
        cost_matrix = misclass_cost * (1 - np.eye(n_classes))
        
        # Add rejection action (last row)
        rejection_row = rejection_cost * np.ones(n_classes)
        cost_matrix = np.vstack([cost_matrix, rejection_row])
        
        return cost_matrix
    
    def optimal_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        false_positive_cost: float,
        false_negative_cost: float
    ) -> float:
        """
        Find optimal decision threshold for cost-sensitive binary classification.
        
        Args:
            scores: Predicted scores/probabilities
            labels: True binary labels
            false_positive_cost: Cost of FP
            false_negative_cost: Cost of FN
            
        Returns:
            Optimal threshold
        """
        thresholds = np.unique(scores)
        best_cost = np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            predictions = (scores >= threshold).astype(int)
            
            # Calculate costs
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            total_cost = fp * false_positive_cost + fn * false_negative_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
        
        return best_threshold


class BayesianDecision:
    """
    Bayesian decision theory implementation.
    """
    
    def __init__(self, prior: Optional[np.ndarray] = None):
        """
        Initialize Bayesian decision maker.
        
        Args:
            prior: Prior probabilities over classes
        """
        self.prior = prior
    
    def bayes_risk(
        self,
        likelihood: np.ndarray,
        cost_matrix: np.ndarray,
        prior: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Bayes risk.
        
        Args:
            likelihood: Likelihood P(X|Y)
            cost_matrix: Cost matrix
            prior: Prior probabilities (uses stored if None)
            
        Returns:
            Bayes risk
        """
        if prior is None:
            prior = self.prior
            
        if prior is None:
            # Uniform prior
            n_classes = cost_matrix.shape[1]
            prior = np.ones(n_classes) / n_classes
        
        # Posterior probabilities
        evidence = likelihood @ prior
        posterior = (likelihood * prior) / evidence
        
        # Expected cost for each action
        expected_costs = cost_matrix @ posterior
        
        # Bayes risk is minimum expected cost
        return np.min(expected_costs)
    
    def minimax_decision(
        self,
        cost_matrix: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Minimax decision rule.
        
        Minimizes maximum possible loss.
        
        Args:
            cost_matrix: Cost matrix C[a, y]
            
        Returns:
            Tuple of (minimax strategy, minimax value)
        """
        n_actions = cost_matrix.shape[0]
        
        # Solve linear program for minimax strategy
        from scipy.optimize import linprog
        
        # Variables: [p_1, ..., p_n, v]
        # Minimize v subject to:
        # sum_i p_i * C[i, j] <= v for all j
        # sum_i p_i = 1
        # p_i >= 0
        
        c = np.zeros(n_actions + 1)
        c[-1] = 1  # Minimize v
        
        # Inequality constraints
        A_ub = np.hstack([cost_matrix.T, -np.ones((cost_matrix.shape[1], 1))])
        b_ub = np.zeros(cost_matrix.shape[1])
        
        # Equality constraint
        A_eq = np.zeros((1, n_actions + 1))
        A_eq[0, :n_actions] = 1
        b_eq = np.array([1])
        
        # Bounds
        bounds = [(0, None)] * n_actions + [(None, None)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')
        
        if result.success:
            strategy = result.x[:n_actions]
            value = result.x[-1]
            return strategy, value
        else:
            warnings.warn("Minimax optimization failed")
            return np.ones(n_actions) / n_actions, np.inf


class PortfolioDecision:
    """
    Portfolio optimization under uncertainty.
    """
    
    def __init__(self):
        """Initialize portfolio decision maker."""
        pass
    
    def mean_variance_portfolio(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float = 1.0,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Mean-variance portfolio optimization.
        
        Args:
            returns: Expected returns for each asset
            covariance: Covariance matrix of returns
            risk_aversion: Risk aversion parameter
            constraints: Additional constraints
            
        Returns:
            Optimal portfolio weights
        """
        n_assets = len(returns)
        
        def objective(weights):
            """Portfolio utility: return - risk_aversion * variance."""
            portfolio_return = weights @ returns
            portfolio_variance = weights @ covariance @ weights
            return -(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Sum to 1
        
        if constraints:
            if 'min_weight' in constraints:
                cons.append({
                    'type': 'ineq',
                    'fun': lambda w: w - constraints['min_weight']
                })
            if 'max_weight' in constraints:
                cons.append({
                    'type': 'ineq',
                    'fun': lambda w: constraints['max_weight'] - w
                })
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Bounds
        bounds = [(0, 1)] * n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x
        else:
            warnings.warn("Portfolio optimization failed")
            return x0
    
    def cvar_portfolio(
        self,
        return_samples: np.ndarray,
        alpha: float = 0.95,
        target_return: Optional[float] = None
    ) -> np.ndarray:
        """
        CVaR-based portfolio optimization.
        
        Minimizes Conditional Value at Risk.
        
        Args:
            return_samples: Sample returns (n_samples, n_assets)
            alpha: Confidence level for CVaR
            target_return: Minimum target return
            
        Returns:
            Optimal portfolio weights
        """
        n_samples, n_assets = return_samples.shape
        
        def cvar_objective(params):
            """CVaR objective function."""
            weights = params[:n_assets]
            var = params[-1]
            
            portfolio_returns = return_samples @ weights
            shortfall = np.maximum(var - portfolio_returns, 0)
            
            cvar = var + np.mean(shortfall) / (1 - alpha)
            return cvar
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda p: np.sum(p[:n_assets]) - 1}
        ]
        
        if target_return is not None:
            mean_returns = np.mean(return_samples, axis=0)
            cons.append({
                'type': 'ineq',
                'fun': lambda p: p[:n_assets] @ mean_returns - target_return
            })
        
        # Initial guess
        x0 = np.concatenate([
            np.ones(n_assets) / n_assets,
            [np.quantile(return_samples @ (np.ones(n_assets) / n_assets), alpha)]
        ])
        
        # Bounds
        bounds = [(0, 1)] * n_assets + [(None, None)]
        
        # Optimize
        result = minimize(cvar_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x[:n_assets]
        else:
            warnings.warn("CVaR optimization failed")
            return np.ones(n_assets) / n_assets


def decision_with_fallback(
    primary_prediction: np.ndarray,
    uncertainty: np.ndarray,
    fallback_prediction: np.ndarray,
    threshold: float = 0.3
) -> np.ndarray:
    """
    Decision with fallback to alternative model.
    
    Args:
        primary_prediction: Primary model predictions
        uncertainty: Uncertainty estimates
        fallback_prediction: Fallback model predictions
        threshold: Uncertainty threshold for fallback
        
    Returns:
        Final decisions
    """
    use_fallback = uncertainty > threshold
    
    decisions = primary_prediction.copy()
    decisions[use_fallback] = fallback_prediction[use_fallback]
    
    return decisions