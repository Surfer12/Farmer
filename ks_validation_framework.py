#!/usr/bin/env python3
"""
Kolmogorov-Smirnov Validation Framework for Hybrid Symbolic-Neural Systems

This module implements K-S testing to validate synthetic data quality and 
distribution fidelity in the context of the Hybrid Symbolic-Neural Accuracy 
Functional. It addresses the critical balance between synthetic scalability 
and empirical grounding discussed in the research.

Key Features:
- One-sample K-S tests for distribution validation
- Two-sample K-S tests for synthetic-real comparison
- Progressive validation gates for hybrid training
- Confidence-aware blending based on K-S statistics
- Integration with the existing hybrid functional framework
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Results from K-S validation testing."""
    ks_statistic: float
    p_value: float
    is_valid: bool
    confidence_level: float
    recommendation: str
    distribution_similarity: float

class KSValidationFramework:
    """
    Kolmogorov-Smirnov validation framework for hybrid AI systems.
    
    This class implements sophisticated validation techniques to ensure
    synthetic data maintains fidelity to real-world distributions while
    supporting the hybrid symbolic-neural approach.
    """
    
    def __init__(self, 
                 alpha: float = 0.05,
                 confidence_threshold: float = 0.7,
                 max_synthetic_ratio: float = 0.5):
        """
        Initialize the K-S validation framework.
        
        Args:
            alpha: Significance level for K-S tests (default: 0.05)
            confidence_threshold: Minimum confidence for synthetic data acceptance
            max_synthetic_ratio: Maximum allowed synthetic data proportion
        """
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.max_synthetic_ratio = max_synthetic_ratio
        self.validation_history = []
        
    def one_sample_ks_test(self, 
                          data: np.ndarray, 
                          reference_dist: str = 'norm',
                          dist_params: Optional[Tuple] = None) -> ValidationResult:
        """
        Perform one-sample K-S test against a reference distribution.
        
        Args:
            data: Sample data to test
            reference_dist: Reference distribution ('norm', 'uniform', 'exponential')
            dist_params: Parameters for the reference distribution
            
        Returns:
            ValidationResult with test statistics and recommendations
        """
        # Perform K-S test
        if dist_params:
            ks_stat, p_value = stats.kstest(data, reference_dist, args=dist_params)
        else:
            ks_stat, p_value = stats.kstest(data, reference_dist)
        
        # Determine validation status
        is_valid = p_value > self.alpha
        confidence_level = 1 - ks_stat  # Higher confidence for smaller K-S statistic
        
        # Generate recommendation
        if is_valid and confidence_level > self.confidence_threshold:
            recommendation = "ACCEPT: Data follows expected distribution"
        elif is_valid:
            recommendation = "CAUTION: Statistically valid but low confidence"
        else:
            recommendation = "REJECT: Significant deviation from expected distribution"
        
        return ValidationResult(
            ks_statistic=ks_stat,
            p_value=p_value,
            is_valid=is_valid,
            confidence_level=confidence_level,
            recommendation=recommendation,
            distribution_similarity=1 - ks_stat
        )
    
    def two_sample_ks_test(self, 
                          synthetic_data: np.ndarray, 
                          real_data: np.ndarray) -> ValidationResult:
        """
        Perform two-sample K-S test between synthetic and real data.
        
        Args:
            synthetic_data: Synthetic dataset
            real_data: Real-world reference dataset
            
        Returns:
            ValidationResult with comparison statistics
        """
        # Perform two-sample K-S test
        ks_stat, p_value = stats.ks_2samp(synthetic_data, real_data)
        
        # Determine validation status
        is_valid = p_value > self.alpha
        confidence_level = 1 - ks_stat
        distribution_similarity = 1 - ks_stat
        
        # Generate recommendation based on research insights
        if is_valid and confidence_level > self.confidence_threshold:
            recommendation = "ACCEPT: Synthetic data maintains real-world fidelity"
        elif is_valid and confidence_level > 0.5:
            recommendation = "CONDITIONAL: Acceptable with monitoring"
        elif ks_stat < 0.2:  # Small practical difference
            recommendation = "CAUTION: Statistically different but practically similar"
        else:
            recommendation = "REJECT: Significant distribution mismatch - risk of model collapse"
        
        return ValidationResult(
            ks_statistic=ks_stat,
            p_value=p_value,
            is_valid=is_valid,
            confidence_level=confidence_level,
            recommendation=recommendation,
            distribution_similarity=distribution_similarity
        )
    
    def progressive_validation_gate(self, 
                                  synthetic_data: np.ndarray,
                                  real_data: np.ndarray,
                                  stage: str = "pre-training") -> Dict:
        """
        Implement progressive validation gates for multi-stage training.
        
        This addresses the research finding that progressive hybrid training
        with validation gates achieves superior performance.
        
        Args:
            synthetic_data: Synthetic training data
            real_data: Real validation data
            stage: Training stage ("pre-training", "fine-tuning", "validation")
            
        Returns:
            Dictionary with validation results and mixing recommendations
        """
        # Perform K-S validation
        ks_result = self.two_sample_ks_test(synthetic_data, real_data)
        
        # Stage-specific validation criteria
        stage_thresholds = {
            "pre-training": {"min_similarity": 0.6, "max_synthetic": 0.8},
            "fine-tuning": {"min_similarity": 0.7, "max_synthetic": 0.5},
            "validation": {"min_similarity": 0.8, "max_synthetic": 0.3}
        }
        
        threshold = stage_thresholds.get(stage, stage_thresholds["fine-tuning"])
        
        # Determine optimal mixing ratio based on K-S results
        if ks_result.distribution_similarity >= threshold["min_similarity"]:
            synthetic_ratio = min(threshold["max_synthetic"], 
                                self.max_synthetic_ratio * ks_result.distribution_similarity)
        else:
            # Reduce synthetic ratio for poor similarity
            synthetic_ratio = max(0.1, threshold["max_synthetic"] * ks_result.distribution_similarity)
        
        # Generate stage-specific recommendations
        gate_status = "PASS" if ks_result.distribution_similarity >= threshold["min_similarity"] else "CONDITIONAL"
        
        return {
            "ks_result": ks_result,
            "gate_status": gate_status,
            "recommended_synthetic_ratio": synthetic_ratio,
            "stage": stage,
            "threshold_met": ks_result.distribution_similarity >= threshold["min_similarity"],
            "quality_score": ks_result.distribution_similarity
        }
    
    def confidence_aware_blending(self, 
                                synthetic_accuracies: np.ndarray,
                                real_accuracies: np.ndarray,
                                model_confidences: np.ndarray) -> np.ndarray:
        """
        Implement confidence-aware label smoothing based on K-S validation.
        
        This implements the CALS approach mentioned in the research, dynamically
        adjusting synthetic-to-real ratios based on model confidence and K-S statistics.
        
        Args:
            synthetic_accuracies: Accuracy scores on synthetic data
            real_accuracies: Accuracy scores on real data  
            model_confidences: Model confidence scores
            
        Returns:
            Optimal blending weights for each sample
        """
        # Validate distribution similarity
        ks_result = self.two_sample_ks_test(synthetic_accuracies, real_accuracies)
        
        # Calculate confidence-aware weights
        base_synthetic_weight = ks_result.distribution_similarity
        
        # Adjust weights based on model confidence
        # High confidence on real data → reduce synthetic reliance
        confidence_adjustment = 1 - model_confidences
        synthetic_weights = base_synthetic_weight * confidence_adjustment
        
        # Ensure weights are bounded and sum appropriately
        synthetic_weights = np.clip(synthetic_weights, 0, self.max_synthetic_ratio)
        real_weights = 1 - synthetic_weights
        
        return np.column_stack([synthetic_weights, real_weights])
    
    def adversarial_validation(self, 
                             synthetic_data: np.ndarray,
                             real_data: np.ndarray,
                             n_bootstrap: int = 1000) -> Dict:
        """
        Perform adversarial validation using bootstrap K-S testing.
        
        This implements continuous evaluation to detect distribution drift
        and prevent model collapse as mentioned in the research.
        
        Args:
            synthetic_data: Synthetic dataset
            real_data: Real dataset
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Comprehensive validation results with stability metrics
        """
        bootstrap_stats = []
        bootstrap_pvals = []
        
        # Bootstrap sampling for robust validation
        for _ in range(n_bootstrap):
            # Sample with replacement
            synth_sample = np.random.choice(synthetic_data, size=len(synthetic_data)//2, replace=True)
            real_sample = np.random.choice(real_data, size=len(real_data)//2, replace=True)
            
            # Perform K-S test
            ks_stat, p_val = stats.ks_2samp(synth_sample, real_sample)
            bootstrap_stats.append(ks_stat)
            bootstrap_pvals.append(p_val)
        
        bootstrap_stats = np.array(bootstrap_stats)
        bootstrap_pvals = np.array(bootstrap_pvals)
        
        # Calculate stability metrics
        stat_stability = 1 - np.std(bootstrap_stats)  # Lower std = higher stability
        pval_stability = np.mean(bootstrap_pvals > self.alpha)  # Proportion of valid tests
        
        # Overall validation score
        validation_score = (stat_stability + pval_stability) / 2
        
        # Risk assessment for model collapse
        collapse_risk = "LOW" if validation_score > 0.8 else "MEDIUM" if validation_score > 0.6 else "HIGH"
        
        return {
            "mean_ks_statistic": np.mean(bootstrap_stats),
            "ks_statistic_std": np.std(bootstrap_stats),
            "mean_p_value": np.mean(bootstrap_pvals),
            "validation_stability": stat_stability,
            "proportion_valid": pval_stability,
            "overall_score": validation_score,
            "collapse_risk": collapse_risk,
            "recommendation": self._generate_adversarial_recommendation(validation_score, collapse_risk)
        }
    
    def _generate_adversarial_recommendation(self, score: float, risk: str) -> str:
        """Generate recommendations based on adversarial validation results."""
        if score > 0.8 and risk == "LOW":
            return "OPTIMAL: Synthetic data maintains excellent fidelity"
        elif score > 0.6 and risk in ["LOW", "MEDIUM"]:
            return "ACCEPTABLE: Monitor for distribution drift"
        elif risk == "MEDIUM":
            return "CAUTION: Increase real data proportion and implement grounding"
        else:
            return "CRITICAL: High collapse risk - prioritize real data and empirical validation"
    
    def visualize_validation_results(self, 
                                   synthetic_data: np.ndarray,
                                   real_data: np.ndarray,
                                   save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of K-S validation results.
        
        Args:
            synthetic_data: Synthetic dataset
            real_data: Real dataset
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution comparison
        axes[0, 0].hist(real_data, bins=50, alpha=0.7, label='Real Data', density=True)
        axes[0, 0].hist(synthetic_data, bins=50, alpha=0.7, label='Synthetic Data', density=True)
        axes[0, 0].set_title('Distribution Comparison')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        
        # 2. Empirical CDFs
        real_sorted = np.sort(real_data)
        synth_sorted = np.sort(synthetic_data)
        real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
        synth_cdf = np.arange(1, len(synth_sorted) + 1) / len(synth_sorted)
        
        axes[0, 1].plot(real_sorted, real_cdf, label='Real Data CDF', linewidth=2)
        axes[0, 1].plot(synth_sorted, synth_cdf, label='Synthetic Data CDF', linewidth=2)
        axes[0, 1].set_title('Empirical Cumulative Distribution Functions')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].legend()
        
        # 3. K-S test results
        ks_result = self.two_sample_ks_test(synthetic_data, real_data)
        
        # Create text summary
        summary_text = f"""
        K-S Statistic: {ks_result.ks_statistic:.4f}
        P-value: {ks_result.p_value:.4f}
        Valid: {ks_result.is_valid}
        Confidence: {ks_result.confidence_level:.3f}
        Similarity: {ks_result.distribution_similarity:.3f}
        
        {ks_result.recommendation}
        """
        
        axes[1, 0].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title('K-S Test Results')
        axes[1, 0].axis('off')
        
        # 4. Quality metrics over time (if validation history exists)
        if self.validation_history:
            quality_scores = [result['quality_score'] for result in self.validation_history]
            axes[1, 1].plot(quality_scores, marker='o', linewidth=2)
            axes[1, 1].axhline(y=self.confidence_threshold, color='r', linestyle='--', 
                              label=f'Confidence Threshold ({self.confidence_threshold})')
            axes[1, 1].set_title('Validation Quality Over Time')
            axes[1, 1].set_xlabel('Validation Cycle')
            axes[1, 1].set_ylabel('Quality Score')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No validation history available', 
                           ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('Validation History')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation visualization saved to {save_path}")
        
        plt.show()

def demonstrate_ks_validation():
    """
    Demonstrate the K-S validation framework with synthetic examples.
    
    This function shows how to integrate K-S testing with the hybrid
    symbolic-neural approach for robust AI system validation.
    """
    print("=== Kolmogorov-Smirnov Validation Framework Demo ===\n")
    
    # Initialize validation framework
    validator = KSValidationFramework(alpha=0.05, confidence_threshold=0.7)
    
    # Generate example datasets
    np.random.seed(42)
    
    # Real data (ground truth)
    real_data = np.random.normal(0, 1, 1000)
    
    # High-quality synthetic data (similar distribution)
    good_synthetic = np.random.normal(0.1, 1.1, 1000)
    
    # Poor-quality synthetic data (different distribution)
    poor_synthetic = np.random.exponential(2, 1000)
    
    print("1. Testing High-Quality Synthetic Data:")
    print("-" * 40)
    good_result = validator.two_sample_ks_test(good_synthetic, real_data)
    print(f"K-S Statistic: {good_result.ks_statistic:.4f}")
    print(f"P-value: {good_result.p_value:.4f}")
    print(f"Distribution Similarity: {good_result.distribution_similarity:.3f}")
    print(f"Recommendation: {good_result.recommendation}\n")
    
    print("2. Testing Poor-Quality Synthetic Data:")
    print("-" * 40)
    poor_result = validator.two_sample_ks_test(poor_synthetic, real_data)
    print(f"K-S Statistic: {poor_result.ks_statistic:.4f}")
    print(f"P-value: {poor_result.p_value:.4f}")
    print(f"Distribution Similarity: {poor_result.distribution_similarity:.3f}")
    print(f"Recommendation: {poor_result.recommendation}\n")
    
    print("3. Progressive Validation Gate Analysis:")
    print("-" * 40)
    for stage in ["pre-training", "fine-tuning", "validation"]:
        gate_result = validator.progressive_validation_gate(good_synthetic, real_data, stage)
        print(f"{stage.upper()}:")
        print(f"  Gate Status: {gate_result['gate_status']}")
        print(f"  Recommended Synthetic Ratio: {gate_result['recommended_synthetic_ratio']:.3f}")
        print(f"  Quality Score: {gate_result['quality_score']:.3f}")
    
    print("\n4. Adversarial Validation Results:")
    print("-" * 40)
    adv_result = validator.adversarial_validation(good_synthetic, real_data, n_bootstrap=100)
    print(f"Mean K-S Statistic: {adv_result['mean_ks_statistic']:.4f}")
    print(f"Validation Stability: {adv_result['validation_stability']:.3f}")
    print(f"Proportion Valid: {adv_result['proportion_valid']:.3f}")
    print(f"Overall Score: {adv_result['overall_score']:.3f}")
    print(f"Collapse Risk: {adv_result['collapse_risk']}")
    print(f"Recommendation: {adv_result['recommendation']}")
    
    # Create visualization
    print("\n5. Generating Validation Visualization...")
    validator.visualize_validation_results(good_synthetic, real_data, 
                                         save_path="ks_validation_results.png")
    
    print("\n=== Integration with Hybrid Functional ===")
    print("This K-S validation framework can be integrated with the")
    print("Hybrid Symbolic-Neural Accuracy Functional to ensure:")
    print("• Synthetic data maintains empirical grounding")
    print("• Progressive training stages meet quality gates")
    print("• Model collapse risks are minimized")
    print("• Confidence-aware blending optimizes performance")

if __name__ == "__main__":
    demonstrate_ks_validation()
