# Practical Examples

This document provides real-world examples and use cases for the Ψ framework and uncertainty quantification tools.

## Table of Contents

1. [Example 1: News Article Verification](#example-1-news-article-verification)
2. [Example 2: Scientific Paper Assessment](#example-2-scientific-paper-assessment)
3. [Example 3: Financial Risk Analysis](#example-3-financial-risk-analysis)
4. [Example 4: Medical Diagnosis Support](#example-4-medical-diagnosis-support)
5. [Example 5: Social Media Content Moderation](#example-5-social-media-content-moderation)
6. [Example 6: Real-time Fraud Detection](#example-6-real-time-fraud-detection)
7. [Example 7: Academic Research Evaluation](#example-7-academic-research-evaluation)
8. [Example 8: Legal Evidence Assessment](#example-8-legal-evidence-assessment)

---

## Example 1: News Article Verification

**Scenario:** A news organization wants to automatically assess the reliability of incoming articles before publication.

### Java Implementation

```java
// NewsVerificationExample.java
import java.util.*;

public class NewsVerificationExample {
    
    static class NewsArticle {
        String title;
        String source;
        double authorityScore;      // Source credibility (0-1)
        double verifiabilityScore;  // How easy to verify (0-1)
        double factCheckScore;      // Automated fact-check score (0-1)
        boolean hasCanonicalSources; // Links to primary sources
        
        NewsArticle(String title, String source, double authority, 
                   double verifiability, double factCheck, boolean canonical) {
            this.title = title;
            this.source = source;
            this.authorityScore = authority;
            this.verifiabilityScore = verifiability;
            this.factCheckScore = factCheck;
            this.hasCanonicalSources = canonical;
        }
    }
    
    public static void main(String[] args) {
        System.out.println("=== News Article Verification System ===\n");
        
        // Create sample articles
        NewsArticle[] articles = {
            new NewsArticle("Climate Study Shows Warming Trend", "Science Journal", 
                           0.92, 0.85, 0.88, true),
            new NewsArticle("Celebrity Spotted at Restaurant", "Gossip Blog", 
                           0.35, 0.20, 0.45, false),
            new NewsArticle("Election Results Updated", "Reuters", 
                           0.95, 0.90, 0.92, true),
            new NewsArticle("Miracle Cure Discovered", "Unknown Blog", 
                           0.15, 0.10, 0.25, false),
            new NewsArticle("Stock Market Analysis", "Financial Times", 
                           0.88, 0.75, 0.82, true)
        };
        
        HierarchicalBayesianModel model = new HierarchicalBayesianModel();
        
        for (NewsArticle article : articles) {
            evaluateArticle(model, article);
        }
    }
    
    private static void evaluateArticle(HierarchicalBayesianModel model, NewsArticle article) {
        // Convert article features to Ψ framework inputs
        ClaimData claim = new ClaimData(
            article.title,
            true, // Assume true for evaluation (would be unknown in practice)
            1.0 - article.authorityScore,      // Authority risk (inverted)
            1.0 - article.verifiabilityScore,  // Verifiability risk (inverted)
            article.factCheckScore              // Base posterior
        );
        
        // Set parameters based on article characteristics
        double S = article.factCheckScore;  // Internal signal from fact-checking
        double N = article.hasCanonicalSources ? 0.90 : 0.30;  // Canonical evidence
        double alpha = article.hasCanonicalSources ? 0.10 : 0.40;  // Trust canonical more
        double beta = 1.0 + (article.authorityScore * 0.3);  // Authority boosts confidence
        
        ModelParameters params = new ModelParameters(S, N, alpha, beta);
        double psi = model.calculatePsi(claim, params);
        
        // Generate recommendation
        String recommendation;
        String reasoning;
        
        if (psi > 0.85) {
            recommendation = "PUBLISH - High confidence";
            reasoning = "Strong evidence, reliable source, easily verifiable";
        } else if (psi > 0.70) {
            recommendation = "REVIEW - Moderate confidence";
            reasoning = "Good evidence but may need additional verification";
        } else if (psi > 0.50) {
            recommendation = "INVESTIGATE - Low confidence";
            reasoning = "Weak evidence or unreliable source - needs fact-checking";
        } else {
            recommendation = "REJECT - Very low confidence";
            reasoning = "Insufficient evidence or unreliable source";
        }
        
        System.out.println("Article: " + article.title);
        System.out.println("Source: " + article.source);
        System.out.printf("Authority: %.2f, Verifiability: %.2f, Fact-check: %.2f\n",
                         article.authorityScore, article.verifiabilityScore, article.factCheckScore);
        System.out.printf("Ψ Score: %.3f\n", psi);
        System.out.println("Recommendation: " + recommendation);
        System.out.println("Reasoning: " + reasoning);
        System.out.println();
    }
}
```

### Swift Implementation

```swift
// NewsVerificationApp.swift
import UOIFCore

struct NewsArticle {
    let title: String
    let source: String
    let authorityScore: Double      // Source credibility (0-1)
    let verifiabilityScore: Double  // How easy to verify (0-1)
    let factCheckScore: Double      // Automated fact-check score (0-1)
    let hasCanonicalSources: Bool   // Links to primary sources
}

struct VerificationResult {
    let psi: Double
    let recommendation: String
    let reasoning: String
    let confidence: Double
}

class NewsVerificationService {
    func evaluateArticle(_ article: NewsArticle) -> VerificationResult {
        // Convert article features to Ψ framework inputs
        let inputs = PsiInputs(
            alpha: article.hasCanonicalSources ? 0.10 : 0.40,  // Trust canonical more
            S_symbolic: article.factCheckScore,                 // Internal signal
            N_external: article.hasCanonicalSources ? 0.90 : 0.30,  // Canonical evidence
            lambdaAuthority: 0.85,
            lambdaVerifiability: 0.15,
            riskAuthority: 1.0 - article.authorityScore,       // Authority risk (inverted)
            riskVerifiability: 1.0 - article.verifiabilityScore,  // Verifiability risk
            basePosterior: article.factCheckScore,
            betaUplift: 1.0 + (article.authorityScore * 0.3)   // Authority boosts confidence
        )
        
        let outcome = PsiModel.computePsi(inputs: inputs)
        
        // Generate recommendation
        let (recommendation, reasoning): (String, String)
        
        if outcome.psi > 0.85 {
            recommendation = "PUBLISH - High confidence"
            reasoning = "Strong evidence, reliable source, easily verifiable"
        } else if outcome.psi > 0.70 {
            recommendation = "REVIEW - Moderate confidence"
            reasoning = "Good evidence but may need additional verification"
        } else if outcome.psi > 0.50 {
            recommendation = "INVESTIGATE - Low confidence"
            reasoning = "Weak evidence or unreliable source - needs fact-checking"
        } else {
            recommendation = "REJECT - Very low confidence"
            reasoning = "Insufficient evidence or unreliable source"
        }
        
        // Calculate overall confidence
        let confidence = ConfidenceHeuristics.overall(
            sources: article.authorityScore,
            hybrid: (outcome.hybrid - 0.3) / 0.7,  // Normalize to [0,1]
            penalty: outcome.penalty,
            posterior: article.factCheckScore
        )
        
        return VerificationResult(
            psi: outcome.psi,
            recommendation: recommendation,
            reasoning: reasoning,
            confidence: confidence
        )
    }
}

// Usage example
func runNewsVerificationExample() {
    let articles = [
        NewsArticle(title: "Climate Study Shows Warming Trend", source: "Science Journal",
                   authorityScore: 0.92, verifiabilityScore: 0.85, factCheckScore: 0.88,
                   hasCanonicalSources: true),
        NewsArticle(title: "Celebrity Spotted at Restaurant", source: "Gossip Blog",
                   authorityScore: 0.35, verifiabilityScore: 0.20, factCheckScore: 0.45,
                   hasCanonicalSources: false),
        NewsArticle(title: "Election Results Updated", source: "Reuters",
                   authorityScore: 0.95, verifiabilityScore: 0.90, factCheckScore: 0.92,
                   hasCanonicalSources: true)
    ]
    
    let service = NewsVerificationService()
    
    print("=== News Article Verification System ===\n")
    
    for article in articles {
        let result = service.evaluateArticle(article)
        
        print("Article: \(article.title)")
        print("Source: \(article.source)")
        print("Authority: \(String(format: "%.2f", article.authorityScore)), " +
              "Verifiability: \(String(format: "%.2f", article.verifiabilityScore)), " +
              "Fact-check: \(String(format: "%.2f", article.factCheckScore))")
        print("Ψ Score: \(String(format: "%.3f", result.psi))")
        print("Confidence: \(String(format: "%.2f", result.confidence))")
        print("Recommendation: \(result.recommendation)")
        print("Reasoning: \(result.reasoning)")
        print()
    }
}
```

---

## Example 2: Scientific Paper Assessment

**Scenario:** A journal editor wants to prioritize peer review assignments based on paper quality indicators.

### Python Implementation

```python
# scientific_paper_assessment.py
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from uncertainty_quantification import QuickStartUQPipeline, RiskBasedDecisionFramework

@dataclass
class ScientificPaper:
    title: str
    authors: List[str]
    institution_rank: float  # 0-1, higher is better
    citation_potential: float  # Predicted citations (ML model output)
    methodology_score: float  # 0-1, peer assessment
    novelty_score: float     # 0-1, automated assessment
    reproducibility_score: float  # 0-1, based on data/code availability
    author_h_index: float    # Average H-index of authors
    has_preprint: bool       # Available as preprint
    data_available: bool     # Data/code publicly available

class PaperAssessmentSystem:
    def __init__(self):
        self.uq_pipeline = QuickStartUQPipeline()
        self.risk_framework = RiskBasedDecisionFramework()
        self.is_trained = False
    
    def train_quality_predictor(self, historical_papers: List[Dict]):
        """Train UQ model on historical paper data."""
        print("Training quality prediction model...")
        
        # Extract features
        X = []
        y = []  # Quality score (e.g., citations after 2 years)
        
        for paper in historical_papers:
            features = [
                paper['institution_rank'],
                paper['methodology_score'],
                paper['novelty_score'],
                paper['reproducibility_score'],
                paper['author_h_index'],
                int(paper['has_preprint']),
                int(paper['data_available'])
            ]
            X.append(features)
            y.append(paper['quality_outcome'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split for training and calibration
        split = int(0.7 * len(X))
        X_train, X_cal = X[:split], X[split:]
        y_train, y_cal = y[:split], y[split:]
        
        # Train UQ pipeline
        self.uq_pipeline.fit(X_train, y_train, X_cal, y_cal)
        self.is_trained = True
        print("Model training completed.")
    
    def assess_paper(self, paper: ScientificPaper) -> Dict:
        """Assess a single paper and return recommendations."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Extract features
        features = np.array([[
            paper.institution_rank,
            paper.methodology_score,
            paper.novelty_score,
            paper.reproducibility_score,
            paper.author_h_index,
            int(paper.has_preprint),
            int(paper.data_available)
        ]])
        
        # Get quality prediction with uncertainty
        results = self.uq_pipeline.predict_with_risk_analysis(features)
        
        prediction = results['predictions'][0]
        uncertainty = results['uncertainty'].std_total[0]
        
        # Risk analysis
        samples = np.random.normal(prediction, uncertainty, 1000)
        var_95 = self.risk_framework.compute_var(samples, 0.05)
        cvar_95 = self.risk_framework.compute_cvar(samples, 0.05)
        
        # Generate recommendations
        if prediction > 0.8 and uncertainty < 0.2:
            priority = "HIGH"
            reviewer_type = "Senior Expert"
            timeline = "Fast Track (2 weeks)"
        elif prediction > 0.6 and uncertainty < 0.3:
            priority = "MEDIUM"
            reviewer_type = "Standard Expert"
            timeline = "Standard (4 weeks)"
        elif prediction > 0.4 or uncertainty > 0.4:
            priority = "LOW"
            reviewer_type = "Junior/Training"
            timeline = "Extended (6 weeks)"
        else:
            priority = "REJECT"
            reviewer_type = "Editorial Decision"
            timeline = "Immediate"
        
        return {
            'paper': paper,
            'quality_prediction': prediction,
            'uncertainty': uncertainty,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'priority': priority,
            'reviewer_type': reviewer_type,
            'timeline': timeline,
            'confidence_interval': results.get('conformal_intervals', (None, None))
        }
    
    def batch_assess_papers(self, papers: List[ScientificPaper]) -> List[Dict]:
        """Assess multiple papers and rank by priority."""
        assessments = [self.assess_paper(paper) for paper in papers]
        
        # Sort by quality prediction (descending)
        assessments.sort(key=lambda x: x['quality_prediction'], reverse=True)
        
        return assessments

# Example usage
def run_paper_assessment_example():
    print("=== Scientific Paper Assessment System ===\n")
    
    # Create assessment system
    system = PaperAssessmentSystem()
    
    # Generate synthetic historical data for training
    np.random.seed(42)
    historical_papers = []
    for i in range(500):
        # Generate realistic paper characteristics
        institution_rank = np.random.beta(2, 3)  # Skewed toward lower ranks
        methodology_score = np.random.beta(3, 2)  # Skewed toward higher scores
        novelty_score = np.random.beta(2, 2)     # Uniform-ish
        reproducibility_score = np.random.beta(1.5, 2)  # Skewed lower
        author_h_index = np.random.gamma(2, 10)   # Right-skewed
        has_preprint = np.random.random() > 0.7   # 30% have preprints
        data_available = np.random.random() > 0.6  # 40% have data
        
        # Quality outcome (citations, impact) - correlated with inputs
        quality_outcome = (
            0.3 * institution_rank +
            0.25 * methodology_score +
            0.2 * novelty_score +
            0.15 * reproducibility_score +
            0.1 * min(author_h_index / 50, 1.0) +
            0.05 * has_preprint +
            0.05 * data_available +
            np.random.normal(0, 0.1)  # Noise
        )
        quality_outcome = max(0, min(1, quality_outcome))  # Clamp to [0,1]
        
        historical_papers.append({
            'institution_rank': institution_rank,
            'methodology_score': methodology_score,
            'novelty_score': novelty_score,
            'reproducibility_score': reproducibility_score,
            'author_h_index': author_h_index,
            'has_preprint': has_preprint,
            'data_available': data_available,
            'quality_outcome': quality_outcome
        })
    
    # Train the system
    system.train_quality_predictor(historical_papers)
    
    # Create sample papers for assessment
    sample_papers = [
        ScientificPaper(
            title="Novel Machine Learning Approach for Climate Modeling",
            authors=["Dr. Smith", "Dr. Johnson"],
            institution_rank=0.85,
            citation_potential=45.2,
            methodology_score=0.90,
            novelty_score=0.80,
            reproducibility_score=0.85,
            author_h_index=25.5,
            has_preprint=True,
            data_available=True
        ),
        ScientificPaper(
            title="Survey of Existing Methods in Data Analysis",
            authors=["Dr. Brown"],
            institution_rank=0.60,
            citation_potential=12.1,
            methodology_score=0.70,
            novelty_score=0.30,
            reproducibility_score=0.60,
            author_h_index=15.2,
            has_preprint=False,
            data_available=False
        ),
        ScientificPaper(
            title="Breakthrough in Quantum Computing Error Correction",
            authors=["Dr. Wilson", "Dr. Davis", "Dr. Miller"],
            institution_rank=0.95,
            citation_potential=78.9,
            methodology_score=0.95,
            novelty_score=0.95,
            reproducibility_score=0.70,
            author_h_index=42.1,
            has_preprint=True,
            data_available=True
        )
    ]
    
    # Assess papers
    assessments = system.batch_assess_papers(sample_papers)
    
    # Display results
    print("Paper Assessment Results (ranked by predicted quality):\n")
    for i, assessment in enumerate(assessments, 1):
        paper = assessment['paper']
        print(f"{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors)}")
        print(f"   Quality Prediction: {assessment['quality_prediction']:.3f} ± {assessment['uncertainty']:.3f}")
        print(f"   Risk Metrics: VaR₉₅={assessment['var_95']:.3f}, CVaR₉₅={assessment['cvar_95']:.3f}")
        print(f"   Priority: {assessment['priority']}")
        print(f"   Reviewer Type: {assessment['reviewer_type']}")
        print(f"   Timeline: {assessment['timeline']}")
        print()
    
    return assessments

if __name__ == "__main__":
    assessments = run_paper_assessment_example()
```

---

## Example 3: Financial Risk Analysis

**Scenario:** A bank wants to assess loan default risk with uncertainty quantification for better capital allocation.

### Implementation

```python
# financial_risk_analysis.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from uncertainty_quantification import (
    DeepEnsemble, ConformalPredictor, RiskBasedDecisionFramework,
    UQEvaluationMetrics
)

@dataclass
class LoanApplication:
    applicant_id: str
    credit_score: int
    annual_income: float
    debt_to_income: float
    loan_amount: float
    employment_years: float
    home_ownership: str  # "own", "rent", "mortgage"
    loan_purpose: str    # "home", "auto", "personal", etc.
    previous_defaults: int
    
class LoanRiskAssessment:
    def __init__(self):
        self.ensemble = None
        self.conformal = None
        self.risk_framework = RiskBasedDecisionFramework()
        self.feature_names = [
            'credit_score', 'annual_income', 'debt_to_income', 
            'loan_amount', 'employment_years', 'home_own', 
            'home_rent', 'purpose_auto', 'purpose_home', 
            'purpose_personal', 'previous_defaults'
        ]
    
    def preprocess_application(self, app: LoanApplication) -> np.ndarray:
        """Convert loan application to feature vector."""
        features = [
            app.credit_score / 850.0,  # Normalize credit score
            np.log1p(app.annual_income) / 15.0,  # Log-normalize income
            app.debt_to_income,
            np.log1p(app.loan_amount) / 15.0,  # Log-normalize loan amount
            min(app.employment_years, 20) / 20.0,  # Cap and normalize
            1.0 if app.home_ownership == "own" else 0.0,
            1.0 if app.home_ownership == "rent" else 0.0,
            1.0 if app.loan_purpose == "auto" else 0.0,
            1.0 if app.loan_purpose == "home" else 0.0,
            1.0 if app.loan_purpose == "personal" else 0.0,
            min(app.previous_defaults, 5) / 5.0  # Cap and normalize
        ]
        return np.array(features)
    
    def train_model(self, applications: List[LoanApplication], 
                   outcomes: List[bool]) -> None:
        """Train the risk assessment model."""
        print("Training loan risk assessment model...")
        
        # Preprocess applications
        X = np.array([self.preprocess_application(app) for app in applications])
        y = np.array([1.0 if defaulted else 0.0 for defaulted in outcomes])
        
        # Split for training and calibration
        split = int(0.7 * len(X))
        X_train, X_cal = X[:split], X[split:]
        y_train, y_cal = y[:split], y[split:]
        
        # Train deep ensemble for uncertainty quantification
        from sklearn.ensemble import RandomForestClassifier
        self.ensemble = DeepEnsemble(
            RandomForestClassifier, 
            n_models=10, 
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        self.ensemble.fit(X_train, y_train)
        
        # Add conformal prediction for calibrated probabilities
        class EnsembleWrapper:
            def __init__(self, ensemble):
                self.ensemble = ensemble
            def predict(self, X):
                return self.ensemble.predict_with_uncertainty(X).mean
        
        wrapper = EnsembleWrapper(self.ensemble)
        self.conformal = ConformalPredictor(wrapper, alpha=0.1)
        self.conformal.fit(X_cal, y_cal)
        
        print("Model training completed.")
    
    def assess_default_risk(self, application: LoanApplication) -> Dict:
        """Assess default risk for a single application."""
        if self.ensemble is None:
            raise ValueError("Model must be trained first")
        
        # Preprocess application
        features = self.preprocess_application(application).reshape(1, -1)
        
        # Get uncertainty estimate
        uncertainty_est = self.ensemble.predict_with_uncertainty(features)
        default_prob = uncertainty_est.mean[0]
        epistemic_uncertainty = np.sqrt(uncertainty_est.epistemic[0])
        
        # Get conformal prediction intervals
        conf_lower, conf_upper = self.conformal.predict_intervals(features)
        
        # Risk analysis - sample from predictive distribution
        samples = np.random.normal(default_prob, epistemic_uncertainty, 1000)
        samples = np.clip(samples, 0, 1)  # Clamp to valid probability range
        
        # Calculate risk metrics
        var_95 = self.risk_framework.compute_var(samples, 0.05)
        cvar_95 = self.risk_framework.compute_cvar(samples, 0.05)
        tail_prob_80 = self.risk_framework.tail_probability(samples, 0.8)
        
        # Generate loan decision
        decision, interest_rate = self._make_loan_decision(
            default_prob, epistemic_uncertainty, application.loan_amount
        )
        
        return {
            'applicant_id': application.applicant_id,
            'default_probability': default_prob,
            'epistemic_uncertainty': epistemic_uncertainty,
            'confidence_interval': (conf_lower[0], conf_upper[0]),
            'var_95': var_95,
            'cvar_95': cvar_95,
            'tail_prob_80': tail_prob_80,
            'decision': decision,
            'interest_rate': interest_rate,
            'expected_loss': default_prob * application.loan_amount,
            'risk_adjusted_return': self._calculate_risk_adjusted_return(
                default_prob, interest_rate, application.loan_amount
            )
        }
    
    def _make_loan_decision(self, default_prob: float, uncertainty: float, 
                          loan_amount: float) -> Tuple[str, float]:
        """Make loan approval decision based on risk assessment."""
        base_rate = 0.05  # Base interest rate
        
        if default_prob < 0.1 and uncertainty < 0.05:
            return "APPROVE", base_rate + 0.02  # Low risk premium
        elif default_prob < 0.2 and uncertainty < 0.1:
            return "APPROVE", base_rate + 0.05  # Medium risk premium
        elif default_prob < 0.3 and uncertainty < 0.15:
            return "APPROVE", base_rate + 0.10  # High risk premium
        elif default_prob < 0.4:
            return "MANUAL_REVIEW", base_rate + 0.15  # Requires human review
        else:
            return "DENY", 0.0  # Too risky
    
    def _calculate_risk_adjusted_return(self, default_prob: float, 
                                      interest_rate: float, 
                                      loan_amount: float) -> float:
        """Calculate expected risk-adjusted return."""
        expected_interest = loan_amount * interest_rate
        expected_loss = loan_amount * default_prob
        return expected_interest - expected_loss

# Example usage
def run_financial_risk_example():
    print("=== Financial Risk Analysis System ===\n")
    
    # Create risk assessment system
    risk_system = LoanRiskAssessment()
    
    # Generate synthetic training data
    np.random.seed(42)
    training_applications = []
    training_outcomes = []
    
    for i in range(1000):
        # Generate realistic loan application
        credit_score = int(np.random.normal(700, 100))
        credit_score = max(300, min(850, credit_score))
        
        annual_income = np.random.lognormal(11, 0.5)  # ~$60k median
        debt_to_income = np.random.beta(2, 5) * 0.8   # Skewed toward lower ratios
        loan_amount = np.random.lognormal(10, 0.8)    # ~$22k median
        employment_years = np.random.exponential(5)
        
        home_ownership = np.random.choice(["own", "rent", "mortgage"], 
                                        p=[0.3, 0.4, 0.3])
        loan_purpose = np.random.choice(["home", "auto", "personal"], 
                                      p=[0.4, 0.3, 0.3])
        previous_defaults = np.random.poisson(0.2)
        
        app = LoanApplication(
            applicant_id=f"app_{i}",
            credit_score=credit_score,
            annual_income=annual_income,
            debt_to_income=debt_to_income,
            loan_amount=loan_amount,
            employment_years=employment_years,
            home_ownership=home_ownership,
            loan_purpose=loan_purpose,
            previous_defaults=previous_defaults
        )
        
        # Simulate default outcome (correlated with risk factors)
        default_prob = (
            0.5 * (1 - credit_score / 850) +
            0.2 * min(debt_to_income, 1.0) +
            0.1 * (1 - min(annual_income / 100000, 1.0)) +
            0.1 * min(previous_defaults / 3, 1.0) +
            0.1 * np.random.random()  # Random component
        )
        defaulted = np.random.random() < default_prob
        
        training_applications.append(app)
        training_outcomes.append(defaulted)
    
    # Train the model
    risk_system.train_model(training_applications, training_outcomes)
    
    # Create test applications
    test_applications = [
        LoanApplication(
            applicant_id="test_001",
            credit_score=750,
            annual_income=80000,
            debt_to_income=0.25,
            loan_amount=25000,
            employment_years=5.0,
            home_ownership="own",
            loan_purpose="home",
            previous_defaults=0
        ),
        LoanApplication(
            applicant_id="test_002",
            credit_score=620,
            annual_income=45000,
            debt_to_income=0.45,
            loan_amount=15000,
            employment_years=2.0,
            home_ownership="rent",
            loan_purpose="personal",
            previous_defaults=1
        ),
        LoanApplication(
            applicant_id="test_003",
            credit_score=580,
            annual_income=35000,
            debt_to_income=0.60,
            loan_amount=30000,
            employment_years=1.0,
            home_ownership="rent",
            loan_purpose="auto",
            previous_defaults=2
        )
    ]
    
    # Assess applications
    print("Loan Risk Assessment Results:\n")
    
    for app in test_applications:
        assessment = risk_system.assess_default_risk(app)
        
        print(f"Applicant: {assessment['applicant_id']}")
        print(f"Credit Score: {app.credit_score}")
        print(f"Annual Income: ${app.annual_income:,.0f}")
        print(f"Debt-to-Income: {app.debt_to_income:.2%}")
        print(f"Loan Amount: ${app.loan_amount:,.0f}")
        print(f"Default Probability: {assessment['default_probability']:.3f} ± {assessment['epistemic_uncertainty']:.3f}")
        print(f"95% Confidence Interval: [{assessment['confidence_interval'][0]:.3f}, {assessment['confidence_interval'][1]:.3f}]")
        print(f"Risk Metrics: VaR₉₅={assessment['var_95']:.3f}, CVaR₉₅={assessment['cvar_95']:.3f}")
        print(f"P(default > 80%): {assessment['tail_prob_80']:.4f}")
        print(f"Decision: {assessment['decision']}")
        if assessment['interest_rate'] > 0:
            print(f"Interest Rate: {assessment['interest_rate']:.2%}")
        print(f"Expected Loss: ${assessment['expected_loss']:,.0f}")
        print(f"Risk-Adjusted Return: ${assessment['risk_adjusted_return']:,.0f}")
        print()
    
    return [risk_system.assess_default_risk(app) for app in test_applications]

if __name__ == "__main__":
    assessments = run_financial_risk_example()
```

---

## Example 4: Medical Diagnosis Support

**Scenario:** A medical AI system assists doctors by providing diagnostic suggestions with uncertainty estimates.

### Implementation

```python
# medical_diagnosis_support.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from uncertainty_quantification import (
    DeepEnsemble, MCDropout, ConformalPredictor, 
    CalibrationMethods, UQEvaluationMetrics
)

@dataclass
class PatientCase:
    patient_id: str
    age: int
    gender: str  # "M" or "F"
    symptoms: List[str]
    vital_signs: Dict[str, float]  # {"temperature": 38.5, "bp_systolic": 120, ...}
    lab_results: Dict[str, float]  # {"wbc": 8.5, "glucose": 95, ...}
    medical_history: List[str]
    current_medications: List[str]
    
@dataclass
class DiagnosticSuggestion:
    condition: str
    probability: float
    confidence_interval: tuple
    epistemic_uncertainty: float
    evidence_strength: str
    recommended_tests: List[str]
    urgency_level: str

class MedicalDiagnosisAI:
    def __init__(self):
        self.ensemble = None
        self.conformal = None
        self.condition_names = [
            "Common Cold", "Influenza", "Pneumonia", "COVID-19",
            "Hypertension", "Diabetes", "Heart Disease", "Anxiety",
            "Depression", "Migraine", "Other/Unknown"
        ]
        
    def preprocess_patient_case(self, case: PatientCase) -> np.ndarray:
        """Convert patient case to feature vector."""
        features = []
        
        # Demographics
        features.append(case.age / 100.0)  # Normalize age
        features.append(1.0 if case.gender == "M" else 0.0)
        
        # Symptom encoding (one-hot for common symptoms)
        common_symptoms = [
            "fever", "cough", "fatigue", "headache", "nausea",
            "chest_pain", "shortness_of_breath", "dizziness"
        ]
        for symptom in common_symptoms:
            features.append(1.0 if symptom in case.symptoms else 0.0)
        
        # Vital signs (normalized)
        features.append(case.vital_signs.get("temperature", 37.0) / 42.0)
        features.append(case.vital_signs.get("bp_systolic", 120) / 200.0)
        features.append(case.vital_signs.get("bp_diastolic", 80) / 120.0)
        features.append(case.vital_signs.get("heart_rate", 70) / 150.0)
        
        # Lab results (normalized)
        features.append(case.lab_results.get("wbc", 7.0) / 15.0)
        features.append(case.lab_results.get("glucose", 100) / 300.0)
        features.append(case.lab_results.get("creatinine", 1.0) / 5.0)
        
        # Medical history encoding
        chronic_conditions = [
            "diabetes", "hypertension", "heart_disease", "asthma"
        ]
        for condition in chronic_conditions:
            features.append(1.0 if condition in case.medical_history else 0.0)
        
        return np.array(features)
    
    def train_diagnostic_model(self, training_cases: List[PatientCase], 
                             diagnoses: List[str]) -> None:
        """Train the diagnostic model with uncertainty quantification."""
        print("Training medical diagnosis model...")
        
        # Preprocess cases
        X = np.array([self.preprocess_patient_case(case) for case in training_cases])
        
        # Encode diagnoses
        y = np.array([self.condition_names.index(diagnosis) 
                     if diagnosis in self.condition_names 
                     else len(self.condition_names)-1 
                     for diagnosis in diagnoses])
        
        # Split for training and calibration
        split = int(0.7 * len(X))
        X_train, X_cal = X[:split], X[split:]
        y_train, y_cal = y[:split], y[split:]
        
        # Train ensemble for epistemic uncertainty
        from sklearn.ensemble import RandomForestClassifier
        self.ensemble = DeepEnsemble(
            RandomForestClassifier,
            n_models=15,  # More models for medical applications
            n_estimators=200,
            max_depth=12,
            random_state=42
        )
        self.ensemble.fit(X_train, y_train)
        
        # Add conformal prediction for calibrated confidence
        class EnsembleWrapper:
            def __init__(self, ensemble):
                self.ensemble = ensemble
            def predict(self, X):
                return self.ensemble.predict_with_uncertainty(X).mean
        
        wrapper = EnsembleWrapper(self.ensemble)
        self.conformal = ConformalPredictor(wrapper, alpha=0.2)  # 80% confidence
        self.conformal.fit(X_cal, y_cal)
        
        print("Medical diagnosis model training completed.")
    
    def diagnose_patient(self, case: PatientCase) -> List[DiagnosticSuggestion]:
        """Provide diagnostic suggestions with uncertainty estimates."""
        if self.ensemble is None:
            raise ValueError("Model must be trained first")
        
        # Preprocess case
        features = self.preprocess_patient_case(case).reshape(1, -1)
        
        # Get predictions from ensemble
        uncertainty_est = self.ensemble.predict_with_uncertainty(features)
        
        # For classification, we need to handle the output differently
        # Get predictions from all models in ensemble
        all_predictions = []
        for model in self.ensemble.models:
            pred_proba = model.predict_proba(features)[0]
            all_predictions.append(pred_proba)
        
        all_predictions = np.array(all_predictions)
        mean_proba = np.mean(all_predictions, axis=0)
        epistemic_uncertainty = np.std(all_predictions, axis=0)
        
        # Get conformal prediction sets
        conf_lower, conf_upper = self.conformal.predict_intervals(features)
        
        # Generate suggestions for top conditions
        suggestions = []
        top_indices = np.argsort(mean_proba)[::-1][:5]  # Top 5 conditions
        
        for idx in top_indices:
            if mean_proba[idx] > 0.05:  # Only suggest if probability > 5%
                condition = self.condition_names[idx]
                probability = mean_proba[idx]
                epistemic_unc = epistemic_uncertainty[idx]
                
                # Determine evidence strength
                if epistemic_unc < 0.1:
                    evidence_strength = "Strong"
                elif epistemic_unc < 0.2:
                    evidence_strength = "Moderate"
                else:
                    evidence_strength = "Weak"
                
                # Recommend additional tests based on condition and uncertainty
                recommended_tests = self._get_recommended_tests(condition, epistemic_unc)
                
                # Determine urgency
                urgency = self._assess_urgency(condition, probability, case)
                
                suggestion = DiagnosticSuggestion(
                    condition=condition,
                    probability=probability,
                    confidence_interval=(
                        max(0, probability - 1.96 * epistemic_unc),
                        min(1, probability + 1.96 * epistemic_unc)
                    ),
                    epistemic_uncertainty=epistemic_unc,
                    evidence_strength=evidence_strength,
                    recommended_tests=recommended_tests,
                    urgency_level=urgency
                )
                
                suggestions.append(suggestion)
        
        return suggestions
    
    def _get_recommended_tests(self, condition: str, uncertainty: float) -> List[str]:
        """Recommend additional tests based on condition and uncertainty."""
        test_recommendations = {
            "Pneumonia": ["Chest X-ray", "Blood culture", "Sputum culture"],
            "COVID-19": ["PCR test", "Antigen test", "Chest CT"],
            "Heart Disease": ["ECG", "Echocardiogram", "Stress test"],
            "Diabetes": ["HbA1c", "Glucose tolerance test", "C-peptide"],
            "Hypertension": ["24-hour BP monitor", "Echocardiogram", "Kidney function tests"]
        }
        
        base_tests = test_recommendations.get(condition, ["Complete blood panel"])
        
        # Add more tests if uncertainty is high
        if uncertainty > 0.2:
            base_tests.extend(["Additional specialist consultation", "Extended monitoring"])
        
        return base_tests
    
    def _assess_urgency(self, condition: str, probability: float, case: PatientCase) -> str:
        """Assess urgency level based on condition and patient factors."""
        high_urgency_conditions = ["Pneumonia", "Heart Disease", "COVID-19"]
        
        # High fever or concerning vital signs
        high_fever = case.vital_signs.get("temperature", 37.0) > 39.0
        high_bp = case.vital_signs.get("bp_systolic", 120) > 160
        
        if condition in high_urgency_conditions and probability > 0.3:
            return "HIGH"
        elif high_fever or high_bp or probability > 0.5:
            return "MEDIUM"
        else:
            return "LOW"

# Example usage
def run_medical_diagnosis_example():
    print("=== Medical Diagnosis Support System ===\n")
    
    # Create diagnosis system
    diagnosis_ai = MedicalDiagnosisAI()
    
    # Generate synthetic training data
    np.random.seed(42)
    training_cases = []
    training_diagnoses = []
    
    for i in range(2000):
        # Generate realistic patient case
        age = int(np.random.normal(50, 20))
        age = max(1, min(100, age))
        gender = np.random.choice(["M", "F"])
        
        # Generate symptoms based on condition
        condition = np.random.choice(diagnosis_ai.condition_names[:-1])  # Exclude "Other"
        
        if condition == "Common Cold":
            symptoms = ["cough", "fatigue"] + np.random.choice(
                ["headache", "nausea"], size=np.random.randint(0, 2), replace=False
            ).tolist()
            temp = np.random.normal(37.5, 0.5)
        elif condition == "Pneumonia":
            symptoms = ["fever", "cough", "chest_pain", "shortness_of_breath"]
            temp = np.random.normal(39.0, 1.0)
        else:  # Default case
            symptoms = np.random.choice(
                ["fever", "cough", "fatigue", "headache"], 
                size=np.random.randint(1, 3), replace=False
            ).tolist()
            temp = np.random.normal(37.8, 0.8)
        
        case = PatientCase(
            patient_id=f"patient_{i}",
            age=age,
            gender=gender,
            symptoms=symptoms,
            vital_signs={
                "temperature": max(36.0, min(42.0, temp)),
                "bp_systolic": int(np.random.normal(120, 20)),
                "bp_diastolic": int(np.random.normal(80, 10)),
                "heart_rate": int(np.random.normal(70, 15))
            },
            lab_results={
                "wbc": max(3.0, np.random.normal(7.0, 2.0)),
                "glucose": max(70, np.random.normal(100, 20)),
                "creatinine": max(0.5, np.random.normal(1.0, 0.3))
            },
            medical_history=np.random.choice(
                ["diabetes", "hypertension"], size=np.random.randint(0, 2), replace=False
            ).tolist(),
            current_medications=[]
        )
        
        training_cases.append(case)
        training_diagnoses.append(condition)
    
    # Train the model
    diagnosis_ai.train_diagnostic_model(training_cases, training_diagnoses)
    
    # Create test cases
    test_cases = [
        PatientCase(
            patient_id="test_001",
            age=45,
            gender="M",
            symptoms=["fever", "cough", "chest_pain", "shortness_of_breath"],
            vital_signs={
                "temperature": 39.2,
                "bp_systolic": 130,
                "bp_diastolic": 85,
                "heart_rate": 95
            },
            lab_results={
                "wbc": 12.5,
                "glucose": 110,
                "creatinine": 1.1
            },
            medical_history=["hypertension"],
            current_medications=["lisinopril"]
        ),
        PatientCase(
            patient_id="test_002",
            age=28,
            gender="F",
            symptoms=["cough", "fatigue", "headache"],
            vital_signs={
                "temperature": 37.8,
                "bp_systolic": 115,
                "bp_diastolic": 75,
                "heart_rate": 80
            },
            lab_results={
                "wbc": 6.8,
                "glucose": 95,
                "creatinine": 0.9
            },
            medical_history=[],
            current_medications=[]
        )
    ]
    
    # Generate diagnostic suggestions
    print("Medical Diagnostic Suggestions:\n")
    
    for case in test_cases:
        print(f"Patient: {case.patient_id}")
        print(f"Age: {case.age}, Gender: {case.gender}")
        print(f"Symptoms: {', '.join(case.symptoms)}")
        print(f"Temperature: {case.vital_signs['temperature']:.1f}°C")
        print(f"Blood Pressure: {case.vital_signs['bp_systolic']}/{case.vital_signs['bp_diastolic']} mmHg")
        
        suggestions = diagnosis_ai.diagnose_patient(case)
        
        print("\nDiagnostic Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion.condition}")
            print(f"   Probability: {suggestion.probability:.3f} "
                  f"[{suggestion.confidence_interval[0]:.3f}, {suggestion.confidence_interval[1]:.3f}]")
            print(f"   Uncertainty: ±{suggestion.epistemic_uncertainty:.3f}")
            print(f"   Evidence Strength: {suggestion.evidence_strength}")
            print(f"   Urgency: {suggestion.urgency_level}")
            print(f"   Recommended Tests: {', '.join(suggestion.recommended_tests)}")
        
        print("\n" + "="*60 + "\n")
    
    return test_cases

if __name__ == "__main__":
    test_cases = run_medical_diagnosis_example()
```

These examples demonstrate practical applications of the Ψ framework and uncertainty quantification tools across various domains. Each example shows how to:

1. **Model domain-specific problems** using the appropriate framework
2. **Quantify different types of uncertainty** (epistemic, aleatoric)
3. **Make risk-aware decisions** based on uncertainty estimates
4. **Provide actionable recommendations** with confidence bounds
5. **Handle real-world constraints** and requirements

The examples can be extended and customized for specific use cases, and they demonstrate the flexibility and power of combining the Ψ framework with modern uncertainty quantification techniques.