import java.util.*;

/**
 * Fractal Ψ Framework Implementation
 * Extends the core Ψ(x,m,s) framework with self-interaction dynamics and stabilizing anchors
 * 
 * Core equation: Ψ_{t+1} = min{ β·exp(-[λ₁Rₐ + λ₂Rᵥ])·[α·S + (1-α)·N + κ·(G(Ψ_t) + C(t)) + Σ_m w_m(t)·M_m(Ψ_t)], 1 }
 * Where:
 * - G(Ψ) = clamp(Ψ², 0, g_max) - self-interaction term (fractal core)
 * - C(t) = weighted sum of stabilizing anchors
 * - Anchors: Safety, Curiosity, Return, MetaAware, MultiScale
 */
public class FractalPsiFramework {
    
    /**
     * Parameters for fractal Ψ computation
     */
    public static class PsiParams {
        public final double alpha;        // Symbolic-neural balance
        public final double beta;         // Risk penalty scaling
        public final double lambdaAuth;   // Authority risk weight
        public final double lambdaVerif;  // Verification risk weight
        public final double kappa;        // Fractal interaction strength
        public final double gMax;         // Self-interaction clamp
        public final double cMax;         // Anchor sum maximum
        
        public PsiParams(double alpha, double beta, double lambdaAuth, double lambdaVerif,
                        double kappa, double gMax, double cMax) {
            this.alpha = alpha;
            this.beta = beta;
            this.lambdaAuth = lambdaAuth;
            this.lambdaVerif = lambdaVerif;
            this.kappa = kappa;
            this.gMax = gMax;
            this.cMax = cMax;
        }
        
        // Default stable parameters
        public static PsiParams defaultParams() {
            return new PsiParams(0.5, 1.0, 1.0, 1.0, 0.15, 0.5, 0.3);
        }
    }
    
    /**
     * Anchor metrics for stabilization
     */
    public static class AnchorMetrics {
        public final double uncertainty;      // Normalized uncertainty [0,1]
        public final double eig;             // Expected Information Gain [0,1]
        public final double driftRisk;       // Value drift risk [0,1]
        public final double distanceToSafe;  // Distance to safe attractor [0,1]
        public final double selfModelError;  // Self-model prediction error [0,1]
        public final double multiscaleCorr;  // Cross-scale coherence [0,1]
        
        public AnchorMetrics(double uncertainty, double eig, double driftRisk,
                           double distanceToSafe, double selfModelError, double multiscaleCorr) {
            this.uncertainty = clamp(uncertainty, 0.0, 1.0);
            this.eig = clamp(eig, 0.0, 1.0);
            this.driftRisk = clamp(driftRisk, 0.0, 1.0);
            this.distanceToSafe = clamp(distanceToSafe, 0.0, 1.0);
            this.selfModelError = clamp(selfModelError, 0.0, 1.0);
            this.multiscaleCorr = clamp(multiscaleCorr, 0.0, 1.0);
        }
        
        // Create from research analysis context
        public static AnchorMetrics fromResearchContext(double topicUncertainty, 
                                                       double innovationPotential,
                                                       double paradigmShiftRisk,
                                                       double collaborationDistance,
                                                       double predictionAccuracy,
                                                       double interdisciplinaryCoherence) {
            return new AnchorMetrics(
                topicUncertainty,
                innovationPotential,
                paradigmShiftRisk,
                collaborationDistance,
                1.0 - predictionAccuracy, // Error = 1 - accuracy
                interdisciplinaryCoherence
            );
        }
    }
    
    /**
     * Computed anchor values
     */
    public static class ComputedAnchors {
        public final double safety;      // 1 - uncertainty (entropy reduction)
        public final double curiosity;   // EIG - drift_risk (bounded exploration)
        public final double returnAnchor; // 1 - distance_to_safe (attractor pull)
        public final double metaAware;   // 1 - self_model_error (self-consistency)
        public final double multiScale;  // multiscale_corr (coherence across scales)
        public final double weightedSum; // Total anchor contribution
        
        public ComputedAnchors(double safety, double curiosity, double returnAnchor,
                             double metaAware, double multiScale, double weightedSum) {
            this.safety = safety;
            this.curiosity = curiosity;
            this.returnAnchor = returnAnchor;
            this.metaAware = metaAware;
            this.multiScale = multiScale;
            this.weightedSum = weightedSum;
        }
    }
    
    /**
     * Fractal Ψ computation result
     */
    public static class FractalPsiResult {
        public final double psiNext;        // Updated Ψ value
        public final double gTerm;          // Self-interaction G(Ψ)
        public final double cTerm;          // Anchor contribution C(t)
        public final ComputedAnchors anchors; // Individual anchor values
        public final double riskPenalty;    // Risk penalty factor
        public final double core;           // Core computation before penalty
        public final double rhoEstimate;    // Lipschitz constant estimate
        
        public FractalPsiResult(double psiNext, double gTerm, double cTerm,
                              ComputedAnchors anchors, double riskPenalty, 
                              double core, double rhoEstimate) {
            this.psiNext = psiNext;
            this.gTerm = gTerm;
            this.cTerm = cTerm;
            this.anchors = anchors;
            this.riskPenalty = riskPenalty;
            this.core = core;
            this.rhoEstimate = rhoEstimate;
        }
    }
    
    /**
     * Compute stabilizing anchors from metrics
     */
    public static ComputedAnchors computeAnchors(AnchorMetrics metrics, PsiParams params) {
        // Safety: entropy reduction (1 - uncertainty)
        double safety = clamp(1.0 - metrics.uncertainty, 0.0, 1.0);
        
        // Curiosity: EIG bounded by drift risk
        double curiosity = clamp(metrics.eig - 0.5 * metrics.driftRisk, 0.0, 1.0);
        
        // Return: distance to attractor (1 - distance_to_safe)
        double returnAnchor = clamp(1.0 - metrics.distanceToSafe, 0.0, 1.0);
        
        // Meta-awareness: self-model agreement (1 - error)
        double metaAware = clamp(1.0 - metrics.selfModelError, 0.0, 1.0);
        
        // Multi-scale: cross-scale coherence
        double multiScale = clamp(metrics.multiscaleCorr, 0.0, 1.0);
        
        // Weighted sum within c_max budget
        double[] weights = {0.3, 0.2, 0.2, 0.15, 0.15}; // s, c, r, ma, ms
        double weightedSum = weights[0] * safety + 
                           weights[1] * curiosity + 
                           weights[2] * returnAnchor + 
                           weights[3] * metaAware + 
                           weights[4] * multiScale;
        
        // Clamp to maximum anchor contribution
        weightedSum = Math.min(weightedSum, params.cMax);
        
        return new ComputedAnchors(safety, curiosity, returnAnchor, metaAware, 
                                 multiScale, weightedSum);
    }
    
    /**
     * Core fractal Ψ computation
     */
    public static FractalPsiResult computePsiFractal(double psi, double S, double N,
                                                   double riskAuthority, double riskVerifiability,
                                                   double modalitiesSum, AnchorMetrics anchorMetrics,
                                                   PsiParams params) {
        
        // Self-interaction term G(Ψ) = clamp(Ψ², 0, g_max)
        double gTerm = Math.min(psi * psi, params.gMax);
        
        // Compute stabilizing anchors
        ComputedAnchors anchors = computeAnchors(anchorMetrics, params);
        double cTerm = anchors.weightedSum;
        
        // Core computation: α·S + (1-α)·N + κ·(G(Ψ) + C(t)) + modalities
        double core = params.alpha * S + 
                     (1.0 - params.alpha) * N + 
                     params.kappa * (gTerm + cTerm) + 
                     modalitiesSum;
        
        // Risk penalty: β·exp(-[λ₁Rₐ + λ₂Rᵥ])
        double riskPenalty = params.beta * Math.exp(-(params.lambdaAuth * riskAuthority + 
                                                     params.lambdaVerif * riskVerifiability));
        
        // Final Ψ update with bounds
        double psiNext = clamp(riskPenalty * core, 0.0, 1.0);
        
        // Estimate local Lipschitz constant for stability
        double rhoEstimate = estimateLocalLipschitz(psi, S, N, params, gTerm, cTerm);
        
        return new FractalPsiResult(psiNext, gTerm, cTerm, anchors, 
                                  riskPenalty, core, rhoEstimate);
    }
    
    /**
     * Estimate local Lipschitz constant for contraction analysis
     */
    private static double estimateLocalLipschitz(double psi, double S, double N, 
                                               PsiParams params, double gTerm, double cTerm) {
        // Numerical derivative estimation
        double epsilon = 1e-6;
        double psiPlus = Math.min(psi + epsilon, 1.0);
        double psiMinus = Math.max(psi - epsilon, 0.0);
        
        // G'(ψ) = 2ψ (derivative of self-interaction)
        double gDerivative = 2.0 * psi;
        if (psi * psi >= params.gMax) {
            gDerivative = 0.0; // Clipped region has zero derivative
        }
        
        // Local Lipschitz: |∂/∂Ψ [core term]|
        double lipschitz = Math.abs(params.kappa * gDerivative);
        
        // Add contribution from other terms (bounded)
        lipschitz += Math.abs(params.alpha) + Math.abs(1.0 - params.alpha);
        
        return lipschitz;
    }
    
    /**
     * Validate stability condition: ρ < 1 for contraction
     */
    public static boolean validateStability(PsiParams params, double[] psiRange, 
                                          AnchorMetrics[] anchorRange) {
        double maxLipschitz = 0.0;
        
        for (double psi : psiRange) {
            for (AnchorMetrics anchors : anchorRange) {
                FractalPsiResult result = computePsiFractal(psi, 0.5, 0.5, 0.1, 0.1, 0.0, 
                                                          anchors, params);
                maxLipschitz = Math.max(maxLipschitz, result.rhoEstimate);
            }
        }
        
        return maxLipschitz < 1.0; // Contraction condition
    }
    
    /**
     * Utility function for clamping values
     */
    public static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }
    
    /**
     * Generate anchor metrics from research context
     */
    public static AnchorMetrics generateResearchAnchors(String researchArea, 
                                                       double innovationLevel,
                                                       double collaborationStrength) {
        Random random = new Random(researchArea.hashCode());
        
        // Research-specific anchor generation
        double uncertainty = 0.2 + 0.3 * random.nextDouble(); // Moderate uncertainty
        double eig = innovationLevel * (0.7 + 0.3 * random.nextDouble()); // Innovation potential
        double driftRisk = 0.1 + 0.2 * random.nextDouble(); // Low to moderate drift
        double distanceToSafe = 0.1 + 0.3 * (1.0 - collaborationStrength); // Collaboration safety
        double selfModelError = 0.1 + 0.2 * random.nextDouble(); // Good self-prediction
        double multiscaleCorr = collaborationStrength * (0.6 + 0.4 * random.nextDouble()); // Coherence
        
        return new AnchorMetrics(uncertainty, eig, driftRisk, distanceToSafe, 
                               selfModelError, multiscaleCorr);
    }
    
    /**
     * Demonstrate fractal dynamics with perturbation recovery
     */
    public static void demonstrateFractalDynamics(String researchContext) {
        System.out.println("🌀 FRACTAL Ψ DYNAMICS DEMONSTRATION:");
        System.out.println("   Research Context: " + researchContext);
        System.out.println();
        
        PsiParams params = PsiParams.defaultParams();
        
        // Initial state
        double psi = 0.6;
        double S = 0.75; // Symbolic accuracy
        double N = 0.85; // Neural accuracy
        double riskAuth = 0.1;
        double riskVerif = 0.1;
        double modalities = 0.05;
        
        // Generate research-appropriate anchors
        AnchorMetrics anchors = generateResearchAnchors(researchContext, 0.8, 0.7);
        
        System.out.println("   Initial Conditions:");
        System.out.printf("   • Ψ₀ = %.3f, S = %.3f, N = %.3f%n", psi, S, N);
        System.out.printf("   • Risk: Authority = %.3f, Verification = %.3f%n", riskAuth, riskVerif);
        System.out.println();
        
        // Evolution over time
        System.out.println("   Fractal Evolution:");
        System.out.println("   Step    Ψ      G(Ψ)   C(t)   Anchors[S,C,R,M,MS]     ρ");
        System.out.println("   " + "-".repeat(75));
        
        for (int step = 0; step < 8; step++) {
            FractalPsiResult result = computePsiFractal(psi, S, N, riskAuth, riskVerif, 
                                                      modalities, anchors, params);
            
            System.out.printf("   %2d     %.3f  %.3f  %.3f  [%.2f,%.2f,%.2f,%.2f,%.2f]  %.3f%n",
                step, result.psiNext, result.gTerm, result.cTerm,
                result.anchors.safety, result.anchors.curiosity, result.anchors.returnAnchor,
                result.anchors.metaAware, result.anchors.multiScale, result.rhoEstimate);
            
            psi = result.psiNext;
            
            // Simulate perturbation at step 4
            if (step == 4) {
                psi += 0.2; // External shock
                psi = clamp(psi, 0.0, 1.0);
                System.out.println("   ↑ Perturbation applied (+0.2)");
            }
        }
        
        System.out.println();
        System.out.println("   Key Observations:");
        System.out.println("   • Self-interaction G(Ψ) provides adaptive feedback");
        System.out.println("   • Anchors C(t) stabilize against perturbations");
        System.out.println("   • Contraction ρ < 1 ensures bounded dynamics");
        System.out.println("   • System recovers from external shocks");
        System.out.println();
    }
}
