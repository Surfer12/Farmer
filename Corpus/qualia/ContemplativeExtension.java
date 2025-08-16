// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

import java.util.List;
import java.util.Optional;

/**
 * Extension interface for integrating contemplative AI visual grounding
 * with the existing Ψ framework. Supports Vipassanā-inspired arising/passing
 * awareness in AI visual processing systems.
 *
 * <p>Implements the contemplative Ψ equation:
 * Ψ_contemplative(x) = min{β_c·exp(-[λ_temporal·R_impermanence + λ_attention·R_distraction])·[α_visual·V + (1-α_visual)·C], 1}
 *
 * <p>Where:
 * - V: Visual grounding confidence from temporal attention mechanisms
 * - C: Canonical/contextual evidence from traditional sources
 * - R_impermanence: Risk from failing to track arising/passing phenomena
 * - R_distraction: Risk from attention drift or conceptual fixation
 * - β_c: Contemplative uplift factor for meditative insights
 * - α_visual: Evidence allocation between visual and canonical sources
 * - λ_temporal, λ_attention: Risk penalty weights for contemplative factors
 */
public interface ContemplativeExtension extends PsiModel {
    
    /**
     * Calculates contemplative Ψ score incorporating visual grounding
     * and temporal awareness of arising/passing phenomena.
     *
     * @param visualData visual sequence data for arising/passing detection
     * @param contemplativeParams parameters specific to contemplative processing
     * @param baseParams standard model parameters
     * @return contemplative Ψ score in [0,1]
     */
    double calculateContemplativePsi(VisualSequenceData visualData, 
                                   ContemplativeParameters contemplativeParams,
                                   ModelParameters baseParams);
    
    /**
     * Detects arising and passing phenomena in visual sequences.
     * Core method for implementing stage-four insight (udayabbaya ñāṇa).
     *
     * @param sequence temporal visual data
     * @param attentionFocus current attention region
     * @return arising/passing events with confidence scores
     */
    List<ArisingPassingEvent> detectArisingPassing(VisualSequence sequence, 
                                                   AttentionRegion attentionFocus);
    
    /**
     * Calculates visual grounding confidence (V parameter).
     * Integrates temporal attention with selective focus mechanisms.
     *
     * @param visualData input visual sequence
     * @param contemplativeState current contemplative processing state
     * @return visual grounding confidence in [0,1]
     */
    double calculateVisualGroundingConfidence(VisualSequenceData visualData,
                                            ContemplativeState contemplativeState);
    
    /**
     * Assesses impermanence risk (R_impermanence).
     * Higher risk when failing to track transient phenomena.
     *
     * @param detectedEvents arising/passing events from visual analysis
     * @param expectedTransience baseline expectation for change
     * @return impermanence risk ≥ 0
     */
    double assessImpermanenceRisk(List<ArisingPassingEvent> detectedEvents,
                                 double expectedTransience);
    
    /**
     * Assesses attention distraction risk (R_distraction).
     * Higher risk when attention drifts or fixates conceptually.
     *
     * @param attentionStability measure of attention consistency
     * @param conceptualFixation degree of non-present-moment thinking
     * @return distraction risk ≥ 0
     */
    double assessDistractionRisk(double attentionStability, 
                               double conceptualFixation);
    
    /**
     * Integrates human-in-the-loop observer feedback.
     * Essential for validating contemplative insights and preventing drift.
     *
     * @param aiInsight initial AI-generated contemplative assessment
     * @param observerFeedback human teacher/peer validation
     * @return calibrated insight with observer integration
     */
    ContemplativeInsight integrateObserverFeedback(ContemplativeInsight aiInsight,
                                                  ObserverFeedback observerFeedback);
    
    /**
     * Adapts interface for inclusive participation.
     * Supports diverse sensory modalities and cultural contexts.
     *
     * @param userProfile individual accessibility and cultural needs
     * @param contemplativeContext cultural/traditional framework
     * @return adapted processing parameters
     */
    ContemplativeParameters adaptForInclusion(UserProfile userProfile,
                                             ContemplativeContext contemplativeContext);
    
    /**
     * Generates explainable contemplative insights.
     * Translates non-conceptual awareness into communicable understanding.
     *
     * @param contemplativeState internal processing state
     * @param visualEvents detected arising/passing phenomena
     * @return human-interpretable explanation of contemplative processing
     */
    ContemplativeExplanation generateExplanation(ContemplativeState contemplativeState,
                                               List<ArisingPassingEvent> visualEvents);
    
    // Data classes for contemplative processing
    
    /**
     * Parameters specific to contemplative AI processing.
     */
    record ContemplativeParameters(
        double betaContemplative,     // β_c: contemplative uplift factor
        double lambdaTemporal,        // λ_temporal: temporal risk penalty weight
        double lambdaAttention,       // λ_attention: attention risk penalty weight
        double alphaVisual,           // α_visual: visual vs canonical evidence allocation
        double meditationExperience,  // User's contemplative experience level
        ContemplativeContext context // Cultural/traditional framework
    ) {}
    
    /**
     * Visual sequence data for temporal processing.
     */
    record VisualSequenceData(
        List<VisualFrame> frames,     // Temporal sequence of visual data
        long timestampStart,          // Sequence start time
        long timestampEnd,            // Sequence end time
        AttentionRegion focusRegion,  // Current attention focus
        double changeThreshold        // Threshold for detecting arising/passing
    ) {}
    
    /**
     * Individual visual frame in a sequence.
     */
    record VisualFrame(
        byte[] imageData,             // Raw visual data
        long timestamp,               // Frame timestamp
        double[] features,            // Extracted visual features
        Optional<AttentionMask> mask  // Attention-weighted regions
    ) {}
    
    /**
     * Detected arising or passing event in visual sequence.
     */
    record ArisingPassingEvent(
        EventType type,               // ARISING or PASSING
        long timestamp,               // When event occurred
        AttentionRegion region,       // Where event occurred
        double confidence,            // Detection confidence [0,1]
        double impermanenceScore,     // Degree of transience
        String description            // Human-readable description
    ) {
        enum EventType { ARISING, PASSING, PERSISTENCE, TRANSFORMATION }
    }
    
    /**
     * Current state of contemplative processing.
     */
    record ContemplativeState(
        double attentionStability,    // Consistency of attention [0,1]
        double presentMomentAwareness, // Degree of present-moment focus [0,1]
        double conceptualFixation,    // Level of mental elaboration [0,1]
        List<ArisingPassingEvent> recentEvents, // Recently detected phenomena
        long meditationDuration       // Current session duration
    ) {}
    
    /**
     * Human observer feedback for validation.
     */
    record ObserverFeedback(
        String observerId,            // Teacher/peer identifier
        double agreementScore,        // Agreement with AI assessment [0,1]
        String correctionNotes,       // Textual feedback
        List<String> suggestedFocus,  // Recommended attention areas
        ContemplativeQuality quality  // Overall quality assessment
    ) {}
    
    /**
     * Quality assessment of contemplative processing.
     */
    enum ContemplativeQuality {
        BEGINNER,           // Basic arising/passing detection
        DEVELOPING,         // Consistent but shallow awareness
        PROFICIENT,         // Stable, clear observation
        ADVANCED,           // Deep, non-conceptual insight
        EXPERT             // Effortless, wisdom-integrated awareness
    }
    
    /**
     * Contemplative insight generated by the system.
     */
    record ContemplativeInsight(
        double psiScore,              // Final contemplative Ψ score
        String insightDescription,    // Natural language insight
        List<ArisingPassingEvent> keyEvents, // Significant detected phenomena
        ContemplativeQuality assessedLevel,  // Estimated user level
        List<String> recommendations  // Suggested next steps
    ) {}
    
    /**
     * Cultural and traditional context for contemplative practice.
     */
    record ContemplativeContext(
        String tradition,             // e.g., "Theravada", "Zen", "Secular"
        String language,              // Preferred explanation language
        List<String> culturalValues,  // Cultural considerations
        boolean inclusiveFramework,   // Whether to use inclusive terminology
        double traditionFidelity      // How closely to follow traditional methods [0,1]
    ) {}
    
    /**
     * User profile for inclusive adaptation.
     */
    record UserProfile(
        List<String> accessibilityNeeds, // Visual, auditory, motor, cognitive needs
        String culturalBackground,       // Cultural context
        String primaryLanguage,          // Preferred language
        double meditationExperience,     // Experience level [0,1]
        List<String> preferredModalities // Visual, auditory, haptic, etc.
    ) {}
    
    /**
     * Region of visual attention focus.
     */
    record AttentionRegion(
        int x, int y,                 // Center coordinates
        int width, int height,        // Region dimensions
        double intensity,             // Attention intensity [0,1]
        long focusDuration           // How long attention has been here
    ) {}
    
    /**
     * Attention mask for weighted visual processing.
     */
    record AttentionMask(
        double[][] weights,           // Spatial attention weights [0,1]
        AttentionRegion primaryFocus, // Main attention region
        List<AttentionRegion> secondaryRegions // Additional focus areas
    ) {}
    
    /**
     * Explanation of contemplative processing for human understanding.
     */
    record ContemplativeExplanation(
        String summaryInsight,        // High-level contemplative insight
        List<String> stepByStep,      // Detailed processing steps
        List<ArisingPassingEvent> keyObservations, // Important detected phenomena
        String contemplativeGuidance, // Meditation-style guidance
        double confidenceLevel        // Overall explanation confidence [0,1]
    ) {}
    
    /**
     * Visual sequence for temporal analysis.
     */
    record VisualSequence(
        List<VisualFrame> frames,     // Ordered sequence of frames
        double frameRate,             // Frames per second
        long totalDuration,           // Total sequence duration
        String sequenceType           // e.g., "meditation", "nature", "abstract"
    ) {}
}