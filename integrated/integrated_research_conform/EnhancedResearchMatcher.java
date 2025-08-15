import java.util.*;
import java.util.stream.Collectors;
import java.io.*;
import java.nio.file.*;

/**
 * Enhanced Research Matching System integrating Academic Network Analysis
 * with Oates' LSTM Hidden State Convergence Theorem for predictive collaboration
 */
public class EnhancedResearchMatcher {
    
    // Core components
    private AcademicNetworkAnalyzer networkAnalyzer;
    private LSTMChaosPredictionEngine lstmEngine;
    private HybridFunctionalCalculator hybridCalculator;
    private ResearchTrajectoryPredictor trajectoryPredictor;
    
    // Configuration parameters
    private double alpha = 0.6; // Symbolic-neural balance
    private double lambda1 = 0.75; // Cognitive penalty weight
    private double lambda2 = 0.25; // Efficiency penalty weight
    private double beta = 1.2; // Bias correction factor
    
    // Performance metrics
    private Map<String, Double> confidenceScores;
    private Map<String, Double> errorBounds;
    private List<PredictionResult> predictionHistory;
    
    public EnhancedResearchMatcher() {
        this.networkAnalyzer = new AcademicNetworkAnalyzer();
        this.lstmEngine = new LSTMChaosPredictionEngine();
        this.hybridCalculator = new HybridFunctionalCalculator(lambda1, lambda2, beta);
        this.trajectoryPredictor = new ResearchTrajectoryPredictor();
        this.confidenceScores = new HashMap<>();
        this.errorBounds = new HashMap<>();
        this.predictionHistory = new ArrayList<>();
    }
    
    /**
     * Enhanced research matching incorporating LSTM predictions and network analysis
     */
    public List<CollaborationMatch> findEnhancedMatches(String researcherId, 
                                                       int numMatches) throws IOException {
        
        System.out.println("=== Enhanced Research Matching Analysis ===");
        
        // Step 1: Network-based analysis
        System.out.println("1. Performing network topology analysis...");
        networkAnalyzer.loadResearchData("publications.csv");
        networkAnalyzer.performTopicModeling();
        networkAnalyzer.buildSimilarityMatrix();
        networkAnalyzer.buildNetworkAndDetectCommunities(0.25);
        
        // Step 2: LSTM trajectory prediction
        System.out.println("2. Generating LSTM-based research trajectory predictions...");
        ResearchTrajectory currentTrajectory = extractResearchTrajectory(researcherId);
        List<ResearchTrajectory> predictedTrajectories = 
            lstmEngine.predictFutureTrajectories(currentTrajectory, 12); // 12 months ahead
        
        // Step 3: Hybrid functional calculation
        System.out.println("3. Computing hybrid symbolic-neural scores...");
        List<CollaborationCandidate> candidates = identifyCollaborationCandidates(researcherId);
        
        List<CollaborationMatch> matches = new ArrayList<>();
        for (CollaborationCandidate candidate : candidates) {
            CollaborationMatch match = evaluateCollaborationMatch(
                researcherId, candidate, predictedTrajectories);
            matches.add(match);
        }
        
        // Step 4: Rank and filter matches
        matches.sort((a, b) -> Double.compare(b.getHybridScore(), a.getHybridScore()));
        
        // Step 5: Apply confidence filtering
        matches = matches.stream()
            .filter(m -> m.getConfidenceScore() >= 0.85)
            .limit(numMatches)
            .collect(Collectors.toList());
        
        System.out.println("4. Analysis complete. Found " + matches.size() + " high-confidence matches");
        
        return matches;
    }
    
    /**
     * Extract research trajectory from publication history
     */
    private ResearchTrajectory extractResearchTrajectory(String researcherId) {
        Researcher researcher = networkAnalyzer.getResearcher(researcherId);
        if (researcher == null) {
            throw new IllegalArgumentException("Researcher not found: " + researcherId);
        }
        
        List<Publication> publications = researcher.getPublications();
        publications.sort((a, b) -> a.getId().compareTo(b.getId())); // Assume chronological by ID
        
        // Extract topic evolution over time
        List<double[]> topicSequence = publications.stream()
            .map(Publication::getTopicDistribution)
            .collect(Collectors.toList());
        
        // Calculate research velocity and acceleration
        List<Double> velocities = calculateTopicVelocities(topicSequence);
        List<Double> accelerations = calculateTopicAccelerations(velocities);
        
        return new ResearchTrajectory(researcherId, topicSequence, velocities, accelerations);
    }
    
    /**
     * Calculate topic velocities (rate of change in research focus)
     */
    private List<Double> calculateTopicVelocities(List<double[]> topicSequence) {
        List<Double> velocities = new ArrayList<>();
        
        for (int i = 1; i < topicSequence.size(); i++) {
            double[] prev = topicSequence.get(i - 1);
            double[] curr = topicSequence.get(i);
            
            double velocity = 0.0;
            for (int j = 0; j < prev.length; j++) {
                velocity += Math.pow(curr[j] - prev[j], 2);
            }
            velocities.add(Math.sqrt(velocity));
        }
        
        return velocities;
    }
    
    /**
     * Calculate topic accelerations (rate of change in research velocity)
     */
    private List<Double> calculateTopicAccelerations(List<Double> velocities) {
        List<Double> accelerations = new ArrayList<>();
        
        for (int i = 1; i < velocities.size(); i++) {
            double acceleration = velocities.get(i) - velocities.get(i - 1);
            accelerations.add(acceleration);
        }
        
        return accelerations;
    }
    
    /**
     * Identify potential collaboration candidates from network analysis
     */
    private List<CollaborationCandidate> identifyCollaborationCandidates(String researcherId) {
        List<CollaborationCandidate> candidates = new ArrayList<>();
        
        // Get researchers from same communities
        List<Community> communities = networkAnalyzer.getCommunities();
        Set<String> communityMembers = new HashSet<>();
        
        for (Community community : communities) {
            if (community.containsMember(researcherId)) {
                communityMembers.addAll(community.getMembers());
            }
        }
        communityMembers.remove(researcherId); // Remove self
        
        // Get researchers with high similarity scores
        List<NetworkEdge> edges = networkAnalyzer.getNetworkEdges();
        Map<String, Double> similarities = new HashMap<>();
        
        for (NetworkEdge edge : edges) {
            if (edge.getSourceId().equals(researcherId)) {
                similarities.put(edge.getTargetId(), edge.getWeight());
            } else if (edge.getTargetId().equals(researcherId)) {
                similarities.put(edge.getSourceId(), edge.getWeight());
            }
        }
        
        // Combine community and similarity information
        for (String candidateId : communityMembers) {
            double similarity = similarities.getOrDefault(candidateId, 0.0);
            if (similarity > 0.3) { // Minimum similarity threshold
                candidates.add(new CollaborationCandidate(candidateId, similarity));
            }
        }
        
        return candidates;
    }
    
    /**
     * Evaluate collaboration match using hybrid functional
     */
    private CollaborationMatch evaluateCollaborationMatch(String researcherId,
                                                        CollaborationCandidate candidate,
                                                        List<ResearchTrajectory> predictedTrajectories) {
        
        // Extract candidate trajectory
        ResearchTrajectory candidateTrajectory = extractResearchTrajectory(candidate.getId());
        
        // Predict collaboration trajectory using LSTM
        ResearchTrajectory collaborationTrajectory = 
            lstmEngine.predictCollaborationTrajectory(
                extractResearchTrajectory(researcherId), candidateTrajectory);
        
        // Calculate symbolic accuracy (network-based metrics)
        double symbolicAccuracy = calculateSymbolicAccuracy(researcherId, candidate);
        
        // Calculate neural accuracy (LSTM prediction confidence)
        double neuralAccuracy = calculateNeuralAccuracy(collaborationTrajectory);
        
        // Calculate hybrid functional score
        double hybridScore = hybridCalculator.computeHybridScore(
            symbolicAccuracy, neuralAccuracy, alpha);
        
        // Calculate confidence using Oates' theorem
        double confidenceScore = calculateOatesConfidence(collaborationTrajectory);
        
        // Calculate error bounds
        double errorBound = calculateErrorBound(collaborationTrajectory.getSequenceLength());
        
        return new CollaborationMatch(
            researcherId, candidate.getId(), hybridScore, confidenceScore, 
            errorBound, symbolicAccuracy, neuralAccuracy, collaborationTrajectory);
    }
    
    /**
     * Calculate symbolic accuracy based on network topology
     */
    private double calculateSymbolicAccuracy(String researcherId, CollaborationCandidate candidate) {
        // Network-based metrics
        double topicSimilarity = candidate.getSimilarity();
        double communityOverlap = calculateCommunityOverlap(researcherId, candidate.getId());
        double publicationComplementarity = calculatePublicationComplementarity(
            researcherId, candidate.getId());
        
        // Weighted combination
        return 0.4 * topicSimilarity + 0.3 * communityOverlap + 0.3 * publicationComplementarity;
    }
    
    /**
     * Calculate neural accuracy using LSTM predictions
     */
    private double calculateNeuralAccuracy(ResearchTrajectory trajectory) {
        // LSTM-based prediction confidence
        double trajectoryStability = calculateTrajectoryStability(trajectory);
        double predictionConsistency = calculatePredictionConsistency(trajectory);
        double chaoticBehavior = calculateChaoticBehavior(trajectory);
        
        // Combine metrics (higher stability and consistency, lower chaos = higher accuracy)
        return 0.4 * trajectoryStability + 0.4 * predictionConsistency + 0.2 * (1.0 - chaoticBehavior);
    }
    
    /**
     * Calculate confidence using Oates' LSTM Hidden State Convergence Theorem
     */
    private double calculateOatesConfidence(ResearchTrajectory trajectory) {
        int T = trajectory.getSequenceLength();
        
        // Error bound: O(1/√T)
        double errorBound = 1.0 / Math.sqrt(T);
        
        // Confidence measure C(p) = P(||x̂_{t+1} - x_{t+1}|| ≤ η | E)
        double eta = 0.1; // Tolerance threshold
        double evidence = calculateEvidenceStrength(trajectory);
        
        // Lipschitz continuity factor for gates
        double lipschitzBound = calculateLipschitzBound(trajectory);
        
        // Expected confidence: E[C(p)] ≥ 1 - ε
        double epsilon = Math.pow(0.01, 4) + errorBound; // O(h^4) + δ_LSTM
        double expectedConfidence = Math.max(0.0, 1.0 - epsilon);
        
        // Adjust for evidence strength and Lipschitz bound
        return Math.min(1.0, expectedConfidence * evidence * lipschitzBound);
    }
    
    /**
     * Calculate error bound using Oates' theorem: O(1/√T)
     */
    private double calculateErrorBound(int sequenceLength) {
        return 1.0 / Math.sqrt(sequenceLength);
    }
    
    // Helper methods for trajectory analysis
    private double calculateCommunityOverlap(String researcher1, String researcher2) {
        List<Community> communities = networkAnalyzer.getCommunities();
        int sharedCommunities = 0;
        int totalCommunities = 0;
        
        for (Community community : communities) {
            boolean has1 = community.containsMember(researcher1);
            boolean has2 = community.containsMember(researcher2);
            
            if (has1 || has2) {
                totalCommunities++;
                if (has1 && has2) {
                    sharedCommunities++;
                }
            }
        }
        
        return totalCommunities > 0 ? (double) sharedCommunities / totalCommunities : 0.0;
    }
    
    private double calculatePublicationComplementarity(String researcher1, String researcher2) {
        Researcher r1 = networkAnalyzer.getResearcher(researcher1);
        Researcher r2 = networkAnalyzer.getResearcher(researcher2);
        
        if (r1 == null || r2 == null) return 0.0;
        
        double[] topics1 = r1.getTopicDistribution();
        double[] topics2 = r2.getTopicDistribution();
        
        if (topics1 == null || topics2 == null) return 0.0;
        
        // Calculate complementarity (inverse of overlap)
        double overlap = 0.0;
        for (int i = 0; i < Math.min(topics1.length, topics2.length); i++) {
            overlap += Math.min(topics1[i], topics2[i]);
        }
        
        return 1.0 - overlap; // Higher complementarity = less overlap
    }
    
    private double calculateTrajectoryStability(ResearchTrajectory trajectory) {
        List<Double> velocities = trajectory.getVelocities();
        if (velocities.isEmpty()) return 1.0;
        
        double mean = velocities.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = velocities.stream()
            .mapToDouble(v -> Math.pow(v - mean, 2))
            .average().orElse(0.0);
        
        // Lower variance = higher stability
        return 1.0 / (1.0 + variance);
    }
    
    private double calculatePredictionConsistency(ResearchTrajectory trajectory) {
        List<Double> accelerations = trajectory.getAccelerations();
        if (accelerations.isEmpty()) return 1.0;
        
        // Measure consistency as inverse of acceleration variance
        double mean = accelerations.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = accelerations.stream()
            .mapToDouble(a -> Math.pow(a - mean, 2))
            .average().orElse(0.0);
        
        return 1.0 / (1.0 + variance);
    }
    
    private double calculateChaoticBehavior(ResearchTrajectory trajectory) {
        // Estimate Lyapunov exponent from trajectory
        List<double[]> sequence = trajectory.getTopicSequence();
        if (sequence.size() < 3) return 0.0;
        
        double maxDivergence = 0.0;
        for (int i = 1; i < sequence.size() - 1; i++) {
            double[] prev = sequence.get(i - 1);
            double[] curr = sequence.get(i);
            double[] next = sequence.get(i + 1);
            
            double divergence = calculateDivergence(prev, curr, next);
            maxDivergence = Math.max(maxDivergence, divergence);
        }
        
        return Math.tanh(maxDivergence); // Bounded between 0 and 1
    }
    
    private double calculateDivergence(double[] prev, double[] curr, double[] next) {
        double d1 = euclideanDistance(prev, curr);
        double d2 = euclideanDistance(curr, next);
        
        return d1 > 0 ? Math.log(d2 / d1) : 0.0;
    }
    
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
    
    private double calculateEvidenceStrength(ResearchTrajectory trajectory) {
        // Evidence strength based on sequence length and data quality
        int length = trajectory.getSequenceLength();
        double dataQuality = calculateDataQuality(trajectory);
        
        // Stronger evidence with longer sequences and higher quality
        return Math.min(1.0, (length / 100.0) * dataQuality);
    }
    
    private double calculateDataQuality(ResearchTrajectory trajectory) {
        // Assess data quality based on trajectory consistency
        List<double[]> sequence = trajectory.getTopicSequence();
        if (sequence.size() < 2) return 0.5;
        
        double totalVariation = 0.0;
        for (int i = 1; i < sequence.size(); i++) {
            double variation = euclideanDistance(sequence.get(i - 1), sequence.get(i));
            totalVariation += variation;
        }
        
        double avgVariation = totalVariation / (sequence.size() - 1);
        
        // Moderate variation indicates good quality (not too stable, not too chaotic)
        return Math.exp(-Math.pow(avgVariation - 0.3, 2) / 0.1);
    }
    
    private double calculateLipschitzBound(ResearchTrajectory trajectory) {
        // Estimate Lipschitz constant for trajectory smoothness
        List<Double> velocities = trajectory.getVelocities();
        if (velocities.isEmpty()) return 1.0;
        
        double maxVelocity = velocities.stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        
        // Lipschitz bound ensures stability
        return 1.0 / (1.0 + maxVelocity);
    }
    
    /**
     * Export enhanced matching results
     */
    public void exportEnhancedResults(List<CollaborationMatch> matches, String outputDir) 
            throws IOException {
        Path outputPath = Paths.get(outputDir);
        Files.createDirectories(outputPath);
        
        // Export collaboration matches
        try (PrintWriter writer = new PrintWriter(
                Files.newBufferedWriter(outputPath.resolve("enhanced_matches.csv")))) {
            writer.println("researcher_id,candidate_id,hybrid_score,confidence_score," +
                          "error_bound,symbolic_accuracy,neural_accuracy,trajectory_length");
            
            for (CollaborationMatch match : matches) {
                writer.printf("%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%d%n",
                    match.getResearcherId(),
                    match.getCandidateId(),
                    match.getHybridScore(),
                    match.getConfidenceScore(),
                    match.getErrorBound(),
                    match.getSymbolicAccuracy(),
                    match.getNeuralAccuracy(),
                    match.getTrajectory().getSequenceLength());
            }
        }
        
        // Export confidence analysis
        try (PrintWriter writer = new PrintWriter(
                Files.newBufferedWriter(outputPath.resolve("confidence_analysis.csv")))) {
            writer.println("metric,value,description");
            
            double avgConfidence = matches.stream()
                .mapToDouble(CollaborationMatch::getConfidenceScore)
                .average().orElse(0.0);
            
            double avgErrorBound = matches.stream()
                .mapToDouble(CollaborationMatch::getErrorBound)
                .average().orElse(0.0);
            
            writer.printf("average_confidence,%.4f,Average Oates confidence across matches%n", avgConfidence);
            writer.printf("average_error_bound,%.4f,Average O(1/√T) error bound%n", avgErrorBound);
            writer.printf("high_confidence_matches,%d,Matches with confidence ≥ 0.9%n",
                matches.stream().mapToInt(m -> m.getConfidenceScore() >= 0.9 ? 1 : 0).sum());
        }
        
        System.out.println("Enhanced results exported to " + outputDir);
    }
    
    // Getters for integration
    public AcademicNetworkAnalyzer getNetworkAnalyzer() { return networkAnalyzer; }
    public LSTMChaosPredictionEngine getLSTMEngine() { return lstmEngine; }
    public HybridFunctionalCalculator getHybridCalculator() { return hybridCalculator; }
}
