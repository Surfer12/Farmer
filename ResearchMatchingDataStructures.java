import java.util.*;

/**
 * Data structures for the Enhanced Research Matching System
 */

/**
 * Research trajectory representing a researcher's topic evolution over time
 */
class ResearchTrajectory {
    private String researcherId;
    private List<double[]> topicSequence;
    private List<Double> velocities;
    private List<Double> accelerations;
    private long timestamp;
    
    public ResearchTrajectory(String researcherId, List<double[]> topicSequence,
                            List<Double> velocities, List<Double> accelerations) {
        this.researcherId = researcherId;
        this.topicSequence = new ArrayList<>(topicSequence);
        this.velocities = new ArrayList<>(velocities);
        this.accelerations = new ArrayList<>(accelerations);
        this.timestamp = System.currentTimeMillis();
    }
    
    public int getSequenceLength() { return topicSequence.size(); }
    
    // Getters
    public String getResearcherId() { return researcherId; }
    public List<double[]> getTopicSequence() { return topicSequence; }
    public List<Double> getVelocities() { return velocities; }
    public List<Double> getAccelerations() { return accelerations; }
    public long getTimestamp() { return timestamp; }
    
    // Analysis methods
    public double getAverageVelocity() {
        return velocities.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    public double getMaxAcceleration() {
        return accelerations.stream().mapToDouble(Math::abs).max().orElse(0.0);
    }
    
    public double[] getLatestTopicDistribution() {
        return topicSequence.isEmpty() ? new double[0] : 
               topicSequence.get(topicSequence.size() - 1);
    }
}

/**
 * Collaboration candidate with similarity metrics
 */
class CollaborationCandidate {
    private String id;
    private double similarity;
    private Map<String, Double> additionalMetrics;
    
    public CollaborationCandidate(String id, double similarity) {
        this.id = id;
        this.similarity = similarity;
        this.additionalMetrics = new HashMap<>();
    }
    
    public void addMetric(String name, double value) {
        additionalMetrics.put(name, value);
    }
    
    // Getters
    public String getId() { return id; }
    public double getSimilarity() { return similarity; }
    public Map<String, Double> getAdditionalMetrics() { return additionalMetrics; }
}

/**
 * Collaboration match result with comprehensive scoring
 */
class CollaborationMatch {
    private String researcherId;
    private String candidateId;
    private double hybridScore;
    private double confidenceScore;
    private double errorBound;
    private double symbolicAccuracy;
    private double neuralAccuracy;
    private ResearchTrajectory trajectory;
    private Map<String, Object> metadata;
    
    public CollaborationMatch(String researcherId, String candidateId, double hybridScore,
                            double confidenceScore, double errorBound, double symbolicAccuracy,
                            double neuralAccuracy, ResearchTrajectory trajectory) {
        this.researcherId = researcherId;
        this.candidateId = candidateId;
        this.hybridScore = hybridScore;
        this.confidenceScore = confidenceScore;
        this.errorBound = errorBound;
        this.symbolicAccuracy = symbolicAccuracy;
        this.neuralAccuracy = neuralAccuracy;
        this.trajectory = trajectory;
        this.metadata = new HashMap<>();
    }
    
    public void addMetadata(String key, Object value) {
        metadata.put(key, value);
    }
    
    // Getters
    public String getResearcherId() { return researcherId; }
    public String getCandidateId() { return candidateId; }
    public double getHybridScore() { return hybridScore; }
    public double getConfidenceScore() { return confidenceScore; }
    public double getErrorBound() { return errorBound; }
    public double getSymbolicAccuracy() { return symbolicAccuracy; }
    public double getNeuralAccuracy() { return neuralAccuracy; }
    public ResearchTrajectory getTrajectory() { return trajectory; }
    public Map<String, Object> getMetadata() { return metadata; }
    
    @Override
    public String toString() {
        return String.format("Match[%s->%s: hybrid=%.3f, conf=%.3f, err=%.4f]",
                           researcherId, candidateId, hybridScore, confidenceScore, errorBound);
    }
}

/**
 * LSTM model implementation for chaos prediction
 */
class LSTMModel {
    private int inputSize;
    private int hiddenSize;
    private int numLayers;
    
    // LSTM parameters (simplified representation)
    private double[][][] weights; // [layer][gate][input/hidden]
    private double[][] biases;    // [layer][gate]
    private double[][] hiddenStates; // [layer][hidden]
    private double[][] cellStates;   // [layer][hidden]
    
    // Gate indices
    private static final int FORGET_GATE = 0;
    private static final int INPUT_GATE = 1;
    private static final int OUTPUT_GATE = 2;
    private static final int CANDIDATE_GATE = 3;
    
    public LSTMModel(int inputSize, int hiddenSize, int numLayers) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.numLayers = numLayers;
        
        // Initialize weight matrices and biases
        this.weights = new double[numLayers][4][inputSize + hiddenSize];
        this.biases = new double[numLayers][4];
        this.hiddenStates = new double[numLayers][hiddenSize];
        this.cellStates = new double[numLayers][hiddenSize];
    }
    
    public void initializeWeights(Random random, double bound) {
        for (int layer = 0; layer < numLayers; layer++) {
            for (int gate = 0; gate < 4; gate++) {
                for (int i = 0; i < inputSize + hiddenSize; i++) {
                    weights[layer][gate][i] = (random.nextDouble() - 0.5) * 2 * bound;
                }
                biases[layer][gate] = (random.nextDouble() - 0.5) * 2 * bound;
            }
        }
    }
    
    public LSTMOutput forward(List<double[]> inputSequence) {
        // Reset states
        for (int layer = 0; layer < numLayers; layer++) {
            Arrays.fill(hiddenStates[layer], 0.0);
            Arrays.fill(cellStates[layer], 0.0);
        }
        
        double[] finalOutput = null;
        
        // Process sequence
        for (double[] input : inputSequence) {
            finalOutput = forwardStep(input);
        }
        
        return new LSTMOutput(finalOutput != null ? finalOutput : new double[hiddenSize]);
    }
    
    private double[] forwardStep(double[] input) {
        double[] currentInput = input;
        
        for (int layer = 0; layer < numLayers; layer++) {
            // Concatenate input and previous hidden state
            double[] combined = new double[currentInput.length + hiddenSize];
            System.arraycopy(currentInput, 0, combined, 0, currentInput.length);
            System.arraycopy(hiddenStates[layer], 0, combined, currentInput.length, hiddenSize);
            
            // Compute gates
            double[] forgetGate = computeGate(combined, weights[layer][FORGET_GATE], 
                                           biases[layer][FORGET_GATE], true);
            double[] inputGate = computeGate(combined, weights[layer][INPUT_GATE], 
                                          biases[layer][INPUT_GATE], true);
            double[] outputGate = computeGate(combined, weights[layer][OUTPUT_GATE], 
                                           biases[layer][OUTPUT_GATE], true);
            double[] candidateGate = computeGate(combined, weights[layer][CANDIDATE_GATE], 
                                              biases[layer][CANDIDATE_GATE], false);
            
            // Update cell state: c_t = f_t * c_{t-1} + i_t * \tilde{c}_t
            for (int i = 0; i < hiddenSize; i++) {
                cellStates[layer][i] = forgetGate[i] * cellStates[layer][i] + 
                                     inputGate[i] * candidateGate[i];
            }
            
            // Update hidden state: h_t = o_t * tanh(c_t)
            for (int i = 0; i < hiddenSize; i++) {
                hiddenStates[layer][i] = outputGate[i] * Math.tanh(cellStates[layer][i]);
            }
            
            // Output of this layer becomes input to next layer
            currentInput = Arrays.copyOf(hiddenStates[layer], hiddenSize);
        }
        
        return currentInput;
    }
    
    private double[] computeGate(double[] input, double[] weights, double bias, boolean sigmoid) {
        double[] output = new double[hiddenSize];
        
        for (int i = 0; i < hiddenSize; i++) {
            double sum = bias;
            for (int j = 0; j < Math.min(input.length, weights.length); j++) {
                sum += input[j] * weights[j];
            }
            
            output[i] = sigmoid ? sigmoid(sum) : Math.tanh(sum);
        }
        
        return output;
    }
    
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    public void backward(List<double[]> input, double[] target, double learningRate) {
        // Simplified backward pass (in practice, would implement full BPTT)
        // This is a placeholder for the actual backpropagation implementation
        
        // Calculate output error
        LSTMOutput output = forward(input);
        double[] prediction = output.getOutput();
        
        // Simple gradient update (placeholder)
        for (int layer = 0; layer < numLayers; layer++) {
            for (int gate = 0; gate < 4; gate++) {
                for (int i = 0; i < weights[layer][gate].length; i++) {
                    // Simplified gradient update
                    double gradient = calculateSimpleGradient(prediction, target, i);
                    weights[layer][gate][i] -= learningRate * gradient;
                }
            }
        }
    }
    
    private double calculateSimpleGradient(double[] prediction, double[] target, int index) {
        // Simplified gradient calculation
        double error = 0.0;
        for (int i = 0; i < Math.min(prediction.length, target.length); i++) {
            error += (prediction[i] - target[i]);
        }
        return error / Math.min(prediction.length, target.length);
    }
}

/**
 * LSTM output container
 */
class LSTMOutput {
    private double[] output;
    private Map<String, Object> metadata;
    
    public LSTMOutput(double[] output) {
        this.output = Arrays.copyOf(output, output.length);
        this.metadata = new HashMap<>();
    }
    
    public double[] getOutput() { return output; }
    public Map<String, Object> getMetadata() { return metadata; }
    public void addMetadata(String key, Object value) { metadata.put(key, value); }
}

/**
 * Training example for LSTM
 */
class TrainingExample {
    private List<double[]> input;
    private double[] target;
    
    public TrainingExample(List<double[]> input, double[] target) {
        this.input = new ArrayList<>(input);
        this.target = Arrays.copyOf(target, target.length);
    }
    
    public List<double[]> getInput() { return input; }
    public double[] getTarget() { return target; }
}

/**
 * Validation result for model evaluation
 */
class ValidationResult {
    private double averageError;
    private double averageConfidence;
    private double theoreticalBound;
    private boolean satisfiesOatesTheorem;
    private int numValidations;
    
    public ValidationResult(double averageError, double averageConfidence, 
                          double theoreticalBound, boolean satisfiesOatesTheorem,
                          int numValidations) {
        this.averageError = averageError;
        this.averageConfidence = averageConfidence;
        this.theoreticalBound = theoreticalBound;
        this.satisfiesOatesTheorem = satisfiesOatesTheorem;
        this.numValidations = numValidations;
    }
    
    // Getters
    public double getAverageError() { return averageError; }
    public double getAverageConfidence() { return averageConfidence; }
    public double getTheoreticalBound() { return theoreticalBound; }
    public boolean satisfiesOatesTheorem() { return satisfiesOatesTheorem; }
    public int getNumValidations() { return numValidations; }
    
    @Override
    public String toString() {
        return String.format("ValidationResult[error=%.4f, conf=%.3f, bound=%.4f, oates=%s, n=%d]",
                           averageError, averageConfidence, theoreticalBound, 
                           satisfiesOatesTheorem, numValidations);
    }
}

/**
 * Hybrid Functional Calculator implementing the core equation
 */
class HybridFunctionalCalculator {
    private double lambda1; // Cognitive penalty weight
    private double lambda2; // Efficiency penalty weight
    private double beta;    // Bias correction factor
    
    public HybridFunctionalCalculator(double lambda1, double lambda2, double beta) {
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        this.beta = beta;
    }
    
    /**
     * Compute hybrid functional score: Ψ(x) = α(t)S(x) + (1-α(t))N(x) × penalties × P(H|E,β)
     */
    public double computeHybridScore(double symbolicAccuracy, double neuralAccuracy, double alpha) {
        // Core hybrid combination
        double hybridCore = alpha * symbolicAccuracy + (1 - alpha) * neuralAccuracy;
        
        // Calculate penalties
        double cognitivePenalty = calculateCognitivePenalty(symbolicAccuracy, neuralAccuracy);
        double efficiencyPenalty = calculateEfficiencyPenalty(alpha);
        
        // Exponential penalty term
        double penaltyTerm = Math.exp(-(lambda1 * cognitivePenalty + lambda2 * efficiencyPenalty));
        
        // Calibrated probability with bias correction
        double calibratedProb = calculateCalibratedProbability(hybridCore, beta);
        
        // Final hybrid score
        return hybridCore * penaltyTerm * calibratedProb;
    }
    
    /**
     * Calculate cognitive penalty (physics violation, energy drift)
     */
    private double calculateCognitivePenalty(double symbolicAccuracy, double neuralAccuracy) {
        // Penalty for large discrepancy between symbolic and neural components
        double discrepancy = Math.abs(symbolicAccuracy - neuralAccuracy);
        
        // Penalty for low individual accuracies
        double lowAccuracyPenalty = Math.max(0, 0.5 - Math.min(symbolicAccuracy, neuralAccuracy));
        
        return 0.6 * discrepancy + 0.4 * lowAccuracyPenalty;
    }
    
    /**
     * Calculate efficiency penalty (computational cost)
     */
    private double calculateEfficiencyPenalty(double alpha) {
        // Penalty for extreme alpha values (prefer balanced approaches)
        double balancePenalty = 4 * alpha * (1 - alpha); // Maximum at alpha = 0.5
        return 1.0 - balancePenalty; // Convert to penalty (lower is better)
    }
    
    /**
     * Calculate calibrated probability with Platt scaling
     */
    private double calculateCalibratedProbability(double score, double beta) {
        // Platt scaling: P(y=1|f) = 1 / (1 + exp(-β*f))
        return 1.0 / (1.0 + Math.exp(-beta * score));
    }
    
    /**
     * Calculate adaptive weight α(t) based on system characteristics
     */
    public double calculateAdaptiveWeight(double chaosLevel, double dataQuality, double timeStep) {
        // Sigmoid function favoring neural in chaotic regions
        double kappa = 2.0; // Sensitivity parameter
        double localLambda = chaosLevel * (1.0 - dataQuality); // Local chaos measure
        
        // α(t) = σ(-κ·λ_local(t))
        double alpha = 1.0 / (1.0 + Math.exp(kappa * localLambda));
        
        // Temporal adjustment
        double temporalFactor = Math.exp(-timeStep / 100.0); // Decay over time
        
        return alpha * temporalFactor + (1 - temporalFactor) * 0.5; // Converge to balanced
    }
    
    /**
     * Compute full Ψ(x) with temporal averaging
     */
    public double computeTemporalHybridScore(List<Double> symbolicSequence,
                                           List<Double> neuralSequence,
                                           List<Double> alphaSequence) {
        if (symbolicSequence.size() != neuralSequence.size() || 
            neuralSequence.size() != alphaSequence.size()) {
            throw new IllegalArgumentException("Sequence lengths must match");
        }
        
        double sum = 0.0;
        int T = symbolicSequence.size();
        
        for (int t = 0; t < T; t++) {
            double score = computeHybridScore(symbolicSequence.get(t),
                                            neuralSequence.get(t),
                                            alphaSequence.get(t));
            sum += score;
        }
        
        return sum / T; // Temporal average
    }
    
    // Getters and setters
    public double getLambda1() { return lambda1; }
    public double getLambda2() { return lambda2; }
    public double getBeta() { return beta; }
    
    public void setLambda1(double lambda1) { this.lambda1 = lambda1; }
    public void setLambda2(double lambda2) { this.lambda2 = lambda2; }
    public void setBeta(double beta) { this.beta = beta; }
}

/**
 * Research trajectory predictor for future collaboration analysis
 */
class ResearchTrajectoryPredictor {
    private Map<String, ResearchTrajectory> trajectoryCache;
    private double predictionHorizon = 12.0; // months
    
    public ResearchTrajectoryPredictor() {
        this.trajectoryCache = new HashMap<>();
    }
    
    /**
     * Predict research trajectory evolution
     */
    public ResearchTrajectory predictTrajectoryEvolution(ResearchTrajectory currentTrajectory,
                                                       double timeHorizon) {
        // Simple linear extrapolation based on current velocity and acceleration
        List<double[]> currentSequence = currentTrajectory.getTopicSequence();
        List<Double> velocities = currentTrajectory.getVelocities();
        List<Double> accelerations = currentTrajectory.getAccelerations();
        
        if (currentSequence.isEmpty()) {
            return currentTrajectory;
        }
        
        double[] lastPoint = currentSequence.get(currentSequence.size() - 1);
        double lastVelocity = velocities.isEmpty() ? 0.0 : velocities.get(velocities.size() - 1);
        double lastAcceleration = accelerations.isEmpty() ? 0.0 : accelerations.get(accelerations.size() - 1);
        
        // Predict future points
        List<double[]> predictedSequence = new ArrayList<>(currentSequence);
        List<Double> predictedVelocities = new ArrayList<>(velocities);
        List<Double> predictedAccelerations = new ArrayList<>(accelerations);
        
        int steps = (int) Math.ceil(timeHorizon);
        for (int step = 1; step <= steps; step++) {
            double t = step / 12.0; // Convert to years
            
            // Simple kinematic prediction: x(t) = x₀ + v₀t + ½at²
            double[] predictedPoint = new double[lastPoint.length];
            for (int i = 0; i < lastPoint.length; i++) {
                predictedPoint[i] = lastPoint[i] + lastVelocity * t + 0.5 * lastAcceleration * t * t;
                predictedPoint[i] = Math.max(0.0, Math.min(1.0, predictedPoint[i])); // Clamp to [0,1]
            }
            
            predictedSequence.add(predictedPoint);
            
            // Update velocity and acceleration
            double newVelocity = lastVelocity + lastAcceleration * t;
            predictedVelocities.add(newVelocity);
            predictedAccelerations.add(lastAcceleration * 0.9); // Decay acceleration
        }
        
        return new ResearchTrajectory(
            currentTrajectory.getResearcherId() + "_predicted",
            predictedSequence,
            predictedVelocities,
            predictedAccelerations
        );
    }
    
    /**
     * Assess collaboration potential between trajectories
     */
    public double assessCollaborationPotential(ResearchTrajectory traj1, ResearchTrajectory traj2) {
        // Calculate topic complementarity
        double[] topics1 = traj1.getLatestTopicDistribution();
        double[] topics2 = traj2.getLatestTopicDistribution();
        
        double complementarity = calculateComplementarity(topics1, topics2);
        
        // Calculate velocity alignment
        double velocityAlignment = calculateVelocityAlignment(traj1, traj2);
        
        // Calculate trajectory stability
        double stability1 = calculateTrajectoryStability(traj1);
        double stability2 = calculateTrajectoryStability(traj2);
        double avgStability = (stability1 + stability2) / 2.0;
        
        // Combine factors
        return 0.4 * complementarity + 0.3 * velocityAlignment + 0.3 * avgStability;
    }
    
    private double calculateComplementarity(double[] topics1, double[] topics2) {
        if (topics1.length != topics2.length) return 0.0;
        
        double overlap = 0.0;
        for (int i = 0; i < topics1.length; i++) {
            overlap += Math.min(topics1[i], topics2[i]);
        }
        
        return 1.0 - overlap; // Higher complementarity = less overlap
    }
    
    private double calculateVelocityAlignment(ResearchTrajectory traj1, ResearchTrajectory traj2) {
        double vel1 = traj1.getAverageVelocity();
        double vel2 = traj2.getAverageVelocity();
        
        // Prefer similar velocities (not too different research paces)
        double velocityDiff = Math.abs(vel1 - vel2);
        return Math.exp(-velocityDiff * 10.0); // Exponential decay
    }
    
    private double calculateTrajectoryStability(ResearchTrajectory trajectory) {
        List<Double> accelerations = trajectory.getAccelerations();
        if (accelerations.isEmpty()) return 1.0;
        
        double variance = accelerations.stream()
            .mapToDouble(a -> a * a)
            .average().orElse(0.0);
        
        return 1.0 / (1.0 + variance); // Lower variance = higher stability
    }
}
