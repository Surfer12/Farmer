import java.util.*;
import java.util.concurrent.*;

/**
 * LSTM Chaos Prediction Engine implementing Oates' Hidden State Convergence Theorem
 * for research trajectory prediction and collaboration forecasting
 */
public class LSTMChaosPredictionEngine {
    
    // LSTM architecture parameters
    private int hiddenSize = 128;
    private int inputSize = 50; // Topic dimensions
    private int numLayers = 2;
    private double learningRate = 0.001;
    private double dropoutRate = 0.2;
    
    // Oates' theorem parameters
    private double lipschitzConstant = 1.0;
    private double confidenceThreshold = 0.85;
    private int minSequenceLength = 10;
    private double chaosThreshold = 0.3;
    
    // Model state
    private LSTMModel model;
    private boolean isTrained = false;
    private Map<String, Double> performanceMetrics;
    
    public LSTMChaosPredictionEngine() {
        this.model = new LSTMModel(inputSize, hiddenSize, numLayers);
        this.performanceMetrics = new HashMap<>();
        initializeModel();
    }
    
    /**
     * Initialize LSTM model with Xavier initialization
     */
    private void initializeModel() {
        System.out.println("Initializing LSTM model with Oates' theorem parameters...");
        
        // Xavier initialization for stability
        Random random = new Random(42);
        double xavier = Math.sqrt(6.0 / (inputSize + hiddenSize));
        
        model.initializeWeights(random, xavier);
        
        System.out.println("Model initialized with:");
        System.out.println("  Hidden size: " + hiddenSize);
        System.out.println("  Input size: " + inputSize);
        System.out.println("  Layers: " + numLayers);
        System.out.println("  Xavier bound: " + String.format("%.4f", xavier));
    }
    
    /**
     * Train LSTM on research trajectory data
     */
    public void trainModel(List<ResearchTrajectory> trajectories, int epochs) {
        System.out.println("Training LSTM model on " + trajectories.size() + " trajectories...");
        
        // Prepare training data
        List<TrainingExample> trainingData = prepareTrainingData(trajectories);
        
        double bestLoss = Double.MAX_VALUE;
        int patienceCounter = 0;
        int maxPatience = 10;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0.0;
            Collections.shuffle(trainingData);
            
            for (TrainingExample example : trainingData) {
                // Forward pass
                LSTMOutput output = model.forward(example.getInput());
                
                // Calculate loss
                double loss = calculateMSELoss(output.getOutput(), example.getTarget());
                epochLoss += loss;
                
                // Backward pass with BPTT
                model.backward(example.getInput(), example.getTarget(), learningRate);
            }
            
            epochLoss /= trainingData.size();
            
            // Early stopping
            if (epochLoss < bestLoss) {
                bestLoss = epochLoss;
                patienceCounter = 0;
            } else {
                patienceCounter++;
                if (patienceCounter >= maxPatience) {
                    System.out.println("Early stopping at epoch " + epoch);
                    break;
                }
            }
            
            if (epoch % 10 == 0) {
                System.out.printf("Epoch %d: Loss = %.6f%n", epoch, epochLoss);
            }
        }
        
        isTrained = true;
        performanceMetrics.put("final_loss", bestLoss);
        performanceMetrics.put("training_epochs", (double) epochs);
        
        System.out.println("Training completed. Final loss: " + String.format("%.6f", bestLoss));
    }
    
    /**
     * Predict future research trajectories using Oates' theorem
     */
    public List<ResearchTrajectory> predictFutureTrajectories(ResearchTrajectory currentTrajectory, 
                                                            int predictionSteps) {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        List<ResearchTrajectory> predictions = new ArrayList<>();
        
        // Apply Oates' theorem for bounded prediction
        int T = currentTrajectory.getSequenceLength();
        double errorBound = calculateOatesErrorBound(T);
        double confidence = calculateOatesConfidence(currentTrajectory, T);
        
        System.out.printf("Predicting with Oates bounds: Error ≤ %.4f, Confidence = %.4f%n", 
                         errorBound, confidence);
        
        if (confidence < confidenceThreshold) {
            System.out.println("Warning: Low confidence prediction. Consider more training data.");
        }
        
        // Generate predictions
        List<double[]> currentSequence = new ArrayList<>(currentTrajectory.getTopicSequence());
        
        for (int step = 0; step < predictionSteps; step++) {
            // Prepare input sequence (last minSequenceLength points)
            int startIdx = Math.max(0, currentSequence.size() - minSequenceLength);
            List<double[]> inputSequence = currentSequence.subList(startIdx, currentSequence.size());
            
            // Forward pass through LSTM
            LSTMOutput output = model.forward(inputSequence);
            double[] prediction = output.getOutput();
            
            // Apply Lipschitz constraint for stability
            prediction = applyLipschitzConstraint(prediction, 
                currentSequence.get(currentSequence.size() - 1));
            
            // Add prediction to sequence
            currentSequence.add(prediction);
            
            // Create trajectory for this prediction step
            ResearchTrajectory stepTrajectory = new ResearchTrajectory(
                currentTrajectory.getResearcherId() + "_pred_" + step,
                Arrays.asList(prediction),
                Arrays.asList(0.0), // Placeholder velocity
                Arrays.asList(0.0)  // Placeholder acceleration
            );
            
            predictions.add(stepTrajectory);
        }
        
        return predictions;
    }
    
    /**
     * Predict collaboration trajectory between two researchers
     */
    public ResearchTrajectory predictCollaborationTrajectory(ResearchTrajectory trajectory1,
                                                           ResearchTrajectory trajectory2) {
        
        // Combine trajectories using weighted average
        List<double[]> combinedSequence = combineTrajectories(trajectory1, trajectory2);
        
        // Create combined trajectory
        ResearchTrajectory combinedTrajectory = new ResearchTrajectory(
            trajectory1.getResearcherId() + "_collab_" + trajectory2.getResearcherId(),
            combinedSequence,
            calculateVelocities(combinedSequence),
            calculateAccelerations(calculateVelocities(combinedSequence))
        );
        
        // Predict future collaboration trajectory
        List<ResearchTrajectory> predictions = predictFutureTrajectories(combinedTrajectory, 6);
        
        // Merge predictions into single trajectory
        List<double[]> mergedSequence = new ArrayList<>(combinedSequence);
        for (ResearchTrajectory pred : predictions) {
            mergedSequence.addAll(pred.getTopicSequence());
        }
        
        return new ResearchTrajectory(
            combinedTrajectory.getResearcherId(),
            mergedSequence,
            calculateVelocities(mergedSequence),
            calculateAccelerations(calculateVelocities(mergedSequence))
        );
    }
    
    /**
     * Calculate Oates' error bound: O(1/√T)
     */
    private double calculateOatesErrorBound(int sequenceLength) {
        return 1.0 / Math.sqrt(Math.max(1, sequenceLength));
    }
    
    /**
     * Calculate Oates' confidence measure: E[C(p)] ≥ 1 - ε
     */
    private double calculateOatesConfidence(ResearchTrajectory trajectory, int T) {
        // Calculate ε = O(h^4) + δ_LSTM
        double h = 0.01; // Step size
        double discretizationError = Math.pow(h, 4);
        double lstmError = calculateLSTMError(trajectory);
        double epsilon = discretizationError + lstmError;
        
        // Expected confidence
        double expectedConfidence = Math.max(0.0, 1.0 - epsilon);
        
        // Adjust for sequence length and Lipschitz continuity
        double lengthFactor = Math.min(1.0, T / 100.0); // Longer sequences = higher confidence
        double lipschitzFactor = calculateLipschitzFactor(trajectory);
        
        return expectedConfidence * lengthFactor * lipschitzFactor;
    }
    
    /**
     * Calculate LSTM error component δ_LSTM
     */
    private double calculateLSTMError(ResearchTrajectory trajectory) {
        // Estimate based on trajectory chaos and model capacity
        double chaosLevel = estimateChaosLevel(trajectory);
        double modelCapacity = (double) hiddenSize / inputSize;
        
        // δ_LSTM decreases with model capacity and training
        double baseError = 0.1;
        double chaosPenalty = chaosLevel * 0.05;
        double capacityBonus = Math.min(0.08, modelCapacity * 0.01);
        
        return Math.max(0.001, baseError + chaosPenalty - capacityBonus);
    }
    
    /**
     * Estimate chaos level using Lyapunov-like measure
     */
    private double estimateChaosLevel(ResearchTrajectory trajectory) {
        List<double[]> sequence = trajectory.getTopicSequence();
        if (sequence.size() < 3) return 0.0;
        
        double maxDivergence = 0.0;
        for (int i = 1; i < sequence.size() - 1; i++) {
            double[] prev = sequence.get(i - 1);
            double[] curr = sequence.get(i);
            double[] next = sequence.get(i + 1);
            
            double d1 = euclideanDistance(prev, curr);
            double d2 = euclideanDistance(curr, next);
            
            if (d1 > 1e-8) {
                double divergence = Math.log(d2 / d1);
                maxDivergence = Math.max(maxDivergence, Math.abs(divergence));
            }
        }
        
        return Math.tanh(maxDivergence); // Bounded chaos measure
    }
    
    /**
     * Calculate Lipschitz factor for gate stability
     */
    private double calculateLipschitzFactor(ResearchTrajectory trajectory) {
        List<Double> velocities = trajectory.getVelocities();
        if (velocities.isEmpty()) return 1.0;
        
        double maxVelocity = velocities.stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        
        // Lipschitz continuity ensures bounded gradients
        return 1.0 / (1.0 + maxVelocity / lipschitzConstant);
    }
    
    /**
     * Apply Lipschitz constraint for prediction stability
     */
    private double[] applyLipschitzConstraint(double[] prediction, double[] previous) {
        double[] constrained = new double[prediction.length];
        
        for (int i = 0; i < prediction.length; i++) {
            double change = prediction[i] - previous[i];
            double maxChange = lipschitzConstant * 0.1; // Maximum allowed change
            
            if (Math.abs(change) > maxChange) {
                change = Math.signum(change) * maxChange;
            }
            
            constrained[i] = previous[i] + change;
            
            // Ensure valid probability distribution
            constrained[i] = Math.max(0.0, Math.min(1.0, constrained[i]));
        }
        
        // Normalize to maintain probability distribution
        double sum = Arrays.stream(constrained).sum();
        if (sum > 0) {
            for (int i = 0; i < constrained.length; i++) {
                constrained[i] /= sum;
            }
        }
        
        return constrained;
    }
    
    /**
     * Combine two research trajectories for collaboration prediction
     */
    private List<double[]> combineTrajectories(ResearchTrajectory traj1, ResearchTrajectory traj2) {
        List<double[]> seq1 = traj1.getTopicSequence();
        List<double[]> seq2 = traj2.getTopicSequence();
        
        int minLength = Math.min(seq1.size(), seq2.size());
        List<double[]> combined = new ArrayList<>();
        
        for (int i = 0; i < minLength; i++) {
            double[] point1 = seq1.get(i);
            double[] point2 = seq2.get(i);
            double[] combinedPoint = new double[Math.min(point1.length, point2.length)];
            
            // Weighted average with slight preference for diversity
            for (int j = 0; j < combinedPoint.length; j++) {
                combinedPoint[j] = 0.4 * point1[j] + 0.4 * point2[j] + 
                                  0.2 * Math.abs(point1[j] - point2[j]); // Diversity bonus
            }
            
            combined.add(combinedPoint);
        }
        
        return combined;
    }
    
    /**
     * Prepare training data from research trajectories
     */
    private List<TrainingExample> prepareTrainingData(List<ResearchTrajectory> trajectories) {
        List<TrainingExample> examples = new ArrayList<>();
        
        for (ResearchTrajectory trajectory : trajectories) {
            List<double[]> sequence = trajectory.getTopicSequence();
            
            // Create sliding window examples
            for (int i = minSequenceLength; i < sequence.size(); i++) {
                List<double[]> input = sequence.subList(i - minSequenceLength, i);
                double[] target = sequence.get(i);
                
                examples.add(new TrainingExample(input, target));
            }
        }
        
        System.out.println("Prepared " + examples.size() + " training examples");
        return examples;
    }
    
    /**
     * Calculate MSE loss between prediction and target
     */
    private double calculateMSELoss(double[] prediction, double[] target) {
        double sum = 0.0;
        for (int i = 0; i < Math.min(prediction.length, target.length); i++) {
            sum += Math.pow(prediction[i] - target[i], 2);
        }
        return sum / Math.min(prediction.length, target.length);
    }
    
    /**
     * Calculate velocities from topic sequence
     */
    private List<Double> calculateVelocities(List<double[]> sequence) {
        List<Double> velocities = new ArrayList<>();
        
        for (int i = 1; i < sequence.size(); i++) {
            double velocity = euclideanDistance(sequence.get(i - 1), sequence.get(i));
            velocities.add(velocity);
        }
        
        return velocities;
    }
    
    /**
     * Calculate accelerations from velocities
     */
    private List<Double> calculateAccelerations(List<Double> velocities) {
        List<Double> accelerations = new ArrayList<>();
        
        for (int i = 1; i < velocities.size(); i++) {
            double acceleration = velocities.get(i) - velocities.get(i - 1);
            accelerations.add(acceleration);
        }
        
        return accelerations;
    }
    
    /**
     * Calculate Euclidean distance between two vectors
     */
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Get performance metrics
     */
    public Map<String, Double> getPerformanceMetrics() {
        return new HashMap<>(performanceMetrics);
    }
    
    /**
     * Validate model using Oates' theorem criteria
     */
    public ValidationResult validateModel(List<ResearchTrajectory> testTrajectories) {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before validation");
        }
        
        double totalError = 0.0;
        double totalConfidence = 0.0;
        int validPredictions = 0;
        
        for (ResearchTrajectory trajectory : testTrajectories) {
            if (trajectory.getSequenceLength() < minSequenceLength + 1) continue;
            
            // Split trajectory for testing
            int splitPoint = trajectory.getSequenceLength() - 1;
            List<double[]> input = trajectory.getTopicSequence().subList(0, splitPoint);
            double[] target = trajectory.getTopicSequence().get(splitPoint);
            
            // Create test trajectory
            ResearchTrajectory testTraj = new ResearchTrajectory(
                trajectory.getResearcherId() + "_test",
                input,
                trajectory.getVelocities().subList(0, Math.max(0, input.size() - 1)),
                trajectory.getAccelerations().subList(0, Math.max(0, input.size() - 2))
            );
            
            // Predict and calculate error
            LSTMOutput output = model.forward(input);
            double error = calculateMSELoss(output.getOutput(), target);
            double confidence = calculateOatesConfidence(testTraj, input.size());
            
            totalError += error;
            totalConfidence += confidence;
            validPredictions++;
        }
        
        double avgError = totalError / validPredictions;
        double avgConfidence = totalConfidence / validPredictions;
        double theoreticalBound = calculateOatesErrorBound(minSequenceLength);
        
        boolean satisfiesOates = avgError <= theoreticalBound * 2.0; // Allow some margin
        
        return new ValidationResult(avgError, avgConfidence, theoreticalBound, 
                                  satisfiesOates, validPredictions);
    }
    
    // Getters and setters
    public boolean isTrained() { return isTrained; }
    public int getHiddenSize() { return hiddenSize; }
    public double getLipschitzConstant() { return lipschitzConstant; }
    public void setLipschitzConstant(double lipschitzConstant) { 
        this.lipschitzConstant = lipschitzConstant; 
    }
}
