// Add these methods to the AcademicNetworkAnalyzer class:

/**
 * Get researcher by ID
 */
public Researcher getResearcher(String researcherId) {
    return researcherMap.get(researcherId);
}

/**
 * Get all researchers
 */
public List<Researcher> getResearchers() {
    return new ArrayList<>(researchers);
}

/**
 * Get all communities
 */
public List<Community> getCommunities() {
    return new ArrayList<>(communities);
}

/**
 * Get all network edges
 */
public List<NetworkEdge> getNetworkEdges() {
    return new ArrayList<>(networkEdges);
}

/**
 * Get similarity matrix
 */
public double[][] getSimilarityMatrix() {
    return similarityMatrix;
}

// Missing PredictionResult class
class PredictionResult {
    private String predictionId;
    private double accuracy;
    private double confidence;
    private long timestamp;
    
    public PredictionResult(String predictionId, double accuracy, double confidence) {
        this.predictionId = predictionId;
        this.accuracy = accuracy;
        this.confidence = confidence;
        this.timestamp = System.currentTimeMillis();
    }
    
    // Getters
    public String getPredictionId() { return predictionId; }
    public double getAccuracy() { return accuracy; }
    public double getConfidence() { return confidence; }
    public long getTimestamp() { return timestamp; }
}
