/**
 * Extensions to AcademicNetworkAnalyzer to support enhanced research matching
 * These methods should be added to the main AcademicNetworkAnalyzer class
 */

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

/**
 * Calculate researcher similarity
 */
public double calculateResearcherSimilarity(String researcher1, String researcher2) {
    Researcher r1 = researcherMap.get(researcher1);
    Researcher r2 = researcherMap.get(researcher2);
    
    if (r1 == null || r2 == null) return 0.0;
    
    double[] dist1 = r1.getTopicDistribution();
    double[] dist2 = r2.getTopicDistribution();
    
    if (dist1 == null || dist2 == null) return 0.0;
    
    return 1.0 - jensenShannonDivergence(dist1, dist2);
}

/**
 * Get researchers in same community
 */
public List<String> getCommunityMembers(String researcherId) {
    for (Community community : communities) {
        if (community.containsMember(researcherId)) {
            return new ArrayList<>(community.getMembers());
        }
    }
    return new ArrayList<>();
}

/**
 * Calculate network density
 */
public double calculateNetworkDensity() {
    int n = researchers.size();
    if (n < 2) return 0.0;
    
    int possibleEdges = n * (n - 1) / 2;
    return (double) networkEdges.size() / possibleEdges;
}

/**
 * Get researcher publication timeline
 */
public List<Publication> getResearcherTimeline(String researcherId) {
    Researcher researcher = researcherMap.get(researcherId);
    if (researcher == null) return new ArrayList<>();
    
    List<Publication> timeline = new ArrayList<>(researcher.getPublications());
    timeline.sort((a, b) -> a.getId().compareTo(b.getId())); // Assume chronological by ID
    return timeline;
}
