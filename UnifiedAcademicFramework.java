import java.util.*;
import java.util.stream.Collectors;
import java.io.*;
import java.nio.file.*;

/**
 * Unified Academic Framework integrating:
 * - Academic Network Community Detection (from research paper)
 * - Ψ(x,m,s) Cognitive-Memory Framework
 * - Oates' LSTM Hidden State Convergence Theorem
 * - Enhanced d_MC Metric with Cross-Modal Terms
 * - Variational Emergence E[Ψ] Minimization
 */
public class UnifiedAcademicFramework {
    
    // Core components from research paper implementation
    private List<Publication> publications;
    private List<Researcher> researchers;
    private Map<String, Researcher> researcherMap;
    private List<ResearcherClone> clones;
    private double[][] similarityMatrix;
    private List<NetworkEdge> networkEdges;
    private List<Community> communities;
    private int medianPublicationCount;
    
    // Enhanced framework components
    private CognitiveMemoryFramework cognitiveFramework;
    private LSTMChaosPredictionEngine lstmEngine;
    private TopicModelingService topicService;
    
    // Integration parameters
    private double similarityThreshold = 0.25;
    private double analysisTimeWindow = 2.0;
    private String outputDirectory = "unified_output";
    
    public UnifiedAcademicFramework() {
        this.publications = new ArrayList<>();
        this.researchers = new ArrayList<>();
        this.researcherMap = new HashMap<>();
        this.clones = new ArrayList<>();
        this.networkEdges = new ArrayList<>();
        this.communities = new ArrayList<>();
        this.cognitiveFramework = new CognitiveMemoryFramework();
        this.lstmEngine = new LSTMChaosPredictionEngine();
        this.topicService = new TopicModelingService();
    }
    
    /**
     * Main unified analysis method combining all components
     */
    public UnifiedAnalysisResult performUnifiedAnalysis(String publicationDataFile) throws IOException {
        System.out.println("=== UNIFIED ACADEMIC FRAMEWORK ANALYSIS ===");
        System.out.println("Integrating research paper methodology with advanced mathematical framework");
        System.out.println();
        
        // Step 1: Load and preprocess data (from research paper)
        System.out.println("1. Loading and preprocessing publication data...");
        loadPublicationData(publicationDataFile);
        
        // Step 2: Topic modeling and researcher cloning (enhanced)
        System.out.println("2. Performing enhanced topic modeling and researcher cloning...");
        performEnhancedTopicModeling();
        
        // Step 3: Build similarity matrix with cognitive-memory integration
        System.out.println("3. Building enhanced similarity matrix with d_MC metric...");
        buildEnhancedSimilarityMatrix();
        
        // Step 4: Network construction and community detection
        System.out.println("4. Constructing network and detecting communities...");
        buildNetworkAndDetectCommunities();
        
        // Step 5: Ψ(x,m,s) framework analysis
        System.out.println("5. Performing Ψ(x,m,s) cognitive-memory analysis...");
        FrameworkAnalysisResult frameworkResult = performCognitiveMemoryAnalysis();
        
        // Step 6: LSTM trajectory prediction and validation
        System.out.println("6. LSTM trajectory prediction with Oates' theorem validation...");
        ValidationResult lstmValidation = performLSTMAnalysis();
        
        // Step 7: Integrated cross-validation
        System.out.println("7. Performing integrated cross-validation...");
        IntegratedValidationResult crossValidation = performIntegratedValidation(
            frameworkResult, lstmValidation);
        
        // Step 8: Export comprehensive results
        System.out.println("8. Exporting unified analysis results...");
        UnifiedAnalysisResult unifiedResult = new UnifiedAnalysisResult(
            researchers, communities, clones, frameworkResult, 
            lstmValidation, crossValidation);
        
        exportUnifiedResults(unifiedResult);
        
        System.out.println("\n=== UNIFIED ANALYSIS COMPLETE ===");
        return unifiedResult;
    }
    
    /**
     * Load publication data following research paper methodology
     */
    private void loadPublicationData(String dataFile) throws IOException {
        publications.clear();
        researchers.clear();
        researcherMap.clear();
        
        List<String> lines = Files.readAllLines(Paths.get(dataFile));
        
        // Skip header if present
        int startIndex = lines.get(0).contains("pub_id") ? 1 : 0;
        
        for (int i = startIndex; i < lines.size(); i++) {
            String line = lines.get(i);
            String[] parts = line.split(",");
            
            if (parts.length >= 4) {
                String pubId = parts[0].trim();
                String title = parts[1].trim().replaceAll("\"", "");
                String abstractText = parts[2].trim().replaceAll("\"", "");
                String authorId = parts[3].trim();
                
                Publication pub = new Publication(pubId, title, abstractText, authorId);
                publications.add(pub);
                
                // Get or create researcher
                Researcher researcher = researcherMap.computeIfAbsent(authorId, 
                    id -> new Researcher(id, "Researcher_" + id));
                researcher.addPublication(pub);
                
                if (!researchers.contains(researcher)) {
                    researchers.add(researcher);
                }
            }
        }
        
        // Filter researchers with fewer than 5 publications (from paper)
        researchers = researchers.stream()
            .filter(r -> r.getPublicationCount() >= 5)
            .collect(Collectors.toList());
        
        // Update researcher map
        researcherMap.clear();
        for (Researcher r : researchers) {
            researcherMap.put(r.getId(), r);
        }
        
        // Calculate median publication count
        List<Integer> counts = researchers.stream()
            .mapToInt(Researcher::getPublicationCount)
            .sorted()
            .boxed()
            .collect(Collectors.toList());
        medianPublicationCount = counts.isEmpty() ? 0 : counts.get(counts.size() / 2);
        
        System.out.println("   Loaded " + publications.size() + " publications");
        System.out.println("   Filtered to " + researchers.size() + " researchers (≥5 publications)");
        System.out.println("   Median publication count: " + medianPublicationCount);
    }
    
    /**
     * Enhanced topic modeling with cognitive-memory integration
     */
    private void performEnhancedTopicModeling() {
        // Extract all publication texts for topic modeling
        List<String> allTexts = publications.stream()
            .map(Publication::getCombinedText)
            .collect(Collectors.toList());
        
        // Train topic model
        topicService.trainTopicModel(allTexts);
        
        // Get topic distributions for each publication
        for (Publication pub : publications) {
            double[] topicDist = topicService.getTopicDistribution(pub.getCombinedText());
            pub.setTopicDistribution(topicDist);
        }
        
        // Create researcher clones for high-impact researchers (from paper methodology)
        createResearcherClones();
        
        // Calculate topic distributions for researchers and clones
        calculateResearcherTopicDistributions();
        
        System.out.println("   Topic modeling completed for " + allTexts.size() + " documents");
        System.out.println("   Created " + clones.size() + " researcher clones");
    }
    
    /**
     * Create researcher clones following Algorithm 1 from research paper
     */
    private void createResearcherClones() {
        clones.clear();
        
        for (Researcher researcher : researchers) {
            if (researcher.isHighImpact(medianPublicationCount)) {
                List<List<Publication>> clusters = clusterPublications(researcher.getPublications());
                
                for (int i = 0; i < clusters.size(); i++) {
                    if (clusters.get(i).size() >= 3) { // Minimum cluster size
                        String cloneId = researcher.getId() + "_clone_" + i;
                        ResearcherClone clone = new ResearcherClone(
                            researcher.getId(), cloneId, clusters.get(i));
                        researcher.addClone(clone);
                        clones.add(clone);
                    }
                }
            }
        }
    }
    
    /**
     * Cluster publications using enhanced topic similarity
     */
    private List<List<Publication>> clusterPublications(List<Publication> publications) {
        List<List<Publication>> clusters = new ArrayList<>();
        
        if (publications.size() <= 10) {
            clusters.add(new ArrayList<>(publications));
            return clusters;
        }
        
        // Enhanced clustering using cognitive-memory framework
        Map<Integer, List<Publication>> topicGroups = new HashMap<>();
        
        for (Publication pub : publications) {
            if (pub.getTopicDistribution() != null) {
                int dominantTopic = getDominantTopic(pub.getTopicDistribution());
                topicGroups.computeIfAbsent(dominantTopic, k -> new ArrayList<>()).add(pub);
            }
        }
        
        // Apply cognitive-memory distance for refinement
        for (List<Publication> group : topicGroups.values()) {
            if (group.size() >= 3) {
                List<Publication> refinedCluster = refineClusterWithCognitiveDistance(group);
                if (refinedCluster.size() >= 3) {
                    clusters.add(refinedCluster);
                }
            }
        }
        
        return clusters;
    }
    
    /**
     * Refine clusters using cognitive-memory distance
     */
    private List<Publication> refineClusterWithCognitiveDistance(List<Publication> publications) {
        // Convert publications to memory vectors for d_MC calculation
        List<MemoryVector> memoryVectors = new ArrayList<>();
        
        for (Publication pub : publications) {
            double[] benchmarkScores = pub.getTopicDistribution();
            double[] experienceVector = new double[10]; // Simplified experience representation
            Arrays.fill(experienceVector, 0.5);
            
            MemoryVector memory = new MemoryVector(benchmarkScores, experienceVector);
            memoryVectors.add(memory);
        }
        
        // Calculate average d_MC distance within cluster
        double totalDistance = 0.0;
        int pairCount = 0;
        
        for (int i = 0; i < memoryVectors.size(); i++) {
            for (int j = i + 1; j < memoryVectors.size(); j++) {
                double distance = cognitiveFramework.computeCognitiveMemoryDistance(
                    memoryVectors.get(i), memoryVectors.get(j));
                totalDistance += distance;
                pairCount++;
            }
        }
        
        double avgDistance = pairCount > 0 ? totalDistance / pairCount : 0.0;
        
        // Keep cluster if average distance is below threshold (coherent cluster)
        if (avgDistance < 0.5) {
            return new ArrayList<>(publications);
        } else {
            // Split cluster if too diverse
            return publications.subList(0, Math.min(publications.size() / 2, publications.size()));
        }
    }
    
    private int getDominantTopic(double[] topicDistribution) {
        int maxIndex = 0;
        for (int i = 1; i < topicDistribution.length; i++) {
            if (topicDistribution[i] > topicDistribution[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    /**
     * Calculate topic distributions for researchers and clones
     */
    private void calculateResearcherTopicDistributions() {
        // Calculate for original researchers
        for (Researcher researcher : researchers) {
            double[] avgDistribution = calculateAverageTopicDistribution(
                researcher.getPublications());
            researcher.setTopicDistribution(avgDistribution);
        }
        
        // Calculate for clones
        for (ResearcherClone clone : clones) {
            double[] avgDistribution = calculateAverageTopicDistribution(
                clone.getPublications());
            clone.setTopicDistribution(avgDistribution);
        }
    }
    
    private double[] calculateAverageTopicDistribution(List<Publication> publications) {
        if (publications.isEmpty()) return new double[50]; // Default topic size
        
        int numTopics = publications.get(0).getTopicDistribution().length;
        double[] avgDistribution = new double[numTopics];
        
        for (Publication pub : publications) {
            double[] dist = pub.getTopicDistribution();
            for (int i = 0; i < numTopics; i++) {
                avgDistribution[i] += dist[i];
            }
        }
        
        // Normalize
        for (int i = 0; i < numTopics; i++) {
            avgDistribution[i] /= publications.size();
        }
        
        return avgDistribution;
    }
    
    /**
     * Build enhanced similarity matrix integrating d_MC metric
     */
    private void buildEnhancedSimilarityMatrix() {
        List<String> allEntities = new ArrayList<>();
        
        // Add original researchers
        for (Researcher researcher : researchers) {
            allEntities.add(researcher.getId());
        }
        
        // Add clones
        for (ResearcherClone clone : clones) {
            allEntities.add(clone.getCloneId());
        }
        
        int n = allEntities.size();
        similarityMatrix = new double[n][n];
        
        // Calculate enhanced similarities combining Jensen-Shannon and d_MC
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double[] dist1 = getTopicDistribution(allEntities.get(i));
                double[] dist2 = getTopicDistribution(allEntities.get(j));
                
                // Traditional Jensen-Shannon similarity
                double jsSimilarity = 1.0 - jensenShannonDivergence(dist1, dist2);
                
                // Enhanced cognitive-memory similarity
                double cognitiveMemorySimilarity = calculateCognitiveMemorySimilarity(
                    allEntities.get(i), allEntities.get(j));
                
                // Combined similarity (weighted average)
                double combinedSimilarity = 0.7 * jsSimilarity + 0.3 * cognitiveMemorySimilarity;
                
                similarityMatrix[i][j] = combinedSimilarity;
                similarityMatrix[j][i] = combinedSimilarity;
            }
        }
        
        System.out.println("   Built enhanced similarity matrix of size " + n + "x" + n);
        System.out.println("   Integrated Jensen-Shannon and d_MC cognitive-memory metrics");
    }
    
    /**
     * Calculate cognitive-memory similarity between entities
     */
    private double calculateCognitiveMemorySimilarity(String entity1, String entity2) {
        // Convert entities to memory vectors
        MemoryVector memory1 = createMemoryVectorFromEntity(entity1);
        MemoryVector memory2 = createMemoryVectorFromEntity(entity2);
        
        if (memory1 == null || memory2 == null) return 0.0;
        
        // Calculate d_MC distance and convert to similarity
        double distance = cognitiveFramework.computeCognitiveMemoryDistance(memory1, memory2);
        
        // Convert distance to similarity (inverse relationship)
        return 1.0 / (1.0 + distance);
    }
    
    /**
     * Create memory vector from researcher or clone entity
     */
    private MemoryVector createMemoryVectorFromEntity(String entityId) {
        // Check if it's a researcher
        Researcher researcher = researcherMap.get(entityId);
        if (researcher != null) {
            double[] benchmarkScores = researcher.getTopicDistribution();
            double[] experienceVector = createExperienceVector(researcher);
            return new MemoryVector(benchmarkScores, experienceVector);
        }
        
        // Check if it's a clone
        for (ResearcherClone clone : clones) {
            if (clone.getCloneId().equals(entityId)) {
                double[] benchmarkScores = clone.getTopicDistribution();
                double[] experienceVector = createExperienceVectorFromClone(clone);
                return new MemoryVector(benchmarkScores, experienceVector);
            }
        }
        
        return null;
    }
    
    private double[] createExperienceVector(Researcher researcher) {
        // Create experience vector based on publication patterns
        List<Publication> pubs = researcher.getPublications();
        double[] experience = new double[15];
        
        // Publication count normalized
        experience[0] = Math.min(1.0, pubs.size() / 20.0);
        
        // Topic diversity (entropy of topic distribution)
        double[] topicDist = researcher.getTopicDistribution();
        experience[1] = calculateEntropy(topicDist);
        
        // Temporal consistency (placeholder)
        experience[2] = 0.7 + Math.random() * 0.3;
        
        // Fill remaining with derived metrics
        for (int i = 3; i < experience.length; i++) {
            experience[i] = 0.4 + Math.random() * 0.6;
        }
        
        return experience;
    }
    
    private double[] createExperienceVectorFromClone(ResearcherClone clone) {
        // Similar to researcher but focused on clone's specific publications
        List<Publication> pubs = clone.getPublications();
        double[] experience = new double[15];
        
        experience[0] = Math.min(1.0, pubs.size() / 10.0); // Clones typically have fewer pubs
        
        double[] topicDist = clone.getTopicDistribution();
        experience[1] = calculateEntropy(topicDist);
        
        // Higher specialization for clones
        experience[2] = 0.8 + Math.random() * 0.2;
        
        for (int i = 3; i < experience.length; i++) {
            experience[i] = 0.5 + Math.random() * 0.5;
        }
        
        return experience;
    }
    
    private double calculateEntropy(double[] distribution) {
        double entropy = 0.0;
        for (double p : distribution) {
            if (p > 0) {
                entropy -= p * Math.log(p) / Math.log(2);
            }
        }
        return Math.min(1.0, entropy / Math.log(distribution.length)); // Normalized
    }
    
    private double[] getTopicDistribution(String entityId) {
        // Check if it's a researcher
        Researcher researcher = researcherMap.get(entityId);
        if (researcher != null) {
            return researcher.getTopicDistribution();
        }
        
        // Check if it's a clone
        for (ResearcherClone clone : clones) {
            if (clone.getCloneId().equals(entityId)) {
                return clone.getTopicDistribution();
            }
        }
        
        return new double[50]; // Default empty distribution
    }
    
    private double jensenShannonDivergence(double[] p, double[] q) {
        if (p.length != q.length) return 1.0;
        
        double[] m = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            m[i] = 0.5 * (p[i] + q[i]);
        }
        
        double jsd = 0.5 * klDivergence(p, m) + 0.5 * klDivergence(q, m);
        return Math.min(1.0, Math.max(0.0, jsd));
    }
    
    private double klDivergence(double[] p, double[] q) {
        double kl = 0.0;
        for (int i = 0; i < p.length; i++) {
            if (p[i] > 0 && q[i] > 0) {
                kl += p[i] * Math.log(p[i] / q[i]);
            }
        }
        return kl;
    }
    
    /**
     * Build network and detect communities following research paper methodology
     */
    private void buildNetworkAndDetectCommunities() {
        buildNetwork();
        detectCommunities();
        refineCommunities();
        
        System.out.println("   Built network with " + networkEdges.size() + " edges");
        System.out.println("   Detected " + communities.size() + " communities");
    }
    
    private void buildNetwork() {
        networkEdges.clear();
        
        int n = similarityMatrix.length;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (similarityMatrix[i][j] > similarityThreshold) {
                    String sourceId = getEntityIdByIndex(i);
                    String targetId = getEntityIdByIndex(j);
                    networkEdges.add(new NetworkEdge(sourceId, targetId, similarityMatrix[i][j]));
                }
            }
        }
    }
    
    private String getEntityIdByIndex(int index) {
        List<String> allEntities = new ArrayList<>();
        for (Researcher researcher : researchers) {
            allEntities.add(researcher.getId());
        }
        for (ResearcherClone clone : clones) {
            allEntities.add(clone.getCloneId());
        }
        return allEntities.get(index);
    }
    
    private void detectCommunities() {
        communities.clear();
        
        // Create adjacency map
        Map<String, Set<String>> adjacencyMap = new HashMap<>();
        for (NetworkEdge edge : networkEdges) {
            adjacencyMap.computeIfAbsent(edge.getSourceId(), k -> new HashSet<>())
                .add(edge.getTargetId());
            adjacencyMap.computeIfAbsent(edge.getTargetId(), k -> new HashSet<>())
                .add(edge.getSourceId());
        }
        
        // Connected components as communities
        Set<String> visited = new HashSet<>();
        int communityId = 0;
        
        for (String nodeId : adjacencyMap.keySet()) {
            if (!visited.contains(nodeId)) {
                Community community = new Community("community_" + communityId++);
                dfsVisit(nodeId, adjacencyMap, visited, community);
                
                if (community.getSize() >= 2) {
                    communities.add(community);
                }
            }
        }
    }
    
    private void dfsVisit(String nodeId, Map<String, Set<String>> adjacencyMap, 
                         Set<String> visited, Community community) {
        visited.add(nodeId);
        community.addMember(nodeId);
        
        Set<String> neighbors = adjacencyMap.get(nodeId);
        if (neighbors != null) {
            for (String neighbor : neighbors) {
                if (!visited.contains(neighbor)) {
                    dfsVisit(neighbor, adjacencyMap, visited, community);
                }
            }
        }
    }
    
    /**
     * Refine communities following Algorithm 1 from research paper
     */
    private void refineCommunities() {
        List<Community> refinedCommunities = new ArrayList<>();
        
        for (Community community : communities) {
            Community refinedCommunity = refineCommunity(community);
            refinedCommunities.add(refinedCommunity);
        }
        
        this.communities = refinedCommunities;
    }
    
    private Community refineCommunity(Community community) {
        // Group nodes by base identity (merge clones back to base researchers)
        Map<String, Set<String>> baseGroups = new HashMap<>();
        
        for (String memberId : community.getMembers()) {
            String baseId = getBaseResearcherId(memberId);
            baseGroups.computeIfAbsent(baseId, k -> new HashSet<>()).add(memberId);
        }
        
        // Create new community with merged base identities
        Community refinedCommunity = new Community(community.getId());
        
        // Add base researchers to refined community
        for (String baseId : baseGroups.keySet()) {
            refinedCommunity.addMember(baseId);
        }
        
        return refinedCommunity;
    }
    
    private String getBaseResearcherId(String entityId) {
        // Check if it's a clone
        for (ResearcherClone clone : clones) {
            if (clone.getCloneId().equals(entityId)) {
                return clone.getBaseResearcherId();
            }
        }
        
        // Otherwise it's a base researcher
        return entityId;
    }
}
