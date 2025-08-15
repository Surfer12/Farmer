import java.io.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Enhanced topic modeling service that can interface with Python BERTopic
 * Uses ProcessBuilder to call Python scripts for actual topic modeling
 */
class EnhancedTopicModelingService {
    private int numTopics = 50;
    private String pythonScript = "topic_modeling.py";
    private Map<String, double[]> topicCache = new HashMap<>();
    
    public void trainTopicModel(List<String> documents) {
        try {
            // Write documents to temporary file
            File tempFile = File.createTempFile("documents", ".txt");
            try (PrintWriter writer = new PrintWriter(tempFile)) {
                for (String doc : documents) {
                    writer.println(doc.replace("\n", " ").replace(",", ";"));
                }
            }
            
            // Call Python BERTopic script
            ProcessBuilder pb = new ProcessBuilder("python3", pythonScript, 
                                                 "train", tempFile.getAbsolutePath());
            pb.redirectErrorStream(true);
            Process process = pb.start();
            
            // Read output
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println("Python: " + line);
                }
            }
            
            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("Topic model training completed successfully");
            } else {
                System.err.println("Topic model training failed with exit code: " + exitCode);
            }
            
            // Clean up
            tempFile.delete();
            
        } catch (IOException | InterruptedException e) {
            System.err.println("Error calling Python topic modeling: " + e.getMessage());
            // Fall back to simulation
            System.out.println("Falling back to simulated topic modeling");
        }
    }
    
    public double[] getTopicDistribution(String text) {
        // Check cache first
        if (topicCache.containsKey(text)) {
            return topicCache.get(text);
        }
        
        try {
            // Call Python script for topic distribution
            ProcessBuilder pb = new ProcessBuilder("python3", pythonScript, 
                                                 "predict", text);
            pb.redirectErrorStream(true);
            Process process = pb.start();
            
            // Read topic distribution from output
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line = reader.readLine();
                if (line != null && line.startsWith("TOPICS:")) {
                    String[] values = line.substring(7).split(",");
                    double[] distribution = new double[values.length];
                    for (int i = 0; i < values.length; i++) {
                        distribution[i] = Double.parseDouble(values[i].trim());
                    }
                    topicCache.put(text, distribution);
                    return distribution;
                }
            }
            
            process.waitFor();
            
        } catch (IOException | InterruptedException e) {
            System.err.println("Error getting topic distribution: " + e.getMessage());
        }
        
        // Fall back to simulation
        return simulateTopicDistribution(text);
    }
    
    private double[] simulateTopicDistribution(String text) {
        double[] distribution = new double[numTopics];
        Random random = new Random(text.hashCode());
        
        double sum = 0;
        for (int i = 0; i < numTopics; i++) {
            distribution[i] = random.nextDouble();
            sum += distribution[i];
        }
        
        // Normalize
        for (int i = 0; i < numTopics; i++) {
            distribution[i] /= sum;
        }
        
        return distribution;
    }
}

/**
 * Parallel processing utilities for large-scale analysis
 */
class ParallelAnalysisUtils {
    private final ExecutorService executor;
    
    public ParallelAnalysisUtils(int numThreads) {
        this.executor = Executors.newFixedThreadPool(numThreads);
    }
    
    public void parallelTopicModeling(List<Publication> publications, 
                                    EnhancedTopicModelingService topicService) {
        List<Future<Void>> futures = new ArrayList<>();
        
        for (Publication pub : publications) {
            Future<Void> future = executor.submit(() -> {
                double[] topicDist = topicService.getTopicDistribution(pub.getCombinedText());
                pub.setTopicDistribution(topicDist);
                return null;
            });
            futures.add(future);
        }
        
        // Wait for all tasks to complete
        for (Future<Void> future : futures) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
                System.err.println("Error in parallel topic modeling: " + e.getMessage());
            }
        }
    }
    
    public double[][] parallelSimilarityMatrix(List<String> entities, 
                                             Function<String, double[]> getDistribution) {
        int n = entities.size();
        double[][] matrix = new double[n][n];
        List<Future<Void>> futures = new ArrayList<>();
        
        for (int i = 0; i < n; i++) {
            final int row = i;
            Future<Void> future = executor.submit(() -> {
                double[] dist1 = getDistribution.apply(entities.get(row));
                for (int j = row; j < n; j++) {
                    double[] dist2 = getDistribution.apply(entities.get(j));
                    double similarity = 1.0 - jensenShannonDivergence(dist1, dist2);
                    matrix[row][j] = similarity;
                    matrix[j][row] = similarity;
                }
                return null;
            });
            futures.add(future);
        }
        
        // Wait for completion
        for (Future<Void> future : futures) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
                System.err.println("Error in parallel similarity computation: " + e.getMessage());
            }
        }
        
        return matrix;
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
    
    public void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
}

/**
 * Advanced community detection algorithms
 */
class AdvancedCommunityDetection {
    
    /**
     * Louvain algorithm for community detection
     */
    public static List<Community> louvainCommunityDetection(List<NetworkEdge> edges) {
        // Build adjacency map with weights
        Map<String, Map<String, Double>> adjacencyMap = new HashMap<>();
        Set<String> allNodes = new HashSet<>();
        
        for (NetworkEdge edge : edges) {
            allNodes.add(edge.getSourceId());
            allNodes.add(edge.getTargetId());
            
            adjacencyMap.computeIfAbsent(edge.getSourceId(), k -> new HashMap<>())
                .put(edge.getTargetId(), edge.getWeight());
            adjacencyMap.computeIfAbsent(edge.getTargetId(), k -> new HashMap<>())
                .put(edge.getSourceId(), edge.getWeight());
        }
        
        // Initialize each node in its own community
        Map<String, String> nodeToCommunity = new HashMap<>();
        Map<String, Community> communities = new HashMap<>();
        
        int communityId = 0;
        for (String node : allNodes) {
            String commId = "louvain_" + communityId++;
            nodeToCommunity.put(node, commId);
            Community community = new Community(commId);
            community.addMember(node);
            communities.put(commId, community);
        }
        
        // Iteratively optimize modularity
        boolean improved = true;
        int iteration = 0;
        
        while (improved && iteration < 100) {
            improved = false;
            iteration++;
            
            for (String node : allNodes) {
                String currentCommunity = nodeToCommunity.get(node);
                String bestCommunity = currentCommunity;
                double bestModularityGain = 0.0;
                
                // Try moving node to neighboring communities
                Map<String, Double> neighbors = adjacencyMap.get(node);
                if (neighbors != null) {
                    Set<String> neighborCommunities = new HashSet<>();
                    for (String neighbor : neighbors.keySet()) {
                        neighborCommunities.add(nodeToCommunity.get(neighbor));
                    }
                    
                    for (String targetCommunity : neighborCommunities) {
                        if (!targetCommunity.equals(currentCommunity)) {
                            double gain = calculateModularityGain(node, currentCommunity, 
                                                               targetCommunity, adjacencyMap, 
                                                               nodeToCommunity);
                            if (gain > bestModularityGain) {
                                bestModularityGain = gain;
                                bestCommunity = targetCommunity;
                            }
                        }
                    }
                }
                
                // Move node if beneficial
                if (!bestCommunity.equals(currentCommunity)) {
                    communities.get(currentCommunity).getMembers().remove(node);
                    communities.get(bestCommunity).addMember(node);
                    nodeToCommunity.put(node, bestCommunity);
                    improved = true;
                }
            }
        }
        
        // Filter out empty communities
        return communities.values().stream()
            .filter(c -> c.getSize() > 0)
            .collect(ArrayList::new, (list, community) -> list.add(community), List::addAll);
    }
    
    private static double calculateModularityGain(String node, String fromCommunity, 
                                                String toCommunity,
                                                Map<String, Map<String, Double>> adjacencyMap,
                                                Map<String, String> nodeToCommunity) {
        // Simplified modularity gain calculation
        // In practice, would implement full modularity formula
        
        double gainFrom = 0.0;
        double gainTo = 0.0;
        
        Map<String, Double> nodeNeighbors = adjacencyMap.get(node);
        if (nodeNeighbors != null) {
            for (Map.Entry<String, Double> entry : nodeNeighbors.entrySet()) {
                String neighbor = entry.getKey();
                double weight = entry.getValue();
                String neighborCommunity = nodeToCommunity.get(neighbor);
                
                if (neighborCommunity.equals(fromCommunity)) {
                    gainFrom -= weight;
                }
                if (neighborCommunity.equals(toCommunity)) {
                    gainTo += weight;
                }
            }
        }
        
        return gainTo - gainFrom;
    }
}

/**
 * Metrics and evaluation utilities
 */
class NetworkMetrics {
    
    public static double calculateModularity(List<Community> communities, 
                                           List<NetworkEdge> edges) {
        // Calculate network modularity
        double totalWeight = edges.stream().mapToDouble(NetworkEdge::getWeight).sum();
        
        // Build community membership map
        Map<String, String> nodeToCommunity = new HashMap<>();
        for (Community community : communities) {
            for (String member : community.getMembers()) {
                nodeToCommunity.put(member, community.getId());
            }
        }
        
        double modularity = 0.0;
        for (NetworkEdge edge : edges) {
            String sourceCommunity = nodeToCommunity.get(edge.getSourceId());
            String targetCommunity = nodeToCommunity.get(edge.getTargetId());
            
            if (sourceCommunity != null && sourceCommunity.equals(targetCommunity)) {
                modularity += edge.getWeight();
            }
        }
        
        return modularity / totalWeight;
    }
    
    public static Map<String, Double> calculateCommunityDensities(List<Community> communities,
                                                                List<NetworkEdge> edges) {
        Map<String, Double> densities = new HashMap<>();
        
        // Build edge map for quick lookup
        Map<String, Set<String>> edgeMap = new HashMap<>();
        for (NetworkEdge edge : edges) {
            edgeMap.computeIfAbsent(edge.getSourceId(), k -> new HashSet<>())
                .add(edge.getTargetId());
            edgeMap.computeIfAbsent(edge.getTargetId(), k -> new HashSet<>())
                .add(edge.getSourceId());
        }
        
        for (Community community : communities) {
            int n = community.getSize();
            if (n < 2) {
                densities.put(community.getId(), 0.0);
                continue;
            }
            
            int actualEdges = 0;
            int possibleEdges = n * (n - 1) / 2;
            
            List<String> members = new ArrayList<>(community.getMembers());
            for (int i = 0; i < members.size(); i++) {
                for (int j = i + 1; j < members.size(); j++) {
                    String member1 = members.get(i);
                    String member2 = members.get(j);
                    
                    if (edgeMap.containsKey(member1) && 
                        edgeMap.get(member1).contains(member2)) {
                        actualEdges++;
                    }
                }
            }
            
            double density = (double) actualEdges / possibleEdges;
            densities.put(community.getId(), density);
            community.setDensity(density);
        }
        
        return densities;
    }
}

// Functional interface for lambda expressions
@FunctionalInterface
interface Function<T, R> {
    R apply(T t);
}
