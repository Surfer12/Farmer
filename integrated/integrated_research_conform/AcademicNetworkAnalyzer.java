import java.util.*;
import java.util.stream.Collectors;
import java.io.*;
import java.nio.file.*;

// Core data structures
class Publication {
    private String id;
    private String title;
    private String abstractText;
    private String authorId;
    private double[] topicDistribution;
    
    public Publication(String id, String title, String abstractText, String authorId) {
        this.id = id;
        this.title = title;
        this.abstractText = abstractText;
        this.authorId = authorId;
    }
    
    // Getters and setters
    public String getId() { return id; }
    public String getTitle() { return title; }
    public String getAbstractText() { return abstractText; }
    public String getAuthorId() { return authorId; }
    public double[] getTopicDistribution() { return topicDistribution; }
    public void setTopicDistribution(double[] topicDistribution) { 
        this.topicDistribution = topicDistribution; 
    }
    
    public String getCombinedText() {
        return title + " " + abstractText;
    }
}

class Researcher {
    private String id;
    private String name;
    private List<Publication> publications;
    private double[] topicDistribution;
    private List<ResearcherClone> clones;
    
    public Researcher(String id, String name) {
        this.id = id;
        this.name = name;
        this.publications = new ArrayList<>();
        this.clones = new ArrayList<>();
    }
    
    public void addPublication(Publication publication) {
        publications.add(publication);
    }
    
    public int getPublicationCount() {
        return publications.size();
    }
    
    public boolean isHighImpact(int medianCount) {
        return publications.size() > 1.5 * medianCount;
    }
    
    // Getters and setters
    public String getId() { return id; }
    public String getName() { return name; }
    public List<Publication> getPublications() { return publications; }
    public double[] getTopicDistribution() { return topicDistribution; }
    public void setTopicDistribution(double[] topicDistribution) { 
        this.topicDistribution = topicDistribution; 
    }
    public List<ResearcherClone> getClones() { return clones; }
    public void addClone(ResearcherClone clone) { clones.add(clone); }
}

class ResearcherClone {
    private String baseResearcherId;
    private String cloneId;
    private List<Publication> publications;
    private double[] topicDistribution;
    
    public ResearcherClone(String baseResearcherId, String cloneId, List<Publication> publications) {
        this.baseResearcherId = baseResearcherId;
        this.cloneId = cloneId;
        this.publications = new ArrayList<>(publications);
    }
    
    // Getters and setters
    public String getBaseResearcherId() { return baseResearcherId; }
    public String getCloneId() { return cloneId; }
    public List<Publication> getPublications() { return publications; }
    public double[] getTopicDistribution() { return topicDistribution; }
    public void setTopicDistribution(double[] topicDistribution) { 
        this.topicDistribution = topicDistribution; 
    }
}

class NetworkEdge {
    private String sourceId;
    private String targetId;
    private double weight;
    
    public NetworkEdge(String sourceId, String targetId, double weight) {
        this.sourceId = sourceId;
        this.targetId = targetId;
        this.weight = weight;
    }
    
    // Getters
    public String getSourceId() { return sourceId; }
    public String getTargetId() { return targetId; }
    public double getWeight() { return weight; }
    public void setWeight(double weight) { this.weight = weight; }
}

class Community {
    private String id;
    private Set<String> members;
    private double density;
    
    public Community(String id) {
        this.id = id;
        this.members = new HashSet<>();
    }
    
    public void addMember(String researcherId) {
        members.add(researcherId);
    }
    
    public boolean containsMember(String researcherId) {
        return members.contains(researcherId);
    }
    
    // Getters
    public String getId() { return id; }
    public Set<String> getMembers() { return members; }
    public int getSize() { return members.size(); }
    public double getDensity() { return density; }
    public void setDensity(double density) { this.density = density; }
}

// Main implementation class
public class AcademicNetworkAnalyzer {
    private List<Researcher> researchers;
    private Map<String, Researcher> researcherMap;
    private List<ResearcherClone> clones;
    private double[][] similarityMatrix;
    private List<NetworkEdge> networkEdges;
    private List<Community> communities;
    private int medianPublicationCount;
    
    // Topic modeling interface - would integrate with Python BERTopic
    private TopicModelingService topicService;
    
    public AcademicNetworkAnalyzer() {
        this.researchers = new ArrayList<>();
        this.researcherMap = new HashMap<>();
        this.clones = new ArrayList<>();
        this.networkEdges = new ArrayList<>();
        this.communities = new ArrayList<>();
        this.topicService = new TopicModelingService();
    }
    
    // Step 1: Load and preprocess data
    public void loadResearchData(String dataFile) throws IOException {
        // Load publication data (would typically parse JSON/CSV)
        List<String> lines = Files.readAllLines(Paths.get(dataFile));
        
        for (String line : lines) {
            // Parse publication data
            String[] parts = line.split(",");
            if (parts.length >= 4) {
                String pubId = parts[0];
                String title = parts[1];
                String abstractText = parts[2];
                String authorId = parts[3];
                
                Publication pub = new Publication(pubId, title, abstractText, authorId);
                
                // Get or create researcher
                Researcher researcher = researcherMap.computeIfAbsent(authorId, 
                    id -> new Researcher(id, "Researcher_" + id));
                researcher.addPublication(pub);
                
                if (!researchers.contains(researcher)) {
                    researchers.add(researcher);
                }
            }
        }
        
        // Filter researchers with fewer than 5 publications
        researchers = researchers.stream()
            .filter(r -> r.getPublicationCount() >= 5)
            .collect(Collectors.toList());
        
        // Calculate median publication count
        List<Integer> counts = researchers.stream()
            .mapToInt(Researcher::getPublicationCount)
            .sorted()
            .boxed()
            .collect(Collectors.toList());
        medianPublicationCount = counts.get(counts.size() / 2);
        
        System.out.println("Loaded " + researchers.size() + " researchers");
        System.out.println("Median publication count: " + medianPublicationCount);
    }
    
    // Step 2: Topic modeling and cloning
    public void performTopicModeling() {
        // Extract all publication texts
        List<String> allTexts = researchers.stream()
            .flatMap(r -> r.getPublications().stream())
            .map(Publication::getCombinedText)
            .collect(Collectors.toList());
        
        // Perform topic modeling (interface with Python BERTopic)
        topicService.trainTopicModel(allTexts);
        
        // Get topic distributions for each publication
        for (Researcher researcher : researchers) {
            for (Publication pub : researcher.getPublications()) {
                double[] topicDist = topicService.getTopicDistribution(pub.getCombinedText());
                pub.setTopicDistribution(topicDist);
            }
        }
        
        // Clone high-impact researchers
        createResearcherClones();
        
        // Calculate topic distributions for researchers and clones
        calculateResearcherTopicDistributions();
    }
    
    private void createResearcherClones() {
        for (Researcher researcher : researchers) {
            if (researcher.isHighImpact(medianPublicationCount)) {
                List<List<Publication>> clusters = clusterPublications(researcher.getPublications());
                
                for (int i = 0; i < clusters.size(); i++) {
                    if (clusters.get(i).size() > 0) {
                        String cloneId = researcher.getId() + "_clone_" + i;
                        ResearcherClone clone = new ResearcherClone(
                            researcher.getId(), cloneId, clusters.get(i));
                        researcher.addClone(clone);
                        clones.add(clone);
                    }
                }
            }
        }
        
        System.out.println("Created " + clones.size() + " researcher clones");
    }
    
    private List<List<Publication>> clusterPublications(List<Publication> publications) {
        // Simplified clustering based on topic similarity
        // In practice, would use HDBSCAN or similar algorithm
        List<List<Publication>> clusters = new ArrayList<>();
        
        if (publications.size() <= 10) {
            clusters.add(new ArrayList<>(publications));
            return clusters;
        }
        
        // Simple k-means style clustering based on dominant topics
        Map<Integer, List<Publication>> topicGroups = new HashMap<>();
        
        for (Publication pub : publications) {
            if (pub.getTopicDistribution() != null) {
                int dominantTopic = getDominantTopic(pub.getTopicDistribution());
                topicGroups.computeIfAbsent(dominantTopic, k -> new ArrayList<>()).add(pub);
            }
        }
        
        // Filter out small clusters
        for (List<Publication> group : topicGroups.values()) {
            if (group.size() >= 3) {
                clusters.add(group);
            }
        }
        
        return clusters;
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
        if (publications.isEmpty()) return new double[0];
        
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
    
    // Step 3: Build similarity matrix
    public void buildSimilarityMatrix() {
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
        
        // Calculate Jensen-Shannon Divergence similarities
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double[] dist1 = getTopicDistribution(allEntities.get(i));
                double[] dist2 = getTopicDistribution(allEntities.get(j));
                
                double jsd = jensenShannonDivergence(dist1, dist2);
                double similarity = 1.0 - jsd;
                
                similarityMatrix[i][j] = similarity;
                similarityMatrix[j][i] = similarity;
            }
        }
        
        System.out.println("Built similarity matrix of size " + n + "x" + n);
    }
    
    private double[] getTopicDistribution(String entityId) {
        // Check if it's a researcher
        for (Researcher researcher : researchers) {
            if (researcher.getId().equals(entityId)) {
                return researcher.getTopicDistribution();
            }
        }
        
        // Check if it's a clone
        for (ResearcherClone clone : clones) {
            if (clone.getCloneId().equals(entityId)) {
                return clone.getTopicDistribution();
            }
        }
        
        return new double[0];
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
    
    // Step 4: Build network and detect communities
    public void buildNetworkAndDetectCommunities(double threshold) {
        buildNetwork(threshold);
        detectCommunities();
        refineCommunities();
    }
    
    private void buildNetwork(double threshold) {
        networkEdges.clear();
        
        int n = similarityMatrix.length;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (similarityMatrix[i][j] > threshold) {
                    String sourceId = getEntityIdByIndex(i);
                    String targetId = getEntityIdByIndex(j);
                    networkEdges.add(new NetworkEdge(sourceId, targetId, similarityMatrix[i][j]));
                }
            }
        }
        
        System.out.println("Built network with " + networkEdges.size() + " edges");
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
        // Simplified community detection (would use NH-Louvain in practice)
        communities.clear();
        
        // Create adjacency map
        Map<String, Set<String>> adjacencyMap = new HashMap<>();
        for (NetworkEdge edge : networkEdges) {
            adjacencyMap.computeIfAbsent(edge.getSourceId(), k -> new HashSet<>())
                .add(edge.getTargetId());
            adjacencyMap.computeIfAbsent(edge.getTargetId(), k -> new HashSet<>())
                .add(edge.getSourceId());
        }
        
        // Simple connected components as communities
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
        
        System.out.println("Detected " + communities.size() + " communities");
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
    
    // Step 5: Refine communities (Algorithm 1 from paper)
    private void refineCommunities() {
        List<Community> refinedCommunities = new ArrayList<>();
        
        for (Community community : communities) {
            Community refinedCommunity = refineCommunity(community);
            refinedCommunities.add(refinedCommunity);
        }
        
        this.communities = refinedCommunities;
        System.out.println("Refined communities");
    }
    
    private Community refineCommunity(Community community) {
        // Group nodes by base identity
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
    
    // Analysis methods
    public void analyzeResults() {
        System.out.println("\n=== Analysis Results ===");
        System.out.println("Total researchers: " + researchers.size());
        System.out.println("Total clones created: " + clones.size());
        System.out.println("Total communities: " + communities.size());
        
        // Community statistics
        double avgCommunitySize = communities.stream()
            .mapToInt(Community::getSize)
            .average()
            .orElse(0.0);
        
        int maxCommunitySize = communities.stream()
            .mapToInt(Community::getSize)
            .max()
            .orElse(0);
        
        int minCommunitySize = communities.stream()
            .mapToInt(Community::getSize)
            .min()
            .orElse(0);
        
        System.out.println("Average community size: " + String.format("%.2f", avgCommunitySize));
        System.out.println("Max community size: " + maxCommunitySize);
        System.out.println("Min community size: " + minCommunitySize);
        
        // Find overlapping researchers
        Map<String, Integer> membershipCount = new HashMap<>();
        for (Community community : communities) {
            for (String member : community.getMembers()) {
                membershipCount.merge(member, 1, Integer::sum);
            }
        }
        
        long overlappingResearchers = membershipCount.values().stream()
            .mapToInt(Integer::intValue)
            .filter(count -> count > 1)
            .count();
        
        System.out.println("Researchers in multiple communities: " + overlappingResearchers);
    }
    
    public void exportResults(String outputDir) throws IOException {
        Path outputPath = Paths.get(outputDir);
        Files.createDirectories(outputPath);
        
        // Export communities
        try (PrintWriter writer = new PrintWriter(
                Files.newBufferedWriter(outputPath.resolve("communities.csv")))) {
            writer.println("community_id,researcher_id,researcher_name");
            
            for (Community community : communities) {
                for (String member : community.getMembers()) {
                    Researcher researcher = researcherMap.get(member);
                    String name = researcher != null ? researcher.getName() : member;
                    writer.println(community.getId() + "," + member + "," + name);
                }
            }
        }
        
        // Export network edges
        try (PrintWriter writer = new PrintWriter(
                Files.newBufferedWriter(outputPath.resolve("network_edges.csv")))) {
            writer.println("source,target,weight");
            
            for (NetworkEdge edge : networkEdges) {
                writer.println(edge.getSourceId() + "," + 
                              edge.getTargetId() + "," + 
                              edge.getWeight());
            }
        }
        
        System.out.println("Results exported to " + outputDir);
    }

    // ---------------------------------------------------------------------
    // Extension methods moved from AcademicNetworkAnalyzerExtensions.java
    // These expose commonly queried data structures from this class.
    // ---------------------------------------------------------------------
    public Researcher getResearcher(String researcherId) {
        return researcherMap.get(researcherId);
    }

    public List<Researcher> getResearchers() {
        return new ArrayList<>(researchers);
    }

    public List<Community> getCommunities() {
        return new ArrayList<>(communities);
    }

    public List<NetworkEdge> getNetworkEdges() {
        return new ArrayList<>(networkEdges);
    }

    public double[][] getSimilarityMatrix() {
        return similarityMatrix;
    }

    public double calculateResearcherSimilarity(String researcher1, String researcher2) {
        Researcher r1 = researcherMap.get(researcher1);
        Researcher r2 = researcherMap.get(researcher2);
        
        if (r1 == null || r2 == null) return 0.0;
        
        double[] dist1 = r1.getTopicDistribution();
        double[] dist2 = r2.getTopicDistribution();
        
        if (dist1 == null || dist2 == null) return 0.0;
        
        return 1.0 - jensenShannonDivergence(dist1, dist2);
    }

    public List<String> getCommunityMembers(String researcherId) {
        for (Community community : communities) {
            if (community.containsMember(researcherId)) {
                return new ArrayList<>(community.getMembers());
            }
        }
        return new ArrayList<>();
    }

    public double calculateNetworkDensity() {
        int n = researchers.size();
        if (n < 2) return 0.0;
        
        int possibleEdges = n * (n - 1) / 2;
        return (double) networkEdges.size() / possibleEdges;
    }

    public List<Publication> getResearcherTimeline(String researcherId) {
        Researcher researcher = researcherMap.get(researcherId);
        if (researcher == null) return new ArrayList<>();
        
        List<Publication> timeline = new ArrayList<>(researcher.getPublications());
        timeline.sort((a, b) -> a.getId().compareTo(b.getId())); // Assume chronological by ID
        return timeline;
    }
    
    // Create sample data for testing
    public void createSampleData() throws IOException {
        // Create sample publications.csv file
        try (PrintWriter writer = new PrintWriter("publications.csv")) {
            writer.println("pub_id,title,abstract,author_id");
            
            // Sample data with different research areas
            String[][] sampleData = {
                {"1", "Machine Learning Advances", "Deep neural networks for classification", "author1"},
                {"2", "Neural Network Optimization", "Gradient descent improvements", "author1"},
                {"3", "Computer Vision Applications", "Image recognition using CNNs", "author1"},
                {"4", "Natural Language Processing", "Transformer architectures", "author1"},
                {"5", "Reinforcement Learning", "Q-learning algorithms", "author1"},
                {"6", "Data Mining Techniques", "Association rule mining", "author1"},
                
                {"7", "Quantum Computing Theory", "Quantum algorithms for optimization", "author2"},
                {"8", "Quantum Error Correction", "Stabilizer codes", "author2"},
                {"9", "Quantum Machine Learning", "Variational quantum circuits", "author2"},
                {"10", "Quantum Cryptography", "Key distribution protocols", "author2"},
                {"11", "Quantum Simulation", "Many-body systems", "author2"},
                {"12", "Quantum Information", "Entanglement measures", "author2"},
                
                {"13", "Bioinformatics Algorithms", "Sequence alignment methods", "author3"},
                {"14", "Genomic Data Analysis", "SNP detection algorithms", "author3"},
                {"15", "Protein Structure Prediction", "Folding simulation methods", "author3"},
                {"16", "Systems Biology", "Network analysis approaches", "author3"},
                {"17", "Computational Biology", "Phylogenetic reconstruction", "author3"},
                {"18", "Medical Informatics", "Electronic health records", "author3"},
                
                {"19", "Distributed Systems", "Consensus algorithms", "author4"},
                {"20", "Cloud Computing", "Resource allocation strategies", "author4"},
                {"21", "Blockchain Technology", "Smart contract verification", "author4"},
                {"22", "Network Security", "Intrusion detection systems", "author4"},
                {"23", "Parallel Computing", "Load balancing techniques", "author4"},
                {"24", "Edge Computing", "Latency optimization", "author4"}
            };
            
            for (String[] row : sampleData) {
                writer.println(String.join(",", row));
            }
        }
        
        System.out.println("Sample data created in publications.csv");
    }
    
    // Example usage
    public static void main(String[] args) {
        try {
            AcademicNetworkAnalyzer analyzer = new AcademicNetworkAnalyzer();
            
            // Create sample data if file doesn't exist
            if (!Files.exists(Paths.get("publications.csv"))) {
                analyzer.createSampleData();
            }
            
            // Load data
            analyzer.loadResearchData("publications.csv");
            
            // Perform topic modeling and cloning
            analyzer.performTopicModeling();
            
            // Build similarity matrix
            analyzer.buildSimilarityMatrix();
            
            // Build network and detect communities
            analyzer.buildNetworkAndDetectCommunities(0.25);
            
            // Analyze results
            analyzer.analyzeResults();
            
            // Export results
            analyzer.exportResults("output");
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

// Interface for topic modeling service (would integrate with Python BERTopic)
class TopicModelingService {
    private int numTopics = 50; // Default number of topics
    
    public void trainTopicModel(List<String> documents) {
        // This would interface with Python BERTopic
        // For now, simulate with random topic distributions
        System.out.println("Training topic model on " + documents.size() + " documents");
    }
    
    public double[] getTopicDistribution(String text) {
        // Simulate topic distribution
        double[] distribution = new double[numTopics];
        Random random = new Random(text.hashCode()); // Deterministic for same text
        
        // Generate random distribution
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
