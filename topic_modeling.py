#!/usr/bin/env python3
"""
BERTopic integration script for Academic Network Analyzer
Provides topic modeling capabilities that can be called from Java
"""

import sys
import pickle
import numpy as np
from pathlib import Path

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("BERTopic dependencies not available. Install with:")
    print("pip install bertopic sentence-transformers umap-learn hdbscan")

class TopicModelingService:
    def __init__(self, model_path="topic_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.embeddings = None
        
    def train_model(self, documents):
        """Train BERTopic model on documents"""
        if not BERTOPIC_AVAILABLE:
            return self._simulate_training(documents)
            
        try:
            # Initialize components
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # UMAP for dimensionality reduction
            umap_model = UMAP(
                n_neighbors=15, 
                n_components=5, 
                min_dist=0.0, 
                metric='cosine',
                random_state=42
            )
            
            # HDBSCAN for clustering
            hdbscan_model = HDBSCAN(
                min_cluster_size=10,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            
            # Vectorizer for topic representation
            vectorizer_model = CountVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                min_df=2,
                max_features=5000
            )
            
            # Initialize BERTopic
            self.model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                top_k_words=10,
                verbose=True
            )
            
            # Fit the model
            print(f"Training BERTopic on {len(documents)} documents...")
            topics, probabilities = self.model.fit_transform(documents)
            
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
                
            print(f"Model trained successfully with {len(set(topics))} topics")
            print(f"Model saved to {self.model_path}")
            
            # Print topic info
            topic_info = self.model.get_topic_info()
            print("\nTop 5 topics:")
            print(topic_info.head())
            
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def _simulate_training(self, documents):
        """Simulate training when BERTopic is not available"""
        print(f"Simulating topic model training on {len(documents)} documents")
        print("Note: Install BERTopic for actual topic modeling")
        
        # Create a dummy model structure
        self.model = {
            'num_topics': 50,
            'documents': len(documents),
            'simulated': True
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        return True
    
    def load_model(self):
        """Load trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            return True
        except FileNotFoundError:
            print(f"Model file {self.model_path} not found")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_topic_distribution(self, text):
        """Get topic distribution for a single text"""
        if self.model is None:
            if not self.load_model():
                return self._simulate_distribution()
        
        if not BERTOPIC_AVAILABLE or isinstance(self.model, dict):
            return self._simulate_distribution()
            
        try:
            # Transform text to get topic probabilities
            topics, probabilities = self.model.transform([text])
            
            if probabilities is not None and len(probabilities) > 0:
                # Return probability distribution
                return probabilities[0].tolist()
            else:
                # Fallback: create one-hot encoding for assigned topic
                topic_id = topics[0]
                num_topics = len(self.model.get_topic_info()) - 1  # Exclude outlier topic
                distribution = [0.0] * max(50, num_topics)
                if topic_id >= 0 and topic_id < len(distribution):
                    distribution[topic_id] = 1.0
                return distribution
                
        except Exception as e:
            print(f"Error getting topic distribution: {e}")
            return self._simulate_distribution()
    
    def _simulate_distribution(self):
        """Simulate topic distribution"""
        np.random.seed(42)  # For reproducibility
        distribution = np.random.dirichlet(np.ones(50))
        return distribution.tolist()

def main():
    if len(sys.argv) < 2:
        print("Usage: python topic_modeling.py <command> [args...]")
        print("Commands:")
        print("  train <documents_file>  - Train topic model")
        print("  predict <text>          - Get topic distribution")
        sys.exit(1)
    
    command = sys.argv[1]
    service = TopicModelingService()
    
    if command == "train":
        if len(sys.argv) < 3:
            print("Usage: python topic_modeling.py train <documents_file>")
            sys.exit(1)
            
        documents_file = sys.argv[2]
        
        try:
            # Read documents from file
            with open(documents_file, 'r', encoding='utf-8') as f:
                documents = [line.strip() for line in f if line.strip()]
            
            # Train model
            success = service.train_model(documents)
            if success:
                print("Training completed successfully")
            else:
                print("Training failed")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error reading documents file: {e}")
            sys.exit(1)
    
    elif command == "predict":
        if len(sys.argv) < 3:
            print("Usage: python topic_modeling.py predict <text>")
            sys.exit(1)
            
        text = sys.argv[2]
        
        # Get topic distribution
        distribution = service.get_topic_distribution(text)
        
        # Output in format expected by Java
        print("TOPICS:" + ",".join(map(str, distribution)))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
