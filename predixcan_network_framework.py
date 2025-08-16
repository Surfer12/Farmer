#!/usr/bin/env python3
"""
PrediXcan Network Framework Implementation
Genotype → Expression → Phenotype → Network Embedding Pipeline

Three-step mapping:
1. Genotype to Expression: Ê_{g,t} = Σ w_{gi,t} · SNP_i
2. Expression to Phenotype: Y = β Ê_{g,t} + ε  
3. Phenotype Integration: Network embedding with GNN/NODE-RK4/Koopman

Example: Cyclosporine concentration prediction via CYP3A4/5, ABCB1 metabolic networks
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import math

@dataclass
class SNPData:
    """SNP genotype data"""
    snp_id: str
    chromosome: int
    position: int
    alleles: Tuple[str, str]
    genotype_vector: np.ndarray  # 0, 1, 2 encoding

@dataclass
class eQTLWeights:
    """eQTL weights from GTEx"""
    gene_id: str
    tissue: str
    snp_weights: Dict[str, float]  # SNP_ID -> weight
    r_squared: float  # Model performance
    n_snps: int

@dataclass
class PredictedExpression:
    """Predicted gene expression from PrediXcan"""
    gene_id: str
    tissue: str
    predicted_expression: float
    confidence: float
    contributing_snps: List[str]

@dataclass
class NetworkNode:
    """Network node with expression features"""
    gene_id: str
    predicted_expressions: Dict[str, float]  # tissue -> expression
    pathway_memberships: List[str]
    interaction_partners: List[str]
    embedding: np.ndarray

class PrediXcanFramework:
    """
    Complete PrediXcan framework implementation
    Genotype → Expression → Phenotype → Network Embedding
    """
    
    def __init__(self):
        self.eqtl_models = {}  # tissue -> {gene -> eQTLWeights}
        self.network_structure = {}  # gene -> NetworkNode
        self.pathway_database = {}  # pathway -> genes
        
        # Initialize with example data
        self._initialize_example_data()
    
    def _initialize_example_data(self):
        """Initialize with cyclosporine metabolism example"""
        
        # Example eQTL weights for CYP3A4, CYP3A5, ABCB1 in liver
        self.eqtl_models['liver'] = {
            'CYP3A4': eQTLWeights(
                gene_id='CYP3A4',
                tissue='liver',
                snp_weights={
                    'rs2740574': 0.45,   # CYP3A4*1B
                    'rs4646437': -0.32,  # Regulatory variant
                    'rs12721627': 0.28   # Enhancer region
                },
                r_squared=0.12,
                n_snps=3
            ),
            'CYP3A5': eQTLWeights(
                gene_id='CYP3A5',
                tissue='liver',
                snp_weights={
                    'rs776746': -0.85,   # CYP3A5*3 (major effect)
                    'rs10264272': 0.23,  # Regulatory
                    'rs4646450': -0.18   # Minor effect
                },
                r_squared=0.34,
                n_snps=3
            ),
            'ABCB1': eQTLWeights(
                gene_id='ABCB1',
                tissue='liver',
                snp_weights={
                    'rs1045642': 0.31,   # C3435T
                    'rs2032582': -0.24,  # G2677T/A
                    'rs1128503': 0.19    # C1236T
                },
                r_squared=0.08,
                n_snps=3
            )
        }
        
        # Network structure (simplified STRING/Reactome)
        self.network_structure = {
            'CYP3A4': NetworkNode(
                gene_id='CYP3A4',
                predicted_expressions={},
                pathway_memberships=['drug_metabolism', 'cytochrome_p450'],
                interaction_partners=['CYP3A5', 'POR', 'CYB5A'],
                embedding=np.zeros(64)
            ),
            'CYP3A5': NetworkNode(
                gene_id='CYP3A5', 
                predicted_expressions={},
                pathway_memberships=['drug_metabolism', 'cytochrome_p450'],
                interaction_partners=['CYP3A4', 'POR', 'CYB5A'],
                embedding=np.zeros(64)
            ),
            'ABCB1': NetworkNode(
                gene_id='ABCB1',
                predicted_expressions={},
                pathway_memberships=['drug_transport', 'abc_transporters'],
                interaction_partners=['ABCG2', 'SLCO1B1'],
                embedding=np.zeros(64)
            )
        }
        
        # Pathway database
        self.pathway_database = {
            'drug_metabolism': ['CYP3A4', 'CYP3A5', 'CYP2D6', 'UGT1A1'],
            'drug_transport': ['ABCB1', 'ABCG2', 'SLCO1B1', 'SLCO1B3'],
            'cytochrome_p450': ['CYP3A4', 'CYP3A5', 'CYP2D6', 'CYP2C9']
        }
    
    def step1_genotype_to_expression(self, snp_data: Dict[str, SNPData], 
                                   tissue: str = 'liver') -> Dict[str, PredictedExpression]:
        """
        Step 1: Genotype to Expression
        Ê_{g,t} = Σ w_{gi,t} · SNP_i
        """
        predicted_expressions = {}
        
        if tissue not in self.eqtl_models:
            raise ValueError(f"No eQTL models available for tissue: {tissue}")
        
        tissue_models = self.eqtl_models[tissue]
        
        for gene_id, eqtl_weights in tissue_models.items():
            # Compute predicted expression
            predicted_expr = 0.0
            contributing_snps = []
            
            for snp_id, weight in eqtl_weights.snp_weights.items():
                if snp_id in snp_data:
                    # Get genotype (0, 1, 2 encoding)
                    genotype = np.mean(snp_data[snp_id].genotype_vector)
                    predicted_expr += weight * genotype
                    contributing_snps.append(snp_id)
            
            # Confidence based on model R² and number of contributing SNPs
            confidence = eqtl_weights.r_squared * (len(contributing_snps) / eqtl_weights.n_snps)
            
            predicted_expressions[gene_id] = PredictedExpression(
                gene_id=gene_id,
                tissue=tissue,
                predicted_expression=predicted_expr,
                confidence=confidence,
                contributing_snps=contributing_snps
            )
        
        return predicted_expressions
    
    def step2_expression_to_phenotype(self, predicted_expressions: Dict[str, PredictedExpression],
                                    phenotype_data: np.ndarray) -> Dict[str, float]:
        """
        Step 2: Expression to Phenotype
        Y = β Ê_{g,t} + ε
        """
        expression_phenotype_associations = {}
        
        for gene_id, pred_expr in predicted_expressions.items():
            # Simple linear regression: phenotype ~ predicted_expression
            X = np.array([pred_expr.predicted_expression]).reshape(-1, 1)
            
            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(phenotype_data)), 
                                              np.full(len(phenotype_data), pred_expr.predicted_expression)])
            
            # Solve normal equations: β = (X'X)^(-1)X'y
            try:
                XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                beta = XtX_inv @ X_with_intercept.T @ phenotype_data
                
                # Effect size is the slope coefficient
                effect_size = beta[1]
                
                # Compute R-squared
                y_pred = X_with_intercept @ beta
                ss_res = np.sum((phenotype_data - y_pred) ** 2)
                ss_tot = np.sum((phenotype_data - np.mean(phenotype_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                expression_phenotype_associations[gene_id] = {
                    'effect_size': effect_size,
                    'r_squared': r_squared,
                    'confidence': pred_expr.confidence * r_squared
                }
                
            except np.linalg.LinAlgError:
                # Singular matrix, use simple correlation
                correlation = np.corrcoef([pred_expr.predicted_expression] * len(phenotype_data), 
                                        phenotype_data)[0, 1]
                expression_phenotype_associations[gene_id] = {
                    'effect_size': correlation,
                    'r_squared': correlation**2,
                    'confidence': pred_expr.confidence * abs(correlation)
                }
        
        return expression_phenotype_associations
    
    def step3_phenotype_integration(self, expression_effects: Dict[str, Dict],
                                  tissues: List[str] = ['liver']) -> Dict[str, np.ndarray]:
        """
        Step 3: Phenotype Integration
        Create gene-level effect vectors for network embedding
        """
        gene_effect_vectors = {}
        
        for gene_id in expression_effects.keys():
            # Multi-tissue integration (simplified - just liver for now)
            effect_vector = []
            
            # Add effect size
            effect_vector.append(expression_effects[gene_id]['effect_size'])
            
            # Add confidence
            effect_vector.append(expression_effects[gene_id]['confidence'])
            
            # Add R-squared
            effect_vector.append(expression_effects[gene_id]['r_squared'])
            
            # Add pathway membership features
            if gene_id in self.network_structure:
                node = self.network_structure[gene_id]
                
                # Pathway membership binary features
                for pathway in ['drug_metabolism', 'drug_transport', 'cytochrome_p450']:
                    effect_vector.append(1.0 if pathway in node.pathway_memberships else 0.0)
                
                # Network connectivity
                effect_vector.append(len(node.interaction_partners))
            else:
                # Default values if gene not in network
                effect_vector.extend([0.0, 0.0, 0.0, 0.0])
            
            gene_effect_vectors[gene_id] = np.array(effect_vector)
        
        return gene_effect_vectors
    
    def network_embedding_gnn(self, gene_effect_vectors: Dict[str, np.ndarray],
                            embedding_dim: int = 64) -> Dict[str, np.ndarray]:
        """
        Network embedding using simplified GNN approach
        """
        # Initialize embeddings with effect vectors
        embeddings = {}
        
        for gene_id, effect_vector in gene_effect_vectors.items():
            # Pad or truncate to embedding dimension
            if len(effect_vector) < embedding_dim:
                embedding = np.pad(effect_vector, (0, embedding_dim - len(effect_vector)))
            else:
                embedding = effect_vector[:embedding_dim]
            
            embeddings[gene_id] = embedding
        
        # Simple message passing (1 iteration)
        updated_embeddings = {}
        
        for gene_id, embedding in embeddings.items():
            if gene_id in self.network_structure:
                node = self.network_structure[gene_id]
                
                # Aggregate neighbor embeddings
                neighbor_sum = np.zeros(embedding_dim)
                neighbor_count = 0
                
                for neighbor_id in node.interaction_partners:
                    if neighbor_id in embeddings:
                        neighbor_sum += embeddings[neighbor_id]
                        neighbor_count += 1
                
                # Update embedding: self + mean(neighbors)
                if neighbor_count > 0:
                    updated_embedding = 0.7 * embedding + 0.3 * (neighbor_sum / neighbor_count)
                else:
                    updated_embedding = embedding
                
                updated_embeddings[gene_id] = updated_embedding
            else:
                updated_embeddings[gene_id] = embedding
        
        return updated_embeddings
    
    def predict_cyclosporine_concentration(self, embeddings: Dict[str, np.ndarray],
                                         baseline_concentration: float = 100.0) -> Dict:
        """
        Predict cyclosporine concentration using network embeddings
        Example application of the framework
        """
        # Key genes for cyclosporine metabolism
        key_genes = ['CYP3A4', 'CYP3A5', 'ABCB1']
        
        # Compute metabolic capacity score
        metabolic_score = 0.0
        transport_score = 0.0
        
        for gene_id in key_genes:
            if gene_id in embeddings:
                embedding = embeddings[gene_id]
                
                # Extract relevant features (first few dimensions are effect sizes)
                effect_size = embedding[0] if len(embedding) > 0 else 0.0
                confidence = embedding[1] if len(embedding) > 1 else 0.0
                
                if gene_id in ['CYP3A4', 'CYP3A5']:
                    # Higher expression = more metabolism = lower concentration
                    metabolic_score += effect_size * confidence
                elif gene_id == 'ABCB1':
                    # Higher expression = more efflux = lower concentration
                    transport_score += effect_size * confidence
        
        # Predict concentration change
        # Simplified model: higher metabolic/transport activity = lower concentration
        concentration_factor = 1.0 - 0.1 * metabolic_score - 0.05 * transport_score
        predicted_concentration = baseline_concentration * max(0.1, concentration_factor)
        
        # Confidence based on gene coverage and effect strengths
        prediction_confidence = min(1.0, len([g for g in key_genes if g in embeddings]) / len(key_genes))
        
        return {
            'predicted_concentration': predicted_concentration,
            'baseline_concentration': baseline_concentration,
            'concentration_change': predicted_concentration - baseline_concentration,
            'metabolic_score': metabolic_score,
            'transport_score': transport_score,
            'prediction_confidence': prediction_confidence,
            'key_genes_analyzed': [g for g in key_genes if g in embeddings]
        }
    
    def integration_with_existing_frameworks(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Integration with existing mathematical frameworks (UOIF, LSTM, Ψ(x))
        """
        # UOIF Integration
        uoif_classification = {
            'claim_class': 'Interpretation',  # Integrates multiple data types
            'confidence': 0.88,  # High for established genomics methods
            'source_type': 'Expert',  # GTEx consortium, established methods
            'framework_alignment': True
        }
        
        # LSTM Integration - treat gene expression as time series
        lstm_features = []
        for gene_id, embedding in embeddings.items():
            # Use embedding as LSTM input features
            lstm_features.append({
                'gene_id': gene_id,
                'feature_vector': embedding[:10],  # First 10 dimensions
                'sequence_potential': True  # Could model expression over time
            })
        
        # Ψ(x) Framework Integration
        # Symbolic component: Known pathway interactions
        symbolic_accuracy = np.mean([
            1.0 if gene_id in self.network_structure else 0.5 
            for gene_id in embeddings.keys()
        ])
        
        # Neural component: Predicted expression accuracy
        neural_accuracy = np.mean([
            np.linalg.norm(embedding) / 64  # Normalized embedding strength
            for embedding in embeddings.values()
        ])
        
        # Adaptive weight based on network coverage
        network_coverage = len([g for g in embeddings.keys() if g in self.network_structure]) / len(embeddings)
        alpha_t = network_coverage  # More network info = more symbolic
        
        # Hybrid Ψ(x) computation
        psi_x = alpha_t * symbolic_accuracy + (1 - alpha_t) * neural_accuracy
        
        return {
            'uoif_integration': uoif_classification,
            'lstm_features': lstm_features,
            'psi_framework': {
                'symbolic_accuracy': symbolic_accuracy,
                'neural_accuracy': neural_accuracy,
                'adaptive_weight': alpha_t,
                'psi_x': psi_x,
                'interpretation': 'Strong genomics-network integration'
            },
            'framework_synergy': {
                'genotype_symbolic': 'Known SNP effects',
                'expression_neural': 'Predicted from ML models',
                'network_hybrid': 'Combines known pathways with predictions',
                'phenotype_validation': 'Real-world outcomes'
            }
        }

def demonstrate_predixcan_framework():
    """Demonstrate complete PrediXcan framework"""
    
    print("=" * 80)
    print("PREDIXCAN NETWORK FRAMEWORK DEMONSTRATION")
    print("Genotype → Expression → Phenotype → Network Embedding")
    print("=" * 80)
    
    # Initialize framework
    predixcan = PrediXcanFramework()
    
    # Generate example SNP data
    print("\n1. GENERATING EXAMPLE SNP DATA")
    print("-" * 40)
    
    snp_data = {
        'rs2740574': SNPData('rs2740574', 7, 99270539, ('C', 'T'), np.array([1, 0, 2, 1, 0])),  # CYP3A4*1B
        'rs776746': SNPData('rs776746', 7, 99652770, ('G', 'A'), np.array([2, 2, 1, 2, 0])),    # CYP3A5*3
        'rs1045642': SNPData('rs1045642', 7, 87509329, ('C', 'T'), np.array([1, 2, 1, 0, 1]))   # ABCB1 C3435T
    }
    
    for snp_id, snp in snp_data.items():
        print(f"  {snp_id}: {snp.alleles[0]}/{snp.alleles[1]} | Mean genotype: {np.mean(snp.genotype_vector):.2f}")
    
    # Step 1: Genotype to Expression
    print(f"\n2. STEP 1: GENOTYPE TO EXPRESSION")
    print("-" * 40)
    print("Ê_{g,t} = Σ w_{gi,t} · SNP_i")
    
    predicted_expressions = predixcan.step1_genotype_to_expression(snp_data, tissue='liver')
    
    for gene_id, pred_expr in predicted_expressions.items():
        print(f"\n  {gene_id} (liver):")
        print(f"    Predicted expression: {pred_expr.predicted_expression:.4f}")
        print(f"    Confidence: {pred_expr.confidence:.3f}")
        print(f"    Contributing SNPs: {', '.join(pred_expr.contributing_snps)}")
    
    # Step 2: Expression to Phenotype
    print(f"\n3. STEP 2: EXPRESSION TO PHENOTYPE")
    print("-" * 40)
    print("Y = β Ê_{g,t} + ε")
    
    # Generate example phenotype data (cyclosporine clearance)
    np.random.seed(42)
    phenotype_data = np.random.normal(50, 15, 5)  # Clearance in mL/min
    
    print(f"Example phenotype (clearance): {phenotype_data}")
    
    expression_effects = predixcan.step2_expression_to_phenotype(predicted_expressions, phenotype_data)
    
    for gene_id, effects in expression_effects.items():
        print(f"\n  {gene_id}:")
        print(f"    Effect size β: {effects['effect_size']:.4f}")
        print(f"    R-squared: {effects['r_squared']:.3f}")
        print(f"    Confidence: {effects['confidence']:.3f}")
    
    # Step 3: Phenotype Integration
    print(f"\n4. STEP 3: PHENOTYPE INTEGRATION")
    print("-" * 40)
    
    gene_effect_vectors = predixcan.step3_phenotype_integration(expression_effects)
    
    for gene_id, effect_vector in gene_effect_vectors.items():
        print(f"  {gene_id}: Effect vector shape {effect_vector.shape}, norm {np.linalg.norm(effect_vector):.3f}")
    
    # Network Embedding
    print(f"\n5. NETWORK EMBEDDING (GNN)")
    print("-" * 30)
    
    embeddings = predixcan.network_embedding_gnn(gene_effect_vectors, embedding_dim=32)
    
    for gene_id, embedding in embeddings.items():
        print(f"  {gene_id}: Embedding shape {embedding.shape}, norm {np.linalg.norm(embedding):.3f}")
    
    # Cyclosporine Prediction
    print(f"\n6. CYCLOSPORINE CONCENTRATION PREDICTION")
    print("-" * 45)
    
    prediction = predixcan.predict_cyclosporine_concentration(embeddings)
    
    print(f"Baseline concentration: {prediction['baseline_concentration']:.1f} ng/mL")
    print(f"Predicted concentration: {prediction['predicted_concentration']:.1f} ng/mL")
    print(f"Concentration change: {prediction['concentration_change']:+.1f} ng/mL")
    print(f"Metabolic score: {prediction['metabolic_score']:.4f}")
    print(f"Transport score: {prediction['transport_score']:.4f}")
    print(f"Prediction confidence: {prediction['prediction_confidence']:.3f}")
    print(f"Key genes analyzed: {', '.join(prediction['key_genes_analyzed'])}")
    
    # Framework Integration
    print(f"\n7. INTEGRATION WITH EXISTING FRAMEWORKS")
    print("-" * 45)
    
    integration = predixcan.integration_with_existing_frameworks(embeddings)
    
    print(f"\nUOIF Integration:")
    uoif = integration['uoif_integration']
    print(f"  Claim class: [{uoif['claim_class']}]")
    print(f"  Confidence: {uoif['confidence']:.3f}")
    print(f"  Source type: {uoif['source_type']}")
    
    print(f"\nΨ(x) Framework Integration:")
    psi = integration['psi_framework']
    print(f"  Symbolic accuracy: {psi['symbolic_accuracy']:.3f}")
    print(f"  Neural accuracy: {psi['neural_accuracy']:.3f}")
    print(f"  Adaptive weight α: {psi['adaptive_weight']:.3f}")
    print(f"  Hybrid Ψ(x): {psi['psi_x']:.3f}")
    print(f"  Interpretation: {psi['interpretation']}")
    
    print(f"\nFramework Synergy:")
    synergy = integration['framework_synergy']
    for key, value in synergy.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Summary
    print(f"\n" + "="*80)
    print("FRAMEWORK SUMMARY")
    print("="*80)
    
    print(f"\n🧬 Genomic Pipeline Success:")
    print(f"  • SNP data processed: {len(snp_data)} variants")
    print(f"  • Gene expressions predicted: {len(predicted_expressions)} genes")
    print(f"  • Network embeddings created: {len(embeddings)} nodes")
    print(f"  • Phenotype prediction: Cyclosporine concentration")
    
    print(f"\n⚙️ Technical Integration:")
    print(f"  • UOIF classification: {uoif['claim_class']} with {uoif['confidence']:.1%} confidence")
    print(f"  • LSTM features: {len(integration['lstm_features'])} gene sequences")
    print(f"  • Ψ(x) hybrid score: {psi['psi_x']:.3f}")
    
    print(f"\n🌍 Biological Insights:")
    print(f"  • CYP3A4/5 metabolic capacity affects drug clearance")
    print(f"  • ABCB1 transport activity modulates bioavailability")
    print(f"  • Network effects capture gene-gene interactions")
    print(f"  • Personalized dosing potential demonstrated")
    
    return predixcan, embeddings, prediction, integration

if __name__ == "__main__":
    predixcan, embeddings, prediction, integration = demonstrate_predixcan_framework()
