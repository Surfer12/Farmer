#!/usr/bin/env python3
"""
NIST Cybersecurity Framework Integration with Ψ(x) Framework

This module implements the three-phase integration:
Phase 1: Measure Function Enhancement
Phase 2: Threshold Governance  
Phase 3: Profile Development

© 2025 Ryan David Oates. All rights reserved.
Classification: Confidential — Internal Use Only
License: GPLv3. Use restricted by company policy and applicable NDAs.
SPDX-License-Identifier: GPL-3.0-only
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NISTFunction(Enum):
    """NIST Cybersecurity Framework Functions"""
    IDENTIFY = "ID"
    PROTECT = "PR"
    DETECT = "DE"
    RESPOND = "RS"
    RECOVER = "RC"

class RiskCategory(Enum):
    """Risk categories for Ra, Rv mapping"""
    TECHNICAL = "technical"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"
    REPUTATIONAL = "reputational"

@dataclass
class TechnicalCharacteristic:
    """Technical characteristics mapped to S, N parameters"""
    name: str
    source_evidence: float  # S parameter [0,1]
    non_source_evidence: float  # N parameter [0,1]
    confidence: float
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SocioTechnicalRisk:
    """Socio-technical risks mapped to Ra, Rv assessments"""
    name: str
    authority_risk: float  # Ra parameter [0,1]
    verifiability_risk: float  # Rv parameter [0,1]
    category: RiskCategory
    impact_level: str  # HIGH, MEDIUM, LOW
    likelihood: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegulatoryEvidence:
    """Regulatory/canonical evidence for β weighting"""
    source: str
    authority_level: str  # CANONICAL, EXPERT, COMMUNITY
    beta_weight: float  # β parameter
    certification_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PsiMeasurement:
    """Ψ(x) measurement result"""
    value: float
    components: Dict[str, float]
    confidence_level: str
    audit_trail: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class NISTMeasureFunction:
    """
    Phase 1: Measure Function Enhancement
    Implements Ψ(x) as a quantitative metric within NIST's Measure function
    """
    
    def __init__(self):
        self.technical_characteristics: List[TechnicalCharacteristic] = []
        self.socio_technical_risks: List[SocioTechnicalRisk] = []
        self.regulatory_evidence: List[RegulatoryEvidence] = []
        self.lambda1 = 1.0  # Authority risk weight
        self.lambda2 = 1.0  # Verifiability risk weight
        
    def add_technical_characteristic(self, characteristic: TechnicalCharacteristic) -> None:
        """Add technical characteristic for S, N parameter mapping"""
        self.technical_characteristics.append(characteristic)
        logger.info(f"Added technical characteristic: {characteristic.name}")
        
    def add_socio_technical_risk(self, risk: SocioTechnicalRisk) -> None:
        """Add socio-technical risk for Ra, Rv assessment"""
        self.socio_technical_risks.append(risk)
        logger.info(f"Added socio-technical risk: {risk.name}")
        
    def add_regulatory_evidence(self, evidence: RegulatoryEvidence) -> None:
        """Add regulatory evidence for β weighting"""
        self.regulatory_evidence.append(evidence)
        logger.info(f"Added regulatory evidence: {evidence.source}")
        
    def compute_evidence_blend(self, alpha: float) -> Tuple[float, Dict[str, float]]:
        """
        Compute evidence blend O(α) = αS + (1-α)N
        
        Args:
            alpha: Weight parameter for source allocation [0,1]
            
        Returns:
            Tuple of (blend_value, components_dict)
        """
        if not self.technical_characteristics:
            raise ValueError("No technical characteristics available for evidence blend")
            
        # Aggregate S and N across all characteristics
        total_weight = sum(tc.confidence for tc in self.technical_characteristics)
        
        S = sum(tc.source_evidence * tc.confidence for tc in self.technical_characteristics) / total_weight
        N = sum(tc.non_source_evidence * tc.confidence for tc in self.technical_characteristics) / total_weight
        
        blend = alpha * S + (1 - alpha) * N
        
        components = {
            'S': S,
            'N': N,
            'alpha': alpha,
            'blend': blend
        }
        
        return blend, components
        
    def compute_penalty_function(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute penalty function pen = exp(-[λ₁Ra + λ₂Rv])
        
        Returns:
            Tuple of (penalty_value, components_dict)
        """
        if not self.socio_technical_risks:
            raise ValueError("No socio-technical risks available for penalty computation")
            
        # Aggregate Ra and Rv across all risks
        total_weight = sum(risk.likelihood for risk in self.socio_technical_risks)
        
        Ra = sum(risk.authority_risk * risk.likelihood for risk in self.socio_technical_risks) / total_weight
        Rv = sum(risk.verifiability_risk * risk.likelihood for risk in self.socio_technical_risks) / total_weight
        
        penalty = math.exp(-(self.lambda1 * Ra + self.lambda2 * Rv))
        
        components = {
            'Ra': Ra,
            'Rv': Rv,
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'penalty': penalty
        }
        
        return penalty, components
        
    def compute_beta_weight(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute β weight from regulatory/canonical evidence
        
        Returns:
            Tuple of (beta_value, components_dict)
        """
        if not self.regulatory_evidence:
            return 1.0, {'beta': 1.0, 'source': 'default'}
            
        # Weight by authority level and recency
        beta = 1.0
        active_evidence = []
        
        for evidence in self.regulatory_evidence:
            # Check if evidence is still valid
            if evidence.expiry_date and evidence.expiry_date < datetime.now(timezone.utc):
                continue
                
            active_evidence.append(evidence)
            beta = max(beta, evidence.beta_weight)  # Take maximum uplift
            
        components = {
            'beta': beta,
            'active_sources': len(active_evidence),
            'sources': [e.source for e in active_evidence]
        }
        
        return beta, components
        
    def measure_psi(self, alpha: float = 0.7) -> PsiMeasurement:
        """
        Compute Ψ(x) = min{β·O(α)·pen(x), 1}
        
        Args:
            alpha: Evidence allocation weight [0,1]
            
        Returns:
            PsiMeasurement with full audit trail
        """
        audit_trail = []
        components = {}
        
        try:
            # Compute evidence blend
            blend, blend_components = self.compute_evidence_blend(alpha)
            components.update(blend_components)
            audit_trail.append(f"Evidence blend O(α={alpha:.3f}) = {blend:.3f}")
            
            # Compute penalty function
            penalty, penalty_components = self.compute_penalty_function()
            components.update(penalty_components)
            audit_trail.append(f"Penalty function pen = {penalty:.3f}")
            
            # Compute beta weight
            beta, beta_components = self.compute_beta_weight()
            components.update(beta_components)
            audit_trail.append(f"Beta weight β = {beta:.3f}")
            
            # Final Ψ computation
            psi_raw = beta * blend * penalty
            psi = min(psi_raw, 1.0)
            components['psi_raw'] = psi_raw
            components['psi'] = psi
            
            audit_trail.append(f"Ψ(x) = min{{{psi_raw:.3f}, 1}} = {psi:.3f}")
            
            # Determine confidence level
            if psi >= 0.95:
                confidence_level = "Maximum"
            elif psi >= 0.85:
                confidence_level = "High"
            elif psi >= 0.70:
                confidence_level = "Medium"
            elif psi >= 0.50:
                confidence_level = "Low"
            else:
                confidence_level = "Minimal"
                
            return PsiMeasurement(
                value=psi,
                components=components,
                confidence_level=confidence_level,
                audit_trail=audit_trail
            )
            
        except Exception as e:
            logger.error(f"Error computing Ψ(x): {e}")
            raise

class NISTThresholdGovernance:
    """
    Phase 2: Threshold Governance
    Apply threshold transfer principles to NIST's governance requirements
    """
    
    def __init__(self):
        self.thresholds: Dict[str, float] = {}
        self.threshold_history: List[Dict[str, Any]] = []
        
    def set_threshold(self, context: str, threshold: float, justification: str) -> None:
        """Set risk tolerance threshold with audit trail"""
        old_threshold = self.thresholds.get(context)
        self.thresholds[context] = threshold
        
        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'context': context,
            'old_threshold': old_threshold,
            'new_threshold': threshold,
            'justification': justification
        }
        
        self.threshold_history.append(record)
        logger.info(f"Threshold updated for {context}: {old_threshold} → {threshold}")
        
    def transfer_threshold(self, context: str, old_beta: float, new_beta: float) -> float:
        """
        Apply threshold transfer principle: τ' = τ·β/β'
        
        Args:
            context: Risk context identifier
            old_beta: Original β parameter
            new_beta: New β parameter
            
        Returns:
            New threshold value
        """
        if context not in self.thresholds:
            raise ValueError(f"No threshold defined for context: {context}")
            
        old_threshold = self.thresholds[context]
        new_threshold = old_threshold * (old_beta / new_beta)
        
        justification = f"Threshold transfer: β {old_beta:.3f} → {new_beta:.3f}"
        self.set_threshold(context, new_threshold, justification)
        
        return new_threshold
        
    def harmonize_thresholds(self, contexts: List[str], target_context: str) -> Dict[str, float]:
        """Enable cross-organizational threshold harmonization"""
        if target_context not in self.thresholds:
            raise ValueError(f"Target context {target_context} not found")
            
        target_threshold = self.thresholds[target_context]
        harmonized = {}
        
        for context in contexts:
            if context in self.thresholds:
                harmonized[context] = target_threshold
                justification = f"Harmonized with {target_context}"
                self.set_threshold(context, target_threshold, justification)
            else:
                logger.warning(f"Context {context} not found for harmonization")
                
        return harmonized
        
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get complete threshold adjustment audit trail"""
        return self.threshold_history.copy()

class NISTProfileDevelopment:
    """
    Phase 3: Profile Development
    Create sector-specific profiles using Ψ(x) parameterizations
    """
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}
        
    def create_healthcare_profile(self) -> Dict[str, Any]:
        """Healthcare: High β for FDA-approved methods"""
        profile = {
            'sector': 'Healthcare',
            'description': 'Healthcare sector profile with FDA regulatory emphasis',
            'parameters': {
                'default_alpha': 0.8,  # High weight on authoritative sources
                'lambda1': 0.5,  # Lower authority risk sensitivity
                'lambda2': 1.2,  # Higher verifiability risk sensitivity
                'beta_weights': {
                    'FDA_approved': 1.8,
                    'HIPAA_compliant': 1.5,
                    'clinical_guidelines': 1.3,
                    'industry_standard': 1.0
                }
            },
            'risk_mappings': {
                'patient_safety': {'Ra': 0.1, 'Rv': 0.2, 'priority': 'HIGH'},
                'data_privacy': {'Ra': 0.2, 'Rv': 0.1, 'priority': 'HIGH'},
                'regulatory_compliance': {'Ra': 0.1, 'Rv': 0.3, 'priority': 'HIGH'},
                'operational_continuity': {'Ra': 0.3, 'Rv': 0.2, 'priority': 'MEDIUM'}
            },
            'thresholds': {
                'patient_critical': 0.95,
                'privacy_sensitive': 0.90,
                'operational': 0.75,
                'administrative': 0.60
            }
        }
        
        self.profiles['healthcare'] = profile
        return profile
        
    def create_finance_profile(self) -> Dict[str, Any]:
        """Finance: Low Ra for regulatory compliance"""
        profile = {
            'sector': 'Finance',
            'description': 'Financial services profile with regulatory compliance emphasis',
            'parameters': {
                'default_alpha': 0.9,  # Very high weight on regulatory sources
                'lambda1': 2.0,  # High authority risk sensitivity
                'lambda2': 0.8,  # Moderate verifiability risk sensitivity
                'beta_weights': {
                    'regulatory_mandate': 2.0,
                    'industry_regulation': 1.7,
                    'best_practice': 1.2,
                    'vendor_recommendation': 1.0
                }
            },
            'risk_mappings': {
                'regulatory_violation': {'Ra': 0.05, 'Rv': 0.1, 'priority': 'CRITICAL'},
                'financial_loss': {'Ra': 0.1, 'Rv': 0.2, 'priority': 'HIGH'},
                'reputation_damage': {'Ra': 0.2, 'Rv': 0.15, 'priority': 'HIGH'},
                'operational_risk': {'Ra': 0.25, 'Rv': 0.3, 'priority': 'MEDIUM'}
            },
            'thresholds': {
                'regulatory_critical': 0.98,
                'financial_material': 0.92,
                'customer_facing': 0.85,
                'internal_operations': 0.70
            }
        }
        
        self.profiles['finance'] = profile
        return profile
        
    def get_profile(self, sector: str) -> Optional[Dict[str, Any]]:
        """Get sector-specific profile"""
        return self.profiles.get(sector.lower())
        
    def apply_profile(self, sector: str, measure_function: NISTMeasureFunction, 
                     governance: NISTThresholdGovernance) -> None:
        """Apply sector profile to measure function and governance"""
        profile = self.get_profile(sector)
        if not profile:
            raise ValueError(f"Profile not found for sector: {sector}")
            
        # Apply parameter settings
        params = profile['parameters']
        measure_function.lambda1 = params['lambda1']
        measure_function.lambda2 = params['lambda2']
        
        # Set thresholds
        for context, threshold in profile['thresholds'].items():
            governance.set_threshold(
                context, 
                threshold, 
                f"Applied {sector} sector profile"
            )
            
        logger.info(f"Applied {sector} profile successfully")

# Example usage and demonstration
def demonstrate_nist_psi_integration():
    """Demonstrate the three-phase NIST-Ψ(x) integration"""
    
    print("=== NIST Cybersecurity Framework + Ψ(x) Integration Demo ===\n")
    
    # Phase 1: Measure Function Enhancement
    print("Phase 1: Measure Function Enhancement")
    print("-" * 40)
    
    measure = NISTMeasureFunction()
    
    # Add technical characteristics
    measure.add_technical_characteristic(TechnicalCharacteristic(
        name="Encryption Implementation",
        source_evidence=0.95,
        non_source_evidence=0.85,
        confidence=0.9,
        category="Cryptographic Controls"
    ))
    
    measure.add_technical_characteristic(TechnicalCharacteristic(
        name="Access Control System",
        source_evidence=0.88,
        non_source_evidence=0.92,
        confidence=0.85,
        category="Identity Management"
    ))
    
    # Add socio-technical risks
    measure.add_socio_technical_risk(SocioTechnicalRisk(
        name="Insider Threat",
        authority_risk=0.3,
        verifiability_risk=0.4,
        category=RiskCategory.OPERATIONAL,
        impact_level="HIGH",
        likelihood=0.2
    ))
    
    measure.add_socio_technical_risk(SocioTechnicalRisk(
        name="Third-party Integration",
        authority_risk=0.5,
        verifiability_risk=0.3,
        category=RiskCategory.TECHNICAL,
        impact_level="MEDIUM",
        likelihood=0.6
    ))
    
    # Add regulatory evidence
    measure.add_regulatory_evidence(RegulatoryEvidence(
        source="NIST SP 800-53",
        authority_level="CANONICAL",
        beta_weight=1.5,
        certification_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
    ))
    
    # Compute Ψ(x)
    result = measure.measure_psi(alpha=0.7)
    
    print(f"Ψ(x) = {result.value:.3f} ({result.confidence_level} confidence)")
    print("Audit Trail:")
    for step in result.audit_trail:
        print(f"  • {step}")
    print()
    
    # Phase 2: Threshold Governance
    print("Phase 2: Threshold Governance")
    print("-" * 40)
    
    governance = NISTThresholdGovernance()
    governance.set_threshold("critical_infrastructure", 0.90, "Initial risk tolerance")
    governance.set_threshold("business_operations", 0.75, "Standard operations threshold")
    
    # Demonstrate threshold transfer
    old_beta = 1.5
    new_beta = 1.8
    new_threshold = governance.transfer_threshold("critical_infrastructure", old_beta, new_beta)
    print(f"Threshold transfer: {old_beta} → {new_beta}, new threshold: {new_threshold:.3f}")
    
    # Demonstrate harmonization
    harmonized = governance.harmonize_thresholds(["business_operations"], "critical_infrastructure")
    print(f"Harmonized thresholds: {harmonized}")
    print()
    
    # Phase 3: Profile Development
    print("Phase 3: Profile Development")
    print("-" * 40)
    
    profiles = NISTProfileDevelopment()
    
    # Create healthcare profile
    healthcare = profiles.create_healthcare_profile()
    print("Healthcare Profile:")
    print(f"  Default α: {healthcare['parameters']['default_alpha']}")
    print(f"  FDA β weight: {healthcare['parameters']['beta_weights']['FDA_approved']}")
    print(f"  Patient critical threshold: {healthcare['thresholds']['patient_critical']}")
    
    # Create finance profile
    finance = profiles.create_finance_profile()
    print("\nFinance Profile:")
    print(f"  Default α: {finance['parameters']['default_alpha']}")
    print(f"  Regulatory β weight: {finance['parameters']['beta_weights']['regulatory_mandate']}")
    print(f"  Regulatory critical threshold: {finance['thresholds']['regulatory_critical']}")
    
    print("\n=== Integration Complete ===")

if __name__ == "__main__":
    demonstrate_nist_psi_integration()