#!/usr/bin/env python3
"""
NIST Cybersecurity Framework Subcategory Mappings to Ψ(x) Parameters

This module provides detailed mappings between NIST CSF subcategories and Ψ(x) framework parameters,
enabling systematic quantitative assessment of cybersecurity posture.

© 2025 Ryan David Oates. All rights reserved.
Classification: Confidential — Internal Use Only
License: GPLv3. Use restricted by company policy and applicable NDAs.
SPDX-License-Identifier: GPL-3.0-only
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ImplementationTier(Enum):
    """NIST Implementation Tiers"""
    PARTIAL = 1
    RISK_INFORMED = 2
    REPEATABLE = 3
    ADAPTIVE = 4

class OutcomeStatus(Enum):
    """NIST Outcome Achievement Status"""
    NOT_ACHIEVED = "Not Achieved"
    PARTIALLY_ACHIEVED = "Partially Achieved"
    LARGELY_ACHIEVED = "Largely Achieved"
    FULLY_ACHIEVED = "Fully Achieved"

@dataclass
class NISTSubcategoryMapping:
    """Mapping of NIST subcategory to Ψ(x) parameters"""
    subcategory_id: str
    subcategory_name: str
    function: str
    category: str
    
    # Ψ(x) parameter mappings
    source_evidence_weight: float  # Contribution to S parameter
    non_source_evidence_weight: float  # Contribution to N parameter
    authority_risk_factor: float  # Contribution to Ra
    verifiability_risk_factor: float  # Contribution to Rv
    regulatory_beta_multiplier: float  # β enhancement factor
    
    # Implementation guidance
    measurement_criteria: List[str]
    evidence_sources: List[str]
    risk_indicators: List[str]
    
    # Tier-specific adjustments
    tier_adjustments: Dict[ImplementationTier, Dict[str, float]]

class NISTSubcategoryMapper:
    """Maps NIST CSF 2.0 subcategories to Ψ(x) parameters"""
    
    def __init__(self):
        self.mappings: Dict[str, NISTSubcategoryMapping] = {}
        self._initialize_mappings()
        
    def _initialize_mappings(self):
        """Initialize all NIST CSF subcategory mappings"""
        
        # IDENTIFY Function Mappings
        self._add_identify_mappings()
        
        # PROTECT Function Mappings  
        self._add_protect_mappings()
        
        # DETECT Function Mappings
        self._add_detect_mappings()
        
        # RESPOND Function Mappings
        self._add_respond_mappings()
        
        # RECOVER Function Mappings
        self._add_recover_mappings()
        
    def _add_identify_mappings(self):
        """Add IDENTIFY function subcategory mappings"""
        
        # ID.AM - Asset Management
        self.mappings["ID.AM-1"] = NISTSubcategoryMapping(
            subcategory_id="ID.AM-1",
            subcategory_name="Inventories of hardware, software, systems, services, and applications are maintained",
            function="IDENTIFY",
            category="Asset Management",
            source_evidence_weight=0.9,
            non_source_evidence_weight=0.7,
            authority_risk_factor=0.2,
            verifiability_risk_factor=0.1,
            regulatory_beta_multiplier=1.2,
            measurement_criteria=[
                "Completeness of asset inventory (>95%)",
                "Accuracy of asset attributes",
                "Frequency of inventory updates",
                "Integration with configuration management"
            ],
            evidence_sources=[
                "CMDB records",
                "Network discovery tools",
                "Asset management systems",
                "Manual verification audits"
            ],
            risk_indicators=[
                "Unknown/shadow IT assets",
                "Outdated inventory data",
                "Missing critical asset attributes"
            ],
            tier_adjustments={
                ImplementationTier.PARTIAL: {"source_evidence_weight": -0.2, "authority_risk_factor": +0.3},
                ImplementationTier.RISK_INFORMED: {"source_evidence_weight": -0.1, "authority_risk_factor": +0.1},
                ImplementationTier.REPEATABLE: {"source_evidence_weight": 0.0, "authority_risk_factor": 0.0},
                ImplementationTier.ADAPTIVE: {"source_evidence_weight": +0.1, "authority_risk_factor": -0.1}
            }
        )
        
        self.mappings["ID.AM-2"] = NISTSubcategoryMapping(
            subcategory_id="ID.AM-2",
            subcategory_name="Inventories of software platforms and applications are prioritized by criticality",
            function="IDENTIFY",
            category="Asset Management",
            source_evidence_weight=0.85,
            non_source_evidence_weight=0.75,
            authority_risk_factor=0.25,
            verifiability_risk_factor=0.15,
            regulatory_beta_multiplier=1.3,
            measurement_criteria=[
                "Business impact classification accuracy",
                "Criticality assessment completeness",
                "Risk-based prioritization methodology"
            ],
            evidence_sources=[
                "Business impact analysis",
                "Risk assessment reports",
                "Stakeholder interviews"
            ],
            risk_indicators=[
                "Inconsistent criticality ratings",
                "Missing business context",
                "Outdated impact assessments"
            ],
            tier_adjustments={
                ImplementationTier.PARTIAL: {"source_evidence_weight": -0.25, "authority_risk_factor": +0.4},
                ImplementationTier.RISK_INFORMED: {"source_evidence_weight": -0.1, "authority_risk_factor": +0.2},
                ImplementationTier.REPEATABLE: {"source_evidence_weight": 0.0, "authority_risk_factor": 0.0},
                ImplementationTier.ADAPTIVE: {"source_evidence_weight": +0.15, "authority_risk_factor": -0.15}
            }
        )
        
        # ID.GV - Governance
        self.mappings["ID.GV-1"] = NISTSubcategoryMapping(
            subcategory_id="ID.GV-1",
            subcategory_name="Organizational cybersecurity policy is established and communicated",
            function="IDENTIFY",
            category="Governance",
            source_evidence_weight=0.95,
            non_source_evidence_weight=0.8,
            authority_risk_factor=0.1,
            verifiability_risk_factor=0.05,
            regulatory_beta_multiplier=1.8,
            measurement_criteria=[
                "Policy completeness and coverage",
                "Communication effectiveness metrics",
                "Stakeholder acknowledgment rates",
                "Policy review and update frequency"
            ],
            evidence_sources=[
                "Approved policy documents",
                "Communication records",
                "Training completion rates",
                "Management attestations"
            ],
            risk_indicators=[
                "Policy gaps or conflicts",
                "Low awareness levels",
                "Outdated policy content"
            ],
            tier_adjustments={
                ImplementationTier.PARTIAL: {"regulatory_beta_multiplier": -0.4, "authority_risk_factor": +0.3},
                ImplementationTier.RISK_INFORMED: {"regulatory_beta_multiplier": -0.2, "authority_risk_factor": +0.1},
                ImplementationTier.REPEATABLE: {"regulatory_beta_multiplier": 0.0, "authority_risk_factor": 0.0},
                ImplementationTier.ADAPTIVE: {"regulatory_beta_multiplier": +0.2, "authority_risk_factor": -0.05}
            }
        )
        
    def _add_protect_mappings(self):
        """Add PROTECT function subcategory mappings"""
        
        # PR.AC - Identity Management and Access Control
        self.mappings["PR.AC-1"] = NISTSubcategoryMapping(
            subcategory_id="PR.AC-1",
            subcategory_name="Identities and credentials for authorized users are managed",
            function="PROTECT",
            category="Identity Management and Access Control",
            source_evidence_weight=0.9,
            non_source_evidence_weight=0.85,
            authority_risk_factor=0.15,
            verifiability_risk_factor=0.1,
            regulatory_beta_multiplier=1.6,
            measurement_criteria=[
                "Identity lifecycle management completeness",
                "Credential strength and rotation",
                "Access review frequency and effectiveness",
                "Privileged account management"
            ],
            evidence_sources=[
                "Identity management system logs",
                "Access review reports",
                "Credential policy compliance",
                "Audit findings"
            ],
            risk_indicators=[
                "Orphaned accounts",
                "Weak credential practices",
                "Excessive privileged access"
            ],
            tier_adjustments={
                ImplementationTier.PARTIAL: {"source_evidence_weight": -0.3, "authority_risk_factor": +0.4},
                ImplementationTier.RISK_INFORMED: {"source_evidence_weight": -0.15, "authority_risk_factor": +0.2},
                ImplementationTier.REPEATABLE: {"source_evidence_weight": 0.0, "authority_risk_factor": 0.0},
                ImplementationTier.ADAPTIVE: {"source_evidence_weight": +0.1, "authority_risk_factor": -0.1}
            }
        )
        
        # PR.DS - Data Security
        self.mappings["PR.DS-1"] = NISTSubcategoryMapping(
            subcategory_id="PR.DS-1",
            subcategory_name="Data is protected at rest",
            function="PROTECT",
            category="Data Security",
            source_evidence_weight=0.95,
            non_source_evidence_weight=0.8,
            authority_risk_factor=0.1,
            verifiability_risk_factor=0.15,
            regulatory_beta_multiplier=1.7,
            measurement_criteria=[
                "Encryption coverage percentage",
                "Key management effectiveness",
                "Data classification compliance",
                "Storage security controls"
            ],
            evidence_sources=[
                "Encryption implementation reports",
                "Key management audit results",
                "Data loss prevention logs",
                "Storage security assessments"
            ],
            risk_indicators=[
                "Unencrypted sensitive data",
                "Weak encryption algorithms",
                "Poor key management practices"
            ],
            tier_adjustments={
                ImplementationTier.PARTIAL: {"source_evidence_weight": -0.4, "verifiability_risk_factor": +0.3},
                ImplementationTier.RISK_INFORMED: {"source_evidence_weight": -0.2, "verifiability_risk_factor": +0.15},
                ImplementationTier.REPEATABLE: {"source_evidence_weight": 0.0, "verifiability_risk_factor": 0.0},
                ImplementationTier.ADAPTIVE: {"source_evidence_weight": +0.05, "verifiability_risk_factor": -0.1}
            }
        )
        
    def _add_detect_mappings(self):
        """Add DETECT function subcategory mappings"""
        
        # DE.AE - Anomalies and Events
        self.mappings["DE.AE-1"] = NISTSubcategoryMapping(
            subcategory_id="DE.AE-1",
            subcategory_name="Networks and network services are monitored to detect potential cybersecurity events",
            function="DETECT",
            category="Anomalies and Events",
            source_evidence_weight=0.85,
            non_source_evidence_weight=0.9,
            authority_risk_factor=0.2,
            verifiability_risk_factor=0.1,
            regulatory_beta_multiplier=1.4,
            measurement_criteria=[
                "Network monitoring coverage",
                "Event detection accuracy",
                "Alert response times",
                "False positive rates"
            ],
            evidence_sources=[
                "SIEM/SOAR system reports",
                "Network monitoring tools",
                "Incident response metrics",
                "Security operations dashboards"
            ],
            risk_indicators=[
                "Monitoring blind spots",
                "High false positive rates",
                "Delayed threat detection"
            ],
            tier_adjustments={
                ImplementationTier.PARTIAL: {"non_source_evidence_weight": -0.4, "authority_risk_factor": +0.4},
                ImplementationTier.RISK_INFORMED: {"non_source_evidence_weight": -0.2, "authority_risk_factor": +0.2},
                ImplementationTier.REPEATABLE: {"non_source_evidence_weight": 0.0, "authority_risk_factor": 0.0},
                ImplementationTier.ADAPTIVE: {"non_source_evidence_weight": +0.1, "authority_risk_factor": -0.1}
            }
        )
        
    def _add_respond_mappings(self):
        """Add RESPOND function subcategory mappings"""
        
        # RS.RP - Response Planning
        self.mappings["RS.RP-1"] = NISTSubcategoryMapping(
            subcategory_id="RS.RP-1",
            subcategory_name="Response plan is executed during or after an incident",
            function="RESPOND",
            category="Response Planning",
            source_evidence_weight=0.8,
            non_source_evidence_weight=0.85,
            authority_risk_factor=0.25,
            verifiability_risk_factor=0.2,
            regulatory_beta_multiplier=1.5,
            measurement_criteria=[
                "Response plan completeness",
                "Execution effectiveness",
                "Stakeholder coordination",
                "Timeline adherence"
            ],
            evidence_sources=[
                "Incident response reports",
                "Response plan documentation",
                "Exercise and drill results",
                "Post-incident reviews"
            ],
            risk_indicators=[
                "Plan execution failures",
                "Coordination breakdowns",
                "Delayed response actions"
            ],
            tier_adjustments={
                ImplementationTier.PARTIAL: {"source_evidence_weight": -0.3, "authority_risk_factor": +0.5},
                ImplementationTier.RISK_INFORMED: {"source_evidence_weight": -0.15, "authority_risk_factor": +0.25},
                ImplementationTier.REPEATABLE: {"source_evidence_weight": 0.0, "authority_risk_factor": 0.0},
                ImplementationTier.ADAPTIVE: {"source_evidence_weight": +0.2, "authority_risk_factor": -0.15}
            }
        )
        
    def _add_recover_mappings(self):
        """Add RECOVER function subcategory mappings"""
        
        # RC.RP - Recovery Planning
        self.mappings["RC.RP-1"] = NISTSubcategoryMapping(
            subcategory_id="RC.RP-1",
            subcategory_name="Recovery plan is executed during or after a cybersecurity incident",
            function="RECOVER",
            category="Recovery Planning",
            source_evidence_weight=0.85,
            non_source_evidence_weight=0.8,
            authority_risk_factor=0.2,
            verifiability_risk_factor=0.25,
            regulatory_beta_multiplier=1.6,
            measurement_criteria=[
                "Recovery time objectives (RTO)",
                "Recovery point objectives (RPO)",
                "Business continuity effectiveness",
                "Lessons learned integration"
            ],
            evidence_sources=[
                "Recovery execution reports",
                "Business continuity test results",
                "System restoration logs",
                "Stakeholder feedback"
            ],
            risk_indicators=[
                "RTO/RPO exceedances",
                "Incomplete recovery procedures",
                "Poor stakeholder communication"
            ],
            tier_adjustments={
                ImplementationTier.PARTIAL: {"source_evidence_weight": -0.35, "verifiability_risk_factor": +0.4},
                ImplementationTier.RISK_INFORMED: {"source_evidence_weight": -0.2, "verifiability_risk_factor": +0.2},
                ImplementationTier.REPEATABLE: {"source_evidence_weight": 0.0, "verifiability_risk_factor": 0.0},
                ImplementationTier.ADAPTIVE: {"source_evidence_weight": +0.15, "verifiability_risk_factor": -0.15}
            }
        )
        
    def get_mapping(self, subcategory_id: str) -> Optional[NISTSubcategoryMapping]:
        """Get mapping for specific subcategory"""
        return self.mappings.get(subcategory_id)
        
    def get_function_mappings(self, function: str) -> List[NISTSubcategoryMapping]:
        """Get all mappings for a specific function"""
        return [mapping for mapping in self.mappings.values() if mapping.function == function]
        
    def apply_tier_adjustments(self, mapping: NISTSubcategoryMapping, tier: ImplementationTier) -> NISTSubcategoryMapping:
        """Apply implementation tier adjustments to mapping"""
        if tier not in mapping.tier_adjustments:
            return mapping
            
        adjustments = mapping.tier_adjustments[tier]
        
        # Create adjusted mapping
        adjusted_mapping = NISTSubcategoryMapping(
            subcategory_id=mapping.subcategory_id,
            subcategory_name=mapping.subcategory_name,
            function=mapping.function,
            category=mapping.category,
            source_evidence_weight=max(0, min(1, mapping.source_evidence_weight + adjustments.get("source_evidence_weight", 0))),
            non_source_evidence_weight=max(0, min(1, mapping.non_source_evidence_weight + adjustments.get("non_source_evidence_weight", 0))),
            authority_risk_factor=max(0, min(1, mapping.authority_risk_factor + adjustments.get("authority_risk_factor", 0))),
            verifiability_risk_factor=max(0, min(1, mapping.verifiability_risk_factor + adjustments.get("verifiability_risk_factor", 0))),
            regulatory_beta_multiplier=max(0.5, mapping.regulatory_beta_multiplier + adjustments.get("regulatory_beta_multiplier", 0)),
            measurement_criteria=mapping.measurement_criteria,
            evidence_sources=mapping.evidence_sources,
            risk_indicators=mapping.risk_indicators,
            tier_adjustments=mapping.tier_adjustments
        )
        
        return adjusted_mapping
        
    def generate_assessment_template(self, subcategory_ids: List[str]) -> Dict[str, Any]:
        """Generate assessment template for specified subcategories"""
        template = {
            "assessment_metadata": {
                "version": "1.0",
                "framework": "NIST CSF 2.0",
                "psi_framework": "Ψ(x) Integration",
                "subcategories_count": len(subcategory_ids)
            },
            "subcategories": {}
        }
        
        for subcategory_id in subcategory_ids:
            mapping = self.get_mapping(subcategory_id)
            if not mapping:
                continue
                
            template["subcategories"][subcategory_id] = {
                "name": mapping.subcategory_name,
                "function": mapping.function,
                "category": mapping.category,
                "measurement_criteria": mapping.measurement_criteria,
                "evidence_sources": mapping.evidence_sources,
                "risk_indicators": mapping.risk_indicators,
                "psi_parameters": {
                    "source_evidence_weight": mapping.source_evidence_weight,
                    "non_source_evidence_weight": mapping.non_source_evidence_weight,
                    "authority_risk_factor": mapping.authority_risk_factor,
                    "verifiability_risk_factor": mapping.verifiability_risk_factor,
                    "regulatory_beta_multiplier": mapping.regulatory_beta_multiplier
                },
                "assessment_fields": {
                    "implementation_tier": "Select: PARTIAL|RISK_INFORMED|REPEATABLE|ADAPTIVE",
                    "outcome_status": "Select: NOT_ACHIEVED|PARTIALLY_ACHIEVED|LARGELY_ACHIEVED|FULLY_ACHIEVED",
                    "evidence_quality_score": "0.0 to 1.0",
                    "risk_level_assessment": "0.0 to 1.0",
                    "notes": "Assessment notes and observations"
                }
            }
            
        return template

def demonstrate_subcategory_mapping():
    """Demonstrate NIST subcategory mapping functionality"""
    
    print("=== NIST CSF Subcategory to Ψ(x) Parameter Mapping Demo ===\n")
    
    mapper = NISTSubcategoryMapper()
    
    # Show sample mappings
    sample_subcategories = ["ID.AM-1", "PR.AC-1", "DE.AE-1", "RS.RP-1", "RC.RP-1"]
    
    for subcategory_id in sample_subcategories:
        mapping = mapper.get_mapping(subcategory_id)
        if mapping:
            print(f"Subcategory: {mapping.subcategory_id}")
            print(f"Name: {mapping.subcategory_name}")
            print(f"Function: {mapping.function}")
            print(f"Ψ(x) Parameters:")
            print(f"  S weight: {mapping.source_evidence_weight:.2f}")
            print(f"  N weight: {mapping.non_source_evidence_weight:.2f}")
            print(f"  Ra factor: {mapping.authority_risk_factor:.2f}")
            print(f"  Rv factor: {mapping.verifiability_risk_factor:.2f}")
            print(f"  β multiplier: {mapping.regulatory_beta_multiplier:.2f}")
            
            # Show tier adjustments
            print(f"Tier Adjustments for PARTIAL implementation:")
            adjusted = mapper.apply_tier_adjustments(mapping, ImplementationTier.PARTIAL)
            print(f"  Adjusted S weight: {adjusted.source_evidence_weight:.2f}")
            print(f"  Adjusted Ra factor: {adjusted.authority_risk_factor:.2f}")
            print()
    
    # Generate assessment template
    print("Assessment Template Generation:")
    template = mapper.generate_assessment_template(["ID.AM-1", "PR.AC-1"])
    print(f"Generated template for {template['assessment_metadata']['subcategories_count']} subcategories")
    print("Template includes measurement criteria, evidence sources, and Ψ(x) parameters")
    
    print("\n=== Mapping Demonstration Complete ===")

if __name__ == "__main__":
    demonstrate_subcategory_mapping()
