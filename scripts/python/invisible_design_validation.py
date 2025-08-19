#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

"""
Invisible Design Validation Protocol

Tools for conducting double-blind A/B/X testing with objective metrics
for equipment invisibility assessment.
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import ttest_ind, f_oneway
import uuid


@dataclass
class TestSession:
    """Single test session data structure."""
    session_id: str
    variant_id: str  # Blinded identifier (A, B, C, etc.)
    user_id: str
    timestamp: str
    trial_number: int
    duration_seconds: float
    
    # IMU data (platform-mounted, not user-mounted)
    imu_data: List[Dict]  # [{timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z}, ...]
    
    # Subjective ratings
    equipment_invisibility: Optional[int]  # 0-10 scale
    effortlessness: Optional[int]  # 0-10 scale
    disruption_count: Optional[int]  # Number of attention disruptions
    
    # Environmental conditions
    temperature: Optional[float]
    humidity: Optional[float]
    wind_speed: Optional[float]
    surface_condition: Optional[str]


@dataclass
class ValidationResults:
    """Results of validation analysis."""
    variant_rankings: List[Tuple[str, float]]  # (variant_id, score)
    micro_correction_rates: Dict[str, float]  # variant_id -> corrections/minute
    movement_smoothness: Dict[str, float]  # variant_id -> smoothness score
    path_consistency: Dict[str, float]  # variant_id -> consistency score
    invisibility_ratings: Dict[str, float]  # variant_id -> average rating
    statistical_significance: Dict[str, float]  # variant_id -> p-value vs baseline
    recommendation: str


class InvisibilityValidator:
    """Main validation engine for invisible design testing."""
    
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize validator.
        
        Args:
            sampling_rate: IMU sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.sessions: List[TestSession] = []
    
    def add_session(self, session: TestSession) -> None:
        """Add a test session to the dataset."""
        self.sessions.append(session)
    
    def load_sessions_from_jsonl(self, filepath: str) -> None:
        """Load sessions from JSONL file."""
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                session = TestSession(**data)
                self.sessions.append(session)
    
    def save_sessions_to_jsonl(self, filepath: str) -> None:
        """Save sessions to JSONL file."""
        with open(filepath, 'w') as f:
            for session in self.sessions:
                json.dump(asdict(session), f)
                f.write('\n')
    
    def calculate_micro_corrections(self, imu_data: List[Dict]) -> float:
        """
        Calculate micro-correction rate from IMU data.
        
        High-frequency corrections indicate user fighting the equipment.
        
        Args:
            imu_data: List of IMU readings
            
        Returns:
            Corrections per minute
        """
        if len(imu_data) < 10:
            return 0.0
        
        # Extract gyro data (roll/yaw corrections)
        timestamps = np.array([d['timestamp'] for d in imu_data])
        gyro_x = np.array([d['gyro_x'] for d in imu_data])
        gyro_z = np.array([d['gyro_z'] for d in imu_data])
        
        # High-pass filter to isolate corrections (>2 Hz)
        nyquist = self.sampling_rate / 2
        high_cutoff = 2.0 / nyquist
        b, a = signal.butter(4, high_cutoff, btype='high')
        
        gyro_x_filt = signal.filtfilt(b, a, gyro_x)
        gyro_z_filt = signal.filtfilt(b, a, gyro_z)
        
        # Count peaks above threshold (corrections)
        correction_threshold = np.std(gyro_x_filt) * 2  # 2-sigma threshold
        corrections_x = len(signal.find_peaks(np.abs(gyro_x_filt), 
                                            height=correction_threshold)[0])
        corrections_z = len(signal.find_peaks(np.abs(gyro_z_filt), 
                                            height=correction_threshold)[0])
        
        total_corrections = corrections_x + corrections_z
        duration_minutes = (timestamps[-1] - timestamps[0]) / 60.0
        
        return total_corrections / duration_minutes if duration_minutes > 0 else 0.0
    
    def calculate_movement_smoothness(self, imu_data: List[Dict]) -> float:
        """
        Calculate movement smoothness score.
        
        Higher scores indicate smoother, more natural movement.
        
        Args:
            imu_data: List of IMU readings
            
        Returns:
            Smoothness score (0-1, higher is better)
        """
        if len(imu_data) < 10:
            return 0.0
        
        # Extract acceleration data
        accel_x = np.array([d['accel_x'] for d in imu_data])
        accel_y = np.array([d['accel_y'] for d in imu_data])
        accel_z = np.array([d['accel_z'] for d in imu_data])
        
        # Calculate jerk (third derivative of position)
        dt = 1.0 / self.sampling_rate
        jerk_x = np.gradient(np.gradient(np.gradient(accel_x))) / (dt**3)
        jerk_y = np.gradient(np.gradient(np.gradient(accel_y))) / (dt**3)
        jerk_z = np.gradient(np.gradient(np.gradient(accel_z))) / (dt**3)
        
        # RMS jerk as smoothness metric (lower is smoother)
        jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
        rms_jerk = np.sqrt(np.mean(jerk_magnitude**2))
        
        # Convert to 0-1 score (lower jerk = higher smoothness)
        # Normalize against typical range (empirically determined)
        typical_max_jerk = 1000.0  # Adjust based on your application
        smoothness = max(0.0, 1.0 - (rms_jerk / typical_max_jerk))
        
        return min(1.0, smoothness)
    
    def calculate_path_consistency(self, sessions: List[TestSession]) -> Dict[str, float]:
        """
        Calculate path consistency for each variant.
        
        Args:
            sessions: List of test sessions
            
        Returns:
            Dictionary of variant_id -> consistency score
        """
        consistency_scores = {}
        
        # Group sessions by variant
        variant_sessions = {}
        for session in sessions:
            if session.variant_id not in variant_sessions:
                variant_sessions[session.variant_id] = []
            variant_sessions[session.variant_id].append(session)
        
        for variant_id, variant_sessions_list in variant_sessions.items():
            if len(variant_sessions_list) < 2:
                consistency_scores[variant_id] = 0.0
                continue
            
            # Extract path features from each session
            path_features = []
            for session in variant_sessions_list:
                if len(session.imu_data) < 10:
                    continue
                
                # Simple path characterization using gyro integration
                gyro_z = np.array([d['gyro_z'] for d in session.imu_data])
                dt = 1.0 / self.sampling_rate
                heading_change = np.cumsum(gyro_z) * dt
                
                # Path features: total turn, max turn rate, turn smoothness
                total_turn = np.abs(heading_change[-1])
                max_turn_rate = np.max(np.abs(gyro_z))
                turn_smoothness = 1.0 / (1.0 + np.std(np.diff(gyro_z)))
                
                path_features.append([total_turn, max_turn_rate, turn_smoothness])
            
            if len(path_features) < 2:
                consistency_scores[variant_id] = 0.0
                continue
            
            # Calculate coefficient of variation (lower = more consistent)
            features_array = np.array(path_features)
            cv_scores = []
            for i in range(features_array.shape[1]):
                feature_values = features_array[:, i]
                if np.mean(feature_values) > 0:
                    cv = np.std(feature_values) / np.mean(feature_values)
                    cv_scores.append(cv)
            
            # Convert to consistency score (lower CV = higher consistency)
            avg_cv = np.mean(cv_scores) if cv_scores else 1.0
            consistency_scores[variant_id] = max(0.0, 1.0 - avg_cv)
        
        return consistency_scores
    
    def analyze_results(self, baseline_variant: Optional[str] = None) -> ValidationResults:
        """
        Analyze all collected sessions and generate recommendations.
        
        Args:
            baseline_variant: Variant ID to use as baseline for comparison
            
        Returns:
            ValidationResults with rankings and recommendations
        """
        if not self.sessions:
            raise ValueError("No sessions available for analysis")
        
        # Group sessions by variant
        variant_sessions = {}
        for session in self.sessions:
            if session.variant_id not in variant_sessions:
                variant_sessions[session.variant_id] = []
            variant_sessions[session.variant_id].append(session)
        
        # Calculate metrics for each variant
        micro_correction_rates = {}
        movement_smoothness = {}
        invisibility_ratings = {}
        
        for variant_id, sessions in variant_sessions.items():
            # Micro-correction rates
            correction_rates = []
            smoothness_scores = []
            invisibility_scores = []
            
            for session in sessions:
                correction_rate = self.calculate_micro_corrections(session.imu_data)
                correction_rates.append(correction_rate)
                
                smoothness = self.calculate_movement_smoothness(session.imu_data)
                smoothness_scores.append(smoothness)
                
                if session.equipment_invisibility is not None:
                    invisibility_scores.append(session.equipment_invisibility)
            
            micro_correction_rates[variant_id] = np.mean(correction_rates)
            movement_smoothness[variant_id] = np.mean(smoothness_scores)
            invisibility_ratings[variant_id] = np.mean(invisibility_scores) if invisibility_scores else 0.0
        
        # Calculate path consistency
        path_consistency = self.calculate_path_consistency(self.sessions)
        
        # Calculate composite scores (lower micro-corrections + higher invisibility)
        composite_scores = {}
        for variant_id in variant_sessions.keys():
            # Normalize metrics to 0-1 range
            max_corrections = max(micro_correction_rates.values()) if micro_correction_rates else 1.0
            correction_score = 1.0 - (micro_correction_rates[variant_id] / max_corrections) if max_corrections > 0 else 1.0
            
            invisibility_score = invisibility_ratings[variant_id] / 10.0  # Already 0-10 scale
            
            # Weighted composite (micro-corrections weighted more heavily)
            composite_scores[variant_id] = 0.6 * correction_score + 0.4 * invisibility_score
        
        # Rank variants by composite score
        variant_rankings = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Statistical significance testing
        statistical_significance = {}
        if baseline_variant and baseline_variant in variant_sessions:
            baseline_corrections = [
                self.calculate_micro_corrections(s.imu_data) 
                for s in variant_sessions[baseline_variant]
            ]
            
            for variant_id in variant_sessions.keys():
                if variant_id == baseline_variant:
                    continue
                
                variant_corrections = [
                    self.calculate_micro_corrections(s.imu_data) 
                    for s in variant_sessions[variant_id]
                ]
                
                if len(baseline_corrections) > 1 and len(variant_corrections) > 1:
                    _, p_value = ttest_ind(baseline_corrections, variant_corrections)
                    statistical_significance[variant_id] = p_value
                else:
                    statistical_significance[variant_id] = 1.0
        
        # Generate recommendation
        best_variant = variant_rankings[0][0]
        recommendation = f"Recommend variant {best_variant} with lowest micro-correction rate " \
                        f"({micro_correction_rates[best_variant]:.2f} corrections/min) and highest " \
                        f"invisibility rating ({invisibility_ratings[best_variant]:.1f}/10)."
        
        return ValidationResults(
            variant_rankings=variant_rankings,
            micro_correction_rates=micro_correction_rates,
            movement_smoothness=movement_smoothness,
            path_consistency=path_consistency,
            invisibility_ratings=invisibility_ratings,
            statistical_significance=statistical_significance,
            recommendation=recommendation
        )
    
    def generate_report(self, results: ValidationResults, output_file: str) -> None:
        """Generate a detailed validation report."""
        with open(output_file, 'w') as f:
            f.write("# Invisible Design Validation Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"{results.recommendation}\n\n")
            
            f.write("## Variant Rankings\n\n")
            f.write("| Rank | Variant | Composite Score | Micro-Corrections/min | Invisibility Rating | Path Consistency |\n")
            f.write("|------|---------|-----------------|----------------------|-------------------|------------------|\n")
            
            for i, (variant_id, score) in enumerate(results.variant_rankings, 1):
                corrections = results.micro_correction_rates.get(variant_id, 0.0)
                invisibility = results.invisibility_ratings.get(variant_id, 0.0)
                consistency = results.path_consistency.get(variant_id, 0.0)
                
                f.write(f"| {i} | {variant_id} | {score:.3f} | {corrections:.2f} | {invisibility:.1f}/10 | {consistency:.3f} |\n")
            
            f.write("\n## Statistical Significance\n\n")
            if results.statistical_significance:
                f.write("| Variant | p-value vs Baseline | Significant (p<0.05) |\n")
                f.write("|---------|--------------------|-----------------------|\n")
                for variant_id, p_value in results.statistical_significance.items():
                    significant = "Yes" if p_value < 0.05 else "No"
                    f.write(f"| {variant_id} | {p_value:.4f} | {significant} |\n")
            else:
                f.write("No baseline variant specified for comparison.\n")
            
            f.write("\n## Detailed Metrics\n\n")
            f.write("### Micro-Correction Rates (corrections per minute)\n")
            for variant_id, rate in results.micro_correction_rates.items():
                f.write(f"- {variant_id}: {rate:.2f}\n")
            
            f.write("\n### Movement Smoothness Scores (0-1, higher is better)\n")
            for variant_id, smoothness in results.movement_smoothness.items():
                f.write(f"- {variant_id}: {smoothness:.3f}\n")
            
            f.write("\n### Equipment Invisibility Ratings (0-10, higher is better)\n")
            for variant_id, rating in results.invisibility_ratings.items():
                f.write(f"- {variant_id}: {rating:.1f}/10\n")
            
            f.write("\n### Path Consistency Scores (0-1, higher is better)\n")
            for variant_id, consistency in results.path_consistency.items():
                f.write(f"- {variant_id}: {consistency:.3f}\n")


def create_test_session_template() -> TestSession:
    """Create a template test session for data collection."""
    return TestSession(
        session_id=str(uuid.uuid4()),
        variant_id="A",  # Replace with actual variant
        user_id="user_001",  # Replace with actual user ID
        timestamp=datetime.now().isoformat(),
        trial_number=1,
        duration_seconds=0.0,
        imu_data=[],
        equipment_invisibility=None,
        effortlessness=None,
        disruption_count=None,
        temperature=None,
        humidity=None,
        wind_speed=None,
        surface_condition=None
    )


def main():
    """Example usage of the validation system."""
    # Create validator
    validator = InvisibilityValidator(sampling_rate=100.0)
    
    # Example: Load existing data
    # validator.load_sessions_from_jsonl("test_sessions.jsonl")
    
    # Example: Create synthetic test data
    for variant in ['A', 'B', 'C']:
        for trial in range(5):
            session = create_test_session_template()
            session.variant_id = variant
            session.trial_number = trial + 1
            session.duration_seconds = 60.0
            
            # Generate synthetic IMU data
            time_points = np.linspace(0, 60, 6000)  # 100 Hz for 60 seconds
            session.imu_data = []
            
            for t in time_points:
                # Add some noise and variant-specific characteristics
                noise_level = {'A': 0.1, 'B': 0.05, 'C': 0.15}[variant]
                
                imu_point = {
                    'timestamp': t,
                    'accel_x': np.random.normal(0, noise_level),
                    'accel_y': np.random.normal(0, noise_level),
                    'accel_z': 9.81 + np.random.normal(0, noise_level),
                    'gyro_x': np.random.normal(0, noise_level * 10),
                    'gyro_y': np.random.normal(0, noise_level * 10),
                    'gyro_z': np.random.normal(0, noise_level * 10)
                }
                session.imu_data.append(imu_point)
            
            # Synthetic subjective ratings
            invisibility_base = {'A': 7, 'B': 9, 'C': 5}[variant]
            session.equipment_invisibility = invisibility_base + np.random.randint(-1, 2)
            session.effortlessness = invisibility_base + np.random.randint(-1, 2)
            session.disruption_count = max(0, 3 - invisibility_base + np.random.randint(-1, 2))
            
            validator.add_session(session)
    
    # Analyze results
    results = validator.analyze_results(baseline_variant='A')
    
    # Generate report
    validator.generate_report(results, 'validation_report.md')
    
    print("Validation complete. See validation_report.md for results.")
    print(f"Recommendation: {results.recommendation}")


if __name__ == "__main__":
    main()