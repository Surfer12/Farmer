#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""
Invisible Equipment Validation System

Implements the validation protocols for equipment invisibility testing,
including IMU data processing, micro-correction detection, and statistical analysis.
"""

import numpy as np
from scipy import signal, stats
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
from collections import deque
from enum import Enum

class MetricType(Enum):
    """Types of invisibility metrics."""
    MICRO_CORRECTION_RATE = "mcr"
    MOVEMENT_SMOOTHNESS = "smoothness"
    PATH_CONSISTENCY = "path_consistency"
    SLIP_STALL_EVENTS = "slip_stall"
    INVISIBILITY_SCORE = "invisibility"
    EFFORTLESSNESS = "effortlessness"
    DISRUPTION_COUNT = "disruptions"

@dataclass
class IMUData:
    """Container for IMU sensor data."""
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    
    @property
    def accel_magnitude(self) -> float:
        """Calculate acceleration magnitude."""
        return np.sqrt(self.accel_x**2 + self.accel_y**2 + self.accel_z**2)
    
    @property
    def angular_velocity_magnitude(self) -> float:
        """Calculate angular velocity magnitude."""
        return np.sqrt(self.gyro_x**2 + self.gyro_y**2 + self.gyro_z**2)

class MicroCorrectionDetector:
    """Detects micro-corrections in IMU data streams."""
    
    def __init__(self, baseline_window: int = 1000, threshold_sigma: float = 2.0):
        """
        Initialize detector.
        
        Args:
            baseline_window: Number of samples for baseline calculation
            threshold_sigma: Standard deviations above baseline for detection
        """
        self.baseline_window = baseline_window
        self.threshold_sigma = threshold_sigma
        self.baseline_buffer = deque(maxlen=baseline_window)
        self.correction_count = 0
        self.sample_count = 0
        
    def update(self, imu_data: IMUData) -> bool:
        """
        Process new IMU data and detect corrections.
        
        Args:
            imu_data: New IMU sample
            
        Returns:
            True if micro-correction detected
        """
        angular_vel = imu_data.angular_velocity_magnitude
        self.baseline_buffer.append(angular_vel)
        self.sample_count += 1
        
        if len(self.baseline_buffer) < self.baseline_window:
            return False
            
        baseline_mean = np.mean(self.baseline_buffer)
        baseline_std = np.std(self.baseline_buffer)
        threshold = baseline_mean + self.threshold_sigma * baseline_std
        
        if angular_vel > threshold:
            self.correction_count += 1
            return True
        return False
    
    def get_rate_per_minute(self, sample_rate: float = 100.0) -> float:
        """Calculate micro-corrections per minute."""
        if self.sample_count == 0:
            return 0.0
        time_minutes = self.sample_count / sample_rate / 60.0
        return self.correction_count / time_minutes

class MovementSmoothnessAnalyzer:
    """Analyzes movement smoothness using multiple metrics."""
    
    def __init__(self, sample_rate: float = 100.0):
        """
        Initialize analyzer.
        
        Args:
            sample_rate: IMU sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.position_buffer = []
        self.velocity_buffer = []
        self.acceleration_buffer = []
        
    def update(self, imu_data: IMUData):
        """Add new IMU data to buffers."""
        self.acceleration_buffer.append([
            imu_data.accel_x,
            imu_data.accel_y,
            imu_data.accel_z
        ])
        
        # Integrate to get velocity (simplified)
        if len(self.acceleration_buffer) > 1:
            dt = 1.0 / self.sample_rate
            vel = self.velocity_buffer[-1] if self.velocity_buffer else [0, 0, 0]
            new_vel = [
                vel[0] + imu_data.accel_x * dt,
                vel[1] + imu_data.accel_y * dt,
                vel[2] + imu_data.accel_z * dt
            ]
            self.velocity_buffer.append(new_vel)
            
            # Integrate to get position
            if self.position_buffer:
                pos = self.position_buffer[-1]
                new_pos = [
                    pos[0] + new_vel[0] * dt,
                    pos[1] + new_vel[1] * dt,
                    pos[2] + new_vel[2] * dt
                ]
                self.position_buffer.append(new_pos)
            else:
                self.position_buffer.append([0, 0, 0])
    
    def calculate_jerk(self) -> float:
        """Calculate RMS jerk (third derivative of position)."""
        if len(self.acceleration_buffer) < 2:
            return 0.0
            
        accel_array = np.array(self.acceleration_buffer)
        jerk = np.diff(accel_array, axis=0) * self.sample_rate
        rms_jerk = np.sqrt(np.mean(jerk**2))
        return float(rms_jerk)
    
    def calculate_spectral_smoothness(self) -> float:
        """Calculate spectral arc length (SPARC) metric."""
        if len(self.velocity_buffer) < 10:
            return 0.0
            
        vel_magnitude = [np.linalg.norm(v) for v in self.velocity_buffer]
        
        # Normalize velocity
        vel_array = np.array(vel_magnitude)
        if np.max(np.abs(vel_array)) == 0:
            return 0.0
        vel_norm = vel_array / np.max(np.abs(vel_array))
        
        # Calculate FFT
        freq, psd = signal.periodogram(vel_norm, fs=self.sample_rate)
        
        # Normalize PSD
        psd_norm = psd / np.sum(psd)
        
        # Calculate spectral arc length
        arc_length = -np.sum(np.sqrt(1 + np.diff(psd_norm)**2))
        return float(arc_length)

class PathConsistencyAnalyzer:
    """Analyzes consistency of movement paths."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.path_segments = []
        self.entry_speeds = []
        
    def add_path_segment(self, positions: List[List[float]], entry_speed: float):
        """
        Add a path segment for analysis.
        
        Args:
            positions: List of [x, y, z] positions
            entry_speed: Speed at segment entry
        """
        if len(positions) < 3:
            return
            
        # Calculate radius of curvature using three-point method
        radii = []
        for i in range(1, len(positions) - 1):
            p1 = np.array(positions[i-1])
            p2 = np.array(positions[i])
            p3 = np.array(positions[i+1])
            
            # Calculate radius using circumcircle
            radius = self._calculate_radius(p1, p2, p3)
            if radius is not None:
                radii.append(radius)
        
        if radii:
            self.path_segments.append({
                'entry_speed': entry_speed,
                'mean_radius': np.mean(radii),
                'radius_variance': np.var(radii)
            })
            self.entry_speeds.append(entry_speed)
    
    def _calculate_radius(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
        """Calculate radius of curvature from three points."""
        # Calculate side lengths
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        
        # Calculate area using Heron's formula
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        
        if area_sq <= 0:
            return None
            
        area = np.sqrt(area_sq)
        
        # Calculate radius
        if area == 0:
            return None
        radius = (a * b * c) / (4 * area)
        return radius
    
    def get_variance_at_speed(self, target_speed: float, tolerance: float = 0.1) -> Optional[float]:
        """Get path variance at a specific entry speed."""
        matching_segments = [
            seg for seg in self.path_segments
            if abs(seg['entry_speed'] - target_speed) < tolerance
        ]
        
        if not matching_segments:
            return None
            
        radii = [seg['mean_radius'] for seg in matching_segments]
        return float(np.var(radii))

class SlipStallDetector:
    """Detects and analyzes slip/stall events."""
    
    def __init__(self, accel_threshold: float = 3.0, recovery_threshold: float = 0.5):
        """
        Initialize detector.
        
        Args:
            accel_threshold: Acceleration threshold for event detection (in g)
            recovery_threshold: Threshold for recovery detection
        """
        self.accel_threshold = accel_threshold * 9.81  # Convert to m/s²
        self.recovery_threshold = recovery_threshold * 9.81
        self.events = []
        self.in_event = False
        self.event_start_time = None
        self.baseline_accel = None
        
    def update(self, imu_data: IMUData, timestamp: float):
        """Process IMU data for slip/stall detection."""
        accel_mag = imu_data.accel_magnitude
        
        # Maintain baseline
        if self.baseline_accel is None:
            self.baseline_accel = accel_mag
        else:
            self.baseline_accel = 0.95 * self.baseline_accel + 0.05 * accel_mag
        
        # Detect event start
        if not self.in_event and abs(accel_mag - self.baseline_accel) > self.accel_threshold:
            self.in_event = True
            self.event_start_time = timestamp
        
        # Detect recovery
        elif self.in_event and abs(accel_mag - self.baseline_accel) < self.recovery_threshold:
            self.in_event = False
            recovery_time = timestamp - self.event_start_time
            self.events.append({
                'start_time': self.event_start_time,
                'recovery_time': recovery_time,
                'peak_acceleration': accel_mag
            })
    
    def get_event_statistics(self) -> Dict[str, float]:
        """Calculate statistics for detected events."""
        if not self.events:
            return {
                'event_count': 0,
                'mean_recovery_time': 0.0,
                'recovery_time_cv': 0.0
            }
        
        recovery_times = [e['recovery_time'] for e in self.events]
        mean_recovery = np.mean(recovery_times)
        std_recovery = np.std(recovery_times)
        
        return {
            'event_count': len(self.events),
            'mean_recovery_time': float(mean_recovery),
            'recovery_time_cv': float(std_recovery / mean_recovery) if mean_recovery > 0 else 0.0
        }

class InvisibilityValidator:
    """Main validation system for equipment invisibility."""
    
    def __init__(self, sample_rate: float = 100.0):
        """
        Initialize validator.
        
        Args:
            sample_rate: IMU sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.mcr_detector = MicroCorrectionDetector()
        self.smoothness_analyzer = MovementSmoothnessAnalyzer(sample_rate)
        self.path_analyzer = PathConsistencyAnalyzer()
        self.slip_detector = SlipStallDetector()
        
        self.subjective_scores = {
            'invisibility': [],
            'effortlessness': [],
            'disruptions': []
        }
        
        self.current_time = 0.0
        
    def process_imu_sample(self, imu_data: IMUData):
        """Process a single IMU sample."""
        self.mcr_detector.update(imu_data)
        self.smoothness_analyzer.update(imu_data)
        self.slip_detector.update(imu_data, self.current_time)
        self.current_time += 1.0 / self.sample_rate
        
    def add_subjective_rating(self, metric: str, value: float):
        """Add subjective rating."""
        if metric in self.subjective_scores:
            self.subjective_scores[metric].append(value)
    
    def calculate_invisibility_score(self) -> float:
        """
        Calculate overall invisibility score.
        
        Returns:
            Score from 0 to 1, where 1 is perfectly invisible
        """
        # Get objective metrics
        mcr = self.mcr_detector.get_rate_per_minute(self.sample_rate)
        jerk = self.smoothness_analyzer.calculate_jerk()
        slip_stats = self.slip_detector.get_event_statistics()
        
        # Normalize metrics (these thresholds would be empirically determined)
        mcr_score = np.exp(-mcr / 10.0)  # 10 corrections/min -> score ~0.37
        jerk_score = np.exp(-jerk / 5.0)  # 5 m/s³ -> score ~0.37
        slip_score = np.exp(-slip_stats['event_count'] / 3.0)  # 3 events -> score ~0.37
        
        # Get subjective scores
        subj_invisibility = np.mean(self.subjective_scores['invisibility']) / 10.0 if self.subjective_scores['invisibility'] else 0.5
        subj_effortless = np.mean(self.subjective_scores['effortlessness']) / 10.0 if self.subjective_scores['effortlessness'] else 0.5
        
        # Weighted combination
        weights = {
            'mcr': 0.3,
            'jerk': 0.2,
            'slip': 0.1,
            'invisibility': 0.25,
            'effortlessness': 0.15
        }
        
        total_score = (
            weights['mcr'] * mcr_score +
            weights['jerk'] * jerk_score +
            weights['slip'] * slip_score +
            weights['invisibility'] * subj_invisibility +
            weights['effortlessness'] * subj_effortless
        )
        
        return float(total_score)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive validation report."""
        mcr = self.mcr_detector.get_rate_per_minute(self.sample_rate)
        jerk = self.smoothness_analyzer.calculate_jerk()
        spectral = self.smoothness_analyzer.calculate_spectral_smoothness()
        slip_stats = self.slip_detector.get_event_statistics()
        invisibility = self.calculate_invisibility_score()
        
        report = {
            'objective_metrics': {
                'micro_correction_rate': mcr,
                'jerk_rms': jerk,
                'spectral_smoothness': spectral,
                'slip_stall_events': slip_stats
            },
            'subjective_metrics': {
                'invisibility_mean': float(np.mean(self.subjective_scores['invisibility'])) if self.subjective_scores['invisibility'] else None,
                'effortlessness_mean': float(np.mean(self.subjective_scores['effortlessness'])) if self.subjective_scores['effortlessness'] else None,
                'disruption_count': float(np.sum(self.subjective_scores['disruptions'])) if self.subjective_scores['disruptions'] else None
            },
            'overall_invisibility_score': invisibility,
            'recommendation': self._generate_recommendation(invisibility, mcr)
        }
        
        return report
    
    def _generate_recommendation(self, invisibility: float, mcr: float) -> str:
        """Generate recommendation based on metrics."""
        if invisibility >= 0.8 and mcr < 5.0:
            return "ACCEPT: Equipment achieves invisibility targets"
        elif invisibility >= 0.7:
            return "REFINE: Minor adjustments needed for optimal invisibility"
        else:
            return "REJECT: Significant redesign required"

class VariantComparator:
    """Compare multiple equipment variants."""
    
    def __init__(self):
        """Initialize comparator."""
        self.variants = {}
        
    def add_variant(self, name: str, validator: InvisibilityValidator):
        """Add a variant's validation results."""
        self.variants[name] = validator.generate_report()
    
    def select_optimal(self) -> Tuple[str, Dict]:
        """
        Select optimal variant based on invisibility criteria.
        
        Returns:
            Tuple of (variant_name, report)
        """
        if not self.variants:
            return None, None
        
        # Filter candidates meeting minimum criteria
        candidates = {
            name: report for name, report in self.variants.items()
            if report['overall_invisibility_score'] >= 0.8
            and report['objective_metrics']['micro_correction_rate'] < 10.0
        }
        
        if not candidates:
            # If no candidates meet criteria, select best available
            candidates = self.variants
        
        # Select based on lowest MCR with invisibility as tiebreaker
        best_variant = min(
            candidates.items(),
            key=lambda x: (
                x[1]['objective_metrics']['micro_correction_rate'],
                -x[1]['overall_invisibility_score']
            )
        )
        
        return best_variant
    
    def generate_comparison_report(self) -> Dict:
        """Generate comparative analysis of all variants."""
        if not self.variants:
            return {}
        
        optimal_name, optimal_report = self.select_optimal()
        
        comparison = {
            'variant_count': len(self.variants),
            'optimal_variant': optimal_name,
            'variants': {}
        }
        
        for name, report in self.variants.items():
            comparison['variants'][name] = {
                'invisibility_score': report['overall_invisibility_score'],
                'mcr': report['objective_metrics']['micro_correction_rate'],
                'recommendation': report['recommendation'],
                'is_optimal': name == optimal_name
            }
        
        return comparison

def example_validation_session():
    """Example validation session with simulated data."""
    
    # Create validators for three variants
    validators = {
        'variant_a': InvisibilityValidator(),
        'variant_b': InvisibilityValidator(),
        'variant_c': InvisibilityValidator()
    }
    
    # Simulate IMU data for each variant
    np.random.seed(42)
    
    for variant_name, validator in validators.items():
        # Simulate different characteristics for each variant
        if variant_name == 'variant_a':
            noise_level = 0.5  # Low noise (good)
            correction_rate = 0.05  # Low corrections (good)
        elif variant_name == 'variant_b':
            noise_level = 1.0  # Medium noise
            correction_rate = 0.1  # Medium corrections
        else:
            noise_level = 1.5  # High noise (bad)
            correction_rate = 0.15  # High corrections (bad)
        
        # Generate 1000 samples (10 seconds at 100Hz)
        for i in range(1000):
            # Simulate IMU data with varying characteristics
            base_accel = 9.81  # 1g baseline
            
            # Add noise and occasional corrections
            if np.random.random() < correction_rate:
                # Simulate a micro-correction
                gyro_magnitude = np.random.normal(5.0, 1.0)
            else:
                gyro_magnitude = np.random.normal(0.5, 0.1)
            
            imu_data = IMUData(
                timestamp=i/100.0,
                accel_x=np.random.normal(0, noise_level),
                accel_y=np.random.normal(0, noise_level),
                accel_z=base_accel + np.random.normal(0, noise_level),
                gyro_x=np.random.normal(0, gyro_magnitude/3),
                gyro_y=np.random.normal(0, gyro_magnitude/3),
                gyro_z=np.random.normal(0, gyro_magnitude/3)
            )
            
            validator.process_imu_sample(imu_data)
        
        # Add simulated subjective scores
        if variant_name == 'variant_a':
            validator.add_subjective_rating('invisibility', 9.0)
            validator.add_subjective_rating('effortlessness', 8.5)
            validator.add_subjective_rating('disruptions', 0)
        elif variant_name == 'variant_b':
            validator.add_subjective_rating('invisibility', 7.0)
            validator.add_subjective_rating('effortlessness', 7.0)
            validator.add_subjective_rating('disruptions', 2)
        else:
            validator.add_subjective_rating('invisibility', 5.0)
            validator.add_subjective_rating('effortlessness', 5.5)
            validator.add_subjective_rating('disruptions', 5)
    
    # Compare variants
    comparator = VariantComparator()
    for name, validator in validators.items():
        comparator.add_variant(name, validator)
    
    # Generate reports
    comparison = comparator.generate_comparison_report()
    optimal_name, optimal_report = comparator.select_optimal()
    
    print("=" * 60)
    print("EQUIPMENT INVISIBILITY VALIDATION REPORT")
    print("=" * 60)
    print(f"\nOptimal Variant: {optimal_name}")
    print(f"Invisibility Score: {optimal_report['overall_invisibility_score']:.3f}")
    print(f"Micro-Correction Rate: {optimal_report['objective_metrics']['micro_correction_rate']:.1f} /min")
    print(f"Recommendation: {optimal_report['recommendation']}")
    
    print("\n" + "-" * 40)
    print("VARIANT COMPARISON")
    print("-" * 40)
    
    for variant, data in comparison['variants'].items():
        status = "✓ OPTIMAL" if data['is_optimal'] else ""
        print(f"\n{variant.upper()} {status}")
        print(f"  Invisibility: {data['invisibility_score']:.3f}")
        print(f"  MCR: {data['mcr']:.1f} /min")
        print(f"  Status: {data['recommendation']}")
    
    # Save detailed report
    with open('invisibility_validation_report.json', 'w') as f:
        json.dump({
            'comparison': comparison,
            'optimal_variant': optimal_name,
            'detailed_reports': {name: validators[name].generate_report() for name in validators}
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Detailed report saved to: invisibility_validation_report.json")

if __name__ == "__main__":
    example_validation_session()