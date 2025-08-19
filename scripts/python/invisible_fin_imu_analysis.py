#!/usr/bin/env python3
"""
Invisible Fin IMU Analysis Pipeline
Detects micro-corrections and analyzes movement smoothness for fin validation
"""

import numpy as np
from scipy import signal, stats
from scipy.spatial.transform import Rotation
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json

@dataclass
class IMUReading:
    """Single IMU measurement"""
    timestamp: float
    accel: np.ndarray  # [ax, ay, az] in m/s²
    gyro: np.ndarray   # [gx, gy, gz] in rad/s
    quat: np.ndarray   # [qw, qx, qy, qz] quaternion

class MicroCorrectionDetector:
    """Detects high-frequency control corrections indicating cognitive load"""
    
    def __init__(self, sample_rate: float = 100.0):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2.0
        
        # High-pass filter for micro-corrections (>2Hz)
        self.correction_cutoff = 2.0
        self.b_hp, self.a_hp = signal.butter(
            4, self.correction_cutoff / self.nyquist, 'high'
        )
        
        # Low-pass filter for intentional movements (<1Hz)
        self.intent_cutoff = 1.0
        self.b_lp, self.a_lp = signal.butter(
            4, self.intent_cutoff / self.nyquist, 'low'
        )
    
    def detect_corrections(self, gyro_data: np.ndarray, 
                         threshold_deg_s: float = 5.0) -> List[Dict]:
        """
        Detect micro-corrections from gyroscope data
        
        Args:
            gyro_data: Nx3 array of gyroscope readings (rad/s)
            threshold_deg_s: Angular velocity threshold in deg/s
        
        Returns:
            List of detected correction events
        """
        threshold_rad_s = np.deg2rad(threshold_deg_s)
        
        # Filter for high-frequency content
        gyro_hp = signal.filtfilt(self.b_hp, self.a_hp, gyro_data, axis=0)
        
        # Calculate angular velocity magnitude
        angular_vel = np.linalg.norm(gyro_hp, axis=1)
        
        # Find peaks above threshold
        peaks, properties = signal.find_peaks(
            angular_vel,
            height=threshold_rad_s,
            distance=int(0.1 * self.sample_rate)  # Min 100ms between corrections
        )
        
        corrections = []
        for i, peak_idx in enumerate(peaks):
            corrections.append({
                'time_idx': peak_idx,
                'time_s': peak_idx / self.sample_rate,
                'magnitude_deg_s': np.rad2deg(properties['peak_heights'][i]),
                'axis': self._dominant_axis(gyro_hp[peak_idx])
            })
        
        return corrections
    
    def calculate_mcr(self, gyro_data: np.ndarray, 
                     window_seconds: float = 60.0) -> float:
        """
        Calculate Micro-Correction Rate (corrections per minute)
        """
        corrections = self.detect_corrections(gyro_data)
        duration = len(gyro_data) / self.sample_rate
        
        if duration < window_seconds:
            # Scale to per-minute rate
            return len(corrections) * (60.0 / duration)
        else:
            # Sliding window average
            window_samples = int(window_seconds * self.sample_rate)
            rates = []
            
            for start in range(0, len(gyro_data) - window_samples, 
                             int(window_samples / 4)):
                window_data = gyro_data[start:start + window_samples]
                window_corrections = self.detect_corrections(window_data)
                rates.append(len(window_corrections) * (60.0 / window_seconds))
            
            return np.mean(rates) if rates else 0.0
    
    def _dominant_axis(self, gyro_reading: np.ndarray) -> str:
        """Identify dominant rotation axis"""
        abs_gyro = np.abs(gyro_reading)
        axes = ['roll', 'pitch', 'yaw']
        return axes[np.argmax(abs_gyro)]

class MovementSmoothnessAnalyzer:
    """Analyzes movement smoothness using multiple metrics"""
    
    def __init__(self, sample_rate: float = 100.0):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
    
    def calculate_jerk(self, accel_data: np.ndarray) -> np.ndarray:
        """
        Calculate jerk (derivative of acceleration)
        """
        # Use gradient for numerical differentiation
        jerk = np.gradient(accel_data, self.dt, axis=0)
        return jerk
    
    def dimensionless_jerk(self, trajectory: np.ndarray, 
                          duration: Optional[float] = None) -> float:
        """
        Calculate dimensionless jerk metric (Hogan & Sternad, 2009)
        Lower values indicate smoother movement
        """
        if duration is None:
            duration = len(trajectory) * self.dt
        
        # Calculate derivatives
        velocity = np.gradient(trajectory, self.dt, axis=0)
        accel = np.gradient(velocity, self.dt, axis=0)
        jerk = np.gradient(accel, self.dt, axis=0)
        
        # Peak velocity
        peak_vel = np.max(np.linalg.norm(velocity, axis=1))
        
        if peak_vel < 0.01:  # Avoid division by zero
            return float('inf')
        
        # Dimensionless jerk
        jerk_magnitude = np.linalg.norm(jerk, axis=1)
        dj = -np.sqrt(0.5 * duration**5 / peak_vel**2 * 
                     np.mean(jerk_magnitude**2))
        
        return abs(dj)
    
    def spectral_arc_length(self, velocity: np.ndarray) -> float:
        """
        Calculate spectral arc length (SPARC) metric
        More negative values indicate smoother movement
        """
        # Normalize velocity profile
        vel_magnitude = np.linalg.norm(velocity, axis=1)
        
        if np.max(vel_magnitude) < 0.01:
            return 0.0
        
        vel_norm = vel_magnitude / np.max(vel_magnitude)
        
        # Compute FFT
        freq = np.fft.rfftfreq(len(vel_norm), self.dt)
        fft_mag = np.abs(np.fft.rfft(vel_norm))
        
        # Normalize spectrum
        fft_norm = fft_mag / np.max(fft_mag)
        
        # Calculate arc length
        arc_length = -np.sum(np.sqrt(1 + np.diff(fft_norm)**2))
        
        return arc_length
    
    def analyze_smoothness(self, imu_data: List[IMUReading]) -> Dict:
        """
        Complete smoothness analysis
        """
        # Extract arrays
        accel = np.array([r.accel for r in imu_data])
        
        # Integrate for position (simple integration, could use Kalman filter)
        velocity = np.cumsum(accel * self.dt, axis=0)
        position = np.cumsum(velocity * self.dt, axis=0)
        
        # Calculate metrics
        jerk = self.calculate_jerk(accel)
        
        results = {
            'jerk_rms': float(np.sqrt(np.mean(jerk**2))),
            'dimensionless_jerk': float(self.dimensionless_jerk(position)),
            'spectral_smoothness': float(self.spectral_arc_length(velocity)),
            'peak_jerk': float(np.max(np.linalg.norm(jerk, axis=1))),
            'mean_jerk': float(np.mean(np.linalg.norm(jerk, axis=1)))
        }
        
        return results

class PathConsistencyAnalyzer:
    """Analyzes consistency of carving paths"""
    
    def __init__(self):
        self.min_carve_duration = 2.0  # seconds
        self.entry_speed_tolerance = 0.2  # m/s
    
    def segment_carves(self, imu_data: List[IMUReading], 
                      gps_data: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Segment continuous data into individual carves
        """
        # Extract gyro data for turn detection
        gyro = np.array([r.gyro for r in imu_data])
        timestamps = np.array([r.timestamp for r in imu_data])
        
        # Detect sustained turns (yaw rate)
        yaw_rate = gyro[:, 2]  # z-axis rotation
        
        # Smooth to remove noise
        window = int(0.5 * 100)  # 0.5 second window at 100Hz
        yaw_smooth = signal.savgol_filter(yaw_rate, window, 3)
        
        # Find carve segments (sustained yaw rate above threshold)
        threshold = np.deg2rad(10)  # 10 deg/s minimum turn rate
        carving = np.abs(yaw_smooth) > threshold
        
        # Find contiguous segments
        segments = []
        changes = np.diff(np.concatenate(([False], carving, [False])).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        for start, end in zip(starts, ends):
            duration = (timestamps[end] - timestamps[start])
            if duration >= self.min_carve_duration:
                segments.append({
                    'start_idx': start,
                    'end_idx': end,
                    'duration': duration,
                    'data': imu_data[start:end]
                })
        
        return segments
    
    def calculate_carve_radius(self, carve_segment: Dict) -> float:
        """
        Estimate turn radius from IMU data
        """
        data = carve_segment['data']
        
        # Extract velocities and yaw rates
        velocities = []
        yaw_rates = []
        
        for i in range(len(data) - 1):
            dt = data[i+1].timestamp - data[i].timestamp
            
            # Estimate forward velocity from accelerometer integration
            # (simplified - better with GPS or water speed sensor)
            accel = data[i].accel
            vel_est = np.linalg.norm(accel[:2]) * dt * 9.81  # rough estimate
            velocities.append(vel_est)
            
            yaw_rates.append(data[i].gyro[2])
        
        # Average radius: R = V / ω
        avg_velocity = np.mean(velocities)
        avg_yaw_rate = np.mean(np.abs(yaw_rates))
        
        if avg_yaw_rate < 0.01:  # Avoid division by zero
            return float('inf')
        
        radius = avg_velocity / avg_yaw_rate
        return radius
    
    def analyze_consistency(self, segments: List[Dict]) -> Dict:
        """
        Analyze consistency across multiple carves
        """
        if len(segments) < 2:
            return {
                'radius_variance': 0.0,
                'radius_cv': 0.0,
                'n_carves': len(segments)
            }
        
        radii = [self.calculate_carve_radius(seg) for seg in segments]
        radii = [r for r in radii if r < 100]  # Filter unrealistic values
        
        if len(radii) < 2:
            return {
                'radius_variance': 0.0,
                'radius_cv': 0.0,
                'n_carves': len(radii)
            }
        
        return {
            'radius_variance': float(np.var(radii)),
            'radius_cv': float(np.std(radii) / np.mean(radii)),
            'radius_mean': float(np.mean(radii)),
            'radius_std': float(np.std(radii)),
            'n_carves': len(radii)
        }

class SlipStallDetector:
    """Detects loss-of-control events"""
    
    def __init__(self, sample_rate: float = 100.0):
        self.sample_rate = sample_rate
        self.slip_threshold = 0.3  # Lateral acceleration ratio
        self.stall_threshold = 0.5  # Deceleration threshold
    
    def detect_slips(self, imu_data: List[IMUReading]) -> List[Dict]:
        """
        Detect slip events from sudden lateral accelerations
        """
        slips = []
        
        for i in range(1, len(imu_data) - 1):
            accel = imu_data[i].accel
            
            # Lateral vs longitudinal acceleration ratio
            lateral = abs(accel[1])  # y-axis
            forward = abs(accel[0])  # x-axis
            
            if forward > 0.1:  # Avoid division by zero
                ratio = lateral / forward
                
                if ratio > self.slip_threshold:
                    # Check for sudden change
                    prev_accel = imu_data[i-1].accel
                    accel_change = np.linalg.norm(accel - prev_accel)
                    
                    if accel_change > 2.0:  # m/s² threshold
                        slips.append({
                            'time': imu_data[i].timestamp,
                            'index': i,
                            'lateral_ratio': ratio,
                            'accel_change': accel_change
                        })
        
        return slips
    
    def detect_stalls(self, imu_data: List[IMUReading]) -> List[Dict]:
        """
        Detect stall events from sudden decelerations
        """
        stalls = []
        
        for i in range(1, len(imu_data)):
            accel = imu_data[i].accel
            prev_accel = imu_data[i-1].accel
            
            # Forward deceleration
            decel = prev_accel[0] - accel[0]
            
            if decel > self.stall_threshold:
                stalls.append({
                    'time': imu_data[i].timestamp,
                    'index': i,
                    'deceleration': decel
                })
        
        return stalls
    
    def analyze_recovery(self, imu_data: List[IMUReading], 
                        event_idx: int) -> Dict:
        """
        Analyze recovery pattern after an event
        """
        if event_idx >= len(imu_data) - 10:
            return {'recovery_time': None, 'pattern': 'incomplete'}
        
        # Look for return to stable state
        baseline_window = 20  # samples
        if event_idx > baseline_window:
            baseline = imu_data[event_idx - baseline_window:event_idx]
        else:
            baseline = imu_data[:event_idx]
        
        # Calculate baseline statistics
        baseline_gyro = np.array([r.gyro for r in baseline])
        baseline_std = np.std(baseline_gyro, axis=0)
        
        # Find recovery point
        recovery_idx = None
        for i in range(event_idx + 1, min(event_idx + 200, len(imu_data))):
            current_gyro = imu_data[i].gyro
            if np.all(np.abs(current_gyro) < 3 * baseline_std):
                recovery_idx = i
                break
        
        if recovery_idx:
            recovery_time = (imu_data[recovery_idx].timestamp - 
                           imu_data[event_idx].timestamp)
            pattern = 'smooth' if recovery_time < 1.0 else 'delayed'
        else:
            recovery_time = None
            pattern = 'incomplete'
        
        return {
            'recovery_time': recovery_time,
            'pattern': pattern,
            'samples_to_recovery': recovery_idx - event_idx if recovery_idx else None
        }

class InvisibleFinAnalyzer:
    """Main analysis pipeline for invisible fin validation"""
    
    def __init__(self, sample_rate: float = 100.0):
        self.sample_rate = sample_rate
        self.correction_detector = MicroCorrectionDetector(sample_rate)
        self.smoothness_analyzer = MovementSmoothnessAnalyzer(sample_rate)
        self.consistency_analyzer = PathConsistencyAnalyzer()
        self.event_detector = SlipStallDetector(sample_rate)
    
    def process_session(self, imu_data: List[IMUReading]) -> Dict:
        """
        Complete analysis of a surf session
        """
        # Extract arrays
        gyro_data = np.array([r.gyro for r in imu_data])
        
        # Micro-corrections
        mcr = self.correction_detector.calculate_mcr(gyro_data)
        corrections = self.correction_detector.detect_corrections(gyro_data)
        
        # Movement smoothness
        smoothness = self.smoothness_analyzer.analyze_smoothness(imu_data)
        
        # Path consistency
        segments = self.consistency_analyzer.segment_carves(imu_data)
        consistency = self.consistency_analyzer.analyze_consistency(segments)
        
        # Slip/stall events
        slips = self.event_detector.detect_slips(imu_data)
        stalls = self.event_detector.detect_stalls(imu_data)
        
        # Recovery analysis for events
        recoveries = []
        for slip in slips[:5]:  # Analyze first 5 slips
            recovery = self.event_detector.analyze_recovery(
                imu_data, slip['index']
            )
            recoveries.append(recovery)
        
        return {
            'micro_correction_rate': mcr,
            'total_corrections': len(corrections),
            'correction_distribution': self._analyze_correction_distribution(corrections),
            'smoothness_metrics': smoothness,
            'path_consistency': consistency,
            'slip_events': len(slips),
            'stall_events': len(stalls),
            'recovery_patterns': recoveries,
            'session_duration': imu_data[-1].timestamp - imu_data[0].timestamp
        }
    
    def _analyze_correction_distribution(self, corrections: List[Dict]) -> Dict:
        """Analyze the distribution of corrections by axis"""
        if not corrections:
            return {'roll': 0, 'pitch': 0, 'yaw': 0}
        
        axes = [c['axis'] for c in corrections]
        distribution = {
            'roll': axes.count('roll'),
            'pitch': axes.count('pitch'),
            'yaw': axes.count('yaw')
        }
        
        return distribution
    
    def compare_variants(self, results: Dict[str, List[Dict]]) -> Dict:
        """
        Statistical comparison of fin variants
        """
        comparison = {}
        
        # Extract MCR values for each variant
        mcr_data = {}
        for variant, sessions in results.items():
            mcr_data[variant] = [s['micro_correction_rate'] for s in sessions]
        
        # ANOVA test
        if len(mcr_data) >= 2:
            variants = list(mcr_data.keys())
            f_stat, p_value = stats.f_oneway(*mcr_data.values())
            
            comparison['anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # Pairwise comparisons if significant
            if p_value < 0.05 and len(variants) > 2:
                comparison['pairwise'] = {}
                for i, v1 in enumerate(variants):
                    for v2 in variants[i+1:]:
                        t_stat, p_val = stats.ttest_ind(
                            mcr_data[v1], mcr_data[v2]
                        )
                        comparison['pairwise'][f"{v1}_vs_{v2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'mean_diff': np.mean(mcr_data[v1]) - np.mean(mcr_data[v2])
                        }
        
        # Summary statistics
        comparison['summary'] = {}
        for variant, sessions in results.items():
            mcr_values = [s['micro_correction_rate'] for s in sessions]
            smoothness_values = [s['smoothness_metrics']['dimensionless_jerk'] 
                               for s in sessions]
            
            comparison['summary'][variant] = {
                'mcr_mean': np.mean(mcr_values),
                'mcr_std': np.std(mcr_values),
                'smoothness_mean': np.mean(smoothness_values),
                'smoothness_std': np.std(smoothness_values),
                'n_sessions': len(sessions)
            }
        
        return comparison
    
    def select_optimal(self, results: Dict[str, List[Dict]], 
                      weights: Optional[Dict] = None) -> str:
        """
        Select optimal fin variant based on weighted criteria
        """
        if weights is None:
            weights = {
                'mcr': 0.4,
                'smoothness': 0.3,
                'consistency': 0.15,
                'events': 0.15
            }
        
        scores = {}
        
        for variant, sessions in results.items():
            # Average metrics across sessions
            avg_mcr = np.mean([s['micro_correction_rate'] for s in sessions])
            avg_smoothness = np.mean([s['smoothness_metrics']['dimensionless_jerk'] 
                                     for s in sessions])
            avg_consistency = np.mean([s['path_consistency']['radius_cv'] 
                                      for s in sessions if s['path_consistency']['n_carves'] > 0])
            avg_events = np.mean([s['slip_events'] + s['stall_events'] 
                                 for s in sessions])
            
            # Normalize (lower is better for all metrics)
            # Use inverse scaling
            mcr_score = 1.0 / (1.0 + avg_mcr / 10.0)  # Normalize to ~0-1
            smoothness_score = 1.0 / (1.0 + avg_smoothness)
            consistency_score = 1.0 / (1.0 + avg_consistency)
            events_score = 1.0 / (1.0 + avg_events / 5.0)
            
            # Weighted sum
            scores[variant] = (
                weights['mcr'] * mcr_score +
                weights['smoothness'] * smoothness_score +
                weights['consistency'] * consistency_score +
                weights['events'] * events_score
            )
        
        # Return variant with highest score
        return max(scores, key=scores.get)

def load_imu_data(filepath: str) -> List[IMUReading]:
    """Load IMU data from file"""
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                reading = json.loads(line)
                data.append(IMUReading(
                    timestamp=reading['timestamp'],
                    accel=np.array(reading['accel']),
                    gyro=np.array(reading['gyro']),
                    quat=np.array(reading.get('quat', [1, 0, 0, 0]))
                ))
    
    return data

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze IMU data for fin invisibility')
    parser.add_argument('input', help='Input JSONL file with IMU data')
    parser.add_argument('--variant', help='Fin variant identifier')
    parser.add_argument('--output', help='Output JSON file for results')
    parser.add_argument('--sample-rate', type=float, default=100.0,
                       help='IMU sample rate in Hz')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading IMU data from {args.input}...")
    imu_data = load_imu_data(args.input)
    print(f"Loaded {len(imu_data)} samples")
    
    # Analyze
    analyzer = InvisibleFinAnalyzer(args.sample_rate)
    results = analyzer.process_session(imu_data)
    
    # Add variant identifier if provided
    if args.variant:
        results['variant'] = args.variant
    
    # Display results
    print("\n=== Invisibility Metrics ===")
    print(f"Micro-Correction Rate: {results['micro_correction_rate']:.2f} /min")
    print(f"Total Corrections: {results['total_corrections']}")
    print(f"Correction Distribution: {results['correction_distribution']}")
    
    print("\n=== Smoothness Metrics ===")
    for metric, value in results['smoothness_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== Path Consistency ===")
    for metric, value in results['path_consistency'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== Event Detection ===")
    print(f"Slip Events: {results['slip_events']}")
    print(f"Stall Events: {results['stall_events']}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()