#!/usr/bin/env python3
"""
SPDX-License-Identifier: GPL-3.0-only
SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

IMU Flow Analysis Tools for Invisible Equipment Optimization

Analyzes IMU data to detect micro-corrections and flow disruption patterns
for equipment invisibility validation.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FlowMetrics:
    """Container for flow state analysis results."""
    micro_correction_rate: float  # corrections per minute
    movement_smoothness: float    # RMS jerk (m/s³)
    path_consistency: float       # carve radius variance
    stall_event_count: int        # slip/stall events
    spectral_smoothness: float    # power concentration in low frequencies
    invisibility_score: float    # subjective rating if available


class IMUFlowAnalyzer:
    """Analyzes IMU data for equipment invisibility metrics."""
    
    def __init__(self, sample_rate: float = 100.0):
        """
        Initialize analyzer.
        
        Args:
            sample_rate: IMU sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # Analysis parameters
        self.correction_threshold_deg = 5.0  # degrees
        self.correction_time_window = 0.5    # seconds
        self.high_freq_cutoff = 2.0          # Hz for micro-corrections
        self.stall_accel_threshold = 2.0     # m/s² for stall detection
        
    def load_imu_data(self, filepath: str) -> pd.DataFrame:
        """
        Load IMU data from file.
        
        Expected columns: timestamp, accel_x, accel_y, accel_z, 
                         gyro_x, gyro_y, gyro_z, [optional: gps_lat, gps_lon, speed]
        """
        df = pd.read_csv(filepath)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate time deltas if not present
        if 'time_sec' not in df.columns:
            df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        
        return df
    
    def calculate_micro_corrections(self, df: pd.DataFrame) -> Dict:
        """
        Calculate micro-correction rate from gyroscope data.
        
        Detects high-frequency roll/yaw inputs exceeding threshold.
        """
        # Extract roll and yaw rates (assuming standard IMU orientation)
        roll_rate = df['gyro_x'].values  # rad/s
        yaw_rate = df['gyro_z'].values   # rad/s
        
        # Convert to degrees/second
        roll_rate_deg = np.degrees(roll_rate)
        yaw_rate_deg = np.degrees(yaw_rate)
        
        # High-pass filter to isolate corrections
        sos = signal.butter(4, self.high_freq_cutoff, btype='high', 
                          fs=self.sample_rate, output='sos')
        
        roll_filtered = signal.sosfilt(sos, roll_rate_deg)
        yaw_filtered = signal.sosfilt(sos, yaw_rate_deg)
        
        # Detect corrections exceeding threshold
        correction_magnitude = np.sqrt(roll_filtered**2 + yaw_filtered**2)
        correction_events = correction_magnitude > self.correction_threshold_deg
        
        # Count corrections with minimum separation
        min_separation_samples = int(self.correction_time_window * self.sample_rate)
        correction_indices = self._find_separated_peaks(
            correction_magnitude, correction_events, min_separation_samples
        )
        
        # Calculate rate per minute
        total_time_min = (df['time_sec'].iloc[-1] - df['time_sec'].iloc[0]) / 60.0
        correction_rate = len(correction_indices) / total_time_min
        
        return {
            'correction_rate_per_min': correction_rate,
            'total_corrections': len(correction_indices),
            'correction_timestamps': df['time_sec'].iloc[correction_indices].tolist(),
            'correction_magnitudes': correction_magnitude[correction_indices].tolist()
        }
    
    def calculate_movement_smoothness(self, df: pd.DataFrame) -> Dict:
        """
        Calculate movement smoothness using jerk analysis.
        """
        # Calculate velocity from acceleration (integrate)
        accel_x = df['accel_x'].values
        accel_y = df['accel_y'].values
        accel_z = df['accel_z'].values
        
        # Remove gravity component (assuming z is up)
        accel_z_corrected = accel_z - 9.81
        
        # Integrate to get velocity
        vel_x = np.cumsum(accel_x) * self.dt
        vel_y = np.cumsum(accel_y) * self.dt
        vel_z = np.cumsum(accel_z_corrected) * self.dt
        
        # Calculate jerk (derivative of acceleration)
        jerk_x = np.gradient(accel_x, self.dt)
        jerk_y = np.gradient(accel_y, self.dt)
        jerk_z = np.gradient(accel_z_corrected, self.dt)
        
        # Total jerk magnitude
        jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
        
        # RMS jerk as smoothness metric (lower is smoother)
        rms_jerk = np.sqrt(np.mean(jerk_magnitude**2))
        
        # Spectral smoothness (power in low frequencies)
        jerk_fft = fft(jerk_magnitude)
        freqs = fftfreq(len(jerk_magnitude), self.dt)
        power_spectrum = np.abs(jerk_fft)**2
        
        # Calculate power below 1 Hz vs total power
        low_freq_mask = np.abs(freqs) <= 1.0
        low_freq_power = np.sum(power_spectrum[low_freq_mask])
        total_power = np.sum(power_spectrum)
        spectral_smoothness = low_freq_power / total_power if total_power > 0 else 0
        
        return {
            'rms_jerk': rms_jerk,
            'spectral_smoothness': spectral_smoothness,
            'mean_jerk': np.mean(jerk_magnitude),
            'max_jerk': np.max(jerk_magnitude)
        }
    
    def calculate_path_consistency(self, df: pd.DataFrame) -> Dict:
        """
        Calculate path consistency from GPS data if available.
        """
        if not all(col in df.columns for col in ['gps_lat', 'gps_lon', 'speed']):
            return {'path_consistency': None, 'note': 'GPS data not available'}
        
        # Convert GPS to local coordinates (simplified)
        lat = df['gps_lat'].values
        lon = df['gps_lon'].values
        speed = df['speed'].values
        
        # Calculate heading and turn radius
        dx = np.gradient(lon)
        dy = np.gradient(lat)
        heading = np.arctan2(dy, dx)
        heading_rate = np.gradient(heading, self.dt)
        
        # Turn radius calculation (R = v / ω)
        turn_radius = np.abs(speed / (heading_rate + 1e-10))  # avoid division by zero
        
        # Filter out straight-line segments (low turn rates)
        turning_mask = np.abs(heading_rate) > 0.01  # rad/s threshold
        
        if np.sum(turning_mask) < 10:
            return {'path_consistency': None, 'note': 'Insufficient turning data'}
        
        # Group by speed bands and calculate radius consistency
        speed_bins = np.linspace(np.min(speed), np.max(speed), 5)
        radius_variance_by_speed = []
        
        for i in range(len(speed_bins) - 1):
            speed_mask = (speed >= speed_bins[i]) & (speed < speed_bins[i + 1])
            combined_mask = turning_mask & speed_mask
            
            if np.sum(combined_mask) > 3:
                radius_in_bin = turn_radius[combined_mask]
                radius_variance_by_speed.append(np.var(radius_in_bin))
        
        # Overall path consistency (lower variance = more consistent)
        path_consistency = np.mean(radius_variance_by_speed) if radius_variance_by_speed else None
        
        return {
            'path_consistency': path_consistency,
            'turn_radius_variance': np.var(turn_radius[turning_mask]),
            'mean_turn_radius': np.mean(turn_radius[turning_mask]),
            'turning_segments': np.sum(turning_mask)
        }
    
    def detect_stall_events(self, df: pd.DataFrame) -> Dict:
        """
        Detect slip/stall events from acceleration data.
        """
        # Calculate total acceleration magnitude
        accel_x = df['accel_x'].values
        accel_y = df['accel_y'].values
        accel_z = df['accel_z'].values
        
        accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Remove gravity baseline
        accel_baseline = np.median(accel_magnitude)
        accel_deviation = np.abs(accel_magnitude - accel_baseline)
        
        # Detect sudden acceleration changes
        stall_events = accel_deviation > self.stall_accel_threshold
        
        # Find event clusters
        min_event_separation = int(1.0 * self.sample_rate)  # 1 second minimum
        stall_indices = self._find_separated_peaks(
            accel_deviation, stall_events, min_event_separation
        )
        
        # Analyze recovery characteristics
        recovery_times = []
        for idx in stall_indices:
            # Find when acceleration returns to baseline
            recovery_window = accel_deviation[idx:idx + min_event_separation]
            recovery_idx = np.where(recovery_window < self.stall_accel_threshold / 2)[0]
            
            if len(recovery_idx) > 0:
                recovery_times.append(recovery_idx[0] * self.dt)
        
        return {
            'stall_event_count': len(stall_indices),
            'stall_timestamps': df['time_sec'].iloc[stall_indices].tolist(),
            'mean_recovery_time': np.mean(recovery_times) if recovery_times else None,
            'recovery_consistency': np.std(recovery_times) if len(recovery_times) > 1 else None
        }
    
    def analyze_session(self, df: pd.DataFrame, 
                       subjective_scores: Optional[Dict] = None) -> FlowMetrics:
        """
        Complete flow analysis of a session.
        """
        # Calculate all metrics
        corrections = self.calculate_micro_corrections(df)
        smoothness = self.calculate_movement_smoothness(df)
        consistency = self.calculate_path_consistency(df)
        stalls = self.detect_stall_events(df)
        
        # Extract subjective scores if provided
        invisibility_score = subjective_scores.get('invisibility', 0.0) if subjective_scores else 0.0
        
        return FlowMetrics(
            micro_correction_rate=corrections['correction_rate_per_min'],
            movement_smoothness=smoothness['rms_jerk'],
            path_consistency=consistency.get('path_consistency', 0.0) or 0.0,
            stall_event_count=stalls['stall_event_count'],
            spectral_smoothness=smoothness['spectral_smoothness'],
            invisibility_score=invisibility_score
        )
    
    def compare_configurations(self, results: Dict[str, FlowMetrics]) -> Dict:
        """
        Compare multiple equipment configurations.
        """
        comparison = {}
        
        # Extract metrics for comparison
        configs = list(results.keys())
        metrics = [
            'micro_correction_rate', 'movement_smoothness', 'path_consistency',
            'stall_event_count', 'spectral_smoothness', 'invisibility_score'
        ]
        
        for metric in metrics:
            values = [getattr(results[config], metric) for config in configs]
            
            # Find best configuration for this metric
            if metric in ['micro_correction_rate', 'movement_smoothness', 
                         'path_consistency', 'stall_event_count']:
                # Lower is better
                best_idx = np.argmin(values)
            else:
                # Higher is better
                best_idx = np.argmax(values)
            
            comparison[metric] = {
                'values': dict(zip(configs, values)),
                'best_config': configs[best_idx],
                'best_value': values[best_idx]
            }
        
        # Overall ranking based on invisibility priority
        rankings = {}
        for config in configs:
            score = 0
            # Weight invisibility and micro-corrections most heavily
            score += 3 * (10 - results[config].invisibility_score)  # invert for minimization
            score += 2 * results[config].micro_correction_rate
            score += results[config].movement_smoothness
            score += results[config].stall_event_count
            score -= results[config].spectral_smoothness  # higher is better
            
            rankings[config] = score
        
        best_overall = min(rankings.keys(), key=lambda k: rankings[k])
        
        comparison['overall_ranking'] = {
            'scores': rankings,
            'best_config': best_overall,
            'recommendation': f"Configuration '{best_overall}' shows best invisibility characteristics"
        }
        
        return comparison
    
    def _find_separated_peaks(self, signal: np.ndarray, events: np.ndarray, 
                            min_separation: int) -> List[int]:
        """Find peaks with minimum separation constraint."""
        event_indices = np.where(events)[0]
        
        if len(event_indices) == 0:
            return []
        
        # Filter for minimum separation
        separated_indices = [event_indices[0]]
        
        for idx in event_indices[1:]:
            if idx - separated_indices[-1] >= min_separation:
                separated_indices.append(idx)
        
        return separated_indices


def main():
    """Example usage of the IMU flow analyzer."""
    analyzer = IMUFlowAnalyzer(sample_rate=100.0)
    
    # Example: analyze multiple configurations
    configs = ['baseline', 'variant_a', 'variant_b']
    results = {}
    
    for config in configs:
        # Load IMU data (example filename pattern)
        filepath = f"/workspace/data/logs/imu_{config}.csv"
        
        try:
            df = analyzer.load_imu_data(filepath)
            
            # Add subjective scores (would come from post-session survey)
            subjective = {
                'invisibility': 7.5,  # Example score
                'effortlessness': 8.0
            }
            
            # Analyze session
            metrics = analyzer.analyze_session(df, subjective)
            results[config] = metrics
            
            print(f"\nConfiguration: {config}")
            print(f"  Micro-corrections: {metrics.micro_correction_rate:.2f}/min")
            print(f"  Movement smoothness: {metrics.movement_smoothness:.3f} m/s³")
            print(f"  Stall events: {metrics.stall_event_count}")
            print(f"  Invisibility score: {metrics.invisibility_score}/10")
            
        except FileNotFoundError:
            print(f"Data file not found for {config}: {filepath}")
            continue
    
    # Compare configurations
    if len(results) > 1:
        comparison = analyzer.compare_configurations(results)
        print(f"\nRecommendation: {comparison['overall_ranking']['recommendation']}")


if __name__ == "__main__":
    main()