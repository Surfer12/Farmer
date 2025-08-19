#!/usr/bin/env python3
"""
SPDX-License-Identifier: GPL-3.0-only
SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

Double-Blind A/B/X Test Protocol for Invisible Equipment Validation

Manages randomization, data collection, and analysis for equipment
invisibility testing with proper blinding protocols.
"""

import numpy as np
import pandas as pd
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid


@dataclass
class TestConfiguration:
    """Test equipment configuration."""
    config_id: str
    description: str
    physical_specs: Dict
    test_code: str  # Blind code for rider
    

@dataclass
class TestSession:
    """Individual test session data."""
    session_id: str
    config_id: str
    test_code: str
    rider_id: str
    timestamp: datetime
    duration_minutes: float
    conditions: Dict
    imu_file: Optional[str] = None
    gps_file: Optional[str] = None
    subjective_scores: Optional[Dict] = None
    notes: Optional[str] = None


@dataclass
class SubjectiveRating:
    """Post-session subjective ratings."""
    invisibility_score: float  # 0-10, how often noticed equipment
    effortlessness_score: float  # 0-10, how automatic turns felt
    disruption_count: int  # number of times attention went to equipment
    confidence: float  # 0-10, confidence in ratings
    additional_notes: str = ""


class BlindTestProtocol:
    """Manages double-blind testing protocol for equipment validation."""
    
    def __init__(self, test_name: str, base_dir: str = "/workspace/data/blind_tests"):
        """
        Initialize blind test protocol.
        
        Args:
            test_name: Name/identifier for this test series
            base_dir: Base directory for test data storage
        """
        self.test_name = test_name
        self.base_dir = Path(base_dir)
        self.test_dir = self.base_dir / test_name
        
        # Create directory structure
        self.test_dir.mkdir(parents=True, exist_ok=True)
        (self.test_dir / "configs").mkdir(exist_ok=True)
        (self.test_dir / "sessions").mkdir(exist_ok=True)
        (self.test_dir / "data").mkdir(exist_ok=True)
        (self.test_dir / "analysis").mkdir(exist_ok=True)
        
        # Initialize test state
        self.configurations: Dict[str, TestConfiguration] = {}
        self.sessions: List[TestSession] = []
        self.randomization_seed: Optional[int] = None
        
        # Load existing test if present
        self._load_test_state()
    
    def add_configuration(self, config_id: str, description: str, 
                         physical_specs: Dict) -> str:
        """
        Add a test configuration with blind coding.
        
        Returns:
            test_code: Blind code for this configuration
        """
        # Generate random test code
        test_code = self._generate_test_code()
        
        config = TestConfiguration(
            config_id=config_id,
            description=description,
            physical_specs=physical_specs,
            test_code=test_code
        )
        
        self.configurations[config_id] = config
        self._save_test_state()
        
        print(f"Added configuration '{config_id}' with test code '{test_code}'")
        return test_code
    
    def generate_session_plan(self, rider_id: str, sessions_per_config: int = 5,
                            session_duration: float = 30.0,
                            randomization_seed: Optional[int] = None) -> List[Dict]:
        """
        Generate randomized session plan for blind testing.
        
        Args:
            rider_id: Identifier for the test rider
            sessions_per_config: Number of sessions per configuration
            session_duration: Session duration in minutes
            randomization_seed: Seed for reproducible randomization
        
        Returns:
            List of session plans with randomized order
        """
        if randomization_seed is not None:
            self.randomization_seed = randomization_seed
            random.seed(randomization_seed)
            np.random.seed(randomization_seed)
        
        # Create session list
        session_plan = []
        config_ids = list(self.configurations.keys())
        
        for config_id in config_ids:
            for session_num in range(sessions_per_config):
                session_plan.append({
                    'config_id': config_id,
                    'test_code': self.configurations[config_id].test_code,
                    'session_number': session_num + 1,
                    'rider_id': rider_id,
                    'duration_minutes': session_duration
                })
        
        # Randomize order
        random.shuffle(session_plan)
        
        # Add session IDs and suggested timing
        start_time = datetime.now()
        for i, session in enumerate(session_plan):
            session['session_id'] = f"{rider_id}_{i+1:03d}"
            session['suggested_start'] = start_time + timedelta(minutes=i * (session_duration + 15))
            session['order'] = i + 1
        
        # Save session plan
        plan_file = self.test_dir / f"session_plan_{rider_id}.json"
        with open(plan_file, 'w') as f:
            json.dump(session_plan, f, indent=2, default=str)
        
        print(f"Generated {len(session_plan)} sessions for rider {rider_id}")
        print(f"Session plan saved to: {plan_file}")
        
        return session_plan
    
    def start_session(self, session_id: str, config_id: str, rider_id: str,
                     conditions: Dict) -> TestSession:
        """
        Start a test session and return session object.
        """
        if config_id not in self.configurations:
            raise ValueError(f"Configuration {config_id} not found")
        
        session = TestSession(
            session_id=session_id,
            config_id=config_id,
            test_code=self.configurations[config_id].test_code,
            rider_id=rider_id,
            timestamp=datetime.now(),
            duration_minutes=0.0,  # Will be updated on completion
            conditions=conditions
        )
        
        self.sessions.append(session)
        
        print(f"Started session {session_id} with test code {session.test_code}")
        print(f"Conditions: {conditions}")
        
        return session
    
    def complete_session(self, session_id: str, duration_minutes: float,
                        imu_file: Optional[str] = None,
                        gps_file: Optional[str] = None,
                        notes: Optional[str] = None) -> None:
        """
        Mark session as complete and record data files.
        """
        session = self._find_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        
        session.duration_minutes = duration_minutes
        session.imu_file = imu_file
        session.gps_file = gps_file
        session.notes = notes
        
        self._save_test_state()
        print(f"Completed session {session_id} ({duration_minutes:.1f} minutes)")
    
    def collect_subjective_ratings(self, session_id: str, 
                                 ratings: SubjectiveRating) -> None:
        """
        Collect post-session subjective ratings.
        """
        session = self._find_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        
        session.subjective_scores = asdict(ratings)
        self._save_test_state()
        
        print(f"Collected subjective ratings for session {session_id}")
        print(f"  Invisibility: {ratings.invisibility_score}/10")
        print(f"  Effortlessness: {ratings.effortlessness_score}/10")
        print(f"  Disruptions: {ratings.disruption_count}")
    
    def get_rider_session_card(self, session_id: str) -> Dict:
        """
        Get session card for rider (blind information only).
        """
        session = self._find_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        
        return {
            'session_id': session_id,
            'test_code': session.test_code,
            'rider_id': session.rider_id,
            'suggested_duration': session.duration_minutes,
            'conditions': session.conditions,
            'instructions': self._get_rider_instructions()
        }
    
    def unblind_results(self) -> Dict:
        """
        Unblind test results for analysis (use only after all sessions complete).
        """
        if not all(s.subjective_scores for s in self.sessions):
            incomplete = [s.session_id for s in self.sessions if not s.subjective_scores]
            print(f"Warning: {len(incomplete)} sessions missing subjective scores")
        
        # Create unblinding map
        unblinding = {}
        for config_id, config in self.configurations.items():
            unblinding[config.test_code] = {
                'config_id': config_id,
                'description': config.description,
                'physical_specs': config.physical_specs
            }
        
        # Organize sessions by configuration
        sessions_by_config = {}
        for session in self.sessions:
            config_id = session.config_id
            if config_id not in sessions_by_config:
                sessions_by_config[config_id] = []
            sessions_by_config[config_id].append(asdict(session))
        
        results = {
            'test_name': self.test_name,
            'randomization_seed': self.randomization_seed,
            'unblinding_map': unblinding,
            'sessions_by_config': sessions_by_config,
            'total_sessions': len(self.sessions),
            'configurations_tested': len(self.configurations)
        }
        
        # Save unblinded results
        results_file = self.test_dir / "analysis" / "unblinded_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Unblinded results saved to: {results_file}")
        return results
    
    def generate_analysis_summary(self) -> Dict:
        """
        Generate summary statistics for test analysis.
        """
        if not self.sessions:
            return {"error": "No sessions completed"}
        
        # Group sessions by configuration
        config_stats = {}
        
        for config_id in self.configurations:
            config_sessions = [s for s in self.sessions if s.config_id == config_id]
            
            if not config_sessions:
                continue
            
            # Extract subjective scores
            invisibility_scores = []
            effortlessness_scores = []
            disruption_counts = []
            
            for session in config_sessions:
                if session.subjective_scores:
                    invisibility_scores.append(session.subjective_scores['invisibility_score'])
                    effortlessness_scores.append(session.subjective_scores['effortlessness_score'])
                    disruption_counts.append(session.subjective_scores['disruption_count'])
            
            config_stats[config_id] = {
                'test_code': self.configurations[config_id].test_code,
                'total_sessions': len(config_sessions),
                'completed_ratings': len(invisibility_scores),
                'invisibility': {
                    'mean': np.mean(invisibility_scores) if invisibility_scores else None,
                    'std': np.std(invisibility_scores) if invisibility_scores else None,
                    'values': invisibility_scores
                },
                'effortlessness': {
                    'mean': np.mean(effortlessness_scores) if effortlessness_scores else None,
                    'std': np.std(effortlessness_scores) if effortlessness_scores else None,
                    'values': effortlessness_scores
                },
                'disruptions': {
                    'mean': np.mean(disruption_counts) if disruption_counts else None,
                    'std': np.std(disruption_counts) if disruption_counts else None,
                    'values': disruption_counts
                }
            }
        
        # Find best configuration based on invisibility
        best_config = None
        best_invisibility = -1
        
        for config_id, stats in config_stats.items():
            if stats['invisibility']['mean'] is not None:
                if stats['invisibility']['mean'] > best_invisibility:
                    best_invisibility = stats['invisibility']['mean']
                    best_config = config_id
        
        summary = {
            'test_summary': {
                'test_name': self.test_name,
                'total_configurations': len(self.configurations),
                'total_sessions': len(self.sessions),
                'best_configuration': best_config,
                'best_invisibility_score': best_invisibility
            },
            'configuration_stats': config_stats
        }
        
        # Save summary
        summary_file = self.test_dir / "analysis" / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def _generate_test_code(self) -> str:
        """Generate random test code for blinding."""
        # Use 3-letter codes to avoid obvious patterns
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        while True:
            code = ''.join(random.choices(letters, k=3))
            # Ensure code is not already used
            if not any(config.test_code == code for config in self.configurations.values()):
                return code
    
    def _find_session(self, session_id: str) -> Optional[TestSession]:
        """Find session by ID."""
        for session in self.sessions:
            if session.session_id == session_id:
                return session
        return None
    
    def _get_rider_instructions(self) -> str:
        """Get standardized rider instructions."""
        return """
        BLIND TEST SESSION INSTRUCTIONS:
        
        1. You will test equipment identified only by the test code shown.
        2. Do not discuss or compare equipment during testing.
        3. Ride normally - do not try to analyze or test the equipment.
        4. Focus on your normal riding goals and techniques.
        5. After the session, you will rate:
           - How often you noticed the equipment (0-10)
           - How automatic/effortless turns felt (0-10)
           - Number of times attention went to equipment
        
        IMPORTANT: Ride for flow, not analysis.
        """
    
    def _save_test_state(self) -> None:
        """Save current test state to disk."""
        state = {
            'test_name': self.test_name,
            'randomization_seed': self.randomization_seed,
            'configurations': {k: asdict(v) for k, v in self.configurations.items()},
            'sessions': [asdict(s) for s in self.sessions]
        }
        
        state_file = self.test_dir / "test_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _load_test_state(self) -> None:
        """Load existing test state if present."""
        state_file = self.test_dir / "test_state.json"
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.randomization_seed = state.get('randomization_seed')
            
            # Load configurations
            for config_id, config_data in state.get('configurations', {}).items():
                self.configurations[config_id] = TestConfiguration(**config_data)
            
            # Load sessions
            for session_data in state.get('sessions', []):
                session_data['timestamp'] = datetime.fromisoformat(session_data['timestamp'])
                self.sessions.append(TestSession(**session_data))
            
            print(f"Loaded existing test state: {len(self.configurations)} configs, {len(self.sessions)} sessions")
            
        except Exception as e:
            print(f"Error loading test state: {e}")


def main():
    """Example usage of the blind test protocol."""
    
    # Initialize test
    test = BlindTestProtocol("fin_stiffness_comparison")
    
    # Add configurations
    test.add_configuration(
        config_id="baseline",
        description="Current production fin",
        physical_specs={"stiffness": 100, "material": "fiberglass"}
    )
    
    test.add_configuration(
        config_id="softer",
        description="20% softer flex",
        physical_specs={"stiffness": 80, "material": "fiberglass"}
    )
    
    test.add_configuration(
        config_id="stiffer", 
        description="20% stiffer flex",
        physical_specs={"stiffness": 120, "material": "fiberglass"}
    )
    
    # Generate session plan
    session_plan = test.generate_session_plan(
        rider_id="rider_001",
        sessions_per_config=5,
        session_duration=30.0,
        randomization_seed=42
    )
    
    print(f"\nGenerated {len(session_plan)} sessions")
    print("First few sessions:")
    for session in session_plan[:3]:
        print(f"  {session['order']}: {session['test_code']} ({session['config_id']})")
    
    # Example session execution
    print("\nExample session execution:")
    session = test.start_session(
        session_id="rider_001_001",
        config_id="baseline", 
        rider_id="rider_001",
        conditions={"wind_speed": 15, "wave_height": 1.2, "water_temp": 22}
    )
    
    # Get rider card (blinded info)
    card = test.get_rider_session_card("rider_001_001")
    print(f"Rider sees test code: {card['test_code']}")
    
    # Complete session
    test.complete_session(
        session_id="rider_001_001",
        duration_minutes=28.5,
        imu_file="imu_rider_001_001.csv",
        notes="Light wind, good conditions"
    )
    
    # Collect ratings
    ratings = SubjectiveRating(
        invisibility_score=8.5,
        effortlessness_score=7.0,
        disruption_count=2,
        confidence=8.0,
        additional_notes="Felt very natural"
    )
    
    test.collect_subjective_ratings("rider_001_001", ratings)
    
    # Generate summary (would do after all sessions)
    summary = test.generate_analysis_summary()
    print(f"\nTest summary: {summary['test_summary']}")


if __name__ == "__main__":
    main()