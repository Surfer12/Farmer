#!/usr/bin/env python3
"""
Blind A/B/X Testing Protocol for Invisible Fin Validation
Manages randomization, data collection, and unblinding
"""

import random
import hashlib
import json
import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path

@dataclass
class FinVariant:
    """Fin variant specification"""
    true_id: str  # Actual variant (A, B, or X)
    blind_code: str  # Randomized code for blind testing
    specifications: Dict  # Physical parameters
    manufacture_date: str
    serial_number: str

@dataclass
class TestSession:
    """Single test session data"""
    session_id: str
    date: str
    time_start: str
    time_end: str
    rider_id: str
    blind_code: str  # Which fin was used (blind)
    true_variant: Optional[str]  # Revealed after unblinding
    conditions: Dict
    subjective_ratings: Dict
    objective_metrics: Optional[Dict]
    notes: str

class BlindTestProtocol:
    """Manages blind testing protocol"""
    
    def __init__(self, test_name: str, seed: Optional[int] = None):
        self.test_name = test_name
        self.seed = seed or random.randint(0, 999999)
        random.seed(self.seed)
        
        self.variants = {}
        self.sessions = []
        self.randomization_key = None
        self.is_blinded = True
        
        # Create test directory
        self.test_dir = Path(f"data/blind_tests/{test_name}")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.variants_file = self.test_dir / "variants.json"
        self.sessions_file = self.test_dir / "sessions.jsonl"
        self.key_file = self.test_dir / ".randomization_key.json"
        self.results_file = self.test_dir / "results.json"
    
    def register_variants(self, variants: List[Dict]):
        """
        Register fin variants and generate blind codes
        
        Args:
            variants: List of variant specifications with 'id' and 'specs'
        """
        # Generate random codes
        codes = self._generate_blind_codes(len(variants))
        
        for variant, code in zip(variants, codes):
            fin = FinVariant(
                true_id=variant['id'],
                blind_code=code,
                specifications=variant['specs'],
                manufacture_date=variant.get('manufacture_date', ''),
                serial_number=variant.get('serial_number', '')
            )
            self.variants[code] = fin
        
        # Save randomization key (encrypted)
        self._save_randomization_key()
        
        # Save public variant info (without true IDs)
        self._save_variants()
        
        print(f"Registered {len(variants)} variants with blind codes:")
        for code in sorted(self.variants.keys()):
            print(f"  {code}")
    
    def _generate_blind_codes(self, n: int) -> List[str]:
        """Generate random alphanumeric codes"""
        codes = []
        chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'  # Avoid confusing characters
        
        for _ in range(n):
            code = ''.join(random.choices(chars, k=6))
            while code in codes:  # Ensure uniqueness
                code = ''.join(random.choices(chars, k=6))
            codes.append(code)
        
        return codes
    
    def generate_test_sequence(self, rider_id: str, n_sessions: int = 9) -> List[str]:
        """
        Generate randomized test sequence for a rider
        Uses Latin square design for balanced ordering
        
        Args:
            rider_id: Unique rider identifier
            n_sessions: Number of test sessions (should be multiple of n_variants)
        
        Returns:
            List of blind codes in test order
        """
        if not self.variants:
            raise ValueError("No variants registered")
        
        codes = list(self.variants.keys())
        n_variants = len(codes)
        
        if n_sessions % n_variants != 0:
            print(f"Warning: {n_sessions} sessions not divisible by {n_variants} variants")
        
        # Generate Latin square for first n_variants sessions
        latin_square = self._generate_latin_square(codes)
        
        # Use rider_id to select row
        rider_hash = int(hashlib.md5(rider_id.encode()).hexdigest(), 16)
        row_idx = rider_hash % len(latin_square)
        
        sequence = latin_square[row_idx]
        
        # Repeat and shuffle for additional sessions
        n_complete = n_sessions // n_variants
        full_sequence = []
        
        for i in range(n_complete):
            if i == 0:
                full_sequence.extend(sequence)
            else:
                # Shuffle for subsequent rounds
                shuffled = sequence.copy()
                random.shuffle(shuffled)
                full_sequence.extend(shuffled)
        
        # Add partial if needed
        remaining = n_sessions % n_variants
        if remaining > 0:
            partial = sequence[:remaining]
            full_sequence.extend(partial)
        
        # Save sequence
        sequence_file = self.test_dir / f"sequence_{rider_id}.json"
        with open(sequence_file, 'w') as f:
            json.dump({
                'rider_id': rider_id,
                'sequence': full_sequence,
                'generated': datetime.datetime.now().isoformat()
            }, f, indent=2)
        
        return full_sequence
    
    def _generate_latin_square(self, items: List[str]) -> List[List[str]]:
        """Generate Latin square for balanced ordering"""
        n = len(items)
        square = []
        
        for i in range(n):
            row = items[i:] + items[:i]
            square.append(row)
        
        return square
    
    def record_session(self, session_data: Dict):
        """
        Record a test session
        
        Args:
            session_data: Dictionary with session information
        """
        # Create session object
        session = TestSession(
            session_id=session_data.get('session_id', 
                                       datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
            date=session_data.get('date', datetime.date.today().isoformat()),
            time_start=session_data['time_start'],
            time_end=session_data['time_end'],
            rider_id=session_data['rider_id'],
            blind_code=session_data['blind_code'],
            true_variant=None,  # Will be filled during unblinding
            conditions=session_data.get('conditions', {}),
            subjective_ratings=session_data['subjective_ratings'],
            objective_metrics=session_data.get('objective_metrics'),
            notes=session_data.get('notes', '')
        )
        
        self.sessions.append(session)
        
        # Append to sessions file
        with open(self.sessions_file, 'a') as f:
            f.write(json.dumps(asdict(session), default=str) + '\n')
        
        print(f"Recorded session {session.session_id}")
    
    def collect_subjective_ratings(self) -> Dict:
        """
        Interactive collection of subjective ratings
        """
        print("\n=== Subjective Ratings ===")
        
        ratings = {}
        
        # Equipment Invisibility
        while True:
            try:
                invisibility = float(input(
                    "Equipment Invisibility (0=never noticed, 10=constantly aware): "
                ))
                if 0 <= invisibility <= 10:
                    ratings['invisibility'] = invisibility
                    break
                else:
                    print("Please enter a value between 0 and 10")
            except ValueError:
                print("Please enter a number")
        
        # Effortlessness
        while True:
            try:
                effortlessness = float(input(
                    "Effortlessness (0=constant control, 10=completely automatic): "
                ))
                if 0 <= effortlessness <= 10:
                    ratings['effortlessness'] = effortlessness
                    break
                else:
                    print("Please enter a value between 0 and 10")
            except ValueError:
                print("Please enter a number")
        
        # Disruption Count
        while True:
            try:
                disruptions = int(input(
                    "Number of times attention went to equipment: "
                ))
                if disruptions >= 0:
                    ratings['disruption_count'] = disruptions
                    break
                else:
                    print("Please enter a non-negative integer")
            except ValueError:
                print("Please enter a whole number")
        
        # Optional: Additional comments
        comments = input("Any additional comments (optional): ")
        if comments:
            ratings['comments'] = comments
        
        return ratings
    
    def unblind(self, key: Optional[str] = None):
        """
        Unblind the test results
        
        Args:
            key: Optional unblinding key (for verification)
        """
        if not self.is_blinded:
            print("Test is already unblinded")
            return
        
        # Load randomization key
        if not self.key_file.exists():
            raise ValueError("Randomization key not found")
        
        with open(self.key_file, 'r') as f:
            key_data = json.load(f)
        
        # Verify key if provided
        if key and key != key_data['verification']:
            raise ValueError("Invalid unblinding key")
        
        # Unblind sessions
        for session in self.sessions:
            if session.blind_code in self.variants:
                session.true_variant = self.variants[session.blind_code].true_id
        
        self.is_blinded = False
        
        # Save unblinded results
        self._save_results()
        
        print("Test unblinded successfully")
    
    def analyze_results(self) -> Dict:
        """
        Analyze test results after unblinding
        """
        if self.is_blinded:
            print("Warning: Analyzing blinded results")
        
        # Load all sessions
        sessions_df = self._load_sessions_dataframe()
        
        if sessions_df.empty:
            return {"error": "No sessions recorded"}
        
        results = {
            'test_name': self.test_name,
            'n_sessions': len(sessions_df),
            'n_riders': sessions_df['rider_id'].nunique(),
            'variants': {}
        }
        
        # Group by variant
        variant_col = 'true_variant' if not self.is_blinded else 'blind_code'
        
        for variant in sessions_df[variant_col].unique():
            if pd.isna(variant):
                continue
                
            variant_data = sessions_df[sessions_df[variant_col] == variant]
            
            # Subjective metrics
            subjective = {}
            for metric in ['invisibility', 'effortlessness', 'disruption_count']:
                values = []
                for _, row in variant_data.iterrows():
                    if metric in row['subjective_ratings']:
                        values.append(row['subjective_ratings'][metric])
                
                if values:
                    subjective[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'n': len(values)
                    }
            
            # Objective metrics (if available)
            objective = {}
            if 'objective_metrics' in variant_data.columns:
                for metric in ['micro_correction_rate', 'smoothness', 'consistency']:
                    values = []
                    for _, row in variant_data.iterrows():
                        if row['objective_metrics'] and metric in row['objective_metrics']:
                            values.append(row['objective_metrics'][metric])
                    
                    if values:
                        objective[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'median': np.median(values),
                            'n': len(values)
                        }
            
            results['variants'][variant] = {
                'n_sessions': len(variant_data),
                'subjective': subjective,
                'objective': objective
            }
        
        # Statistical comparisons
        if len(results['variants']) >= 2 and not self.is_blinded:
            results['comparisons'] = self._statistical_comparisons(sessions_df)
        
        # Selection recommendation
        if not self.is_blinded:
            results['recommendation'] = self._select_optimal_variant(results)
        
        return results
    
    def _statistical_comparisons(self, df: pd.DataFrame) -> Dict:
        """Perform statistical comparisons between variants"""
        from scipy import stats
        
        comparisons = {}
        
        # Invisibility comparison
        invisibility_by_variant = {}
        for variant in df['true_variant'].unique():
            if pd.isna(variant):
                continue
            variant_data = df[df['true_variant'] == variant]
            values = [r['invisibility'] for _, row in variant_data.iterrows() 
                     for r in [row['subjective_ratings']] if 'invisibility' in r]
            if values:
                invisibility_by_variant[variant] = values
        
        if len(invisibility_by_variant) >= 2:
            # ANOVA
            f_stat, p_value = stats.f_oneway(*invisibility_by_variant.values())
            comparisons['invisibility_anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # Pairwise t-tests
            if p_value < 0.05:
                comparisons['invisibility_pairwise'] = {}
                variants = list(invisibility_by_variant.keys())
                for i, v1 in enumerate(variants):
                    for v2 in variants[i+1:]:
                        t_stat, p_val = stats.ttest_ind(
                            invisibility_by_variant[v1],
                            invisibility_by_variant[v2]
                        )
                        comparisons['invisibility_pairwise'][f"{v1}_vs_{v2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'mean_diff': (np.mean(invisibility_by_variant[v1]) - 
                                        np.mean(invisibility_by_variant[v2]))
                        }
        
        return comparisons
    
    def _select_optimal_variant(self, results: Dict) -> Dict:
        """Select optimal variant based on invisibility criteria"""
        scores = {}
        
        for variant, data in results['variants'].items():
            score = 0.0
            weight_sum = 0.0
            
            # Invisibility (lower is better, weight 0.4)
            if 'invisibility' in data['subjective']:
                inv_score = 10.0 - data['subjective']['invisibility']['mean']
                score += 0.4 * (inv_score / 10.0)
                weight_sum += 0.4
            
            # Effortlessness (higher is better, weight 0.3)
            if 'effortlessness' in data['subjective']:
                eff_score = data['subjective']['effortlessness']['mean']
                score += 0.3 * (eff_score / 10.0)
                weight_sum += 0.3
            
            # Disruption count (lower is better, weight 0.3)
            if 'disruption_count' in data['subjective']:
                # Normalize assuming max 10 disruptions
                disr_score = max(0, 10 - data['subjective']['disruption_count']['mean'])
                score += 0.3 * (disr_score / 10.0)
                weight_sum += 0.3
            
            # Normalize by actual weight sum
            if weight_sum > 0:
                scores[variant] = score / weight_sum
            else:
                scores[variant] = 0.0
        
        if scores:
            optimal = max(scores, key=scores.get)
            return {
                'selected_variant': optimal,
                'score': scores[optimal],
                'all_scores': scores,
                'reasoning': f"Variant {optimal} had the best combined invisibility score"
            }
        else:
            return {'error': 'No valid scores calculated'}
    
    def _load_sessions_dataframe(self) -> pd.DataFrame:
        """Load sessions into a DataFrame"""
        if not self.sessions_file.exists():
            return pd.DataFrame()
        
        sessions = []
        with open(self.sessions_file, 'r') as f:
            for line in f:
                if line.strip():
                    sessions.append(json.loads(line))
        
        return pd.DataFrame(sessions)
    
    def _save_randomization_key(self):
        """Save encrypted randomization key"""
        key_data = {
            'test_name': self.test_name,
            'seed': self.seed,
            'generated': datetime.datetime.now().isoformat(),
            'verification': hashlib.sha256(
                f"{self.test_name}_{self.seed}".encode()
            ).hexdigest()[:8],
            'mapping': {v.blind_code: v.true_id for v in self.variants.values()}
        }
        
        with open(self.key_file, 'w') as f:
            json.dump(key_data, f, indent=2)
        
        # Set file permissions to read-only
        self.key_file.chmod(0o400)
    
    def _save_variants(self):
        """Save variant information (without true IDs if blinded)"""
        variants_data = []
        
        for code, variant in self.variants.items():
            data = {
                'blind_code': code,
                'specifications': variant.specifications,
                'manufacture_date': variant.manufacture_date,
                'serial_number': variant.serial_number
            }
            
            if not self.is_blinded:
                data['true_id'] = variant.true_id
            
            variants_data.append(data)
        
        with open(self.variants_file, 'w') as f:
            json.dump(variants_data, f, indent=2)
    
    def _save_results(self):
        """Save analysis results"""
        results = self.analyze_results()
        
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def generate_report(self) -> str:
        """Generate human-readable test report"""
        results = self.analyze_results()
        
        report = []
        report.append(f"# Blind Test Report: {self.test_name}")
        report.append(f"Generated: {datetime.datetime.now().isoformat()}")
        report.append(f"Status: {'UNBLINDED' if not self.is_blinded else 'BLINDED'}")
        report.append("")
        
        report.append(f"## Test Summary")
        report.append(f"- Total Sessions: {results['n_sessions']}")
        report.append(f"- Number of Riders: {results['n_riders']}")
        report.append(f"- Number of Variants: {len(results['variants'])}")
        report.append("")
        
        report.append("## Variant Results")
        for variant, data in results['variants'].items():
            report.append(f"\n### Variant: {variant}")
            report.append(f"Sessions: {data['n_sessions']}")
            
            if data['subjective']:
                report.append("\n**Subjective Metrics:**")
                for metric, values in data['subjective'].items():
                    report.append(f"- {metric}: {values['mean']:.2f} ± {values['std']:.2f}")
            
            if data['objective']:
                report.append("\n**Objective Metrics:**")
                for metric, values in data['objective'].items():
                    report.append(f"- {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        
        if 'comparisons' in results:
            report.append("\n## Statistical Comparisons")
            if 'invisibility_anova' in results['comparisons']:
                anova = results['comparisons']['invisibility_anova']
                report.append(f"**Invisibility ANOVA:**")
                report.append(f"- F-statistic: {anova['f_statistic']:.3f}")
                report.append(f"- p-value: {anova['p_value']:.4f}")
                report.append(f"- Significant: {anova['significant']}")
        
        if 'recommendation' in results:
            report.append("\n## Recommendation")
            rec = results['recommendation']
            if 'selected_variant' in rec:
                report.append(f"**Selected Variant:** {rec['selected_variant']}")
                report.append(f"**Score:** {rec['score']:.3f}")
                report.append(f"**Reasoning:** {rec['reasoning']}")
        
        return "\n".join(report)

def main():
    """Example usage and CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Blind A/B/X Testing Protocol')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Initialize test
    init_parser = subparsers.add_parser('init', help='Initialize new test')
    init_parser.add_argument('name', help='Test name')
    init_parser.add_argument('--seed', type=int, help='Random seed')
    
    # Register variants
    reg_parser = subparsers.add_parser('register', help='Register fin variants')
    reg_parser.add_argument('test', help='Test name')
    reg_parser.add_argument('--variants', nargs='+', required=True,
                           help='Variant IDs (e.g., A B X)')
    
    # Generate sequence
    seq_parser = subparsers.add_parser('sequence', help='Generate test sequence')
    seq_parser.add_argument('test', help='Test name')
    seq_parser.add_argument('rider', help='Rider ID')
    seq_parser.add_argument('--sessions', type=int, default=9,
                           help='Number of sessions')
    
    # Record session
    rec_parser = subparsers.add_parser('record', help='Record test session')
    rec_parser.add_argument('test', help='Test name')
    rec_parser.add_argument('--interactive', action='store_true',
                           help='Interactive rating collection')
    
    # Unblind
    unblind_parser = subparsers.add_parser('unblind', help='Unblind test results')
    unblind_parser.add_argument('test', help='Test name')
    unblind_parser.add_argument('--key', help='Verification key')
    
    # Analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument('test', help='Test name')
    analyze_parser.add_argument('--report', action='store_true',
                               help='Generate full report')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        protocol = BlindTestProtocol(args.name, args.seed)
        print(f"Initialized test: {args.name}")
        print(f"Random seed: {protocol.seed}")
        
    elif args.command == 'register':
        protocol = BlindTestProtocol(args.test)
        variants = [{'id': v, 'specs': {}} for v in args.variants]
        protocol.register_variants(variants)
        
    elif args.command == 'sequence':
        protocol = BlindTestProtocol(args.test)
        sequence = protocol.generate_test_sequence(args.rider, args.sessions)
        print(f"Test sequence for rider {args.rider}:")
        for i, code in enumerate(sequence, 1):
            print(f"  Session {i}: {code}")
    
    elif args.command == 'record':
        protocol = BlindTestProtocol(args.test)
        
        if args.interactive:
            # Interactive session recording
            print("=== Record Test Session ===")
            session_data = {
                'rider_id': input("Rider ID: "),
                'blind_code': input("Fin code: "),
                'time_start': input("Start time (HH:MM): "),
                'time_end': input("End time (HH:MM): "),
                'subjective_ratings': protocol.collect_subjective_ratings(),
                'conditions': {
                    'wave_height': input("Wave height (ft): "),
                    'wind': input("Wind conditions: "),
                    'tide': input("Tide: ")
                },
                'notes': input("Session notes: ")
            }
            protocol.record_session(session_data)
        else:
            print("Use --interactive flag for interactive recording")
    
    elif args.command == 'unblind':
        protocol = BlindTestProtocol(args.test)
        protocol.unblind(args.key)
        
    elif args.command == 'analyze':
        protocol = BlindTestProtocol(args.test)
        
        if args.report:
            report = protocol.generate_report()
            print(report)
            
            # Save report
            report_file = protocol.test_dir / "report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {report_file}")
        else:
            results = protocol.analyze_results()
            print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()