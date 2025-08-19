<!--
SPDX-License-Identifier: LicenseRef-Internal-Use-Only
SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
-->

# Invisible Equipment Design System

> "Flow emerges when the equipment disappears. Optimize for 'no surprises,' not 'more assistance.'"

A comprehensive framework for designing sports equipment that becomes invisible to the user through perfect predictability, enabling flow states by eliminating cognitive load.

## Quick Start

### 1. Design Framework
Start with the core design philosophy in [`docs/notes/invisible_equipment_design.md`](docs/notes/invisible_equipment_design.md):
- **Predictability**: Linear, monotonic responses with no surprises
- **Consistency**: Stable behavior across all conditions  
- **Single timescale**: Unified response dynamics
- **Low salience**: No sensory novelty or attention-grabbing features
- **Schema congruence**: Matches established motor expectations

### 2. Validation Protocol
Use the double-blind A/B/X testing methodology:

```bash
# Initialize new test
./scripts/sh/build_test_loop.sh -t "fin_test_v1" -r "rider_001" setup

# Generate randomized session plan  
./scripts/sh/build_test_loop.sh -t "fin_test_v1" -r "rider_001" generate-plan

# Run complete iteration cycle
./scripts/sh/build_test_loop.sh -t "fin_test_v1" -r "rider_001" iterate
```

### 3. Objective Analysis
Analyze IMU data for invisibility metrics:

```python
from scripts.python.imu_flow_analysis import IMUFlowAnalyzer

analyzer = IMUFlowAnalyzer(sample_rate=100.0)
df = analyzer.load_imu_data("session_data.csv")
metrics = analyzer.analyze_session(df, subjective_scores)

print(f"Micro-corrections: {metrics.micro_correction_rate:.2f}/min")
print(f"Invisibility score: {metrics.invisibility_score}/10")
```

## Core Components

### Design Framework
- **[Invisible Equipment Design](docs/notes/invisible_equipment_design.md)**: Complete design philosophy and criteria
- **[Fin Optimization Heuristics](docs/notes/fin_optimization_heuristics.md)**: Specific guidelines for fin design
- **Anti-features**: What to actively avoid (cognitive streams, mode switching, non-linear surprises)

### Validation Tools
- **[IMU Flow Analysis](scripts/python/imu_flow_analysis.py)**: Objective measurement of flow disruption
- **[Blind Test Protocol](scripts/python/blind_test_protocol.py)**: Double-blind A/B/X testing framework
- **[Build-Test Loop](scripts/sh/build_test_loop.sh)**: Automated iteration workflow

### Key Metrics

#### Primary (Objective)
- **Micro-correction rate**: High-frequency control inputs (target: <50% of baseline)
- **Movement smoothness**: RMS jerk analysis (lower is better)
- **Path consistency**: Variance in turning behavior (lower is better)
- **Stall event frequency**: Slip/recovery incidents (minimize)

#### Secondary (Subjective)
- **Equipment Invisibility**: "How often did you notice the fin?" (target: >8/10)
- **Effortlessness**: "How automatic did turns feel?" (target: >8/10)  
- **Disruption count**: Times attention went to equipment (minimize)

## Selection Criteria

**Primary Rule**: Choose configuration with lowest micro-corrections AND highest invisibility score, even if peak performance metrics are similar.

**Why**: Flow state preservation trumps performance optimization. Equipment that requires less active management enables better overall performance through reduced cognitive load.

## Fin-Specific Guidelines

### Mechanical Properties
- **Linear flex response**: <5% deviation from linear over 80% of working range
- **Critical damping**: Return to neutral in <0.2s without overshoot
- **Temperature stability**: <10% property change from 5°C to 35°C

### Geometry Tolerances
- **Rake angle**: ±1.0° for consistent feel across speeds
- **Cant angle**: ±0.5° for stable turning characteristics  
- **Toe angle**: ±0.25° for predictable tracking
- **Surface finish**: Ra 0.8-1.6 μm uniformly

### Manufacturing Quality Control
- **100% dimensional inspection** of critical features
- **Statistical sampling** of surface finish and mechanical properties
- **Blind rider validation** of production units

## Example Workflow

### Phase 1: Configuration Setup
```bash
# Create test configurations
cat > fin_test_config.json << EOF
{
    "configurations": [
        {
            "config_id": "baseline",
            "description": "Current production fin",
            "physical_specs": {"stiffness": 100, "material": "fiberglass"}
        },
        {
            "config_id": "softer", 
            "description": "20% softer flex",
            "physical_specs": {"stiffness": 80, "material": "fiberglass"}
        }
    ]
}
EOF

# Initialize test
./scripts/sh/build_test_loop.sh -t "stiffness_test" -r "rider_001" -c fin_test_config.json setup
```

### Phase 2: Data Collection
```bash
# Generate randomized session plan (5 sessions per config)
./scripts/sh/build_test_loop.sh -t "stiffness_test" -r "rider_001" generate-plan

# Run sessions (script guides through each session)
./scripts/sh/build_test_loop.sh -t "stiffness_test" run-session rider_001_001
```

### Phase 3: Analysis
```bash
# Analyze all collected data
./scripts/sh/build_test_loop.sh -t "stiffness_test" analyze

# Results saved to: data/blind_tests/stiffness_test/analysis/
```

## Success Indicators

### Equipment Becomes Invisible When:
- Micro-correction rate drops significantly (<50% of baseline)
- Riders consistently rate invisibility >8/10
- Behavioral consistency across manufacturing units (<5% variation)
- No property drift over extended use (<3% over 100 sessions)

### Warning Signs:
- High disruption counts (attention drawn to equipment)
- Inconsistent subjective ratings across sessions
- Manufacturing variation affecting rider perception
- Property degradation over time

## File Structure

```
docs/notes/
├── invisible_equipment_design.md      # Core design framework
└── fin_optimization_heuristics.md     # Fin-specific guidelines

scripts/python/
├── imu_flow_analysis.py              # Objective flow metrics
└── blind_test_protocol.py            # Double-blind testing

scripts/sh/
└── build_test_loop.sh                # Automated workflow

data/
├── blind_tests/                      # Test data and results
└── analysis/                         # Analysis outputs
```

## Integration with Existing Workflow

This system integrates with the broader project's focus on:
- **Ψ Framework**: Equipment invisibility as a measurable confidence metric
- **MCDA Integration**: Invisibility as a key criterion in multi-criteria decisions
- **Bayesian Analysis**: Uncertainty quantification in equipment performance
- **HMC Sampling**: Robust parameter estimation for design optimization

## Next Steps

1. **Validate Framework**: Test with known equipment configurations
2. **Expand Metrics**: Add sport-specific objective measurements  
3. **Automate Analysis**: Real-time feedback during testing
4. **Scale Manufacturing**: Quality control for production volumes
5. **Cross-Sport Application**: Adapt framework to other equipment types

---

*The goal is not better equipment, but invisible equipment. When the tool disappears, flow emerges.*