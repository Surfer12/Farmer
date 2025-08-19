<!-- SPDX-License-Identifier: LicenseRef-Internal-Use-Only -->

# Build-Test Loop Template for Invisible Design

*Systematic approach to iterative equipment design optimization*

## Overview

This template provides a structured methodology for iteratively improving equipment design based on the invisible design framework. The goal is to minimize user awareness of the equipment while maximizing performance through objective measurement.

## Pre-Loop Setup

### Define Success Criteria
- **Primary Metric**: Micro-correction rate (corrections/minute)
- **Secondary Metric**: Equipment invisibility rating (0-10 scale)
- **Consistency Threshold**: ±5% performance variation across conditions
- **Statistical Power**: Minimum 5 trials per variant for significance

### Establish Baseline
- **Current User Favorite**: Document existing equipment specifications
- **Baseline Performance**: Record current micro-correction rate and invisibility rating
- **Environmental Conditions**: Standard test conditions (temperature, surface, etc.)
- **User Calibration**: Ensure user is familiar with test protocol

### Design Variable Selection
- **Single Variable Focus**: Change only one parameter per iteration
- **Measurable Parameters**: Quantifiable specifications (stiffness, damping, geometry)
- **Practical Range**: Variants that can be manufactured and tested
- **Safety Bounds**: Stay within safe operating parameters

## Iteration Protocol

### Phase 1: Variant Design

#### Parameter Bracketing
- **Current Value**: Document baseline parameter value
- **Variant A**: 15-20% below baseline
- **Variant B**: Baseline (control)
- **Variant C**: 15-20% above baseline

#### Manufacturing Specifications
- **Identical Appearance**: No visual identification possible
- **Consistent Non-Varied Parameters**: All other specs identical
- **Quality Control**: Verify manufacturing tolerances
- **Blind Coding**: Random assignment of test codes

#### Documentation Requirements
```
Iteration: [N]
Parameter: [e.g., Torsional Stiffness]
Baseline Value: [X units]
Variant A: [0.8X units] - Code: [Random ID]
Variant B: [X units] - Code: [Random ID] 
Variant C: [1.2X units] - Code: [Random ID]
Manufacturing Date: [Date]
Quality Check: [Pass/Fail with notes]
```

### Phase 2: Test Execution

#### Pre-Test Setup
- **Environment Control**: Consistent temperature, surface, wind
- **Equipment Preparation**: Random order assignment
- **User Preparation**: Standard warm-up routine
- **Data Collection Setup**: IMU calibration and recording

#### Test Protocol
- **Randomized Order**: Different sequence each day
- **Minimum Trials**: 5 trials per variant
- **Rest Periods**: Consistent breaks between trials
- **Blind Testing**: User and operator unaware of variant identity

#### Data Collection Checklist
- [ ] IMU data recording (platform-mounted)
- [ ] Session duration and conditions
- [ ] Subjective ratings (post-session only)
- [ ] Environmental data (temperature, humidity, wind)
- [ ] Any anomalies or equipment issues

### Phase 3: Analysis

#### Objective Metrics Calculation
- **Micro-Correction Rate**: High-frequency gyro corrections per minute
- **Movement Smoothness**: RMS jerk analysis
- **Path Consistency**: Variance in repeated maneuvers
- **Environmental Stability**: Performance across conditions

#### Statistical Analysis
- **Significance Testing**: t-test vs baseline (p < 0.05)
- **Effect Size**: Practical significance of differences
- **Confidence Intervals**: Range of expected performance
- **Power Analysis**: Adequacy of sample size

#### Subjective Analysis
- **Equipment Invisibility**: Average rating across trials
- **Disruption Count**: Frequency of attention to equipment
- **User Preference**: Blind ranking if possible
- **Qualitative Feedback**: Open-ended observations

### Phase 4: Decision Making

#### Selection Criteria (Priority Order)
1. **Lowest Micro-Correction Rate**: Primary objective metric
2. **Highest Invisibility Rating**: Primary subjective metric
3. **Best Consistency**: Lowest variance across trials
4. **Statistical Significance**: Meaningful improvement over baseline

#### Decision Matrix
```
Variant | Corrections/min | Invisibility | Consistency | p-value | Rank
--------|----------------|--------------|-------------|---------|------
A       | [X.XX]         | [X.X/10]     | [X.XXX]     | [X.XXX] | [N]
B       | [X.XX]         | [X.X/10]     | [X.XXX]     | [---]   | [N]
C       | [X.XX]         | [X.X/10]     | [X.XXX]     | [X.XXX] | [N]
```

#### Selection Rules
- **Clear Winner**: >10% improvement in corrections + p<0.05
- **Marginal Improvement**: 5-10% improvement + invisibility increase
- **No Improvement**: Continue with baseline, try different parameter
- **Degradation**: Eliminate variant, investigate cause

## Loop Management

### Convergence Criteria
- **Plateau Detection**: <2% improvement over 3 iterations
- **Consistency Achievement**: <5% variation across conditions
- **User Satisfaction**: Invisibility rating >8.0/10
- **Statistical Confidence**: p<0.01 for final selection

### Documentation Standards

#### Iteration Log
```markdown
## Iteration [N]: [Parameter Name]
**Date**: [YYYY-MM-DD]
**Parameter Range**: [Min] to [Max] [units]
**Test Conditions**: [Temperature, surface, wind, etc.]

### Variants Tested
- Variant A ([Value]): [Performance summary]
- Variant B ([Value]): [Performance summary] 
- Variant C ([Value]): [Performance summary]

### Results
- **Winner**: Variant [X] with [Y]% improvement
- **Statistical Significance**: p = [value]
- **User Feedback**: [Key observations]

### Next Steps
- [Continue with winning variant as new baseline]
- [Try different parameter]
- [Investigate anomaly]
```

#### Parameter History
```
Parameter: [Name]
Iteration 1: [Start] → [Best] ([+/-X%])
Iteration 2: [Start] → [Best] ([+/-X%])
Iteration 3: [Start] → [Best] ([+/-X%])
Final Value: [X units] (±[tolerance])
Total Improvement: [X%] reduction in corrections
```

### Quality Assurance

#### Manufacturing Consistency
- **Tolerance Verification**: Measure actual vs. specified values
- **Batch Consistency**: Multiple units of winning variant
- **Durability Testing**: Performance over extended use
- **User Acceptance**: Final blind validation

#### Long-Term Validation
- **Extended Testing**: 2-4 weeks with optimized variant
- **Condition Robustness**: Performance across weather/surface variations
- **User Adaptation**: Invisibility rating stability over time
- **Wear Characteristics**: Performance degradation patterns

## Common Pitfalls and Solutions

### Design Pitfalls
- **Multiple Parameter Changes**: Test one variable at a time
- **Insufficient Range**: Bracket wide enough to see differences
- **Manufacturing Variation**: Control tolerances on non-varied parameters
- **Visual Cues**: Ensure variants look identical

### Testing Pitfalls
- **Order Effects**: Randomize sequence, sufficient rest between trials
- **Hawthorne Effect**: Minimize user awareness of measurement
- **Environmental Drift**: Control conditions, measure environmental variables
- **Insufficient Power**: Ensure adequate sample size for significance

### Analysis Pitfalls
- **Cherry Picking**: Use pre-defined statistical criteria
- **Multiple Comparisons**: Adjust p-values for multiple tests
- **Practical vs Statistical**: Consider effect size, not just significance
- **Subjective Bias**: Weight objective metrics more heavily

## Tools and Resources

### Required Equipment
- **IMU System**: Platform-mounted, >50Hz sampling rate
- **Environmental Sensors**: Temperature, humidity, wind speed
- **Manufacturing Tools**: Consistent fabrication capability
- **Statistical Software**: Python/R for analysis

### Templates and Scripts
- **Data Collection Sheet**: [Link to template]
- **Analysis Script**: `scripts/python/invisible_design_validation.py`
- **Report Generator**: Automated analysis and reporting
- **Manufacturing Specs**: Tolerance and quality control templates

### Success Metrics Dashboard
- **Current Best**: [Parameter values and performance]
- **Improvement Trajectory**: [Progress over iterations]
- **Confidence Level**: [Statistical certainty of improvements]
- **User Satisfaction**: [Invisibility ratings over time]

## Example: Torsional Stiffness Optimization

### Background
- **Current Setup**: 50 Nm/degree torsional stiffness
- **User Complaint**: "Feels twitchy in turns"
- **Hypothesis**: Reduce stiffness for smoother response

### Iteration 1: Broad Range
- **Variant A**: 35 Nm/deg (-30%)
- **Variant B**: 50 Nm/deg (baseline)
- **Variant C**: 65 Nm/deg (+30%)

### Results
- **Corrections/min**: A=2.1, B=3.4, C=4.2
- **Invisibility**: A=8.2, B=6.8, C=5.9
- **Winner**: Variant A (significant improvement)

### Iteration 2: Fine Tuning
- **New Baseline**: 35 Nm/deg
- **Variant A**: 30 Nm/deg (-14%)
- **Variant B**: 35 Nm/deg (control)
- **Variant C**: 40 Nm/deg (+14%)

### Results
- **Corrections/min**: A=2.3, B=2.1, C=2.8
- **Invisibility**: A=7.9, B=8.2, C=7.1
- **Winner**: Variant B (35 Nm/deg optimal)

### Final Specification
- **Optimal Value**: 35 ±2 Nm/deg
- **Performance Gain**: 38% reduction in corrections
- **User Rating**: 8.2/10 invisibility
- **Manufacturing Tolerance**: ±5% (critical parameter)

---

*"Iterate systematically, measure objectively, decide based on invisibility."*