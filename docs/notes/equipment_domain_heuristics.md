<!-- SPDX-License-Identifier: LicenseRef-Internal-Use-Only -->

# Equipment Domain Heuristics for Invisible Design

*Domain-specific guidelines for achieving equipment invisibility across different applications*

## Overview

While the core principles of invisible design remain constant, different domains require specific technical approaches. This document provides practical heuristics for common equipment categories, focusing on the technical parameters that most affect user awareness.

## General Principles Across Domains

### Universal Invisibility Factors
- **Predictable Response**: Linear input-output relationships
- **Single Timescale**: Unified response time across all functions
- **Environmental Stability**: Consistent behavior across conditions
- **Manufacturing Consistency**: Identical feel across units
- **No Surprises**: Behavior matches user mental models

### Common Anti-Patterns
- **Mode Switching**: Different behaviors in different contexts
- **Threshold Effects**: Sudden changes in response
- **Environmental Sensitivity**: Performance varies with conditions
- **Hysteresis**: Path-dependent behavior
- **Audible/Tactile Signatures**: Sensory distractions

## Domain-Specific Heuristics

### Snow/Water Sports Equipment

#### Flex Characteristics
- **Progressive Flex Pattern**: Smooth, predictable stiffness curve
- **No Flat Spots**: Avoid regions of constant stiffness
- **Consistent Flex Across Speed**: Same feel at 10mph and 40mph
- **Temperature Stability**: Minimal stiffness change with temperature

**Target Parameters**:
- Flexural modulus variation: <10% across operating temperature range
- Flex pattern linearity: R² > 0.95 for load-deflection curve
- Speed independence: <5% stiffness change across speed range

#### Torsional Response
- **Linear Torsional Stiffness**: Constant Nm/degree across range
- **No Snap-Back**: Smooth return to neutral without oscillation
- **Symmetric Response**: Identical left/right turn characteristics
- **Damped Return**: Critical damping or slight overdamping

**Target Parameters**:
- Torsional stiffness: 30-60 Nm/degree (application dependent)
- Return damping: 0.7-1.0 critical damping ratio
- Left/right asymmetry: <3% difference in stiffness

#### Edge/Surface Interface
- **Gentle Stall Onset**: Progressive grip loss, not sudden
- **Consistent Surface Texture**: No variation along length
- **Silent Operation**: No whistling or chattering
- **Predictable Slip Recovery**: Consistent regain of control

**Target Parameters**:
- Stall gradient: <20% grip loss per degree of angle increase
- Surface roughness: Ra < 0.8 μm, consistent ±10%
- Acoustic signature: <40 dB at user ear position

### Musical Instruments

#### Touch Response
- **Linear Velocity Response**: Output proportional to input force/speed
- **No Dead Zones**: Responsive across entire range
- **Consistent Key Weight**: Identical feel across keyboard/fretboard
- **Predictable Dynamics**: Same force always produces same output

**Target Parameters**:
- Key weight variation: <5g across keyboard
- Velocity curve linearity: R² > 0.98
- Dynamic range: >60dB without compression artifacts

#### Mechanical Feedback
- **Consistent Action**: Same travel distance and force curve
- **No Mechanical Noise**: Silent key/string mechanisms
- **Stable Intonation**: Pitch stability across temperature/humidity
- **Predictable Sustain**: Consistent decay characteristics

**Target Parameters**:
- Key travel consistency: ±0.1mm across keys
- Intonation stability: <2 cents across operating conditions
- Sustain variation: <10% across dynamic range

#### Acoustic Response
- **Linear Amplitude Response**: Output proportional to input
- **No Resonant Peaks**: Smooth frequency response
- **Consistent Timbre**: Same harmonic content across range
- **Predictable Projection**: Consistent volume at distance

**Target Parameters**:
- Frequency response: ±3dB across usable range
- Harmonic distortion: <1% THD at normal playing levels
- Dynamic linearity: R² > 0.95 for input vs. output

### Automotive Controls

#### Steering Response
- **Linear Steering Ratio**: Constant wheel-to-road relationship
- **No Dead Band**: Immediate response from center
- **Consistent Effort**: Same force required across speed range
- **Predictable Return**: Self-centering without oscillation

**Target Parameters**:
- Steering ratio variation: <5% across lock-to-lock
- Center dead band: <2 degrees
- Return-to-center time: 1.5-2.5 seconds from 90° input

#### Pedal Feel
- **Progressive Response**: Smooth force-displacement curve
- **No Grab Points**: Linear relationship throughout range
- **Consistent Travel**: Same distance for same effect
- **Predictable Modulation**: Fine control at all positions

**Target Parameters**:
- Pedal force gradient: 50-200 N/m (application dependent)
- Travel consistency: ±2mm across units
- Hysteresis: <5% of full-scale force

#### Control Interfaces
- **Immediate Response**: <100ms latency for all inputs
- **Linear Mapping**: Control position matches function output
- **No Mode Confusion**: Same control always does same thing
- **Tactile Consistency**: Identical feel across similar controls

**Target Parameters**:
- Response latency: <50ms for safety-critical functions
- Control linearity: R² > 0.98 for position vs. function
- Tactile force variation: <10% across similar controls

### Software Interfaces

#### Response Timing
- **Consistent Latency**: Same delay for similar operations
- **Predictable Progress**: Linear progress indicators
- **No Blocking**: Non-blocking UI during background operations
- **Smooth Animations**: 60fps, no frame drops

**Target Parameters**:
- UI response time: <100ms for immediate feedback
- Animation frame rate: >55fps sustained
- Progress indicator accuracy: ±5% of actual completion

#### Interaction Patterns
- **Consistent Gestures**: Same input produces same result
- **No Mode Switching**: Context doesn't change behavior
- **Predictable Navigation**: Clear spatial/hierarchical relationships
- **Error Recovery**: Consistent undo/redo behavior

**Target Parameters**:
- Gesture recognition accuracy: >98% for trained patterns
- Navigation consistency: <3 clicks to any function
- Error recovery: 100% reversible operations

#### Visual Consistency
- **Uniform Spacing**: Consistent grid and alignment
- **Predictable Typography**: Same hierarchy, same meaning
- **Stable Layout**: No unexpected element movement
- **Consistent Color**: Same colors mean same things

**Target Parameters**:
- Grid consistency: ±2px alignment tolerance
- Typography scale: Consistent ratio (e.g., 1.2x) between levels
- Color consistency: <ΔE 2.0 across similar elements

### Power Tools

#### Motor Response
- **Linear Torque Curve**: Output proportional to trigger position
- **No Torque Spikes**: Smooth power delivery
- **Consistent Speed**: RPM stability under varying load
- **Predictable Stall**: Gradual torque increase before stall

**Target Parameters**:
- Torque linearity: R² > 0.95 across operating range
- Speed regulation: ±5% RPM under 50% load variation
- Stall torque gradient: <20% increase per 100 RPM reduction

#### Vibration Control
- **Minimal Hand-Arm Vibration**: Below exposure limits
- **No Resonant Frequencies**: Avoid 8-25 Hz range
- **Consistent Feel**: Same vibration signature across units
- **Isolated Handles**: Vibration doesn't reach user

**Target Parameters**:
- Hand-arm vibration: <2.5 m/s² (8-hour exposure limit)
- Resonant frequency: >30 Hz or <5 Hz
- Handle isolation: >90% vibration reduction

#### Ergonomic Interface
- **Natural Grip**: Fits range of hand sizes
- **Balanced Weight**: Center of gravity near grip
- **No Pressure Points**: Distributed contact force
- **Intuitive Controls**: Switch placement matches expectations

**Target Parameters**:
- Grip diameter: 32-38mm for power tools
- Balance point: Within 50mm of grip center
- Contact pressure: <200 kPa at any point

## Cross-Domain Validation Metrics

### Objective Measurements
- **Micro-Correction Rate**: High-frequency adjustments per minute
- **Response Linearity**: R² value for input-output relationship
- **Temporal Consistency**: Coefficient of variation across trials
- **Environmental Stability**: Performance variation across conditions

### Subjective Assessments
- **Equipment Invisibility**: 0-10 scale, target >8.0
- **Cognitive Load**: Attention required for equipment vs. task
- **Disruption Frequency**: Number of awareness events per session
- **Preference Ranking**: Blind comparison against alternatives

### Statistical Requirements
- **Sample Size**: Minimum 5 trials per condition
- **Significance Level**: p < 0.05 for meaningful differences
- **Effect Size**: >10% improvement for practical significance
- **Confidence Interval**: 95% CI for performance estimates

## Implementation Guidelines

### Design Phase
1. **Identify Critical Parameters**: What affects user awareness most?
2. **Establish Baselines**: Current performance and user expectations
3. **Define Target Values**: Specific, measurable improvement goals
4. **Plan Validation**: How will invisibility be measured?

### Testing Phase
1. **Control Variables**: Change one parameter at a time
2. **Blind Testing**: User unaware of variant identity
3. **Objective Measurement**: IMU, force sensors, timing systems
4. **Statistical Analysis**: Proper significance testing

### Manufacturing Phase
1. **Critical Tolerances**: Tight control on invisibility-affecting parameters
2. **Quality Systems**: Consistent production of optimal specifications
3. **Validation Testing**: Every unit meets invisibility targets
4. **Long-term Monitoring**: Performance stability over product life

## Common Failure Modes

### Design Failures
- **Over-Engineering**: Adding features that increase salience
- **Mode Complexity**: Different behaviors in different contexts
- **Threshold Effects**: Sudden changes in response characteristics
- **Environmental Sensitivity**: Performance varies with conditions

### Manufacturing Failures
- **Tolerance Stack-up**: Accumulated variations affect feel
- **Material Inconsistency**: Different batches behave differently
- **Assembly Variation**: Human factors in production
- **Quality Drift**: Gradual changes in manufacturing process

### Testing Failures
- **Insufficient Power**: Too few trials for statistical significance
- **Confounding Variables**: Multiple changes simultaneously
- **User Bias**: Knowledge of variants affects performance
- **Environmental Variation**: Uncontrolled test conditions

## Success Patterns

### High-Performance Invisibility
- **Consistent Excellence**: Every unit performs identically
- **Predictable Behavior**: User never surprised by equipment response
- **Environmental Robustness**: Same feel across all conditions
- **Long-term Stability**: Performance doesn't degrade with use

### User Adoption Indicators
- **Immediate Familiarity**: No learning curve required
- **Unconscious Use**: User forgets about equipment during task
- **Preference Persistence**: Chosen consistently in blind tests
- **Performance Improvement**: Better task outcomes with invisible equipment

### Manufacturing Success
- **Process Capability**: Cpk > 1.33 for critical parameters
- **Quality Consistency**: <1% variation in key specifications
- **Cost Effectiveness**: Invisibility achieved within budget constraints
- **Scalable Production**: Consistent quality at volume

---

*"Every domain has its own path to invisibility, but all paths lead to the user forgetting the tool exists."*