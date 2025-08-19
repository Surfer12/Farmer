<!--
SPDX-License-Identifier: LicenseRef-Internal-Use-Only
SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
-->

# Invisible Equipment Design Framework

## Core Philosophy: Flow Through Invisibility

**Equipment disappears when it becomes predictable.** The goal is not enhanced performance but elimination of cognitive load through perfect predictability.

### Fundamental Principle
> "Flow emerges when the equipment disappears. Optimize for 'no surprises,' not 'more assistance.'"

## Design Criteria (Invisibility Requirements)

### 1. Predictability
- **Linear, monotonic responses**: No thresholds, modes, or snap-through behavior
- **Smooth load-deflection curves**: Avoid sharp transitions or discontinuities
- **Consistent force-feedback mapping**: Same input always produces same feel

### 2. Consistency Across Conditions
- **Minimal parameter drift**: Stable behavior across speed, load, and temperature variations
- **Wave-to-wave stability**: Consistent feel regardless of environmental changes
- **Tolerance insensitivity**: Manufacturing variations don't affect rider perception

### 3. Single Timescale Dynamics
- **Return-to-neutral without overshoot**: Clean, predictable recovery
- **No dual dynamics**: Avoid fast/slow response combinations that require rider reconciliation
- **Unified response time**: All equipment responses on same temporal scale

### 4. Low Salience
- **No sensory novelty**: Equipment should not draw attention through unusual sensations
- **No audible/vibratory signatures**: Silent operation across speed ranges
- **No UI or mid-session adjustability**: Set-and-forget operation

### 5. Schema Congruence
- **Match established motor expectations**: Build on rider's existing movement patterns
- **Small deviations are worse than "better" behavior**: Consistency trumps optimization
- **Preserve familiar feedback loops**: Don't break existing rider-equipment relationships

## Anti-Features (What to Actively Avoid)

### Cognitive Streams
- ❌ No biometrics displays
- ❌ No haptic feedback systems
- ❌ No real-time performance indicators
- ❌ No adaptive or "smart" features

### Mode Switching
- ❌ No mid-session adjustability
- ❌ If adjustability exists, set pre-session and lock out
- ❌ No automatic mode changes based on conditions

### Non-Linear Surprises
- ❌ Avoid bistability or hysteresis effects
- ❌ No sharp stiffness transitions
- ❌ No sudden drag changes
- ❌ No threshold-based behavior changes

## Validation Methodology (Low-Interference Testing)

### Double-Blind A/B/X Protocol
- **Identical appearance**: Test variants indistinguishable to rider
- **Randomized order**: Eliminate sequence effects
- **No mid-session changes**: Complete rides with single configuration
- **Blinded analysis**: Analyst doesn't know which variant during data processing

### Objective Metrics (IMU-Based)

#### Primary: Micro-Correction Rate
- **Definition**: High-frequency roll/yaw inputs per minute (>2 Hz)
- **Measurement**: IMU on board (not rider) sampling at ≥100 Hz
- **Threshold**: Corrections requiring >5° deflection in <0.5s
- **Target**: Minimize correction frequency

#### Secondary: Movement Smoothness
- **Jerk analysis**: Third derivative of position (m/s³)
- **Spectral smoothness**: Power concentration in low frequencies
- **Calculation**: RMS jerk over ride segments
- **Target**: Lower jerk values indicate smoother control

#### Tertiary: Path Consistency
- **Carve radius variance**: Standard deviation of turn radius for matched entry speeds
- **Speed-radius correlation**: Consistency of turn geometry across speed bands
- **Measurement**: GPS track analysis with speed correlation
- **Target**: Lower variance indicates more predictable handling

#### Quaternary: Slip/Stall Events
- **Detection**: Sudden acceleration changes not explained by rider input
- **Recovery consistency**: Time and smoothness of recovery from stall
- **Frequency**: Events per unit time or distance
- **Target**: Minimize frequency and maximize recovery predictability

### Subjective Metrics (Minimal Collection)

#### Equipment Invisibility Scale (0-10)
- **Question**: "How often did you notice the fin during this session?"
- **Anchors**: 0 = "Never noticed" | 10 = "Constantly aware"
- **Collection**: Single question post-session
- **Target**: Minimize score

#### Effortlessness Scale (0-10)
- **Question**: "How automatic did turns feel?"
- **Anchors**: 0 = "Required constant attention" | 10 = "Completely automatic"
- **Collection**: Single question post-session
- **Target**: Maximize score

#### Disruption Count
- **Question**: "How many times did your attention go to the equipment?"
- **Format**: Integer count
- **Collection**: Immediate post-session recall
- **Target**: Minimize count

### Selection Criteria
**Primary Rule**: Choose setup with lowest micro-corrections AND highest invisibility score, even if peak performance metrics are similar.

**Secondary Considerations**:
- Stable behavior across speed bands
- Consistent manufacturing tolerances
- Long-term durability without behavior drift

## Fin-Specific Design Heuristics

### Flex and Torsional Properties
- **Load-deflection curve**: Smooth, near-linear response
- **Torsional return**: Predictable spring-back without oscillation
- **No snap-back**: Avoid sudden force releases
- **Consistent modulus**: Material properties stable across temperature range

### Damping Characteristics
- **Critical damping target**: Enough to prevent oscillation
- **Quick return**: Fast return-to-neutral without overshoot
- **Speed independence**: Damping ratio constant across speed range
- **No resonance**: Avoid natural frequencies in rider input band (0.1-5 Hz)

### Foil and Edge Design
- **Gentle stall onset**: Progressive lift loss, no sudden drops
- **Surface consistency**: Uniform finish to avoid flow disruption
- **No tonal signatures**: Avoid geometries that create whistles or vibrations
- **Edge radius**: Optimized for consistent flow attachment

### Geometry Optimization
- **Rake angle**: Preserve feel across speed range
- **Cant angle**: Stable turning characteristics
- **Toe angle**: Consistent tracking behavior
- **Tolerance windows**: Tight control of dimensions affecting rider perception

## Build-Test-Iterate Loop

### Phase 1: Variable Isolation
1. **Fix all other variables**: Board, conditions, rider, session timing
2. **Single parameter variation**: Change only one fin characteristic
3. **Bracket current favorite**: Create variants above/below current spec
4. **Fabricate minimum viable set**: 3 variants for statistical validity

### Phase 2: Blind Testing Protocol
1. **Preparation**:
   - Label variants with codes unknown to rider
   - Randomize test order
   - Install IMU system on board
   - Calibrate measurement systems

2. **Testing**:
   - Minimum 5 rides per variant
   - Complete sessions without equipment changes
   - No live feedback to rider
   - Consistent environmental conditions

3. **Data Collection**:
   - Continuous IMU logging
   - Post-session subjective ratings
   - Environmental condition logging
   - No performance feedback to rider

### Phase 3: Analysis and Selection
1. **Objective Analysis**:
   - Calculate micro-correction rates
   - Analyze movement smoothness
   - Assess path consistency
   - Count slip/stall events

2. **Subjective Analysis**:
   - Score invisibility ratings
   - Evaluate effortlessness scores
   - Count disruption events

3. **Selection Criteria**:
   - Primary: Lowest micro-correction rate
   - Secondary: Highest invisibility score
   - Tertiary: Stable across speed bands
   - Final check: Manufacturing feasibility

### Phase 4: Specification Lock
1. **Document final geometry**: Complete dimensional specification
2. **Define tolerance limits**: Critical dimensions for consistent behavior
3. **Manufacturing process**: Procedures to maintain behavioral consistency
4. **Quality control**: Tests to verify behavioral specifications

## Implementation Guidelines

### Manufacturing Priorities
- **Behavioral consistency** over absolute performance
- **Tight tolerances** on perception-critical dimensions
- **Process control** to minimize unit-to-unit variation
- **Material stability** across environmental conditions

### Quality Assurance
- **Behavioral testing** of production units
- **Statistical process control** on critical dimensions
- **Long-term stability** validation
- **Field performance** monitoring

### Continuous Improvement
- **Performance drift** monitoring
- **User feedback** on invisibility maintenance
- **Manufacturing process** refinement
- **Design iteration** based on field data

## Success Metrics

### Primary Success Indicators
- **Micro-correction rate**: <50% of baseline
- **Invisibility score**: >8/10 average
- **Manufacturing consistency**: <5% behavioral variation

### Secondary Success Indicators
- **User retention**: Continued use without seeking alternatives
- **Adaptation time**: Minimal learning curve for new users
- **Cross-condition stability**: Consistent behavior across environments

### Long-Term Validation
- **Behavioral stability**: No drift over extended use
- **User satisfaction**: Maintained invisibility over time
- **Market acceptance**: Adoption by flow-state focused users

---

*This framework prioritizes the elimination of cognitive load through perfect predictability rather than performance enhancement. The goal is equipment that becomes an extension of the rider's intention rather than a tool requiring active management.*