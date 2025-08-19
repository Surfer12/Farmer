<!-- SPDX-License-Identifier: LicenseRef-Internal-Use-Only -->

# Invisible Design Framework

*Design for invisibility, not assistance. Flow emerges when the user disappears.*

## Core Philosophy

Optimize for "no surprises," not "more assistance." The best interface is the one that becomes transparent to the user, allowing pure focus on the task rather than the tool.

## Design Criteria (Make It Invisible)

### Predictability
- **Linear, monotonic responses**: No thresholds, modes, or snap-through behavior
- **Consistent cause-effect mapping**: Same input always produces same output
- **No hidden state changes**: Behavior doesn't depend on history or context

### Consistency Across Conditions
- **Minimal parameter drift**: Stable performance with speed, load, or temperature changes
- **Wave-to-wave stability**: Consistent behavior across repeated actions
- **Environmental robustness**: Performance doesn't degrade with conditions

### Single Timescale
- **Return-to-neutral without overshoot**: Clean, predictable recovery
- **No dual dynamics**: Avoid fast/slow components the user must reconcile
- **Unified response time**: All system responses on same temporal scale

### Low Salience
- **No sensory novelty**: Nothing to draw attention away from task
- **No audible/vibratory signatures**: Silent operation
- **No UI during use**: No mid-session displays or feedback
- **No mid-session adjustability**: Lock settings pre-session

### Schema-Congruent
- **Matches established motor expectations**: Builds on existing muscle memory
- **Small deviations are worse than "better" behavior**: Don't surprise the user
- **Familiar physics**: Behaves like natural systems user already knows

## Anti-Features (What to Avoid)

### No Cognitive Streams
- **No biometrics**: Don't monitor the user
- **No displays**: No information during task execution
- **No haptics**: No artificial feedback systems

### No Mode Switching
- **Pre-session adjustment only**: Set parameters before starting
- **Lock out controls**: No accidental changes during use
- **Single behavior profile**: No adaptive or learning modes

### No Non-Linear Surprises
- **Avoid bistability**: No sudden state changes
- **No hysteresis**: Same input gives same output regardless of path
- **No sharp transitions**: Smooth stiffness/drag curves

## Validation Protocol (Low-Interference)

### Double-Blind A/B/X Testing
- **Identical-looking equipment**: No visual cues about variants
- **Randomized order**: Prevent learning effects
- **No mid-session changes**: Complete isolation of variables
- **Minimum 5 trials per variant**: Statistical significance

### Objective Proxies

#### Micro-Correction Rate
- **Definition**: Count high-frequency roll/yaw inputs per minute
- **Measurement**: IMU on platform, not user
- **Target**: Minimize corrections needed

#### Movement Smoothness
- **Jerk analysis**: Third derivative of position
- **Spectral smoothness**: Frequency domain analysis of IMU traces
- **Target**: Smooth, natural movement patterns

#### Path Consistency
- **Variance of carve radius**: For matched entry speeds
- **Repeatability**: Consistency across trials
- **Target**: Predictable path execution

#### Slip/Stall Events
- **Frequency**: How often control is lost
- **Recovery consistency**: Predictable recovery behavior
- **Target**: Minimize unexpected events

### Subjective Metrics (Minimal)

#### Equipment Invisibility (0–10 scale)
- **Question**: "How often did you notice the equipment?"
- **Target**: Score of 8+ (rarely noticed)

#### Effortlessness (0–10 scale)
- **Question**: "How automatic did actions feel?"
- **Target**: Score of 8+ (highly automatic)

#### Disruption Count
- **Question**: "How many times did attention go to equipment?"
- **Target**: Zero disruptions per session

### Selection Rule
Prefer the setup with:
1. **Lowest micro-corrections** (primary criterion)
2. **Highest invisibility rating** (secondary criterion)
3. **Even if peak performance metrics are similar**

## Equipment-Specific Heuristics

### Flex/Torsion
- **Smooth, near-linear load–deflection**: Predictable response
- **Predictable torsional return**: No snap-back behavior
- **Consistent across speed range**: No speed-dependent characteristics

### Damping
- **Enough to prevent oscillation**: Critically damped or slightly over
- **Quick return without overshoot**: Single timescale response
- **Temperature stable**: Consistent across conditions

### Surface/Edge
- **Gentle stall onset**: Progressive, not sudden
- **Surface finish consistent**: No texture variations
- **No tonal whistles**: Silent operation across speed range

### Geometry
- **Physical specs preserve feel across speed**: Scale-invariant behavior
- **Small tolerance windows**: Prevent perceptual drift between units
- **Manufacturing consistency**: Every unit behaves identically

## Practical Build–Test Loop

### 1. Variable Isolation
- **Fix platform and conditions**: Control all other variables
- **Vary one design variable at a time**: e.g., torsional stiffness only
- **Document baseline**: Current user favorite as reference

### 2. Variant Fabrication
- **3 variants minimum**: Bracket the current favorite
- **Identical appearance**: No visual identification possible
- **Randomized coding**: Blind assignment system

### 3. Blind Testing Protocol
- **≥5 trials each variant**: Statistical significance
- **IMU on platform**: Not on user (less invasive)
- **No live feedback**: No real-time data display
- **Randomized order**: Prevent learning effects

### 4. Selection Criteria
Choose variant with:
- **Lowest micro-correction rate** (objective primary)
- **Highest Equipment Invisibility** (subjective primary)
- **Stable behavior across speed bands** (consistency check)

### 5. Specification Lock
- **Fix final spec**: No further adjustments
- **Prioritize manufacturing tolerances**: Preserve key dynamics
- **Document critical parameters**: What must remain consistent

## Implementation Guidelines

### Pre-Session Setup
- **All adjustments made before use**: No mid-session changes
- **User preferences locked in**: Cannot accidentally change
- **Consistent starting state**: Same setup every time

### Manufacturing Tolerances
- **Identify critical parameters**: What affects invisibility most
- **Tight tolerance on key specs**: Consistency across units
- **Looser tolerance on non-critical**: Cost optimization

### Quality Control
- **Objective testing**: IMU-based consistency checks
- **User acceptance testing**: Invisibility rating validation
- **Long-term stability**: Performance over time/use

## Success Metrics

### Primary Indicators
- **Micro-correction rate < baseline**: Fewer adjustments needed
- **Equipment Invisibility > 8/10**: Rarely noticed
- **Zero attention disruptions**: Complete transparency

### Secondary Indicators
- **Path consistency improved**: More repeatable performance
- **Reduced learning curve**: Immediate familiarity
- **User preference in blind tests**: Chosen without knowing why

## Common Pitfalls

### Over-Engineering
- **Adding "helpful" features**: Usually increases salience
- **Smart adaptive behavior**: Creates unpredictability
- **Multiple adjustment options**: Increases complexity

### Measurement Artifacts
- **Hawthorne effect**: Users perform differently when monitored
- **Placebo from new equipment**: Separate from actual improvement
- **Order effects**: Learning confounds comparison

### Design Assumptions
- **"Better" performance metrics**: May not correlate with invisibility
- **Technical superiority**: Engineering metrics vs. user experience
- **Feature completeness**: More features often reduce invisibility

## Future Directions

### Cross-Domain Applications
- **Musical instruments**: Touch, response, and feedback
- **Automotive interfaces**: Controls and displays
- **Software interfaces**: Interaction patterns and feedback

### Advanced Metrics
- **Neural load indicators**: Cognitive effort measurement
- **Flow state correlation**: Objective flow indicators
- **Long-term adaptation**: How invisibility changes over time

### Manufacturing Scaling
- **Consistency at volume**: Maintaining invisibility across production
- **Cost optimization**: Balancing tolerance and economics
- **Quality systems**: Ensuring consistent user experience

---

*"The user disappears when the tool becomes invisible. Design for that disappearance."*