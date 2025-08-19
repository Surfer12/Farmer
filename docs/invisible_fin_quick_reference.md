# Invisible Fin Design - Quick Reference

## Core Principle
**Design for invisibility, not assistance.** The best fin is the one you never think about.

## Key Metrics (Priority Order)

### 1. Micro-Correction Rate (MCR)
- **Target:** < baseline fin
- **Measurement:** Corrections/minute via IMU @ 100Hz
- **Threshold:** >5°/s angular velocity, >2Hz frequency

### 2. Equipment Invisibility Rating
- **Target:** < 3.0 (0-10 scale)
- **Question:** "How often were you aware of your fins?"
- **Critical:** Primary subjective metric

### 3. Movement Smoothness
- **Dimensionless Jerk:** Lower is better
- **Spectral Arc Length:** More negative = smoother
- **Path Consistency CV:** < 0.15

## Design Targets

### Physical Parameters
```
Torsional Stiffness: 15-25 Nm/rad (linear response)
Damping Ratio: 0.6-0.8 (critical damping)
Return Time: 90% recovery in <0.5s
Hysteresis: <10% energy loss
Stall Angle: 15-18° with >3° transition
Surface Finish: Ra < 0.4μm
```

### Manufacturing Tolerances
```
Rake: ±0.5° max
Cant: ±0.3° max  
Toe: ±0.2° max
Maintain feel across 10-30 km/h
```

## Test Protocol Checklist

### Pre-Test
- [ ] Manufacture 3+ variants (bracket current favorite)
- [ ] Ensure identical appearance
- [ ] Generate randomized blind codes
- [ ] Install board IMU (100Hz minimum)
- [ ] Create test sequences (Latin square)

### During Test
- [ ] No mid-session changes
- [ ] Minimum 5 sessions per variant
- [ ] Consistent conditions (same break, tide, etc.)
- [ ] Collect ratings immediately post-session
- [ ] No discussion between riders

### Post-Test Analysis
1. Calculate MCR for each variant
2. Compute smoothness metrics
3. Aggregate subjective ratings
4. Run statistical comparisons (ANOVA)
5. Select variant with lowest combined cognitive load

## Selection Algorithm
```python
score = 0.4 * (10 - invisibility_rating)/10 + 
        0.3 * normalized_mcr_score +
        0.15 * smoothness_score +
        0.15 * consistency_score
```

## Red Flags (Avoid These)
- ❌ Non-linear stiffness curves
- ❌ Mode transitions or bistability
- ❌ Audible whistles or vibrations
- ❌ Slow return to neutral (>1s)
- ❌ High hysteresis (>15%)
- ❌ Sharp stall characteristics
- ❌ Mid-session adjustability

## Green Flags (Target These)
- ✅ Linear, predictable response
- ✅ Quick, damped return
- ✅ Silent operation
- ✅ Consistent across speeds
- ✅ Gentle stall onset
- ✅ Minimal drift with temperature
- ✅ Reproducible manufacturing

## Quick Commands

### Initialize Test
```bash
python blind_abx_testing_protocol.py init my_fin_test --seed 42
```

### Register Variants
```bash
python blind_abx_testing_protocol.py register my_fin_test --variants A B X
```

### Generate Sequence
```bash
python blind_abx_testing_protocol.py sequence my_fin_test rider_001 --sessions 9
```

### Analyze IMU Data
```bash
python invisible_fin_imu_analysis.py session_001.jsonl --variant A --output results.json
```

### Unblind Results
```bash
python blind_abx_testing_protocol.py unblind my_fin_test
```

## Decision Tree

```
MCR significantly different? 
├─ YES → Select lowest MCR variant
└─ NO → Check invisibility ratings
         ├─ Significant difference? 
         │   ├─ YES → Select lowest rating
         │   └─ NO → Check smoothness
         │            └─ Select smoothest
         └─ All similar → Keep current fin
```

## Remember
**The goal is equipment that disappears into flow.** If riders are thinking about their fins, you haven't achieved invisibility—regardless of performance metrics.