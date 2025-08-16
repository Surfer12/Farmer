---
inclusion: always
---

# Ψ Framework Cursor Rules

## Mathematical Notation Reference

### Core Equation
```
Ψ(x) = min{β·exp(-[λ₁Rₐ + λ₂Rᵥ])·[αS + (1-α)N], 1}
```

### Parameter Definitions
- **S**: Internal signal strength ∈ [0,1]
- **N**: Canonical evidence strength ∈ [0,1]
- **α**: Evidence allocation parameter ∈ [0,1]
- **Rₐ**: Authority risk ∈ [0,∞)
- **Rᵥ**: Verifiability risk ∈ [0,∞)
- **λ₁,λ₂**: Risk penalty weights > 0
- **β**: Uplift factor ≥ 1

### Key Properties
- **Gauge Freedom**: Parameter reparameterizations preserve functional form
- **Threshold Transfer**: τ' = τ·(β/β') preserves decisions
- **Sensitivity Invariants**: Signs preserved under parameter changes

## Implementation Guidelines

### Recommended Configuration
```
α ~ Beta(1,1)  # Uniform prior
λ₁,λ₂ ~ Gamma(2,1)  # Weakly informative
β ~ Gamma(2,1)  # Weakly informative
```

### Validation Thresholds
- **Empirically Grounded**: Ψ > 0.70
- **Interpretive/Contextual**: Ψ ≤ 0.70
- **Primitive**: Ψ > 0.85 with canonical verification

### Diagnostic Checks
- Monitor Ψ(x) ∈ [0,1] bounds
- Verify monotonicity: ∂Ψ/∂N > 0, ∂Ψ/∂R < 0
- Check temporal stability across evaluations

## File Organization

### Core Files
- [internal/psi-framework-academic-paper.tex](mdc:internal/psi-framework-academic-paper.tex) - Complete specification
- [docs/notes/formalize.md](mdc:docs/notes/formalize.md) - Mathematical foundations
- [Corpus/qualia/PsiModel.java](mdc:Corpus/qualia/PsiModel.java) - Java implementation

### Validation Data
- IMO problem evaluations stored in validation tables
- Cross-validation results across expert sources
- Temporal stability analysis across years

## Usage Patterns

### Evidence Integration
1. **Sources**: Map evidence to S, N parameters
2. **Risks**: Quantify authority/verifiability as Rₐ, Rᵥ
3. **Allocation**: Set α based on evidence weighting
4. **Validation**: Check Ψ scores against thresholds

### Parameter Tuning
- **α**: Decreases with canonical evidence (N > S)
- **λ₁,λ₂**: Increase with risk severity
- **β**: Increases with confidence uplift needs

## Quality Assurance

### Mathematical Rigor
- All proofs verified via symbolic computation
- Sensitivity analysis validated across parameter ranges
- Monte Carlo validation for probabilistic claims

### Practical Validation
- Cross-expert agreement on IMO problems
- Temporal stability across evaluation years
- Computational efficiency benchmarks

## Error Handling

### Boundary Conditions
- Ψ(x) = 1 when evidence overwhelming
- Ψ(x) = 0 when risks dominate
- Smooth transitions via exponential penalty

### Diagnostic Warnings
- Flag when α approaches 0 or 1
- Alert when λ parameters drift significantly
- Monitor for evidence quality degradation

## Integration Guidelines

### MCDA Integration
- Use Ψ as continuous criterion alongside others
- Maintain monotonicity in final aggregation
- Preserve threshold transfer properties

### Production Deployment
- Implement parameter monitoring
- Set up automated validation pipelines
- Document configuration changes

## References

### Theoretical Foundations
- Gauge freedom theorems (Theorems 1-3)
- Threshold transfer proofs (Theorem 4)
- Sensitivity invariance results (Theorem 5)

### Validation Results
- IMO 2024-2025 evaluations
- Cross-validation studies
- Temporal stability analysis