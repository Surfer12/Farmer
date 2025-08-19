# Documentation Overview

This directory contains comprehensive documentation for all public APIs, functions, and components in the project.

## Quick Navigation

### 🚀 Getting Started
- **[Getting Started Guide](GETTING_STARTED.md)** - Installation, setup, and your first Ψ evaluation
- **[API Documentation](API_DOCUMENTATION.md)** - Complete reference for all public APIs
- **[Tutorials](TUTORIALS.md)** - Step-by-step tutorials for common use cases

### 📚 Learning Resources
- **[Examples](EXAMPLES.md)** - Real-world examples and practical applications
- **[Architecture Documentation](architecture.md)** - System design and architecture overview
- **[Mathematical Framework](notes/formalize.md)** - Theoretical foundations of the Ψ framework

### 🔧 Development
- **[Contributing Guidelines](../CONTRIBUTING.md)** - How to contribute to the project
- **[Security Policy](../SECURITY.md)** - Security guidelines and reporting
- **[License Information](../LICENSE)** - Project licensing and usage terms

## Documentation Structure

### Core Documentation
- **[README.md](../README.md)** - Main project overview
- **[docs/architecture.md](architecture.md)** - System architecture
- **[SECURITY.md](../SECURITY.md)** - Security policy
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines

### API References
- **Java APIs** - Complete documentation for the Ψ framework core (Corpus/qualia)
- **Swift APIs** - UOIFCore framework for iOS and macOS applications
- **Python APIs** - Uncertainty quantification and risk analysis tools
- **Command Line Interfaces** - CLI usage for all components

### Tutorials and Examples
- **Basic Ψ Evaluation** - Understanding the framework components
- **Bayesian Inference with HMC** - Full probabilistic inference
- **Multi-Chain Convergence Analysis** - Ensuring reliable sampling
- **Uncertainty Quantification Pipeline** - Complete UQ workflows
- **Real-world Applications** - Domain-specific implementations

### Notes and Research
- **[docs/notes/](notes/)** - Analysis and decision documentation
- **[docs/research_logs/](research_logs/)** - Research and exploration logs
- **[internal/](../internal/)** - Internal documentation and status

## Key Features Documented

### Ψ Framework
- **Mathematical Foundation**: Complete specification of the Ψ formula and its components
- **Java Implementation**: HierarchicalBayesianModel with HMC sampling
- **Swift Implementation**: UOIFCore for mobile and desktop applications
- **Validation Thresholds**: Classification criteria and confidence levels

### Uncertainty Quantification
- **Deep Ensembles**: Epistemic uncertainty via model disagreement
- **Monte Carlo Dropout**: Lightweight Bayesian approximation
- **Conformal Prediction**: Distribution-free prediction intervals
- **Risk Analysis**: VaR, CVaR, and tail probability estimation

### Integration Tools
- **Audit Trail System**: Comprehensive logging and monitoring
- **MCDA Integration**: Multi-criteria decision analysis
- **Metrics and Monitoring**: Performance tracking and diagnostics
- **External API Support**: HTTP, JDBC, and file-based integrations

## Usage Patterns

### Quick Start (5 minutes)
1. Follow the [Getting Started Guide](GETTING_STARTED.md)
2. Run your first Ψ evaluation
3. Explore the basic examples

### Complete Learning Path (2-3 hours)
1. **Getting Started** - Setup and basic concepts
2. **Tutorial 1** - Basic Ψ evaluation and interpretation
3. **Tutorial 2** - Bayesian inference with HMC
4. **Tutorial 3** - Multi-chain convergence analysis
5. **Tutorial 4** - Uncertainty quantification pipeline
6. **Examples** - Real-world applications

### Production Deployment
1. Review [API Documentation](API_DOCUMENTATION.md) for integration details
2. Study [Examples](EXAMPLES.md) for domain-specific implementations
3. Follow [Security Policy](../SECURITY.md) for secure deployment
4. Implement monitoring using the audit trail system

## Framework Capabilities

### Ψ (Psi) Framework
The Ψ framework provides a mathematical model for evaluating evidence quality:

```
Ψ(x) = min{β·exp(-[λ₁Rₐ + λ₂Rᵥ])·[αS + (1-α)N], 1}
```

**Key Benefits:**
- **Principled Evidence Evaluation** - Mathematical framework for assessing claim reliability
- **Uncertainty Quantification** - Separates different types of uncertainty
- **Threshold-Based Classification** - Clear decision criteria
- **Gauge Freedom** - Parameter reparameterizations preserve functional form
- **Sensitivity Analysis** - Understanding parameter impact on decisions

### Uncertainty Quantification
Comprehensive tools for quantifying and leveraging uncertainty:

**Techniques Covered:**
- **Aleatoric vs Epistemic** - Separating data noise from model uncertainty
- **Calibration Methods** - Ensuring reliable confidence estimates  
- **Risk Metrics** - Converting uncertainty to actionable risk measures
- **Out-of-Distribution Detection** - Identifying when models are unreliable
- **Decision Frameworks** - Using uncertainty for better decisions

### Multi-Language Support
- **Java** - Core Ψ framework with HMC sampling
- **Swift** - Mobile and desktop applications
- **Python** - Uncertainty quantification and machine learning integration

## Best Practices

### Development Workflow
1. **Start with Examples** - Use provided examples as templates
2. **Test Thoroughly** - Validate on your specific data and use case
3. **Monitor Performance** - Use built-in metrics and diagnostics
4. **Document Decisions** - Record parameter choices and validation results
5. **Follow Security Guidelines** - Implement appropriate safeguards

### Model Validation
1. **Cross-Validation** - Test on held-out data
2. **Convergence Diagnostics** - Ensure reliable sampling (R̂ < 1.1)
3. **Calibration Assessment** - Verify confidence interval coverage
4. **Sensitivity Analysis** - Understand parameter impact
5. **Domain Expert Review** - Validate results with subject matter experts

### Production Deployment
1. **Monitoring Setup** - Implement comprehensive logging
2. **Performance Tracking** - Monitor inference times and accuracy
3. **Model Versioning** - Track model changes and performance
4. **Fallback Strategies** - Handle edge cases and failures gracefully
5. **Regular Validation** - Continuously validate model performance

## Getting Help

### Documentation
- Start with the [Getting Started Guide](GETTING_STARTED.md)
- Check the [API Documentation](API_DOCUMENTATION.md) for specific functions
- Review [Examples](EXAMPLES.md) for similar use cases
- Consult [Tutorials](TUTORIALS.md) for step-by-step guidance

### Troubleshooting
- Check common issues in the Getting Started guide
- Review error messages and diagnostic output
- Validate input data format and ranges
- Ensure all dependencies are installed correctly

### Advanced Topics
- Review [internal documentation](../internal/) for advanced features
- Check [research logs](research_logs/) for experimental features
- Consult [mathematical framework](notes/formalize.md) for theoretical details

## Contributing

We welcome contributions! Please see:
- **[Contributing Guidelines](../CONTRIBUTING.md)** - How to contribute
- **[Security Policy](../SECURITY.md)** - Security considerations
- **[License Information](../LICENSE)** - Usage terms

## License

This project is licensed under GPL-3.0-only. See [LICENSE](../LICENSE) for details.

Internal documentation uses LicenseRef-Internal-Use-Only. See [LICENSES/](../LICENSES/) for complete license texts.

---

**Last Updated:** Generated automatically as part of comprehensive API documentation
**Maintainers:** See [CONTRIBUTORS](../CONTRIBUTORS.md) for current maintainers
**Version:** See [internal/StatusUpdate/status.jsonl](../internal/StatusUpdate/status.jsonl) for current status