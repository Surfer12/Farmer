---
inclusion: manual
---
# Documentation Guidelines

## Documentation Structure

### Core Documentation
- [README.md](mdc:README.md) - Main project overview
- [docs/architecture.md](mdc:docs/architecture.md) - System architecture
- [SECURITY.md](mdc:SECURITY.md) - Security policy
- [CONTRIBUTING.md](mdc:CONTRIBUTING.md) - Contribution guidelines

### Notes and Analysis
- [docs/notes/](mdc:docs/notes/) - Analysis and decision documentation
- [docs/research_logs/](mdc:docs/research_logs/) - Research and exploration logs
- [internal/](mdc:internal/) - Internal documentation and status

### Key Documentation Files
- [docs/notes/how_i_decide_step_by_step.md](mdc:docs/notes/how_i_decide_step_by_step.md) - Decision framework
- [docs/notes/formalization_analysis.md](mdc:docs/notes/formalization_analysis.md) - Mathematical analysis
- [internal/NOTATION.md](mdc:internal/NOTATION.md) - Notation reference

## Writing Guidelines

### Style and Format
- Use clear, concise language
- Structure documents with proper headings (##, ###)
- Include code examples in fenced blocks with language tags
- Use bullet points for lists and action items

### Content Organization
- Start with overview/summary
- Provide context and motivation
- Include concrete examples
- End with actionable conclusions

### Mathematical Content
- Use LaTeX notation for mathematical expressions
- Define symbols and variables clearly
- Provide intuitive explanations alongside formal definitions
- Include numerical examples where helpful

### Cross-References
- Link to related documents using relative paths
- Update links when files are moved or renamed
- Maintain consistency in file references

## Maintenance
- Keep documentation current with code changes
- Review and update outdated information regularly
- Use clear commit messages when updating docs
- Consider documentation impact when refactoring code