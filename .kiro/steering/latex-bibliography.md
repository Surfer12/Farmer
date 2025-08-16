---
inclusion: fileMatch
fileMatchPattern: ['*.tex', '*.sty', '*.bib']
---

# LaTeX and Bibliography Guidelines

## File Organization

### LaTeX Documents
- Main documents: `tex/*.tex`
- Style files: `tex/styles/*.sty`
- Bibliography files: `refs/*.bib`

### Key Files
- [tex/epistemic_model_rationale.tex](mdc:tex/epistemic_model_rationale.tex) - Main epistemic framework document
- [tex/uoif_analysis_2025.tex](mdc:tex/uoif_analysis_2025.tex) - UOIF analysis
- [tex/hybrid_ai_physics_application.tex](mdc:tex/hybrid_ai_physics_application.tex) - Physics applications

### Bibliography Files
- [refs/epistemic_references.bib](mdc:refs/epistemic_references.bib) - Epistemic framework references
- [refs/references.bib](mdc:refs/references.bib) - General references
- [refs/citations.bib](mdc:refs/citations.bib) - Additional citations

## LaTeX Best Practices

### Document Structure
- Use consistent preamble across documents
- Include proper theorem environments (theorem, lemma, definition)
- Use colored boxes for different content types (proofs, reflections, examples)

### Bibliography Management
- Maintain separate .bib files by topic
- Use consistent citation keys
- Include DOI and URL fields when available

### Style Files
- [tex/styles/analogy_equivalence_patterns.sty](mdc:tex/styles/analogy_equivalence_patterns.sty) - Equivalence patterns
- [tex/styles/drafting_latex_document.sty](mdc:tex/styles/drafting_latex_document.sty) - Document drafting
- [tex/styles/focused_response.sty](mdc:tex/styles/focused_response.sty) - Response formatting
- [tex/styles/status_update_summary.sty](mdc:tex/styles/status_update_summary.sty) - Status updates

## Compilation
- Use build scripts from [scripts/sh/](mdc:scripts/sh/) for compilation
- Ensure all references resolve correctly after file reorganization