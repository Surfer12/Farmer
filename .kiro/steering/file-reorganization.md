---
inclusion: fileMatch
fileMatchPattern: ['*.md', '*.tex', '*.py', '*.sh', '*.jsonl', '*.sty', '*.bib', '*.java']
---

# File Reorganization Workflow

## Migration Process

When reorganizing files in the Farmer project:

1. **Audit Phase**: Classify files by type and purpose
2. **Design Phase**: Create coherent directory structure
3. **Normalize Phase**: Clean filenames (remove special chars, spaces → underscores)
4. **Move Phase**: Relocate files to appropriate directories
5. **Update Phase**: Fix all references and imports
6. **Document Phase**: Update README and architecture docs
7. **Commit Phase**: Clear commit message summarizing changes

## File Type Mappings

### LaTeX Files
- `*.tex` → `tex/`
- `*.sty` → `tex/styles/`
- `*.bib` → `refs/`

### Documentation
- Research logs → `docs/research_logs/`
- Notes and analysis → `docs/notes/`
- Decision frameworks → `docs/notes/`

### Scripts and Utilities
- `*.py` → `scripts/python/`
- `*.sh` → `scripts/sh/`

### Data and Logs
- `*.jsonl` → `data/logs/`
- Status files → `internal/StatusUpdate/legacy_logs/`

### Configuration
- Workspace files → `config/`
- IDE settings → `config/`

## Reference Updates Required

After moving files, search and update references in:
- LaTeX `\input{}` and `\include{}` statements
- Python import paths
- Shell script paths
- Build configuration files
- Documentation links

## Filename Normalization Examples

- `"### How I decide, step by step.md"` → `"how_i_decide_step_by_step.md"`
- `"'m gathering references related to MCDA .md"` → `"mcda_references.md"`
- `"Quick status: Expanded the ruleset to co"` → `"status_ruleset_expansion.md"`
- `"Drafting LaTeX Document.sty"` → `"drafting_latex_document.sty"`