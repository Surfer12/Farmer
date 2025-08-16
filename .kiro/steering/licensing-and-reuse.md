---
inclusion: always
---
# Licensing & REUSE

- Default license for code: GPL-3.0-only. SPDX headers must be present on all code files.
- Internal-use-only texts use `LicenseRef-Internal-Use-Only` SPDX.
- License files live under [LICENSES/](mdc:LICENSES). REUSE configuration at [REUSE.toml](mdc:REUSE.toml).
- Notices at [NOTICE](mdc:NOTICE) and [COPYRIGHT](mdc:COPYRIGHT).

Guidelines
- When adding new files, include appropriate SPDX headers:
  - Code: `// SPDX-License-Identifier: GPL-3.0-only` (or language comment equivalent)
  - Internal docs: `SPDX-License-Identifier: LicenseRef-Internal-Use-Only`
- Keep third-party snippets annotated with their licenses and sources.
---
alwaysApply: true
---
# Licensing and REUSE Policy

- Public releases: GPL-3.0-only. Use SPDX headers in all code/docs intended for public distribution.
- Internal artifacts: LicenseRef-Internal-Use-Only; store under `internal/` and mark with SPDX.
- REUSE:
  - License texts under [LICENSES/](mdc:LICENSES)
  - Default/license overrides in [REUSE.toml](mdc:REUSE.toml)
- Top-level [LICENSE](mdc:LICENSE) mirrors GPLv3; [NOTICE](mdc:NOTICE) and [COPYRIGHT](mdc:COPYRIGHT) present.

Key files:
- Internal licensing explainer: [internal/psi-licensing.tex](mdc:internal/psi-licensing.tex)
- GPLv3 text: [LICENSES/GPL-3.0-only.txt](mdc:LICENSES/GPL-3.0-only.txt)
- Internal-use text: [LICENSES/LicenseRef-Internal-Use-Only.txt](mdc:LICENSES/LicenseRef-Internal-Use-Only.txt)
