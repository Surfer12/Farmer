---
inclusion: manual
---

# Licensing & REUSE

- Defaults
  - Public code/docs: GPL-3.0-only (default).
  - Internal-only areas: `internal/**`, `**/*.sty`, `**/audit-logs/**` → `LicenseRef-Internal-Use-Only`.
  - Optional non-code CC0: `examples/**`, `docs/public/**` → `CC0-1.0` (only if explicitly used).
  - See [REUSE.toml](mdc:REUSE.toml) and explainer [internal/psi-licensing.tex](mdc:internal/psi-licensing.tex).

- Do NOT license source code under CC0. CC0 is only allowed for selected non-code assets when directory-scoped and clearly annotated.

- SPDX header examples

```text
// Java (public)
// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions

% LaTeX (internal)
% SPDX-License-Identifier: LicenseRef-Internal-Use-Only

# Markdown (public)
SPDX-License-Identifier: GPL-3.0-only

# Markdown/asset (optional CC0 areas only)
SPDX-License-Identifier: CC0-1.0
```

- License texts live in `LICENSES/`:
  - [GPL-3.0-only.txt](mdc:LICENSES/GPL-3.0-only.txt)
  - [LicenseRef-Internal-Use-Only.txt](mdc:LICENSES/LicenseRef-Internal-Use-Only.txt)
  - [CC0-1.0.txt](mdc:LICENSES/CC0-1.0.txt)

- Notices
  - [NOTICE](mdc:NOTICE) and [COPYRIGHT](mdc:COPYRIGHT) must remain present.

