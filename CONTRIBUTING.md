SPDX-License-Identifier: GPL-3.0-only

# Contributing to the Ψ Framework

Thank you for your interest in contributing! Public contributions are accepted under GPL-3.0-only.

## How to Contribute
- Fork the repo and create a feature branch.
- Add tests and documentation.
- Ensure code passes build/lint.
- Open a pull request describing changes and rationale.

## License and DCO
By contributing, you agree that your contributions are licensed under GPL-3.0-only. Optionally sign off commits using the Developer Certificate of Origin (DCO).

## Security
Please report security issues privately (see SECURITY.md). Do not open a public issue for sensitive reports.

## Technology Governance

To maintain a focused and maintainable codebase, we are implementing a "tool sprawl freeze" as part of our AI Factory initiative. The goal is to build upon a stable, well-defined core platform and avoid introducing new tools, frameworks, or languages without a strong, documented justification.

### Core Technology Stack

- **Computational Core:** Java (21+)
- **Application Layer:** Swift & SwiftUI
- **Scripting & Utilities:** Python 3.11+
- **Build & Automation:** Make, shell scripts

### Process for Proposing New Tools

Proposals for new technologies must be submitted as a pull request modifying `docs/architecture.md` and must include:

1.  **Problem Statement:** A clear description of the problem that the current stack cannot solve.
2.  **Proposed Solution:** The specific tool/framework and why it is the best choice.
3.  **Analysis:** A brief analysis of the new tool's impact on security, maintenance, and operational complexity.
4.  **Alternatives Considered:** A discussion of why other tools or existing solutions are not sufficient.

Proposals will be reviewed by the AI Governance Board. Only once a proposal is approved and merged can the new technology be integrated.


## 📝 License and DCO

By contributing to this project, you agree to the terms of the [GPL-3.0-only license](LICENSE).

Additionally, by submitting a pull request, you confirm that you have read and agree to the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).

## 📝 License and DCO

By contributing to this project, you agree to the terms of the [GPL-3.0-only license](LICENSE).
