# Core Platform

This document defines the official technology stack for the AI Factory initiative. To ensure stability, maintainability, and focus, all new development should adhere to this platform.

## Guiding Principles

- **Stability over Novelty:** We prioritize mature, well-supported technologies that have been proven in production environments.
- **Minimalism:** We aim to use the smallest set of tools that effectively solve our problems.
- **Clear Separation of Concerns:** Each component of the stack has a well-defined purpose.

## Technology Stack

### Computational Core

- **Language:** Java (21+)
- **Rationale:** Java provides a powerful, performant, and mature ecosystem for building the core of our Ψ framework and other computationally intensive services. Its strong typing and extensive libraries make it well-suited for mission-critical code.

### Application Layer

- **Platform:** Swift & SwiftUI
- **Rationale:** For user-facing applications, Swift and SwiftUI offer a modern, safe, and efficient development experience, enabling us to build high-quality native applications.

### Scripting & Utilities

- **Language:** Python (3.11+)
- **Rationale:** Python is our language of choice for a wide range of scripting and utility tasks, including data analysis, automation, and machine learning experimentation. Its extensive scientific computing ecosystem is a key asset.

### Build & Automation

- **Tools:** Make, shell scripts
- **Rationale:** We use a simple, battle-tested combination of Make and shell scripts for our build and automation pipelines. This approach is transparent, flexible, and avoids introducing unnecessary complexity.
