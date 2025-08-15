# Mojo CFD Solver for Vector 3/2 Blackstix+ Fin Hydrodynamics

A high-performance Computational Fluid Dynamics (CFD) solver implemented in Mojo for analyzing the hydrodynamics of surfboard fins, specifically optimized for the Vector 3/2 Blackstix+ fin design.

## Project Overview

This CFD solver implements the Navier-Stokes equations using finite volume methods to simulate fluid flow around surfboard fins. The implementation leverages Mojo's SIMD vectorization and GPU parallelism capabilities to achieve significant performance improvements over traditional Python-based CFD solvers.

### Key Features

- **High-Performance Computing**: Utilizes Mojo's SIMD and GPU parallelism for 10-12x faster computation than Python
- **Navier-Stokes Implementation**: Finite volume discretization of the Cauchy momentum equation
- **Turbulence Modeling**: k-ω SST turbulence model for accurate boundary layer simulation
- **Python Interoperability**: Seamless integration with NumPy and Matplotlib for visualization
- **ML Surrogate Models**: Neural network integration for real-time performance prediction

### Fin Specifications

**Vector 3/2 Blackstix+ Configuration:**
- Side fins: 15.00 sq.in., 6.5° angle, 3/2 foil design
- Center fin: 14.50 sq.in., symmetric profile
- Height: 4.48", Base: 4.63"
- Rake characteristics with 3° cant, 2° toe
- Scimitar tip for reduced drag

## Technical Implementation

### Governing Equations

The solver implements the Cauchy momentum equation:
```
∂(ρu)/∂t + ∇·(ρu⊗u) = -∇p + ∇·τ + ρa
```

Where:
- ρ: fluid density (1000 kg/m³ for water)
- u: velocity field
- p: pressure field
- τ: deviatoric stress tensor
- a: body forces (neglected for simplicity)

### Computational Domain

- 2D cross-section discretization: 100x100 to 1M cells
- Staggered grid for velocity-pressure coupling
- Angle of attack range: 0° to 20°
- Reynolds number: 10⁵ to 10⁶ (typical surfing conditions)

### Performance Targets

- Real-time simulation capability for design iterations
- Accurate lift coefficient prediction (±5% error)
- 30% pressure differential modeling for concave surfaces
- Support for wave conditions: 2-6 ft waves, riders 125-175 lbs

## Project Structure

```
cfd_solver/
├── src/
│   ├── core/           # Core CFD algorithms
│   ├── mesh/           # Mesh generation and handling
│   ├── boundary/       # Boundary condition implementations
│   ├── turbulence/     # Turbulence models
│   └── solvers/        # Main solver implementations
├── tests/              # Unit and integration tests
├── examples/           # Example simulations and tutorials
├── visualization/      # Python visualization scripts
└── docs/              # Documentation and research notes
```

## Getting Started

### Prerequisites

- Mojo compiler (latest version)
- Python 3.8+ with NumPy, Matplotlib, PyTorch
- GPU support (optional but recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd cfd_solver

# Install Python dependencies
pip install -r requirements.txt

# Build Mojo modules
mojo build src/main.mojo
```

### Quick Start

```mojo
from cfd_solver import NavierStokesSolver, FinGeometry

# Create fin geometry
let fin = FinGeometry(
    height=4.48,
    base=4.63,
    angle=6.5,
    foil_type="3/2"
)

# Initialize solver
let solver = NavierStokesSolver(
    mesh_size=(100, 100),
    reynolds_number=1e5,
    angle_of_attack=10.0
)

# Run simulation
let results = solver.solve(fin, timesteps=1000)
```

## Research Applications

This solver supports research into:
- Hydrodynamic optimization for different rider weights
- Cognitive load correlation with fin performance
- Real-time performance prediction via ML surrogates
- Flow state enhancement through optimized lift characteristics

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Futures Fins for fin specifications and performance data
- Modular AI for Mojo language development
- CFD research community for numerical methods