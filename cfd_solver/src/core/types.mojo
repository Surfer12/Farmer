"""
Core data types and constants for CFD solver
Defines fundamental structures for fluid simulation
"""

from tensor import Tensor
from math import sqrt, exp, log
import math

# Physical constants for water at standard conditions
alias WATER_DENSITY: Float32 = 1000.0  # kg/m³
alias WATER_VISCOSITY: Float32 = 1e-6   # m²/s kinematic viscosity
alias GRAVITY: Float32 = 9.81           # m/s²

# Numerical constants
alias SMALL_NUMBER: Float32 = 1e-10
alias PI: Float32 = 3.14159265359
alias CONVERGENCE_TOLERANCE: Float32 = 1e-6

# Grid and solver parameters
alias MAX_ITERATIONS: Int = 1000
alias DEFAULT_MESH_SIZE: Int = 100

@register_passable("trivial")
struct Vector2D:
    """2D vector for velocity and other vector quantities"""
    var x: Float32
    var y: Float32
    
    fn __init__(inout self, x: Float32 = 0.0, y: Float32 = 0.0):
        self.x = x
        self.y = y
    
    fn magnitude(self) -> Float32:
        """Calculate vector magnitude"""
        return sqrt(self.x * self.x + self.y * self.y)
    
    fn normalize(inout self):
        """Normalize vector to unit length"""
        let mag = self.magnitude()
        if mag > SMALL_NUMBER:
            self.x /= mag
            self.y /= mag
    
    fn dot(self, other: Vector2D) -> Float32:
        """Dot product with another vector"""
        return self.x * other.x + self.y * other.y
    
    fn __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x + other.x, self.y + other.y)
    
    fn __sub__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x - other.x, self.y - other.y)
    
    fn __mul__(self, scalar: Float32) -> Vector2D:
        return Vector2D(self.x * scalar, self.y * scalar)

@register_passable("trivial")
struct FluidProperties:
    """Fluid properties for CFD simulation"""
    var density: Float32
    var viscosity: Float32
    var reynolds_number: Float32
    
    fn __init__(inout self, density: Float32 = WATER_DENSITY, 
                viscosity: Float32 = WATER_VISCOSITY,
                reynolds_number: Float32 = 1e5):
        self.density = density
        self.viscosity = viscosity
        self.reynolds_number = reynolds_number

@register_passable("trivial")
struct SimulationParameters:
    """Parameters controlling the CFD simulation"""
    var dt: Float32                    # Time step
    var dx: Float32                    # Grid spacing in x
    var dy: Float32                    # Grid spacing in y
    var angle_of_attack: Float32       # Fin angle of attack in degrees
    var inlet_velocity: Float32        # Freestream velocity
    var max_iterations: Int            # Maximum solver iterations
    var convergence_tolerance: Float32  # Convergence criterion
    
    fn __init__(inout self, 
                dt: Float32 = 0.001,
                dx: Float32 = 0.01,
                dy: Float32 = 0.01,
                angle_of_attack: Float32 = 10.0,
                inlet_velocity: Float32 = 5.0,
                max_iterations: Int = MAX_ITERATIONS,
                convergence_tolerance: Float32 = CONVERGENCE_TOLERANCE):
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.angle_of_attack = angle_of_attack
        self.inlet_velocity = inlet_velocity
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance

struct FlowField:
    """Main data structure holding all flow field variables"""
    var nx: Int  # Grid points in x direction
    var ny: Int  # Grid points in y direction
    
    # Primary variables
    var u: Tensor[DType.float32]      # x-velocity component
    var v: Tensor[DType.float32]      # y-velocity component  
    var p: Tensor[DType.float32]      # pressure field
    
    # Auxiliary variables
    var u_old: Tensor[DType.float32]  # Previous timestep x-velocity
    var v_old: Tensor[DType.float32]  # Previous timestep y-velocity
    var p_old: Tensor[DType.float32]  # Previous timestep pressure
    
    # Turbulence variables (for k-ω SST model)
    var k: Tensor[DType.float32]      # Turbulent kinetic energy
    var omega: Tensor[DType.float32]  # Specific dissipation rate
    
    # Boundary condition markers
    var boundary_mask: Tensor[DType.int32]  # 0=fluid, 1=wall, 2=inlet, 3=outlet
    
    fn __init__(inout self, nx: Int, ny: Int):
        self.nx = nx
        self.ny = ny
        
        # Initialize all tensors with appropriate shapes
        self.u = Tensor[DType.float32](nx, ny)
        self.v = Tensor[DType.float32](nx, ny)
        self.p = Tensor[DType.float32](nx, ny)
        
        self.u_old = Tensor[DType.float32](nx, ny)
        self.v_old = Tensor[DType.float32](nx, ny)
        self.p_old = Tensor[DType.float32](nx, ny)
        
        self.k = Tensor[DType.float32](nx, ny)
        self.omega = Tensor[DType.float32](nx, ny)
        
        self.boundary_mask = Tensor[DType.int32](nx, ny)
        
        # Initialize with default values
        self._initialize_fields()
    
    fn _initialize_fields(inout self):
        """Initialize flow fields with default values"""
        # Zero initial conditions for most variables
        self.u.zero()
        self.v.zero()
        self.p.zero()
        self.u_old.zero()
        self.v_old.zero()
        self.p_old.zero()
        
        # Initialize turbulence variables with small positive values
        for i in range(self.nx):
            for j in range(self.ny):
                self.k[i, j] = 0.01  # Small initial turbulent kinetic energy
                self.omega[i, j] = 1.0  # Initial specific dissipation rate
                self.boundary_mask[i, j] = 0  # Default to fluid cells
    
    fn save_previous_timestep(inout self):
        """Save current values as previous timestep"""
        for i in range(self.nx):
            for j in range(self.ny):
                self.u_old[i, j] = self.u[i, j]
                self.v_old[i, j] = self.v[i, j]
                self.p_old[i, j] = self.p[i, j]
    
    fn calculate_velocity_magnitude(self, i: Int, j: Int) -> Float32:
        """Calculate velocity magnitude at grid point (i,j)"""
        return sqrt(self.u[i, j] * self.u[i, j] + self.v[i, j] * self.v[i, j])
    
    fn get_velocity_vector(self, i: Int, j: Int) -> Vector2D:
        """Get velocity vector at grid point (i,j)"""
        return Vector2D(self.u[i, j], self.v[i, j])

# Boundary condition types
alias FLUID_CELL: Int = 0
alias WALL_CELL: Int = 1
alias INLET_CELL: Int = 2
alias OUTLET_CELL: Int = 3