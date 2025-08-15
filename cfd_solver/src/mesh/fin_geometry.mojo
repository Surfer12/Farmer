"""
Fin geometry definition and mesh generation for Vector 3/2 Blackstix+ fin
Implements the specific fin profile and generates computational mesh
"""

from tensor import Tensor
from math import sin, cos, tan, atan2, sqrt
from ..core.types import Vector2D, PI, FLUID_CELL, WALL_CELL, INLET_CELL, OUTLET_CELL

@register_passable("trivial")
struct FinSpecifications:
    """Specifications for Vector 3/2 Blackstix+ fin"""
    var height: Float32        # 4.48 inches
    var base: Float32          # 4.63 inches  
    var area: Float32          # 15.00 sq.in for side fins, 14.50 for center
    var angle: Float32         # 6.5 degrees
    var cant: Float32          # 3 degrees
    var toe: Float32           # 2 degrees
    var foil_type: String      # "3/2" foil design
    var is_center_fin: Bool    # True for center fin (symmetric)
    
    fn __init__(inout self, 
                height: Float32 = 4.48,
                base: Float32 = 4.63,
                area: Float32 = 15.00,
                angle: Float32 = 6.5,
                cant: Float32 = 3.0,
                toe: Float32 = 2.0,
                foil_type: String = "3/2",
                is_center_fin: Bool = False):
        self.height = height
        self.base = base
        self.area = area
        self.angle = angle
        self.cant = cant
        self.toe = toe
        self.foil_type = foil_type
        self.is_center_fin = is_center_fin

struct FinGeometry:
    """Generates and stores fin geometry for CFD simulation"""
    var specs: FinSpecifications
    var chord_line: Tensor[DType.float32]     # Chord line coordinates
    var upper_surface: Tensor[DType.float32]  # Upper surface coordinates
    var lower_surface: Tensor[DType.float32]  # Lower surface coordinates
    var thickness_distribution: Tensor[DType.float32]  # Thickness along chord
    var n_points: Int                         # Number of profile points
    
    fn __init__(inout self, specs: FinSpecifications, n_points: Int = 100):
        self.specs = specs
        self.n_points = n_points
        
        # Initialize coordinate arrays (n_points x 2 for x,y coordinates)
        self.chord_line = Tensor[DType.float32](n_points, 2)
        self.upper_surface = Tensor[DType.float32](n_points, 2)
        self.lower_surface = Tensor[DType.float32](n_points, 2)
        self.thickness_distribution = Tensor[DType.float32](n_points)
        
        # Generate fin profile
        self._generate_profile()
    
    fn _generate_profile(inout self):
        """Generate Vector 3/2 foil profile with concave characteristics"""
        let chord_length = self.specs.base
        
        for i in range(self.n_points):
            # Normalized chord position (0 at leading edge, 1 at trailing edge)
            let x_norm = Float32(i) / Float32(self.n_points - 1)
            let x_coord = x_norm * chord_length
            
            # Store chord line (straight line for reference)
            self.chord_line[i, 0] = x_coord
            self.chord_line[i, 1] = 0.0
            
            # Generate 3/2 foil thickness distribution
            # This creates the characteristic concave shape on the pressure side
            let thickness = self._calculate_32_foil_thickness(x_norm)
            self.thickness_distribution[i] = thickness
            
            # Apply concave shape for Vector 3/2 design
            let upper_y = self._calculate_upper_surface(x_norm, thickness)
            let lower_y = self._calculate_lower_surface(x_norm, thickness)
            
            # Store surface coordinates
            self.upper_surface[i, 0] = x_coord
            self.upper_surface[i, 1] = upper_y
            self.lower_surface[i, 0] = x_coord
            self.lower_surface[i, 1] = lower_y
    
    fn _calculate_32_foil_thickness(self, x_norm: Float32) -> Float32:
        """Calculate thickness distribution for 3/2 foil design"""
        # 3/2 foil has maximum thickness at ~30% chord, characteristic concave shape
        let max_thickness = 0.12 * self.specs.base  # ~12% thickness-to-chord ratio
        
        # Modified NACA-like distribution for 3/2 foil
        let t = max_thickness * (
            0.2969 * sqrt(x_norm) - 
            0.1260 * x_norm - 
            0.3516 * x_norm * x_norm + 
            0.2843 * x_norm * x_norm * x_norm - 
            0.1015 * x_norm * x_norm * x_norm * x_norm
        )
        
        return t
    
    fn _calculate_upper_surface(self, x_norm: Float32, thickness: Float32) -> Float32:
        """Calculate upper surface y-coordinate (suction side)"""
        # Upper surface is relatively flat for 3/2 foil
        let camber = self._calculate_camber(x_norm)
        return camber + thickness * 0.5
    
    fn _calculate_lower_surface(self, x_norm: Float32, thickness: Float32) -> Float32:
        """Calculate lower surface y-coordinate (pressure side - concave)"""
        # Lower surface has concave shape characteristic of Vector fins
        let camber = self._calculate_camber(x_norm)
        let concave_factor = self._calculate_concave_factor(x_norm)
        
        return camber - thickness * 0.5 * concave_factor
    
    fn _calculate_camber(self, x_norm: Float32) -> Float32:
        """Calculate camber line for asymmetric foil (zero for center fin)"""
        if self.specs.is_center_fin:
            return 0.0  # Symmetric center fin
        
        # Asymmetric side fin with slight camber
        let max_camber = 0.02 * self.specs.base  # 2% camber
        let camber_position = 0.4  # Maximum camber at 40% chord
        
        if x_norm <= camber_position:
            return max_camber * (2.0 * camber_position * x_norm - x_norm * x_norm) / (camber_position * camber_position)
        else:
            let factor = (1.0 - 2.0 * camber_position + 2.0 * camber_position * x_norm - x_norm * x_norm)
            return max_camber * factor / ((1.0 - camber_position) * (1.0 - camber_position))
    
    fn _calculate_concave_factor(self, x_norm: Float32) -> Float32:
        """Calculate concavity factor for pressure side (creates 30% pressure differential)"""
        # Enhanced concavity in the middle section (20-80% chord)
        if x_norm < 0.2 or x_norm > 0.8:
            return 1.0  # Normal thickness at leading/trailing edges
        
        # Increased concavity in middle section
        let concave_peak = 0.5  # Peak concavity at mid-chord
        let distance_from_peak = abs(x_norm - concave_peak)
        let concavity = 1.3 + 0.5 * exp(-10.0 * distance_from_peak)  # Enhanced concave shape
        
        return concavity

struct MeshGenerator:
    """Generates computational mesh around fin geometry"""
    var nx: Int
    var ny: Int
    var domain_width: Float32
    var domain_height: Float32
    var fin_position_x: Float32  # Fin position in domain
    var fin_position_y: Float32
    
    fn __init__(inout self, nx: Int, ny: Int, 
                domain_width: Float32 = 20.0,  # 20 * fin base length
                domain_height: Float32 = 10.0,  # 10 * fin height
                fin_position_x: Float32 = 5.0,
                fin_position_y: Float32 = 0.0):
        self.nx = nx
        self.ny = ny
        self.domain_width = domain_width
        self.domain_height = domain_height
        self.fin_position_x = fin_position_x
        self.fin_position_y = fin_position_y
    
    fn generate_boundary_mask(self, fin_geometry: FinGeometry) -> Tensor[DType.int32]:
        """Generate boundary condition mask for the computational domain"""
        var boundary_mask = Tensor[DType.int32](self.nx, self.ny)
        
        let dx = self.domain_width / Float32(self.nx - 1)
        let dy = self.domain_height / Float32(self.ny - 1)
        
        # Initialize all cells as fluid
        for i in range(self.nx):
            for j in range(self.ny):
                boundary_mask[i, j] = FLUID_CELL
        
        # Set boundary conditions
        for i in range(self.nx):
            for j in range(self.ny):
                let x = Float32(i) * dx
                let y = Float32(j) * dy - self.domain_height * 0.5  # Center domain vertically
                
                # Inlet boundary (left side)
                if i == 0:
                    boundary_mask[i, j] = INLET_CELL
                
                # Outlet boundary (right side)
                elif i == self.nx - 1:
                    boundary_mask[i, j] = OUTLET_CELL
                
                # Top and bottom boundaries (slip walls)
                elif j == 0 or j == self.ny - 1:
                    boundary_mask[i, j] = WALL_CELL
                
                # Check if point is inside fin geometry
                elif self._point_inside_fin(x, y, fin_geometry):
                    boundary_mask[i, j] = WALL_CELL
        
        return boundary_mask
    
    fn _point_inside_fin(self, x: Float32, y: Float32, fin_geometry: FinGeometry) -> Bool:
        """Check if a point is inside the fin geometry using ray casting"""
        # Translate point to fin coordinate system
        let fin_x = x - self.fin_position_x
        let fin_y = y - self.fin_position_y
        
        # Simple bounding box check first
        if fin_x < 0.0 or fin_x > fin_geometry.specs.base:
            return False
        if abs(fin_y) > fin_geometry.specs.height * 0.5:
            return False
        
        # Find corresponding point on fin profile
        let x_norm = fin_x / fin_geometry.specs.base
        let point_index = Int(x_norm * Float32(fin_geometry.n_points - 1))
        
        if point_index >= 0 and point_index < fin_geometry.n_points:
            let upper_y = fin_geometry.upper_surface[point_index, 1]
            let lower_y = fin_geometry.lower_surface[point_index, 1]
            
            # Check if point is between upper and lower surfaces
            return fin_y <= upper_y and fin_y >= lower_y
        
        return False
    
    fn generate_mesh_coordinates(self) -> Tensor[DType.float32]:
        """Generate mesh coordinates for visualization and analysis"""
        var coordinates = Tensor[DType.float32](self.nx, self.ny, 2)
        
        let dx = self.domain_width / Float32(self.nx - 1)
        let dy = self.domain_height / Float32(self.ny - 1)
        
        for i in range(self.nx):
            for j in range(self.ny):
                coordinates[i, j, 0] = Float32(i) * dx  # x-coordinate
                coordinates[i, j, 1] = Float32(j) * dy - self.domain_height * 0.5  # y-coordinate
        
<<<<<<< HEAD
        return coordinates
=======
        return coordinates
>>>>>>> 38a288d (Fix formatting issues by ensuring all files end with a newline character.)
