"""
Flow visualization module for CFD solver results
Provides comprehensive visualization of pressure fields, velocity vectors, and performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class FlowVisualizer:
    """Comprehensive visualization tool for CFD results"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer with default figure settings"""
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Custom colormap for pressure visualization (blue=low, red=high)
        self.pressure_colormap = LinearSegmentedColormap.from_list(
            'pressure', ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000', '#800000']
        )
        
        # Vector 3/2 fin specifications for reference
        self.fin_specs = {
            'height': 4.48,  # inches
            'base': 4.63,    # inches
            'area': 15.00,   # sq.in for side fins
            'angle': 6.5,    # degrees
            'cant': 3.0,     # degrees
            'toe': 2.0       # degrees
        }
    
    def plot_pressure_field(self, 
                           pressure: np.ndarray, 
                           coordinates: np.ndarray,
                           boundary_mask: np.ndarray,
                           title: str = "Pressure Field Around Vector 3/2 Fin",
                           save_path: Optional[str] = None) -> None:
        """
        Plot pressure field with fin geometry overlay
        
        Args:
            pressure: 2D pressure field array
            coordinates: 3D array [nx, ny, 2] with x,y coordinates
            boundary_mask: 2D array marking boundary conditions
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract coordinate arrays
        x_coords = coordinates[:, :, 0]
        y_coords = coordinates[:, :, 1]
        
        # Mask pressure field to hide wall cells
        masked_pressure = np.ma.masked_where(boundary_mask == 1, pressure)
        
        # Create pressure contour plot
        contour = ax.contourf(x_coords, y_coords, masked_pressure, 
                             levels=50, cmap=self.pressure_colormap, extend='both')
        
        # Add pressure contour lines
        contour_lines = ax.contour(x_coords, y_coords, masked_pressure, 
                                  levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        # Highlight fin boundary
        fin_boundary = np.where(boundary_mask == 1)
        if len(fin_boundary[0]) > 0:
            ax.scatter(x_coords[fin_boundary], y_coords[fin_boundary], 
                      c='white', s=1, alpha=0.8, label='Fin Surface')
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('Pressure (Pa)', rotation=270, labelpad=20)
        
        # Formatting
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add pressure differential annotation
        p_max = np.max(masked_pressure)
        p_min = np.min(masked_pressure)
        pressure_diff = ((p_max - p_min) / np.mean(np.abs(masked_pressure))) * 100
        
        ax.text(0.02, 0.98, f'Pressure Differential: {pressure_diff:.1f}%', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_velocity_vectors(self, 
                             u_velocity: np.ndarray, 
                             v_velocity: np.ndarray,
                             coordinates: np.ndarray,
                             boundary_mask: np.ndarray,
                             skip: int = 3,
                             title: str = "Velocity Field Around Vector 3/2 Fin",
                             save_path: Optional[str] = None) -> None:
        """
        Plot velocity vector field with streamlines
        
        Args:
            u_velocity: x-component of velocity
            v_velocity: y-component of velocity  
            coordinates: 3D array [nx, ny, 2] with x,y coordinates
            boundary_mask: 2D array marking boundary conditions
            skip: Skip factor for vector display (every nth vector)
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract coordinate arrays
        x_coords = coordinates[:, :, 0]
        y_coords = coordinates[:, :, 1]
        
        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(u_velocity**2 + v_velocity**2)
        
        # Mask velocity for fluid cells only
        fluid_mask = (boundary_mask == 0)
        masked_u = np.ma.masked_where(~fluid_mask, u_velocity)
        masked_v = np.ma.masked_where(~fluid_mask, v_velocity)
        masked_vel_mag = np.ma.masked_where(~fluid_mask, velocity_magnitude)
        
        # Velocity magnitude contour
        contour = ax.contourf(x_coords, y_coords, masked_vel_mag, 
                             levels=30, cmap='viridis', alpha=0.7)
        
        # Velocity vectors (subsampled)
        ax.quiver(x_coords[::skip, ::skip], y_coords[::skip, ::skip],
                 masked_u[::skip, ::skip], masked_v[::skip, ::skip],
                 scale=None, scale_units='xy', angles='xy', 
                 color='white', alpha=0.8, width=0.002)
        
        # Streamlines
        try:
            ax.streamplot(x_coords, y_coords, masked_u.filled(0), masked_v.filled(0),
                         color='red', density=1, linewidth=1, alpha=0.6)
        except:
            pass  # Skip streamlines if data is problematic
        
        # Highlight fin boundary
        fin_boundary = np.where(boundary_mask == 1)
        if len(fin_boundary[0]) > 0:
            ax.scatter(x_coords[fin_boundary], y_coords[fin_boundary], 
                      c='black', s=2, alpha=1.0, label='Fin Surface')
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('Velocity Magnitude (m/s)', rotation=270, labelpad=20)
        
        # Formatting
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_lift_drag_analysis(self, 
                               angles_of_attack: List[float],
                               lift_coefficients: List[float],
                               drag_coefficients: List[float],
                               reynolds_number: float,
                               title: str = "Vector 3/2 Blackstix+ Performance Analysis",
                               save_path: Optional[str] = None) -> None:
        """
        Plot lift and drag coefficients vs angle of attack
        
        Args:
            angles_of_attack: List of angles in degrees
            lift_coefficients: Corresponding lift coefficients
            drag_coefficients: Corresponding drag coefficients
            reynolds_number: Reynolds number for the analysis
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Lift coefficient plot
        ax1.plot(angles_of_attack, lift_coefficients, 'b-o', linewidth=2, 
                markersize=6, label='CL')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Angle of Attack (°)')
        ax1.set_ylabel('Lift Coefficient (CL)')
        ax1.set_title('Lift vs Angle of Attack')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Highlight optimal lift region (10-15 degrees for surfing)
        optimal_region = [a for a in angles_of_attack if 10 <= a <= 15]
        if optimal_region:
            ax1.axvspan(10, 15, alpha=0.2, color='green', 
                       label='Optimal Surf Range')
        
        # Drag coefficient plot
        ax2.plot(angles_of_attack, drag_coefficients, 'r-s', linewidth=2, 
                markersize=6, label='CD')
        ax2.set_xlabel('Angle of Attack (°)')
        ax2.set_ylabel('Drag Coefficient (CD)')
        ax2.set_title('Drag vs Angle of Attack')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Lift-to-drag ratio
        lift_to_drag = np.array(lift_coefficients) / np.array(drag_coefficients)
        ax3.plot(angles_of_attack, lift_to_drag, 'g-^', linewidth=2, 
                markersize=6, label='L/D')
        ax3.set_xlabel('Angle of Attack (°)')
        ax3.set_ylabel('Lift-to-Drag Ratio')
        ax3.set_title('Aerodynamic Efficiency')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Find maximum L/D ratio
        max_ld_idx = np.argmax(lift_to_drag)
        max_ld_angle = angles_of_attack[max_ld_idx]
        max_ld_value = lift_to_drag[max_ld_idx]
        
        ax3.annotate(f'Max L/D: {max_ld_value:.2f}\nat {max_ld_angle:.1f}°',
                    xy=(max_ld_angle, max_ld_value),
                    xytext=(max_ld_angle + 2, max_ld_value + 1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, ha='left')
        
        # Overall title with Reynolds number
        fig.suptitle(f'{title}\nRe = {reynolds_number:.0e}', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_pressure_plot(self, 
                                       pressure: np.ndarray,
                                       coordinates: np.ndarray,
                                       boundary_mask: np.ndarray,
                                       title: str = "Interactive Pressure Field") -> go.Figure:
        """
        Create interactive pressure field plot using Plotly
        
        Args:
            pressure: 2D pressure field array
            coordinates: 3D array [nx, ny, 2] with x,y coordinates
            boundary_mask: 2D array marking boundary conditions
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        x_coords = coordinates[:, :, 0]
        y_coords = coordinates[:, :, 1]
        
        # Mask pressure field
        masked_pressure = np.where(boundary_mask == 1, np.nan, pressure)
        
        fig = go.Figure()
        
        # Add pressure contour
        fig.add_trace(go.Contour(
            z=masked_pressure,
            x=x_coords[0, :],
            y=y_coords[:, 0],
            colorscale='RdBu_r',
            name='Pressure',
            hovertemplate='x: %{x:.3f}m<br>y: %{y:.3f}m<br>Pressure: %{z:.2f}Pa<extra></extra>'
        ))
        
        # Add fin boundary
        fin_boundary = np.where(boundary_mask == 1)
        if len(fin_boundary[0]) > 0:
            fig.add_trace(go.Scatter(
                x=x_coords[fin_boundary],
                y=y_coords[fin_boundary],
                mode='markers',
                marker=dict(size=2, color='white'),
                name='Fin Surface',
                hovertemplate='Fin Surface<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='x (m)',
            yaxis_title='y (m)',
            width=800,
            height=600,
            showlegend=True
        )
        
        fig.update_yaxis(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def plot_convergence_history(self, 
                               residual_history: List[float],
                               title: str = "CFD Solver Convergence",
                               save_path: Optional[str] = None) -> None:
        """
        Plot convergence history of the CFD solver
        
        Args:
            residual_history: List of residual values over iterations
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(residual_history) + 1)
        
        ax.semilogy(iterations, residual_history, 'b-', linewidth=2, 
                   marker='o', markersize=3, alpha=0.7)
        
        # Add convergence target line
        convergence_target = 1e-6
        ax.axhline(y=convergence_target, color='r', linestyle='--', 
                  alpha=0.7, label=f'Target: {convergence_target:.0e}')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Annotate final residual
        if residual_history:
            final_residual = residual_history[-1]
            ax.annotate(f'Final: {final_residual:.2e}',
                       xy=(len(residual_history), final_residual),
                       xytext=(len(residual_history) * 0.8, final_residual * 10),
                       arrowprops=dict(arrowstyle='->', color='blue'),
                       fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_fin_geometry_plot(self, 
                               upper_surface: np.ndarray,
                               lower_surface: np.ndarray,
                               title: str = "Vector 3/2 Blackstix+ Fin Profile") -> None:
        """
        Plot the fin geometry profile showing the 3/2 foil shape
        
        Args:
            upper_surface: Array of upper surface coordinates [n_points, 2]
            lower_surface: Array of lower surface coordinates [n_points, 2]
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot fin profile
        ax.fill_between(upper_surface[:, 0], upper_surface[:, 1], 
                       lower_surface[:, 1], alpha=0.3, color='lightblue', 
                       label='Fin Cross-Section')
        
        ax.plot(upper_surface[:, 0], upper_surface[:, 1], 'b-', 
               linewidth=2, label='Suction Side (Upper)')
        ax.plot(lower_surface[:, 0], lower_surface[:, 1], 'r-', 
               linewidth=2, label='Pressure Side (Lower)')
        
        # Mark leading and trailing edges
        ax.plot(upper_surface[0, 0], upper_surface[0, 1], 'go', 
               markersize=8, label='Leading Edge')
        ax.plot(upper_surface[-1, 0], upper_surface[-1, 1], 'ro', 
               markersize=8, label='Trailing Edge')
        
        # Add chord line
        chord_line_x = [upper_surface[0, 0], upper_surface[-1, 0]]
        chord_line_y = [0, 0]
        ax.plot(chord_line_x, chord_line_y, 'k--', alpha=0.5, 
               label='Chord Line')
        
        # Annotations for key features
        ax.annotate('Concave Pressure Side\n(30% pressure differential)', 
                   xy=(0.5 * self.fin_specs['base'], 
                       np.min(lower_surface[:, 1])),
                   xytext=(0.7 * self.fin_specs['base'], 
                          np.min(lower_surface[:, 1]) - 0.5),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, ha='center')
        
        ax.set_xlabel('Chord Position (inches)')
        ax.set_ylabel('Thickness (inches)')
        ax.set_title(f'{title}\nHeight: {self.fin_specs["height"]}" | '
                    f'Base: {self.fin_specs["base"]}" | '
                    f'Area: {self.fin_specs["area"]} sq.in')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

def export_results_to_csv(angles: List[float], 
                         lift_coeffs: List[float], 
                         drag_coeffs: List[float],
                         filename: str = "fin_performance_results.csv") -> None:
    """
    Export CFD results to CSV file for further analysis
    
    Args:
        angles: Angles of attack
        lift_coeffs: Lift coefficients
        drag_coeffs: Drag coefficients
        filename: Output filename
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'Angle_of_Attack_deg': angles,
        'Lift_Coefficient': lift_coeffs,
        'Drag_Coefficient': drag_coeffs,
        'Lift_to_Drag_Ratio': np.array(lift_coeffs) / np.array(drag_coeffs)
    })
    
    df.to_csv(filename, index=False)
    print(f"Results exported to {filename}")

# Example usage and testing functions
if __name__ == "__main__":
    # Create sample data for testing visualization
    visualizer = FlowVisualizer()
    
    # Sample pressure field (could be replaced with actual CFD results)
    nx, ny = 100, 50
    x = np.linspace(0, 20, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y)
    
    # Simple pressure field around a cylinder (placeholder)
    pressure = np.sin(X) * np.cos(Y) * 1000
    coordinates = np.stack([X, Y], axis=-1)
    boundary_mask = np.zeros((nx, ny))
    
    # Create sample fin boundary
    fin_x = np.where((X > 5) & (X < 9.63) & (np.abs(Y) < 2.24))
    boundary_mask[fin_x] = 1
    
    print("Flow visualization module loaded successfully!")
    print("Available visualization methods:")
    print("- plot_pressure_field()")
    print("- plot_velocity_vectors()")  
    print("- plot_lift_drag_analysis()")
    print("- create_interactive_pressure_plot()")
    print("- plot_convergence_history()")
<<<<<<< HEAD
    print("- create_fin_geometry_plot()")
=======
    print("- create_fin_geometry_plot()")
>>>>>>> 38a288d (Fix formatting issues by ensuring all files end with a newline character.)
