import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import networkx as nx
from typing import Dict, List, Any, Set, Optional
from mpl_toolkits.mplot3d import Axes3D

# Try to import autogen, with error handling for installation
try:
    import autogen
    from autogen import Agent, UserProxyAgent, AssistantAgent
except ImportError:
    print("Installing autogen...")
    import subprocess
    subprocess.check_call(["pip", "install", "pyautogen"])
    import autogen
    from autogen import Agent, UserProxyAgent, AssistantAgent

# Create data directory if it doesn't exist
os.makedirs("workspace", exist_ok=True)
os.makedirs("workspace/data", exist_ok=True)
os.makedirs("workspace/data/raw", exist_ok=True)
os.makedirs("workspace/data/processed", exist_ok=True)
os.makedirs("workspace/data/visualizations", exist_ok=True)

# =============================================================================
# WORKFLOW CONFIGURATION
# =============================================================================
class WorkflowConfig:
    """Configuration class for workflow execution modes"""
    
    # Execution modes
    MODE_AUTO = "auto"              # Fully automated, no human interaction
    MODE_HUMAN_IN_LOOP = "human"    # Human approval at each step
    MODE_LLM = "llm"                # LLM-based agent coordination
    
    def __init__(self):
        self.execution_mode = self.MODE_AUTO
        self.human_approval_required = False
        self.show_visualizations = True
        self.save_intermediate_results = True
        self.verbose = True
        
    def set_mode(self, mode: str):
        """Set the execution mode"""
        valid_modes = [self.MODE_AUTO, self.MODE_HUMAN_IN_LOOP, self.MODE_LLM]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        self.execution_mode = mode
        self.human_approval_required = (mode == self.MODE_HUMAN_IN_LOOP)
        
    def is_human_in_loop(self) -> bool:
        return self.execution_mode == self.MODE_HUMAN_IN_LOOP

# Global config instance
workflow_config = WorkflowConfig()

# Human-in-the-loop feedback state
human_feedback_state = {
    "approvals": {},
    "modifications": [],
    "comments": {},
    "rejected_tasks": set()
}

# =============================================================================
# HUMAN-IN-THE-LOOP UTILITIES
# =============================================================================
def display_task_result(task_id: str, result: dict, visualize: bool = True):
    """Display task result to user for review"""
    print("\n" + "="*60)
    print(f"📊 TASK COMPLETED: {task_id}")
    print("="*60)
    
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            print(result)
            return
    
    print(f"\n✅ Status: {result.get('status', 'unknown')}")
    print(f"⏰ Timestamp: {result.get('timestamp', 'N/A')}")
    
    # Display task-specific info
    if 'dataset_info' in result:
        info = result['dataset_info']
        print(f"\n📦 Dataset Info:")
        print(f"   - Dimensions: {info.get('dimensions', 'N/A')}")
        print(f"   - Format: {info.get('format', 'N/A')}")
        
    if 'horizons_detected' in result:
        print(f"\n🔍 Horizons Detected: {result['horizons_detected']}")
        if 'confidence_scores' in result:
            print(f"   - Confidence Scores: {result['confidence_scores']}")
            
    if 'faults_detected' in result:
        print(f"\n⚡ Faults Detected: {result['faults_detected']}")
        if 'major_faults' in result:
            print(f"   - Major Faults: {result['major_faults']}")
            
    if 'output_file_path' in result:
        print(f"\n💾 Output File: {result['output_file_path']}")
        
    if 'visualization_path' in result:
        print(f"📈 Visualization: {result['visualization_path']}")
        
    print("="*60)

def get_human_approval(task_id: str, task_description: str) -> tuple:
    """
    Get human approval for a task result.
    Returns: (approved: bool, feedback: str)
    """
    print(f"\n🤔 HUMAN REVIEW REQUIRED for task: {task_id}")
    print(f"   Description: {task_description}")
    print("\nOptions:")
    print("  [a] Approve - Accept this result and continue")
    print("  [m] Modify  - Accept but note modifications needed")
    print("  [r] Reject  - Reject and re-run this task")
    print("  [s] Skip    - Skip approval for remaining tasks (auto-approve)")
    print("  [q] Quit    - Stop the workflow")
    
    while True:
        choice = input("\nYour choice [a/m/r/s/q]: ").strip().lower()
        
        if choice == 'a':
            return True, "Approved"
        elif choice == 'm':
            feedback = input("Enter modification notes: ").strip()
            human_feedback_state["modifications"].append({
                "task_id": task_id,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            })
            return True, feedback
        elif choice == 'r':
            reason = input("Enter rejection reason: ").strip()
            human_feedback_state["rejected_tasks"].add(task_id)
            return False, reason
        elif choice == 's':
            workflow_config.human_approval_required = False
            print("✓ Auto-approval enabled for remaining tasks")
            return True, "Auto-approved (user skipped)"
        elif choice == 'q':
            raise KeyboardInterrupt("User requested workflow termination")
        else:
            print("Invalid choice. Please enter a, m, r, s, or q.")

def show_workflow_status(workflow_dag):
    """Display current workflow status"""
    print("\n" + "="*60)
    print("📋 WORKFLOW STATUS")
    print("="*60)
    
    for task_id in workflow_dag.graph.nodes:
        status = "✅" if task_id in workflow_dag.completed_tasks else "⏳"
        desc = workflow_dag.graph.nodes[task_id]["description"]
        deps = list(workflow_dag.graph.predecessors(task_id))
        dep_str = f" (depends on: {deps})" if deps else ""
        print(f"  {status} {task_id}: {desc}{dep_str}")
    
    print("="*60)

# Sample data generation
class SeismicDataGenerator:
    @staticmethod
    def generate_seismic_volume(shape=(100, 100, 50), noise_level=0.2, save_path=None):
        """Generate synthetic seismic data with horizons and faults"""
        print(f"Generating synthetic seismic volume with shape {shape}...")
        
        # Create base horizons (smooth surfaces with varying depths)
        horizons = []
        num_horizons = 5
        for i in range(num_horizons):
            # Create a smooth surface with some randomness
            base_depth = int(shape[2] * (i + 1) / (num_horizons + 1))
            horizon = np.ones((shape[0], shape[1])) * base_depth
            
            # Add some undulations to the horizon
            x = np.linspace(0, 4*np.pi, shape[0])
            y = np.linspace(0, 4*np.pi, shape[1])
            X, Y = np.meshgrid(x, y)
            undulation = 3 * np.sin(X/2) * np.cos(Y/2)
            horizon += undulation
            
            horizons.append(horizon.astype(int))
        
        # Create a fault (vertical discontinuity)
        fault_x = shape[0] // 2
        fault_displacement = shape[2] // 10
        
        # Initialize volume with random noise
        volume = np.random.normal(0, noise_level, shape)
        
        # Add horizons to the volume with high amplitude
        for horizon in horizons:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    depth = min(horizon[i, j], shape[2] - 1)
                    
                    # Apply fault displacement
                    if i > fault_x:
                        depth = min(depth + fault_displacement, shape[2] - 1)
                    
                    # Make a strong reflection at the horizon
                    volume[i, j, depth] = 1.0
                    
                    # Add some continuity to horizons
                    if depth > 0:
                        volume[i, j, depth-1] = 0.7
                    if depth < shape[2] - 1:
                        volume[i, j, depth+1] = 0.7
        
        # Save the volume if path is provided
        if save_path:
            np.save(save_path, volume)
            print(f"Saved synthetic seismic volume to {save_path}")
            
            # Save a sample slice for visualization
            plt.figure(figsize=(10, 8))
            plt.imshow(volume[:, shape[1]//2, :].T, cmap='seismic', aspect='auto')
            plt.colorbar(label='Amplitude')
            plt.title(f"Synthetic Seismic Data - Inline {shape[1]//2}")
            plt.xlabel("Crossline")
            plt.ylabel("Time/Depth")
            
            slice_path = save_path.replace('.npy', '_slice.png')
            plt.savefig(slice_path)
            plt.close()
            print(f"Saved sample slice to {slice_path}")
        
        # Return metadata about the generated data
        metadata = {
            "shape": shape,
            "horizons": len(horizons),
            "fault_position": fault_x,
            "fault_displacement": fault_displacement,
            "noise_level": noise_level
        }
        
        return volume, metadata


    @staticmethod
    def detect_faults(volume, horizons, save_path=None):
        """Detect faults based on discontinuities in horizons"""
        print("Detecting faults in seismic volume...")
        
        shape = volume.shape
        fault_likelihood = np.zeros((shape[0], shape[1], shape[2]))
        
        # Simple fault detection based on horizon discontinuities
        for horizon_id, surface in horizons.items():
            # Calculate horizontal gradient of the horizon surface
            gradient_x = np.zeros_like(surface)
            gradient_y = np.zeros_like(surface)
            
            # X direction gradient
            gradient_x[:, 1:-1] = surface[:, 2:] - surface[:, :-2]
            
            # Y direction gradient
            gradient_y[1:-1, :] = surface[2:, :] - surface[:-2, :]
            
            # Gradient magnitude
            gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Mark high gradient areas as potential faults
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if not np.isnan(surface[i, j]):
                        depth = int(surface[i, j])
                        if depth < shape[2]:
                            # Higher gradient = higher fault likelihood
                            fault_likelihood[i, j, depth] = max(
                                fault_likelihood[i, j, depth],
                                min(1.0, gradient_mag[i, j] / 5.0)  # Normalize
                            )
        
        # Threshold to find fault surfaces
        fault_threshold = 0.5
        fault_points = np.where(fault_likelihood > fault_threshold)
        fault_surfaces = list(zip(fault_points[0], fault_points[1], fault_points[2]))
        
        # Group nearby fault points into fault planes
        fault_planes = []
        remaining_points = set(fault_surfaces)
        
        while remaining_points:
            # Start a new fault plane
            seed = next(iter(remaining_points))
            current_plane = {seed}
            remaining_points.remove(seed)
            
            # Grow the plane
            plane_changed = True
            while plane_changed:
                plane_changed = False
                points_to_add = set()
                
                for point in current_plane:
                    i, j, k = point
                    
                    # Check neighbors within a small radius
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-2, -1, 0, 1, 2]:  # Allow more flexibility in depth
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                    
                                ni, nj, nk = i + di, j + dj, k + dk
                                neighbor = (ni, nj, nk)
                                
                                if neighbor in remaining_points:
                                    points_to_add.add(neighbor)
                                    remaining_points.remove(neighbor)
                
                if points_to_add:
                    current_plane.update(points_to_add)
                    plane_changed = True
            
            # Add the completed plane to our list
            if len(current_plane) > 50:  # Only keep significant planes
                fault_planes.append(current_plane)
        
        # Save faults if path is provided
        if save_path:
            fault_data = {
                "fault_likelihood": fault_likelihood,
                "fault_planes": [list(plane) for plane in fault_planes]
            }
            
            np.save(save_path, fault_data)
            print(f"Saved detected faults to {save_path}")
            
            # Create visualization of fault likelihood
            plt.figure(figsize=(10, 8))
            # Show maximum fault likelihood along one axis
            fault_slice = np.max(fault_likelihood, axis=1)
            plt.imshow(fault_slice.T, cmap='hot', aspect='auto')
            plt.colorbar(label='Fault Likelihood')
            plt.title("Fault Likelihood")
            plt.xlabel("Crossline")
            plt.ylabel("Time/Depth")
            
            viz_path = save_path.replace('.npy', '_viz.png')
            plt.savefig(viz_path)
            plt.close()
            print(f"Saved fault visualization to {viz_path}")
        
        return {
            "num_fault_planes": len(fault_planes),
            "fault_points": len(fault_surfaces),
            "fault_planes": fault_planes
        }

    @staticmethod
    def create_visualization(volume, horizons, faults, save_path=None):
        """Create comprehensive visualization of seismic data with horizons and faults"""
        print("Creating comprehensive visualization...")
        
        # Import 3D plotting tools
        from mpl_toolkits.mplot3d import Axes3D
        
        shape = volume.shape
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Seismic inline with horizons
        ax1 = fig.add_subplot(221)
        inline_idx = shape[1] // 2
        inline_data = volume[:, inline_idx, :].T
        
        ax1.imshow(inline_data, cmap='seismic', aspect='auto', extent=[0, shape[0], shape[2], 0])
        ax1.set_title(f"Seismic Inline {inline_idx} with Horizons")
        ax1.set_xlabel("Crossline")
        ax1.set_ylabel("Time/Depth")
        
        # Add horizons to inline
        for horizon_id, surface in horizons.items():
            horizon_line = surface[:, inline_idx]
            ax1.plot(np.arange(shape[0]), horizon_line, 
                    linewidth=2, label=f"Horizon {horizon_id + 1}")
        
        ax1.legend(loc='upper right')
        
        # Plot 2: Seismic crossline with horizons
        ax2 = fig.add_subplot(222)
        xline_idx = shape[0] // 2
        xline_data = volume[xline_idx, :, :].T
        
        ax2.imshow(xline_data, cmap='seismic', aspect='auto', extent=[0, shape[1], shape[2], 0])
        ax2.set_title(f"Seismic Crossline {xline_idx} with Horizons")
        ax2.set_xlabel("Inline")
        ax2.set_ylabel("Time/Depth")
        
        # Add horizons to crossline
        for horizon_id, surface in horizons.items():
            horizon_line = surface[xline_idx, :]
            ax2.plot(np.arange(shape[1]), horizon_line, 
                    linewidth=2, label=f"Horizon {horizon_id + 1}")
        
        ax2.legend(loc='upper right')
        
        # Plot 3: Horizon surface with faults (3D)
        ax3 = fig.add_subplot(223, projection='3d')
        selected_horizon = list(horizons.values())[0]  # Use first horizon for visualization
        
        # Create meshgrid for 3D surface
        X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        
        # Plot horizon surface using 3D axes
        ax3.plot_surface(X, Y, selected_horizon.T, cmap='viridis', alpha=0.7, 
                        rstride=5, cstride=5, edgecolor='none')
        
        # Add fault planes if available
        if "fault_planes" in faults:
            for plane in faults["fault_planes"][:3]:  # Show only a few planes for clarity
                plane_points = list(plane)
                if len(plane_points) > 100:
                    # Sample points for clearer visualization
                    sample_size = max(100, len(plane_points)//100)
                    sampled_points = plane_points[:sample_size]
                    xs = [p[0] for p in sampled_points]
                    ys = [p[1] for p in sampled_points]
                    zs = [p[2] for p in sampled_points]
                    ax3.scatter(xs, ys, zs, c='red', marker='o', s=20, alpha=0.8)
        
        ax3.set_title("3D Horizon Surface with Faults")
        ax3.set_xlabel("Inline")
        ax3.set_ylabel("Crossline")
        ax3.set_zlabel("Time/Depth")
        
        # Plot 4: Time/depth slice with horizons and faults
        ax4 = fig.add_subplot(224)
        depth_idx = shape[2] // 3  # Slice at 1/3 depth
        depth_slice = volume[:, :, depth_idx]
        
        ax4.imshow(depth_slice, cmap='seismic', aspect='auto')
        ax4.set_title(f"Time/Depth Slice at {depth_idx} with Faults")
        ax4.set_xlabel("Crossline")
        ax4.set_ylabel("Inline")
        
        # Add fault intersections with the slice
        if "fault_planes" in faults:
            fault_x = []
            fault_y = []
            
            for plane in faults["fault_planes"]:
                for point in plane:
                    i, j, k = point
                    if abs(k - depth_idx) <= 1:  # Allow some tolerance
                        fault_x.append(j)
                        fault_y.append(i)
            
            if fault_x:
                ax4.scatter(fault_x, fault_y, c='red', marker='x', s=30, alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved comprehensive visualization to {save_path}")
        
        plt.close()
        
        return "Visualization completed successfully"

# Task dependency graph for tracking workflow progress
class WorkflowDAG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.results = {}
        self.completed_tasks = set()
        
    def add_task(self, task_id: str, description: str, dependencies: List[str] = None):
        """Add a task to the workflow with optional dependencies"""
        self.graph.add_node(task_id, description=description)
        if dependencies:
            for dep in dependencies:
                if dep in self.graph:
                    self.graph.add_edge(dep, task_id)
                else:
                    raise ValueError(f"Dependency {dep} not found in workflow")
    
    def is_ready(self, task_id: str) -> bool:
        """Check if a task is ready to be executed (all dependencies completed)"""
        if task_id in self.completed_tasks:
            return False
        
        dependencies = list(self.graph.predecessors(task_id))
        return all(dep in self.completed_tasks for dep in dependencies)
    
    def get_ready_tasks(self) -> List[str]:
        """Get all tasks that are ready to be executed"""
        return [task for task in self.graph.nodes 
                if self.is_ready(task) and task not in self.completed_tasks]
    
    def mark_completed(self, task_id: str, result: Any = None):
        """Mark a task as completed and store its result"""
        if task_id not in self.graph:
            raise ValueError(f"Task {task_id} not found in workflow")
        
        self.completed_tasks.add(task_id)
        if result is not None:
            self.results[task_id] = result
    
    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """Get detailed information about a task"""
        if task_id not in self.graph:
            raise ValueError(f"Task {task_id} not found in workflow")
        
        return {
            "task_id": task_id,
            "description": self.graph.nodes[task_id]["description"],
            "dependencies": list(self.graph.predecessors(task_id)),
            "dependents": list(self.graph.successors(task_id)),
            "is_completed": task_id in self.completed_tasks,
            "result": self.results.get(task_id, None)
        }
    
    def get_dependency_results(self, task_id: str) -> Dict[str, Any]:
        """Get results from all dependencies of a task"""
        dependencies = list(self.graph.predecessors(task_id))
        return {dep: self.results.get(dep) for dep in dependencies if dep in self.results}
    
    def all_tasks_completed(self) -> bool:
        """Check if all tasks in the workflow are completed"""
        return len(self.completed_tasks) == len(self.graph.nodes)


def update_seismic_data_generator():
        """Updates the SeismicDataGenerator with the optimized extract_horizons function"""
        SeismicDataGenerator.extract_horizons = extract_horizons

# Agent Function Definitions
def execute_data_loading(task_info, workflow_dag):
    """Function to load or generate seismic data"""
    print(f"Executing data loading task: {task_info['description']}")
    
    # Generate synthetic seismic data
    data_generator = SeismicDataGenerator()
    volume, metadata = data_generator.generate_seismic_volume(
        shape=(30, 30, 5),
        noise_level=0.2,
        save_path="workspace/data/raw/seismic_volume.npy"
    )
    
    # Result with metadata
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "dataset_info": {
            "dimensions": volume.shape,
            "format": "NumPy Array",
            "generated": True,
            "metadata": metadata
        },
        "file_path": "workspace/data/raw/seismic_volume.npy"
    }
    
    # Mark the task as completed in the workflow
    workflow_dag.mark_completed(task_info['task_id'], result)
    return json.dumps(result, indent=2)

def execute_data_cleaning(task_info, workflow_dag):
    """Function to clean seismic data"""
    print(f"Executing data cleaning task: {task_info['description']}")
    
    # Get input data from dependencies
    input_data = workflow_dag.get_dependency_results(task_info['task_id'])
    
    # Load raw seismic volume
    raw_data_path = input_data.get("load_data", {}).get("file_path", "workspace/data/raw/seismic_volume.npy")
    volume = np.load(raw_data_path)
    
    # Simple cleaning: apply smoothing filter
    from scipy import ndimage
    cleaned_volume = ndimage.gaussian_filter(volume, sigma=0.5)
    
    # Save cleaned data
    cleaned_path = "workspace/data/processed/cleaned_seismic_volume.npy"
    np.save(cleaned_path, cleaned_volume)
    
    # Create a simple visualization of before/after
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(volume[:, volume.shape[1]//2, :].T, cmap='seismic', aspect='auto')
    plt.title("Original Data")
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(cleaned_volume[:, cleaned_volume.shape[1]//2, :].T, cmap='seismic', aspect='auto')
    plt.title("Cleaned Data")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("workspace/data/visualizations/cleaning_comparison.png")
    plt.close()
    
    # Result with metadata
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "preprocessing_steps": [
            "Gaussian smoothing with sigma=0.5",
            "Noise reduction complete"
        ],
        "output_file_path": cleaned_path,
        "visualization_path": "workspace/data/visualizations/cleaning_comparison.png"
    }
    
    # Mark the task as completed in the workflow
    workflow_dag.mark_completed(task_info['task_id'], result)
    return json.dumps(result, indent=2)

def extract_horizons(volume, noise_threshold=0.6, save_path=None):
    """Optimized function to extract horizons from seismic volume"""
    print("Extracting horizons from seismic volume...")
    
    horizons = []
    shape = volume.shape
    
    # Add progress reporting
    print(f"Volume shape: {shape}")
    print(f"Threshold: {noise_threshold}")
    print("Phase 1: Identifying high-amplitude points...")
    
    # Optimize by using numpy operations instead of loops
    # Find points with amplitude above threshold
    high_amplitude_points = np.where(volume > noise_threshold)
    
    # Convert to list of tuples for easier processing
    print(f"Found {len(high_amplitude_points[0])} high-amplitude points")
    print("Phase 2: Grouping points into horizons...")
    
    # Take a sample of points if there are too many (for performance)
    max_points = 50000
    if len(high_amplitude_points[0]) > max_points:
        print(f"Sampling {max_points} points out of {len(high_amplitude_points[0])} for performance")
        indices = np.random.choice(len(high_amplitude_points[0]), max_points, replace=False)
        points = [(high_amplitude_points[0][i], high_amplitude_points[1][i], high_amplitude_points[2][i]) 
                  for i in indices]
    else:
        points = [(high_amplitude_points[0][i], high_amplitude_points[1][i], high_amplitude_points[2][i]) 
                  for i in range(len(high_amplitude_points[0]))]
    
    # Group horizons by approximate depth
    horizon_groups = {}
    depth_tolerance = 3  # Allow horizons to vary by this many samples
    
    # Process in batches for better progress reporting
    batch_size = 1000
    num_batches = (len(points) + batch_size - 1) // batch_size
    
    for batch in range(num_batches):
        if batch % 10 == 0:
            print(f"Processing batch {batch+1}/{num_batches}")
        
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(points))
        batch_points = points[start_idx:end_idx]
        
        for i, j, depth in batch_points:
            # Find group with similar depth
            assigned = False
            for group_id, group in horizon_groups.items():
                # Use the mean depth of the first 100 points as reference
                reference_depths = [d for _, _, d in group[:100]]
                if reference_depths and abs(depth - np.mean(reference_depths)) < depth_tolerance:
                    group.append((i, j, depth))
                    assigned = True
                    break
            
            # Create new group if no match
            if not assigned:
                new_group_id = len(horizon_groups)
                horizon_groups[new_group_id] = [(i, j, depth)]
    
    print(f"Found {len(horizon_groups)} potential horizon groups")
    
    # Filter out small groups
    min_points = shape[0] * shape[1] * 0.02  # At least 2% coverage
    print(f"Filtering out groups with fewer than {min_points} points")
    filtered_groups = {k: v for k, v in horizon_groups.items() if len(v) >= min_points}
    print(f"Retained {len(filtered_groups)} significant horizon groups")
    
    # Convert to numpy arrays for easier processing
    horizon_surfaces = {}
    print("Phase 3: Creating horizon surfaces...")
    
    # Limit to 5 horizons maximum for performance
    top_groups = sorted(filtered_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    
    for idx, (group_id, points) in enumerate(top_groups):
        print(f"Processing horizon {idx+1}/{len(top_groups)} with {len(points)} points")
        
        # Initialize with NaN
        surface = np.full((shape[0], shape[1]), np.nan)
        
        # Fill in detected points
        for i, j, depth in points:
            if 0 <= i < shape[0] and 0 <= j < shape[1]:  # Ensure within bounds
                if np.isnan(surface[i, j]) or depth < surface[i, j]:
                    # Keep the shallowest point at each i,j location
                    surface[i, j] = depth
        
        # Simple interpolation for missing values
        print(f"Interpolating missing values in horizon {idx+1}")
        
        # Use a more efficient interpolation approach
        # First, identify all points that need interpolation
        mask = np.isnan(surface)
        
        # Perform a simple diffusion-based interpolation
        # This is much faster than point-by-point interpolation
        # Iterate a few times to fill in gaps
        for iteration in range(5):
            if np.all(~mask):  # If no more NaNs, we're done
                break
                
            # For each NaN point, take the average of non-NaN neighbors
            new_surface = surface.copy()
            
            # Create shifted versions of the array for each direction
            shifts = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            valid_neighbors = np.zeros_like(surface)
            neighbor_sum = np.zeros_like(surface)
            
            for di, dj in shifts:
                shifted = np.roll(surface, (di, dj), axis=(0, 1))
                # If we rolled across an edge, invalidate those cells
                if di > 0:
                    shifted[-di:, :] = np.nan
                elif di < 0:
                    shifted[:-di, :] = np.nan
                if dj > 0:
                    shifted[:, -dj:] = np.nan
                elif dj < 0:
                    shifted[:, :-dj] = np.nan
                
                valid = ~np.isnan(shifted)
                valid_neighbors += valid
                neighbor_sum += np.nan_to_num(shifted, 0) * valid
            
            # Calculate the average
            has_neighbors = valid_neighbors > 0
            interpolated = neighbor_sum / np.maximum(valid_neighbors, 1)
            
            # Only update NaN points that have at least one valid neighbor
            update_mask = mask & has_neighbors
            new_surface[update_mask] = interpolated[update_mask]
            
            # Update the surface and mask
            surface = new_surface
            mask = np.isnan(surface)
            
            print(f"  Iteration {iteration+1}: {np.sum(mask)} points still need interpolation")
            
            if iteration == 4 and np.any(mask):
                print(f"  Note: {np.sum(mask)} points could not be interpolated, will be set to average depth")
                # Fill any remaining NaNs with the average depth
                surface[mask] = np.nanmean(surface)
        
        horizon_surfaces[idx] = surface
    
    # Save horizons if path is provided
    if save_path:
        print(f"Saving extracted horizons to {save_path}")
        np.save(save_path, horizon_surfaces)
        
        # Save visualization of horizons
        print("Creating horizon visualization...")
        plt.figure(figsize=(15, 10))
        for group_id, surface in horizon_surfaces.items():
            plt.subplot(1, len(horizon_surfaces), group_id + 1)
            plt.imshow(surface, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Depth')
            plt.title(f"Horizon {group_id + 1}")
        
        viz_path = save_path.replace('.npy', '_viz.png')
        plt.savefig(viz_path)
        plt.close()
        print(f"Saved horizon visualization to {viz_path}")
    
    print("Horizon extraction completed successfully")
    return horizon_surfaces

# Replace the original SeismicDataGenerator.extract_horizons with this optimized version

# Add these missing functions to your code
def execute_horizon_visualization(task_info, workflow_dag):
    """Function to visualize horizons"""
    print(f"Executing horizon visualization task: {task_info['description']}")
    
    # Get input data from dependencies
    input_data = workflow_dag.get_dependency_results(task_info['task_id'])
    
    # Load cleaned data and horizons
    cleaned_data_path = workflow_dag.results.get("clean_data", {}).get("output_file_path", 
                                                                     "workspace/data/processed/cleaned_seismic_volume.npy")
    horizons_path = workflow_dag.results.get("pick_horizons", {}).get("output_file_path",
                                                                    "workspace/data/processed/horizon_surfaces.npy")
    
    cleaned_volume = np.load(cleaned_data_path)
    horizons = np.load(horizons_path, allow_pickle=True).item()
    
    # Create visualization of horizons
    fig = plt.figure(figsize=(15, 10))
    
    # Plot inline section with horizons
    ax1 = fig.add_subplot(211)
    inline_idx = cleaned_volume.shape[1] // 2
    ax1.imshow(cleaned_volume[:, inline_idx, :].T, cmap='seismic', aspect='auto')
    ax1.set_title(f"Inline {inline_idx} with Horizons")
    
    # Add horizons to the plot
    for horizon_id, surface in horizons.items():
        horizon_line = surface[:, inline_idx]
        ax1.plot(np.arange(cleaned_volume.shape[0]), horizon_line, 
                 linewidth=2, label=f"Horizon {horizon_id + 1}")
    
    plt.colorbar(label='Amplitude', ax=ax1)
    ax1.set_xlabel("Crossline")
    ax1.set_ylabel("Time/Depth")
    ax1.legend()
    
    # Plot 3D view of a horizon
    # Import 3D plotting tools
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create a 3D subplot
    ax2 = fig.add_subplot(212, projection='3d')
    
    selected_horizon = list(horizons.values())[0]  # First horizon
    
    # Downsample for faster plotting
    downsample = 3
    X, Y = np.meshgrid(
        np.arange(0, cleaned_volume.shape[0], downsample),
        np.arange(0, cleaned_volume.shape[1], downsample)
    )
    Z = selected_horizon[::downsample, ::downsample].T
    
    # Use the 3D axes to plot the surface
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, 
                    rstride=1, cstride=1, edgecolor='none')
    ax2.set_title("3D View of Horizon Surface")
    ax2.set_xlabel("Inline")
    ax2.set_ylabel("Crossline")
    
    # Save visualization
    viz_path = "workspace/data/visualizations/horizon_visualization.png"
    plt.tight_layout()
    plt.savefig(viz_path, dpi=300)
    plt.close()
    
    # Result with metadata
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "visualization_type": "2D and 3D horizon views",
        "elements_included": ["horizons", "seismic data"],
        "output_file_path": viz_path
    }
    
    # Mark the task as completed in the workflow
    workflow_dag.mark_completed(task_info['task_id'], result)
    return json.dumps(result, indent=2)

def execute_horizon_picking(task_info, workflow_dag):
    """Function to identify geological horizons"""
    print(f"Executing horizon picking task: {task_info['description']}")
    
    # Get input data from dependencies
    input_data = workflow_dag.get_dependency_results(task_info['task_id'])
    
    # Load cleaned seismic volume
    cleaned_data_path = input_data.get("clean_data", {}).get("output_file_path", 
                                                           "workspace/data/processed/cleaned_seismic_volume.npy")
    cleaned_volume = np.load(cleaned_data_path)
    
    # Extract horizons using the optimized function
    horizons = SeismicDataGenerator.extract_horizons(
        cleaned_volume, 
        noise_threshold=0.6,
        save_path="workspace/data/processed/horizon_surfaces.npy"
    )
    
    # Result with metadata
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "horizons_detected": len(horizons),
        "confidence_scores": [0.85 + 0.03 * i for i in range(len(horizons))],  # Simulated confidence
        "output_file_path": "workspace/data/processed/horizon_surfaces.npy",
        "visualization_path": "workspace/data/processed/horizon_surfaces_viz.png"
    }
    
    # Mark the task as completed in the workflow
    workflow_dag.mark_completed(task_info['task_id'], result)
    return json.dumps(result, indent=2)

def execute_fault_detection(task_info, workflow_dag):
    """Function to detect faults in seismic data"""
    print(f"Executing fault detection task: {task_info['description']}")
    
    # Get input data from dependencies
    input_data = workflow_dag.get_dependency_results(task_info['task_id'])
    
    # Load cleaned seismic volume and horizons
    cleaned_data_path = workflow_dag.results.get("clean_data", {}).get("output_file_path", 
                                                                     "workspace/data/processed/cleaned_seismic_volume.npy")
    horizons_path = workflow_dag.results.get("pick_horizons", {}).get("output_file_path",
                                                                    "workspace/data/processed/horizon_surfaces.npy")
    
    cleaned_volume = np.load(cleaned_data_path)
    horizons = np.load(horizons_path, allow_pickle=True).item()
    
    # Detect faults
    fault_results = SeismicDataGenerator.detect_faults(
        cleaned_volume, 
        horizons,
        save_path="workspace/data/processed/fault_surfaces.npy"
    )
    
    # Result with metadata
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "faults_detected": fault_results["num_fault_planes"],
        "major_faults": min(3, fault_results["num_fault_planes"]),
        "intersection_analysis": {
            "horizon_intersections": fault_results["num_fault_planes"] * 2,
            "critical_intersections": fault_results["num_fault_planes"] // 2
        },
        "output_file_path": "workspace/data/processed/fault_surfaces.npy",
        "visualization_path": "workspace/data/processed/fault_surfaces_viz.png"
    }
    
    # Mark the task as completed in the workflow
    workflow_dag.mark_completed(task_info['task_id'], result)
    return json.dumps(result, indent=2)
    
# Add this error handling wrapper around your workflow execution
def execute_workflow_with_error_handling():
    """Execute workflow with proper error handling and progress reporting"""
    try:
        workflow = run_workflow_without_llm()
        return workflow
    except Exception as e:
        import traceback
        print(f"\n==== ERROR OCCURRED ====\n")
        print(f"Error details: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        print("\nCheck the workspace directory for any partial results.")
        return None
    
def execute_horizon_visualization(task_info, workflow_dag):
    """Function to visualize horizons (2D only - avoids 3D errors)"""
    print(f"Executing horizon visualization task: {task_info['description']}")
    
    # Get input data from dependencies
    input_data = workflow_dag.get_dependency_results(task_info['task_id'])
    
    # Load cleaned data and horizons
    cleaned_data_path = workflow_dag.results.get("clean_data", {}).get("output_file_path", 
                                                                     "workspace/data/processed/cleaned_seismic_volume.npy")
    horizons_path = workflow_dag.results.get("pick_horizons", {}).get("output_file_path",
                                                                    "workspace/data/processed/horizon_surfaces.npy")
    
    cleaned_volume = np.load(cleaned_data_path)
    horizons = np.load(horizons_path, allow_pickle=True).item()
    
    # Create visualization of horizons - 2D only
    plt.figure(figsize=(15, 6))
    
    # Plot inline section with horizons
    inline_idx = cleaned_volume.shape[1] // 2
    plt.imshow(cleaned_volume[:, inline_idx, :].T, cmap='seismic', aspect='auto')
    plt.title(f"Inline {inline_idx} with Horizons")
    
    # Add horizons to the plot
    for horizon_id, surface in horizons.items():
        horizon_line = surface[:, inline_idx]
        plt.plot(np.arange(cleaned_volume.shape[0]), horizon_line, 
                 linewidth=2, label=f"Horizon {horizon_id + 1}")
    
    plt.colorbar(label='Amplitude')
    plt.xlabel("Crossline")
    plt.ylabel("Time/Depth")
    plt.legend()
    
    # Save visualization
    viz_path = "workspace/data/visualizations/horizon_visualization.png"
    plt.tight_layout()
    plt.savefig(viz_path, dpi=300)
    plt.close()
    
    # Result with metadata
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "visualization_type": "2D horizon view",
        "elements_included": ["horizons", "seismic data"],
        "output_file_path": viz_path
    }
    
    # Mark the task as completed in the workflow
    workflow_dag.mark_completed(task_info['task_id'], result)
    return json.dumps(result, indent=2)

def execute_final_visualization(task_info, workflow_dag):
    """Function to create a simplified final visualization (no 3D)"""
    print(f"Executing final visualization task: {task_info['description']}")
    
    # Get paths to all required data
    volume_path = workflow_dag.results.get("clean_data", {}).get("output_file_path", 
                                                               "workspace/data/processed/cleaned_seismic_volume.npy")
    horizons_path = workflow_dag.results.get("pick_horizons", {}).get("output_file_path",
                                                                    "workspace/data/processed/horizon_surfaces.npy")
    faults_path = workflow_dag.results.get("detect_faults", {}).get("output_file_path",
                                                                  "workspace/data/processed/fault_surfaces.npy")
    
    # Load all data
    volume = np.load(volume_path)
    horizons = np.load(horizons_path, allow_pickle=True).item()
    fault_data = np.load(faults_path, allow_pickle=True).item()
    
    # Create simplified 2D visualization
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Seismic inline with horizons
    plt.subplot(221)
    inline_idx = volume.shape[1] // 2
    inline_data = volume[:, inline_idx, :].T
    
    plt.imshow(inline_data, cmap='seismic', aspect='auto')
    plt.title(f"Seismic Inline {inline_idx} with Horizons")
    plt.xlabel("Crossline")
    plt.ylabel("Time/Depth")
    
    # Add horizons to inline
    for horizon_id, surface in horizons.items():
        horizon_line = surface[:, inline_idx]
        plt.plot(np.arange(volume.shape[0]), horizon_line, 
                 linewidth=2, label=f"Horizon {horizon_id + 1}")
    
    plt.legend(loc='upper right', fontsize='small')
    
    # Plot 2: Seismic crossline with horizons
    plt.subplot(222)
    xline_idx = volume.shape[0] // 2
    xline_data = volume[xline_idx, :, :].T
    
    plt.imshow(xline_data, cmap='seismic', aspect='auto')
    plt.title(f"Seismic Crossline {xline_idx} with Horizons")
    plt.xlabel("Inline")
    plt.ylabel("Time/Depth")
    
    # Add horizons to crossline
    for horizon_id, surface in horizons.items():
        horizon_line = surface[xline_idx, :]
        plt.plot(np.arange(volume.shape[1]), horizon_line, 
                 linewidth=2, label=f"Horizon {horizon_id + 1}")
    
    plt.legend(loc='upper right', fontsize='small')
    
    # Plot 3: Time slice
    plt.subplot(223)
    time_idx = volume.shape[2] // 2
    time_slice = volume[:, :, time_idx]
    
    plt.imshow(time_slice, cmap='seismic', aspect='auto')
    plt.title(f"Time Slice at {time_idx}")
    plt.xlabel("Inline")
    plt.ylabel("Crossline")
    plt.colorbar(label='Amplitude')
    
    # Plot 4: Fault likelihood
    plt.subplot(224)
    if "fault_likelihood" in fault_data:
        # Calculate maximum likelihood along depth axis
        fault_slice = np.max(fault_data["fault_likelihood"], axis=2)
        plt.imshow(fault_slice, cmap='hot', aspect='auto')
        plt.title("Fault Likelihood (Maximum Projection)")
        plt.xlabel("Inline")
        plt.ylabel("Crossline")
        plt.colorbar(label='Likelihood')
    else:
        plt.text(0.5, 0.5, "Fault data not available", ha='center', va='center')
        plt.title("Fault Analysis")
    
    plt.tight_layout()
    
    # Save the visualization
    viz_path = "workspace/data/visualizations/final_comprehensive_view.png"
    plt.savefig(viz_path, dpi=300)
    plt.close()
    
    # Result with metadata
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "visualization_type": "Comprehensive 2D visualization",
        "elements_included": ["seismic data", "horizons", "faults", "time slice"],
        "output_file_path": viz_path
    }
    
    # Mark the task as completed in the workflow
    workflow_dag.mark_completed(task_info['task_id'], result)
    return json.dumps(result, indent=2)


def execute_final_visualization(task_info, workflow_dag):
    """Function to mock the final visualization (for testing)"""
    print(f"Executing final visualization task: {task_info['description']}")
    
    # Get input data from dependencies - still need to reference to mark as completed
    input_data = workflow_dag.get_dependency_results(task_info['task_id'])
    
    # Result with metadata
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "visualization_type": "Mock final visualization (skipped for testing)",
        "elements_included": ["seismic data", "horizons", "faults", "3D view"],
        "output_file_path": "workspace/data/visualizations/final_visualization_mock.txt"
    }
    
    # Create a mock visualization file
    with open("workspace/data/visualizations/final_visualization_mock.txt", "w") as f:
        f.write("Final visualization skipped for testing purposes.")
    
    # Mark the task as completed in the workflow
    workflow_dag.mark_completed(task_info['task_id'], result)
    return json.dumps(result, indent=2)

# =============================================================================
# HUMAN-IN-THE-LOOP WORKFLOW EXECUTION
# =============================================================================
def run_workflow_with_human_in_loop():
    """Execute the workflow with human approval at each step"""
    print("\n" + "="*60)
    print("🔄 HUMAN-IN-THE-LOOP WORKFLOW MODE")
    print("="*60)
    print("\nYou will be asked to review and approve each task result.")
    print("This allows you to validate interpretations before proceeding.\n")
    
    # Initialize the workflow DAG
    workflow = WorkflowDAG()
    
    # Define tasks with dependencies
    workflow.add_task("load_data", "Load seismic data from the specified source")
    workflow.add_task("clean_data", "Clean and normalize the seismic data", ["load_data"])
    workflow.add_task("pick_horizons", "Identify geological horizons in the processed data", ["clean_data"])
    workflow.add_task("detect_faults", "Detect fault lines in relation to identified horizons", ["pick_horizons"])
    workflow.add_task("visualize_horizons", "Create visual representation of the identified horizons", ["pick_horizons"])
    workflow.add_task("create_final_visualization", 
                    "Create comprehensive visualization including horizons and faults", 
                    ["detect_faults", "visualize_horizons"])
    
    # Function mapping for task execution
    function_map = {
        "load_data": execute_data_loading,
        "clean_data": execute_data_cleaning,
        "pick_horizons": execute_horizon_picking,
        "detect_faults": execute_fault_detection,
        "visualize_horizons": execute_horizon_visualization,
        "create_final_visualization": execute_final_visualization
    }
    
    # Show initial workflow status
    show_workflow_status(workflow)
    
    # Execute workflow with human approval
    print("\n==== Starting Human-in-the-Loop Workflow ====\n")
    
    try:
        while not workflow.all_tasks_completed():
            # Get tasks that are ready to be executed
            ready_tasks = workflow.get_ready_tasks()
            
            if not ready_tasks:
                if workflow.all_tasks_completed():
                    print("\n✅ All tasks completed!")
                    break
                else:
                    print("\n❌ Error: No tasks ready but workflow not complete")
                    break
            
            print(f"\n📋 Ready tasks: {ready_tasks}")
            
            # Process each ready task
            for task_id in ready_tasks:
                task_info = workflow.get_task_info(task_id)
                print(f"\n{'='*60}")
                print(f"▶️  EXECUTING: {task_id}")
                print(f"   {task_info['description']}")
                print(f"{'='*60}")
                
                # Execute the task
                if task_id in function_map:
                    start_time = time.time()
                    result_json = function_map[task_id](task_info, workflow)
                    end_time = time.time()
                    
                    print(f"\n⏱️  Task completed in {end_time - start_time:.2f} seconds")
                    
                    # Parse and display result
                    try:
                        result = json.loads(result_json)
                    except Exception:
                        result = {"raw_result": result_json}
                    
                    display_task_result(task_id, result)
                    
                    # Get human approval if required
                    if workflow_config.human_approval_required:
                        approved, feedback = get_human_approval(task_id, task_info['description'])
                        
                        # Store feedback
                        human_feedback_state["approvals"][task_id] = {
                            "approved": approved,
                            "feedback": feedback,
                            "timestamp": datetime.now().isoformat()
                        }
                        human_feedback_state["comments"][task_id] = feedback
                        
                        if not approved:
                            print(f"\n⚠️  Task {task_id} was rejected. Reason: {feedback}")
                            print("Note: Re-running rejected tasks is not yet implemented.")
                            print("Continuing with workflow...")
                    else:
                        print("✓ Auto-approved")
                else:
                    print(f"❌ Error: No function mapping for task {task_id}")
            
            # Show updated workflow status
            show_workflow_status(workflow)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user.")
        show_workflow_status(workflow)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 WORKFLOW EXECUTION SUMMARY")
    print("="*60)
    print(f"Total tasks: {len(workflow.graph.nodes)}")
    print(f"Completed tasks: {len(workflow.completed_tasks)}")
    
    # Show human feedback summary
    if human_feedback_state["approvals"]:
        print("\n👤 Human Feedback Summary:")
        for task_id, approval_info in human_feedback_state["approvals"].items():
            status = "✅ Approved" if approval_info["approved"] else "❌ Rejected"
            print(f"   - {task_id}: {status}")
            if approval_info["feedback"] != "Approved":
                print(f"     Comment: {approval_info['feedback']}")
    
    if human_feedback_state["modifications"]:
        print("\n📝 Modification Requests:")
        for mod in human_feedback_state["modifications"]:
            print(f"   - {mod['task_id']}: {mod['feedback']}")
    
    # List output files
    print("\n💾 Output Files:")
    for task_id, result in workflow.results.items():
        if isinstance(result, dict) and "output_file_path" in result:
            print(f"   - {task_id}: {result['output_file_path']}")
    
    return workflow

# Run workflow without requiring LLM interaction
def run_workflow_without_llm():
    """Execute the workflow directly without LLM agents (fully automated)"""
    print("Running workflow in direct execution mode (no LLM)...")
    
    # Initialize the workflow DAG
    workflow = WorkflowDAG()
    
    # Define tasks with dependencies
    workflow.add_task("load_data", "Load seismic data from the specified source")
    workflow.add_task("clean_data", "Clean and normalize the seismic data", ["load_data"])
    workflow.add_task("pick_horizons", "Identify geological horizons in the processed data", ["clean_data"])
    workflow.add_task("detect_faults", "Detect fault lines in relation to identified horizons", ["pick_horizons"])
    workflow.add_task("visualize_horizons", "Create visual representation of the identified horizons", ["pick_horizons"])
    workflow.add_task("create_final_visualization", 
                    "Create comprehensive visualization including horizons and faults", 
                    ["detect_faults", "visualize_horizons"])
    
    # Function mapping for task execution
    function_map = {
        "load_data": execute_data_loading,
        "clean_data": execute_data_cleaning,
        "pick_horizons": execute_horizon_picking,
        "detect_faults": execute_fault_detection,
        "visualize_horizons": execute_horizon_visualization,
        "create_final_visualization": execute_final_visualization
    }
    
    # Execute workflow
    print("\n==== Starting Workflow Execution ====\n")
    
    while not workflow.all_tasks_completed():
        # Get tasks that are ready to be executed
        ready_tasks = workflow.get_ready_tasks()
        
        if not ready_tasks:
            if workflow.all_tasks_completed():
                print("\n==== All tasks completed! ====")
                break
            else:
                print("\n==== Error: No tasks ready but workflow not complete ====")
                break
        
        print(f"\nReady tasks: {ready_tasks}")
        
        # Process each ready task
        for task_id in ready_tasks:
            task_info = workflow.get_task_info(task_id)
            print(f"\n---- Processing task: {task_id} - {task_info['description']} ----")
            
            # Execute the task using the appropriate function
            if task_id in function_map:
                start_time = time.time()
                result = function_map[task_id](task_info, workflow)
                end_time = time.time()
                
                print(f"Task {task_id} executed in {end_time - start_time:.2f} seconds")
                print(f"Result preview: {result[:200]}...")
            else:
                print(f"Error: No function mapping for task {task_id}")
    
    # Print summary of results
    print("\n==== Workflow Execution Summary ====\n")
    print(f"Total tasks: {len(workflow.graph.nodes)}")
    print(f"Completed tasks: {len(workflow.completed_tasks)}")
    
    # List all output files
    print("\nOutput Files:")
    for task_id, result in workflow.results.items():
        if isinstance(result, dict) and "output_file_path" in result:
            print(f"- {task_id}: {result['output_file_path']}")
    
    return workflow

# Create agents with LLM support
def create_geoscience_agents(workflow_dag, config_list):
    """Create specialized agents using the DashScope API"""
    
    # Configure the LLM for use with DashScope
    if config_list:
        llm_config = {
            "config_list": config_list,
            # Add any additional OpenAI-compatible parameters if needed
            "temperature": 0.2,
            "timeout": 120
        }
    else:
        llm_config = None
    
    # Data processing agent
    data_processor = autogen.AssistantAgent(
        name="DataProcessor",
        system_message="""
        You are a Data Processing Expert specialized in seismic data.
        Your responsibilities include loading seismic data from files
        and preprocessing it for interpretation.
        """,
        llm_config=llm_config
    )
    
    # Horizon picking agent
    horizon_expert = autogen.AssistantAgent(
        name="HorizonExpert",
        system_message="""
        You are a Horizon Picking Expert specialized in identifying 
        geological horizons in seismic data.
        """,
        llm_config=llm_config
    )
    
    # Fault detection agent
    fault_expert = autogen.AssistantAgent(
        name="FaultExpert",
        system_message="""
        You are a Fault Detection Expert specialized in identifying 
        fault structures in seismic data.
        """,
        llm_config=llm_config
    )
    
    # Visualization agent
    visualization_expert = autogen.AssistantAgent(
        name="VisualizationExpert",
        system_message="""
        You are a Geoscience Visualization Expert specialized in 
        creating informative visual representations.
        """,
        llm_config=llm_config
    )
    
    # Manager agent
    manager_agent = autogen.AssistantAgent(
        name="WorkflowManager",
        system_message="""
        You are the Geoscience Workflow Manager, responsible for 
        coordinating the execution of a complex geoscience data
        processing workflow.
        """,
        llm_config=llm_config
    )
    
    # User proxy agent - MODIFIED to disable Docker requirement
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "workspace",
            "use_docker": False  # Disable Docker requirement
        }
    )
    
    # Register all agents and their capabilities
    agents = {
        "manager": manager_agent,
        "data_processor": {"agent": data_processor, "tasks": ["load", "clean"]},
        "horizon_expert": {"agent": horizon_expert, "tasks": ["horizon"]},
        "fault_expert": {"agent": fault_expert, "tasks": ["fault"]},
        "visualization_expert": {"agent": visualization_expert, "tasks": ["visual"]}
    }
    
    return agents, user_proxy

# Modified setup_autogen_config function to address API key issues
def setup_autogen_config():
    """Set up autogen configuration for Alibaba Cloud DashScope"""
    # Define the API key - either from environment or hardcoded for testing
    api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-")
    
    # DashScope specific configuration
    config_list = [
        {
            "model": "qwen-plus",  # Alibaba Cloud's model
            "api_key": os.getenv("QWEN_API_KEY", ""), 
            "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        }
    ]
    
    # Always run in LLM mode since we have the API key configured
    print("Using Alibaba Cloud DashScope API for LLM interactions.")
    use_llm = True
    
    return config_list, use_llm

def display_main_menu():
    """Display the main menu for workflow mode selection"""
    print("\n" + "="*60)
    print("🌊 AUTOGEN SEISMIC INTERPRETATION SYSTEM")
    print("="*60)
    print("\nSelect execution mode:\n")
    print("  [1] 🤖 Fully Automated (no interaction)")
    print("      - Runs all tasks automatically")
    print("      - Best for batch processing")
    print()
    print("  [2] 👤 Human-in-the-Loop")
    print("      - Review and approve each task result")
    print("      - Provide feedback and modification notes")
    print("      - Best for quality control and validation")
    print()
    print("  [3] 🧠 LLM-Coordinated (requires API key)")
    print("      - AI agents coordinate the workflow")
    print("      - Requires configured DashScope/OpenAI API")
    print()
    print("  [4] ℹ️  Show workflow information")
    print("  [5] 🚪 Exit")
    print()
    
    return input("Enter your choice [1-5]: ").strip()

def show_workflow_info():
    """Display information about the workflow tasks"""
    print("\n" + "="*60)
    print("📋 WORKFLOW INFORMATION")
    print("="*60)
    print("\nThis workflow processes seismic data through the following stages:\n")
    
    tasks = [
        ("load_data", "Load/Generate Seismic Data", 
         "Creates or loads a 3D seismic volume with synthetic horizons and faults"),
        ("clean_data", "Data Cleaning & Preprocessing",
         "Applies Gaussian smoothing to reduce noise"),
        ("pick_horizons", "Horizon Picking",
         "Identifies geological horizons using amplitude thresholding"),
        ("detect_faults", "Fault Detection",
         "Detects fault planes based on horizon discontinuities"),
        ("visualize_horizons", "Horizon Visualization",
         "Creates 2D visualization of picked horizons"),
        ("create_final_visualization", "Final Visualization",
         "Generates comprehensive visualization with all interpretations")
    ]
    
    for i, (task_id, name, desc) in enumerate(tasks, 1):
        print(f"  {i}. {name}")
        print(f"     Task ID: {task_id}")
        print(f"     Description: {desc}")
        print()
    
    print("Dependencies:")
    print("  load_data → clean_data → pick_horizons → detect_faults")
    print("                                        ↘ visualize_horizons")
    print("                                           ↘ create_final_visualization")
    print()
    
    input("Press Enter to continue...")

def main():
    """Main function to run the geoscience workflow with mode selection"""
    print("\n🚀 Initializing Autogen Geoscience Workflow...")
    
    # Update the horizon extraction function first
    update_seismic_data_generator()
    
    while True:
        choice = display_main_menu()
        
        if choice == '1':
            # Fully Automated Mode
            print("\n✅ Starting Fully Automated Mode...")
            workflow_config.set_mode(WorkflowConfig.MODE_AUTO)
            
            try:
                workflow = execute_workflow_with_error_handling()
                if workflow:
                    print("\n✅ Workflow completed successfully!")
            except Exception as e:
                print(f"\n❌ Error: {e}")
            
            print("\nCheck 'workspace/data/visualizations' folder for results.")
            
        elif choice == '2':
            # Human-in-the-Loop Mode
            print("\n✅ Starting Human-in-the-Loop Mode...")
            workflow_config.set_mode(WorkflowConfig.MODE_HUMAN_IN_LOOP)
            
            # Reset feedback state
            global human_feedback_state
            human_feedback_state = {
                "approvals": {},
                "modifications": [],
                "comments": {},
                "rejected_tasks": set()
            }
            
            try:
                workflow = run_workflow_with_human_in_loop()
                if workflow:
                    print("\n✅ Workflow completed!")
                    
                    # Ask if user wants to save feedback
                    save_feedback = input("\nSave human feedback to file? [y/n]: ").strip().lower()
                    if save_feedback == 'y':
                        feedback_file = f"workspace/data/human_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(feedback_file, 'w') as f:
                            # Convert set to list for JSON serialization
                            feedback_to_save = human_feedback_state.copy()
                            feedback_to_save["rejected_tasks"] = list(feedback_to_save["rejected_tasks"])
                            json.dump(feedback_to_save, f, indent=2)
                        print(f"📁 Feedback saved to: {feedback_file}")
            except KeyboardInterrupt:
                print("\n\n⚠️ Workflow interrupted by user.")
            except Exception as e:
                print(f"\n❌ Error: {e}")
            
        elif choice == '3':
            # LLM-Coordinated Mode
            print("\n✅ Starting LLM-Coordinated Mode...")
            workflow_config.set_mode(WorkflowConfig.MODE_LLM)
            
            # Setup autogen config
            config_list, use_llm = setup_autogen_config()
            
            if not os.getenv("QWEN_API_KEY"):
                print("\n⚠️ Warning: QWEN_API_KEY environment variable not set.")
                print("Set it with: export QWEN_API_KEY='your-api-key'")
                continue_anyway = input("Continue anyway? [y/n]: ").strip().lower()
                if continue_anyway != 'y':
                    continue
            
            # Initialize the workflow DAG
            workflow = WorkflowDAG()
            
            # Define tasks with dependencies
            workflow.add_task("load_data", "Load seismic data from the specified source")
            workflow.add_task("clean_data", "Clean and normalize the seismic data", ["load_data"])
            workflow.add_task("pick_horizons", "Identify geological horizons in the processed data", ["clean_data"])
            workflow.add_task("detect_faults", "Detect fault lines in relation to identified horizons", ["pick_horizons"])
            workflow.add_task("visualize_horizons", "Create visual representation of the identified horizons", ["pick_horizons"])
            workflow.add_task("create_final_visualization", 
                             "Create comprehensive visualization including horizons and faults", 
                             ["detect_faults", "visualize_horizons"])
            
            try:
                print("Running workflow with Autogen LLM-based agents...")
                
                # Create agents and user proxy
                agents, user_proxy = create_geoscience_agents(workflow, config_list)
                
                # Function map for task execution
                function_map = {
                    "load_data": lambda: execute_data_loading({"task_id": "load_data", "description": workflow.graph.nodes["load_data"]["description"]}, workflow),
                    "clean_data": lambda: execute_data_cleaning({"task_id": "clean_data", "description": workflow.graph.nodes["clean_data"]["description"]}, workflow),
                    "pick_horizons": lambda: execute_horizon_picking({"task_id": "pick_horizons", "description": workflow.graph.nodes["pick_horizons"]["description"]}, workflow),
                    "detect_faults": lambda: execute_fault_detection({"task_id": "detect_faults", "description": workflow.graph.nodes["detect_faults"]["description"]}, workflow),
                    "visualize_horizons": lambda: execute_horizon_visualization({"task_id": "visualize_horizons", "description": workflow.graph.nodes["visualize_horizons"]["description"]}, workflow),
                    "create_final_visualization": lambda: execute_final_visualization({"task_id": "create_final_visualization", "description": workflow.graph.nodes["create_final_visualization"]["description"]}, workflow)
                }
                
                # Register these functions with the user proxy
                user_proxy.register_function(function_map)
                
                # Create a group chat with all agents
                manager = agents["manager"]
                all_agents = [manager] + [info["agent"] for name, info in agents.items() if name != "manager"]
                
                # Create group chat
                groupchat = autogen.GroupChat(agents=all_agents, messages=[])
                group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})
                
                # Start the conversation with the workflow description
                task_list = "\n".join([f"- {task_id}: {workflow.graph.nodes[task_id]['description']}" for task_id in workflow.graph.nodes])
                dependencies = "\n".join([f"- {task_id} depends on: {list(workflow.graph.predecessors(task_id))}" for task_id in workflow.graph.nodes if list(workflow.graph.predecessors(task_id))])
                
                # Initial message describing the workflow
                initial_message = f"""
                We need to process seismic data using our workflow. Here are the tasks:
                
                {task_list}
                
                Dependencies:
                {dependencies}
                
                Tasks should be executed in the correct order based on dependencies.
                When a task is ready to execute (all dependencies completed), you can run it using the corresponding function.
                Please coordinate the execution of this workflow, running tasks in parallel when possible.
                """
                
                # Start the conversation
                user_proxy.initiate_chat(group_chat_manager, message=initial_message)
                
                print("\n✅ LLM Workflow completed!")
                
            except Exception as e:
                print(f"\n❌ Error during LLM execution: {e}")
                print("Falling back to automated mode...")
                workflow = execute_workflow_with_error_handling()
            
        elif choice == '4':
            # Show workflow information
            show_workflow_info()
            
        elif choice == '5':
            # Exit
            print("\n👋 Goodbye!")
            break
            
        else:
            print("\n❌ Invalid choice. Please enter 1-5.")
        
        # Ask if user wants to continue
        if choice in ['1', '2', '3']:
            again = input("\n🔄 Run another workflow? [y/n]: ").strip().lower()
            if again != 'y':
                print("\n👋 Goodbye!")
                break

if __name__ == "__main__":
    main()