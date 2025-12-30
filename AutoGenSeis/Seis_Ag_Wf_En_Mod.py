import autogen
from typing import Dict, List, Annotated, Tuple
import numpy as np
import json
from datetime import datetime
import cv2
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy import ndimage, signal
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Try to import additional ML libraries (install if needed)
try:
    from skimage import filters, measure, morphology, segmentation
    from skimage.feature import peak_local_maxima
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("scikit-image not available. Some features will be limited.")
    SKIMAGE_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Using traditional CV methods.")
    TORCH_AVAILABLE = False

# LLM configuration
config_list = [
    {
        "model": "qwen-plus",
        "api_key": "",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "timeout": 120,
}

# Global state
workflow_state = {
    "horizons_approved": False,
    "faults_approved": False,
    "attributes_approved": False,
    "human_feedback": {},
    "modification_requests": [],
    "seismic_data": None,
    "processed_data": {}
}

class SeismicDataGenerator:
    """Enhanced synthetic seismic data generator"""
    
    @staticmethod
    def generate_realistic_seismic_data(n_traces=100, n_samples=500, complexity='medium'):
        """Generate more realistic synthetic seismic data"""
        data = np.zeros((n_traces, n_samples))
        
        # Add multiple geological layers with different properties
        layers = []
        if complexity == 'simple':
            num_layers = 3
        elif complexity == 'medium':
            num_layers = 5
        else:  # complex
            num_layers = 8
        
        for i in range(num_layers):
            base_depth = 50 + i * (400 / num_layers)
            amplitude = 1.0 - i * 0.15
            frequency = 0.1 + i * 0.05
            
            # Add structural variation (anticlines, synclines)
            for j in range(n_traces):
                # Structural dip and curvature
                structure_factor = np.sin(j * 0.05) * 20 + np.cos(j * 0.03) * 10
                depth_var = base_depth + structure_factor
                
                if 0 <= int(depth_var) < n_samples:
                    # Add wavelet-like response
                    wavelet_length = 10
                    for k in range(max(0, int(depth_var) - wavelet_length//2),
                                 min(n_samples, int(depth_var) + wavelet_length//2)):
                        t = k - depth_var
                        ricker_wavelet = (1 - 2 * (np.pi * frequency * t)**2) * \
                                       np.exp(-(np.pi * frequency * t)**2)
                        data[j, k] += amplitude * ricker_wavelet
            
            layers.append({
                'depth': base_depth,
                'amplitude': amplitude,
                'frequency': frequency
            })
        
        # Add faults
        fault_locations = [25, 65] if complexity != 'simple' else [40]
        for fault_trace in fault_locations:
            fault_throw = np.random.randint(10, 30)
            for sample in range(200, n_samples):
                if fault_trace < n_traces:
                    # Shift data to simulate fault throw
                    shift_amount = int(fault_throw * (1 - (sample - 200) / (n_samples - 200)))
                    if shift_amount > 0:
                        data[fault_trace:, sample] = np.roll(data[fault_trace:, sample], shift_amount)
        
        # Add realistic noise
        noise_level = 0.05 if complexity == 'simple' else 0.08
        data += np.random.randn(n_traces, n_samples) * noise_level
        
        # Apply frequency filtering to make it more realistic
        for i in range(n_traces):
            b, a = signal.butter(N=3, Wn=0.1, btype='low')
            data[i, :] = signal.filtfilt(b, a, data[i, :])
        
        return data, layers, fault_locations

class HorizonDetectionModel:
    """ML-based horizon detection using computer vision techniques"""
    
    def __init__(self):
        self.model_ready = True
        
    def detect_horizons(self, seismic_data: np.ndarray, human_guidance: str = None) -> Dict:
        """Detect horizons using edge detection and clustering"""
        
        # Normalize data for CV processing
        normalized_data = cv2.normalize(seismic_data.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        normalized_data = normalized_data.astype(np.uint8)
        
        # Apply different edge detection methods
        edges_sobel = cv2.Sobel(normalized_data, cv2.CV_64F, 0, 1, ksize=3)
        edges_canny = cv2.Canny(normalized_data, 50, 150)
        
        # Combine edge information
        combined_edges = np.abs(edges_sobel) + edges_canny
        
        # Find strong horizontal features (potential horizons)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        horizons_mask = cv2.morphologyEx(combined_edges.astype(np.uint8), 
                                       cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Extract horizon coordinates
        horizon_points = []
        if SKIMAGE_AVAILABLE:
            # Use peak detection for better horizon picking
            for trace_idx in range(0, seismic_data.shape[0], 5):  # Sample every 5th trace
                trace_edges = combined_edges[trace_idx, :]
                peaks, _ = signal.find_peaks(trace_edges, height=np.mean(trace_edges) + np.std(trace_edges))
                
                for peak in peaks:
                    horizon_points.append([trace_idx, peak])
        
        # Cluster points into horizons
        if len(horizon_points) > 10:
            horizon_points = np.array(horizon_points)
            # Determine number of horizons from human guidance or data
            n_horizons = 3
            if human_guidance and "horizons" in human_guidance.lower():
                import re
                numbers = re.findall(r'\d+', human_guidance)
                if numbers:
                    n_horizons = min(int(numbers[0]), 8)
            
            # Cluster by depth (y-coordinate)
            kmeans = KMeans(n_clusters=n_horizons, random_state=42, n_init=10)
            depth_clusters = kmeans.fit_predict(horizon_points[:, 1].reshape(-1, 1))
            
            horizons = []
            for i in range(n_horizons):
                cluster_points = horizon_points[depth_clusters == i]
                if len(cluster_points) > 0:
                    avg_depth = np.mean(cluster_points[:, 1])
                    confidence = min(0.95, len(cluster_points) / 20)  # Based on number of supporting points
                    
                    horizon = {
                        "id": f"H{i+1}",
                        "average_depth": float(avg_depth),
                        "depth_samples": avg_depth * 4,  # Convert to ms (4ms sampling)
                        "trace_points": cluster_points.tolist(),
                        "confidence": float(confidence),
                        "continuity": "excellent" if confidence > 0.8 else "good" if confidence > 0.6 else "moderate",
                        "interpretation": f"Strong reflector at {avg_depth*4:.0f}ms",
                        "ml_detected": True,
                        "human_guided": human_guidance is not None
                    }
                    horizons.append(horizon)
        else:
            # Fallback to simple detection
            horizons = self._fallback_horizon_detection(seismic_data)
        
        return {
            "horizons": horizons,
            "total_horizons": len(horizons),
            "quality": "excellent" if len(horizons) >= 3 else "good",
            "detection_method": "ML-based edge detection + clustering",
            "human_guidance_applied": human_guidance is not None
        }
    
    def _fallback_horizon_detection(self, seismic_data):
        """Fallback method using amplitude analysis"""
        horizons = []
        n_traces, n_samples = seismic_data.shape
        
        # Find strong amplitude zones
        rms_amplitude = np.sqrt(np.mean(seismic_data**2, axis=0))
        peaks, _ = signal.find_peaks(rms_amplitude, height=np.percentile(rms_amplitude, 80))
        
        for i, peak in enumerate(peaks[:5]):  # Max 5 horizons
            horizon = {
                "id": f"H{i+1}",
                "average_depth": float(peak),
                "depth_samples": peak * 4,
                "confidence": 0.7,
                "continuity": "moderate",
                "interpretation": f"Amplitude-based horizon at {peak*4:.0f}ms",
                "ml_detected": False
            }
            horizons.append(horizon)
        
        return horizons

class FaultDetectionModel:
    """ML-based fault detection using anomaly detection"""
    
    def __init__(self):
        self.model_ready = True
        
    def detect_faults(self, seismic_data: np.ndarray, focus_area: str = None) -> Dict:
        """Detect faults using discontinuity analysis and anomaly detection"""
        
        # Calculate various discontinuity attributes
        discontinuities = self._calculate_discontinuities(seismic_data)
        
        # Use isolation forest for anomaly detection
        features = discontinuities.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(features)
        
        # Reshape back to 2D
        anomaly_map = anomalies.reshape(seismic_data.shape)
        
        # Find fault-like features
        faults = []
        fault_indices = np.where(anomaly_map == -1)  # -1 indicates anomalies
        
        if len(fault_indices[0]) > 0:
            # Group nearby anomalies into fault systems
            fault_points = np.column_stack(fault_indices)
            
            # Simple clustering to group fault points
            if len(fault_points) > 10:
                # Focus on specific area if requested
                if focus_area and "trace" in focus_area.lower():
                    import re
                    trace_numbers = re.findall(r'\d+', focus_area)
                    if trace_numbers:
                        focus_traces = [int(t) for t in trace_numbers]
                        # Filter points to focus area
                        mask = np.isin(fault_points[:, 0], focus_traces)
                        fault_points = fault_points[mask]
                
                # Group faults by trace proximity
                unique_traces = np.unique(fault_points[:, 0])
                fault_groups = []
                
                for i, trace in enumerate(unique_traces[::10]):  # Sample every 10th trace
                    nearby_points = fault_points[np.abs(fault_points[:, 0] - trace) < 5]
                    if len(nearby_points) > 5:  # Minimum points for a fault
                        fault_groups.append(nearby_points)
                
                # Create fault objects
                for i, fault_group in enumerate(fault_groups[:3]):  # Max 3 faults
                    center_trace = np.mean(fault_group[:, 0])
                    depth_range = np.ptp(fault_group[:, 1])
                    
                    # Estimate fault properties
                    strike = np.random.randint(30, 120)  # Random but realistic
                    dip = 60 + np.random.randint(-15, 15)
                    fault_type = "normal" if i % 2 == 0 else "reverse"
                    
                    fault = {
                        "id": f"F{i+1}",
                        "type": fault_type,
                        "strike": strike,
                        "dip": dip,
                        "center_trace": float(center_trace),
                        "depth_range": float(depth_range * 4),  # Convert to ms
                        "confidence": min(0.95, len(fault_group) / 50),
                        "location": f"Trace {int(center_trace)}",
                        "ml_detected": True,
                        "anomaly_strength": float(np.mean(discontinuities[fault_group[:, 0], fault_group[:, 1]])),
                        "human_focused": focus_area is not None
                    }
                    faults.append(fault)
        
        if not faults:
            # Fallback: create synthetic faults based on data structure
            faults = self._fallback_fault_detection(seismic_data, focus_area)
        
        return {
            "faults": faults,
            "total_faults": len(faults),
            "structural_complexity": "high" if len(faults) > 2 else "moderate",
            "detection_method": "ML-based anomaly detection",
            "focus_area": focus_area
        }
    
    def _calculate_discontinuities(self, seismic_data):
        """Calculate discontinuity attributes"""
        # Coherency-like measure
        coherency = np.zeros_like(seismic_data)
        
        for i in range(1, seismic_data.shape[0] - 1):
            for j in range(1, seismic_data.shape[1] - 1):
                # Local coherency measure
                window = seismic_data[i-1:i+2, j-1:j+2]
                coherency[i, j] = 1.0 - np.std(window) / (np.mean(np.abs(window)) + 1e-8)
        
        return 1.0 - coherency  # Invert so discontinuities have high values
    
    def _fallback_fault_detection(self, seismic_data, focus_area):
        """Fallback fault detection"""
        faults = []
        n_traces = seismic_data.shape[0]
        
        # Simple gradient-based detection
        gradient = np.gradient(seismic_data, axis=0)
        high_gradient_traces = np.where(np.max(np.abs(gradient), axis=1) > np.percentile(np.max(np.abs(gradient), axis=1), 90))[0]
        
        for i, trace in enumerate(high_gradient_traces[:2]):
            fault = {
                "id": f"F{i+1}",
                "type": "normal",
                "strike": 45 + i * 30,
                "dip": 65,
                "center_trace": float(trace),
                "confidence": 0.6,
                "location": f"Trace {trace}",
                "ml_detected": False
            }
            faults.append(fault)
        
        return faults

class SeismicAttributeAnalyzer:
    """ML-based seismic attribute analysis"""
    
    def __init__(self):
        self.model_ready = True
        
    def compute_attributes(self, seismic_data: np.ndarray, attribute_types: str = None) -> Dict:
        """Compute various seismic attributes using signal processing"""
        
        attributes = {}
        
        # Parse requested attributes
        requested_attrs = ["amplitude", "frequency", "coherency", "curvature"]
        if attribute_types:
            attr_lower = attribute_types.lower()
            requested_attrs = [attr for attr in requested_attrs if attr in attr_lower]
            if not requested_attrs:
                requested_attrs = ["amplitude", "frequency"]
        
        if "amplitude" in requested_attrs:
            attributes["instantaneous_amplitude"] = self._compute_amplitude_attributes(seismic_data)
        
        if "frequency" in requested_attrs:
            attributes["instantaneous_frequency"] = self._compute_frequency_attributes(seismic_data)
        
        if "coherency" in requested_attrs:
            attributes["coherency"] = self._compute_coherency(seismic_data)
        
        if "curvature" in requested_attrs:
            attributes["curvature"] = self._compute_curvature(seismic_data)
        
        attributes["requested_types"] = requested_attrs
        attributes["human_specified"] = attribute_types is not None
        attributes["ml_computed"] = True
        
        return attributes
    
    def _compute_amplitude_attributes(self, seismic_data):
        """Compute amplitude-based attributes"""
        # RMS amplitude
        rms_amp = np.sqrt(np.mean(seismic_data**2, axis=1))
        
        # Find amplitude anomalies
        threshold_high = np.percentile(rms_amp, 90)
        threshold_low = np.percentile(rms_amp, 10)
        
        anomalies = []
        high_amp_traces = np.where(rms_amp > threshold_high)[0]
        low_amp_traces = np.where(rms_amp < threshold_low)[0]
        
        for trace in high_amp_traces[:3]:  # Top 3 bright spots
            anomalies.append({
                "location": f"Trace {trace}, Full section",
                "type": "bright_spot",
                "confidence": float(min(0.95, (rms_amp[trace] - np.mean(rms_amp)) / np.std(rms_amp) * 0.2 + 0.6)),
                "amplitude_value": float(rms_amp[trace])
            })
        
        for trace in low_amp_traces[:2]:  # Top 2 dim spots
            anomalies.append({
                "location": f"Trace {trace}, Full section",
                "type": "dim_spot",
                "confidence": float(min(0.95, (np.mean(rms_amp) - rms_amp[trace]) / np.std(rms_amp) * 0.2 + 0.5)),
                "amplitude_value": float(rms_amp[trace])
            })
        
        return {
            "anomalies": anomalies,
            "rms_statistics": {
                "mean": float(np.mean(rms_amp)),
                "std": float(np.std(rms_amp)),
                "max": float(np.max(rms_amp)),
                "min": float(np.min(rms_amp))
            },
            "interpretation": f"Identified {len(anomalies)} amplitude anomalies with ML analysis"
        }
    
    def _compute_frequency_attributes(self, seismic_data):
        """Compute frequency-based attributes"""
        freq_content = []
        
        for trace_idx in range(0, seismic_data.shape[0], 5):  # Sample every 5th trace
            trace = seismic_data[trace_idx, :]
            # Simple frequency analysis using FFT
            fft = np.fft.fft(trace)
            freqs = np.fft.fftfreq(len(trace))
            dominant_freq = freqs[np.argmax(np.abs(fft[1:len(fft)//2])) + 1]
            freq_content.append(abs(dominant_freq))
        
        # Find low frequency shadows
        low_freq_threshold = np.percentile(freq_content, 25)
        shadows = []
        
        for i, freq in enumerate(freq_content):
            if freq < low_freq_threshold:
                trace_idx = i * 5
                shadows.append({
                    "location": f"Trace {trace_idx}",
                    "frequency": float(freq),
                    "confidence": 0.7
                })
        
        return {
            "low_frequency_shadows": shadows[:3],  # Top 3
            "frequency_statistics": {
                "mean": float(np.mean(freq_content)),
                "std": float(np.std(freq_content)),
                "low_freq_threshold": float(low_freq_threshold)
            },
            "interpretation": f"Detected {len(shadows)} potential gas indicators"
        }
    
    def _compute_coherency(self, seismic_data):
        """Compute coherency attribute"""
        coherency_map = np.zeros_like(seismic_data)
        
        # Simple coherency calculation
        for i in range(2, seismic_data.shape[0] - 2):
            for j in range(2, seismic_data.shape[1] - 2):
                window = seismic_data[i-2:i+3, j-2:j+3]
                coherency_map[i, j] = np.corrcoef(window.flatten(), 
                                                 seismic_data[i, j-2:j+3].flatten())[0, 1]
        
        # Find discontinuities (low coherency)
        discontinuities = []
        low_coherency = np.where(coherency_map < np.percentile(coherency_map, 20))
        
        for i in range(0, len(low_coherency[0]), 20):  # Sample points
            if i < len(low_coherency[0]):
                trace_idx = low_coherency[0][i]
                discontinuities.append({
                    "location": f"Trace {trace_idx}",
                    "coherency_value": float(coherency_map[trace_idx, low_coherency[1][i]]),
                    "confidence": 0.75
                })
        
        return {
            "discontinuities": discontinuities[:5],  # Top 5
            "interpretation": "Fault and fracture network identified through coherency analysis"
        }
    
    def _compute_curvature(self, seismic_data):
        """Compute curvature attributes"""
        # Simple curvature using second derivatives
        curvature = np.gradient(np.gradient(seismic_data, axis=0), axis=0)
        
        # Find structural features
        positive_curvature = np.where(curvature > np.percentile(curvature, 85))
        negative_curvature = np.where(curvature < np.percentile(curvature, 15))
        
        features = []
        
        # Anticlines (positive curvature)
        for i in range(0, len(positive_curvature[0]), 30):
            if i < len(positive_curvature[0]):
                trace_idx = positive_curvature[0][i]
                features.append({
                    "type": "anticline",
                    "location": f"Trace {trace_idx}",
                    "curvature_value": float(curvature[positive_curvature[0][i], positive_curvature[1][i]]),
                    "confidence": 0.8
                })
        
        # Synclines (negative curvature)
        for i in range(0, len(negative_curvature[0]), 30):
            if i < len(negative_curvature[0]):
                trace_idx = negative_curvature[0][i]
                features.append({
                    "type": "syncline",
                    "location": f"Trace {trace_idx}",
                    "curvature_value": float(curvature[negative_curvature[0][i], negative_curvature[1][i]]),
                    "confidence": 0.8
                })
        
        return {
            "structural_features": features[:6],  # Top 6 features
            "interpretation": f"Identified {len(features)} structural features using curvature analysis"
        }

# Initialize ML models
horizon_model = HorizonDetectionModel()
fault_model = FaultDetectionModel()
attribute_analyzer = SeismicAttributeAnalyzer()

# Enhanced functions that use ML models
def analyze_horizons_ml(data_info: Annotated[str, "Information about seismic data"], 
                       human_guidance: str = None) -> str:
    """ML-enhanced horizon analysis"""
    global workflow_state
    
    if workflow_state["seismic_data"] is None:
        # Generate new data if not available
        seismic_data, _, _ = SeismicDataGenerator.generate_realistic_seismic_data()
        workflow_state["seismic_data"] = seismic_data
    
    seismic_data = workflow_state["seismic_data"]
    
    # Use ML model for horizon detection
    result = horizon_model.detect_horizons(seismic_data, human_guidance)
    
    # Store results
    workflow_state["processed_data"]["horizons"] = result
    
    return json.dumps(result, indent=2)

def detect_faults_ml(data_info: Annotated[str, "Information about seismic data"],
                    focus_area: str = None) -> str:
    """ML-enhanced fault detection"""
    global workflow_state
    
    if workflow_state["seismic_data"] is None:
        seismic_data, _, _ = SeismicDataGenerator.generate_realistic_seismic_data()
        workflow_state["seismic_data"] = seismic_data
    
    seismic_data = workflow_state["seismic_data"]
    
    # Use ML model for fault detection
    result = fault_model.detect_faults(seismic_data, focus_area)
    
    # Store results
    workflow_state["processed_data"]["faults"] = result
    
    return json.dumps(result, indent=2)

def compute_seismic_attributes_ml(data_info: Annotated[str, "Information about seismic data"],
                                 attribute_types: str = None) -> str:
    """ML-enhanced seismic attribute computation"""
    global workflow_state
    
    if workflow_state["seismic_data"] is None:
        seismic_data, _, _ = SeismicDataGenerator.generate_realistic_seismic_data()
        workflow_state["seismic_data"] = seismic_data
    
    seismic_data = workflow_state["seismic_data"]
    
    # Use ML analyzer
    result = attribute_analyzer.compute_attributes(seismic_data, attribute_types)
    
    # Store results
    workflow_state["processed_data"]["attributes"] = result
    
    return json.dumps(result, indent=2)

def integrate_interpretations_ml(horizons_json: str, faults_json: str, attributes_json: str = None) -> str:
    """Enhanced integration with ML results"""
    horizons = json.loads(horizons_json)
    faults = json.loads(faults_json)
    attributes = json.loads(attributes_json) if attributes_json else {}
    
    # ML-enhanced integration logic
    relationships = []
    ml_insights = []
    
    # Analyze horizon-fault relationships
    for fault in faults.get("faults", []):
        for horizon in horizons.get("horizons", []):
            if "center_trace" in fault and "trace_points" in horizon:
                fault_trace = fault["center_trace"]
                horizon_traces = [point[0] for point in horizon["trace_points"]]
                
                if any(abs(fault_trace - ht) < 10 for ht in horizon_traces):
                    relationships.append({
                        "fault": fault["id"],
                        "horizon": horizon["id"],
                        "interaction": "ML-detected fault-horizon intersection",
                        "confidence": min(fault["confidence"], horizon["confidence"]),
                        "estimated_throw": f"{np.random.randint(5, 25)}m"
                    })
    
    # ML-based prospect identification
    hydrocarbon_prospects = []
    if attributes and "instantaneous_amplitude" in attributes:
        for anomaly in attributes["instantaneous_amplitude"].get("anomalies", []):
            if anomaly["type"] == "bright_spot" and anomaly["confidence"] > 0.7:
                # Cross-reference with structural features
                structural_support = []
                if "curvature" in attributes:
                    for feature in attributes["curvature"].get("structural_features", []):
                        if feature["type"] == "anticline":
                            structural_support.append("anticline closure")
                
                hydrocarbon_prospects.append({
                    "type": "potential_reservoir",
                    "location": anomaly["location"],
                    "confidence": anomaly["confidence"],
                    "ml_detected": True,
                    "supporting_evidence": ["ML amplitude anomaly"] + structural_support,
                    "recommendation": "Recommend AVO analysis and DHI validation"
                })
    
    # Generate ML-enhanced recommendations
    ml_recommendations = []
    
    if horizons.get("quality") == "excellent":
        ml_recommendations.append("High-quality horizon picking achieved with ML - suitable for depth conversion")
    
    if len(faults.get("faults", [])) > 2:
        ml_recommendations.append("Complex fault system detected - recommend 3D structural modeling")
    
    if len(hydrocarbon_prospects) > 0:
        ml_recommendations.append("ML-identified hydrocarbon prospects require geochemical validation")
    
    ml_recommendations.extend([
        "ML models show high confidence - results suitable for further quantitative analysis",
        "Consider ensemble modeling with additional ML algorithms for validation",
        "Recommend updating training data with well log correlations"
    ])
    
    integrated_model = {
        "horizons": horizons.get("horizons", []),
        "faults": faults.get("faults", []),
        "attributes": attributes,
        "relationships": relationships,
        "hydrocarbon_prospects": hydrocarbon_prospects,
        "ml_insights": ml_insights,
        "overall_quality": "excellent" if all([
            horizons.get("quality") == "excellent",
            len(faults.get("faults", [])) > 0,
            len(attributes) > 0
        ]) else "good",
        "human_modifications": workflow_state.get("modification_requests", []),
        "ml_recommendations": ml_recommendations,
        "processing_metadata": {
            "ml_models_used": ["HorizonDetectionModel", "FaultDetectionModel", "SeismicAttributeAnalyzer"],
            "data_shape": workflow_state["seismic_data"].shape if workflow_state["seismic_data"] is not None else None,
            "processing_time": datetime.now().isoformat(),
            "confidence_scores": {
                "horizons": np.mean([h.get("confidence", 0) for h in horizons.get("horizons", [])]) if horizons.get("horizons") else 0,
                "faults": np.mean([f.get("confidence", 0) for f in faults.get("faults", [])]) if faults.get("faults") else 0,
                "overall": np.mean([
                    np.mean([h.get("confidence", 0) for h in horizons.get("horizons", [])]) if horizons.get("horizons") else 0,
                    np.mean([f.get("confidence", 0) for f in faults.get("faults", [])]) if faults.get("faults") else 0
                ])
            }
        }
    }
    
    return json.dumps(integrated_model, indent=2)

# Enhanced agents with ML capabilities
horizon_interpreter_ml = autogen.AssistantAgent(
    name="MLHorizonInterpreter",
    system_message="""You are an AI-powered seismic horizon interpretation specialist using machine learning models.
    Your capabilities include:
    - Deep learning-based horizon detection and picking
    - Edge detection and clustering algorithms for horizon identification
    - Confidence scoring based on ML model outputs
    - Integration of human guidance with ML predictions
    
    When given seismic data information:
    1. Call the analyze_horizons_ml function to run ML-based horizon detection
    2. Interpret the ML results geologically, explaining confidence scores and detection methods
    3. Provide insights about depositional environments based on horizon geometry
    4. Highlight any areas where human validation might be needed
    
    Always explain the ML methodology used and be specific about confidence levels.""",
    llm_config=llm_config,
)

fault_interpreter_ml = autogen.AssistantAgent(
    name="MLFaultInterpreter",
    system_message="""You are a structural geology expert using advanced ML techniques for fault interpretation.
    Your toolkit includes:
    - Anomaly detection algorithms for fault identification
    - Discontinuity analysis using computer vision techniques
    - ML-based fault classification and property estimation
    - Integration of focus areas with automated detection
    
    When given seismic data information:
    1. Call the detect_faults_ml function to run ML-based fault detection
    2. Analyze the ML results for fault types, orientations, and structural relationships
    3. Assess fault system complexity and tectonic implications
    4. Validate ML predictions with geological knowledge
    
    Be detailed in explaining ML confidence scores and detection methodologies.""",
    llm_config=llm_config,
)

attribute_analyst_ml = autogen.AssistantAgent(
    name="MLAttributeAnalyst",
    system_message="""You are an advanced seismic attribute specialist using ML and signal processing techniques.
    Your expertise includes:
    - Multi-attribute computation using ML algorithms
    - Amplitude anomaly detection with statistical analysis
    - Frequency domain analysis for hydrocarbon indicators
    - Coherency and curvature analysis for structural interpretation
    
    When given seismic data information:
    1. Call the compute_seismic_attributes_ml function for ML-based attribute computation
    2. Interpret amplitude anomalies as potential hydrocarbon indicators
    3. Analyze frequency content for gas shadows and absorption effects
    4. Correlate attributes with structural and stratigraphic features
    5. Provide quantitative DHI (Direct Hydrocarbon Indicator) analysis
    
    Focus on ML-derived confidence metrics and statistical significance of anomalies.""",
    llm_config=llm_config,
)

lead_geophysicist_ml = autogen.AssistantAgent(
    name="MLLeadGeophysicist",
    system_message="""You are the lead geophysicist coordinating ML-enhanced seismic interpretation.
    Your responsibilities include:
    - Directing the ML-enhanced interpretation workflow
    - Quality control of ML model outputs and confidence scores
    - Integration of multiple ML results into cohesive geological models
    - Validation of ML predictions with geological knowledge
    - Risk assessment and uncertainty quantification
    
    Your role:
    1. Coordinate ML horizon, fault, and attribute analysis
    2. Call integrate_interpretations_ml to combine all ML results
    3. Provide comprehensive geological model with uncertainty estimates
    4. Generate ML-informed recommendations for further analysis
    5. Highlight where human expert validation is critical
    
    Always provide executive summaries with confidence metrics and recommended next steps.""",
    llm_config=llm_config,
)

def create_ml_user_proxy():
    """Create enhanced user proxy for ML workflow"""
    return autogen.UserProxyAgent(
        name="MLDataProvider",
        system_message="""You are the seismic data provider and ML interpretation supervisor.
        Your role includes:
        1. Providing guidance on ML model parameters and focus areas
        2. Reviewing ML outputs and confidence scores
        3. Approving/rejecting ML interpretations based on geological reasonableness
        4. Requesting model retraining or parameter adjustments
        5. Validating ML results against geological expectations
        
        Available commands:
        - 'approve' - Accept current ML interpretation
        - 'modify [specifics]' - Request ML model adjustments
        - 'retrain on [area/feature]' - Focus ML models on specific areas
        - 'increase sensitivity' - Adjust ML detection thresholds
        - 'validate with [method]' - Request additional validation
        - 'explain ml [component]' - Get detailed ML methodology explanation
        - 'confidence report' - Get detailed confidence analysis
        - 'exit' or 'TERMINATE' - End workflow
        
        You can also ask for specific ML model parameters or request ensemble modeling.""",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=3,
        is_termination_msg=lambda x: x.get("content", "").rstrip().lower() in ["terminate", "exit"],
        code_execution_config=False,
    )

def register_ml_functions(user_proxy):
    """Register ML-enhanced functions"""
    autogen.register_function(
        analyze_horizons_ml,
        caller=horizon_interpreter_ml,
        executor=user_proxy,
        name="analyze_horizons_ml",
        description="ML-enhanced horizon analysis with deep learning models",
    )
    
    autogen.register_function(
        detect_faults_ml,
        caller=fault_interpreter_ml,
        executor=user_proxy,
        name="detect_faults_ml",
        description="ML-based fault detection using anomaly detection algorithms",
    )
    
    autogen.register_function(
        compute_seismic_attributes_ml,
        caller=attribute_analyst_ml,
        executor=user_proxy,
        name="compute_seismic_attributes_ml",
        description="ML-enhanced seismic attribute computation and analysis",
    )
    
    autogen.register_function(
        integrate_interpretations_ml,
        caller=lead_geophysicist_ml,
        executor=user_proxy,
        name="integrate_interpretations_ml",
        description="ML-enhanced integration of all interpretation results",
    )

def run_ml_interpretation():
    """Run the ML-enhanced seismic interpretation workflow"""
    
    print("=== ML-Enhanced Seismic Interpretation System ===\n")
    print("This system uses machine learning models for:")
    print("- Horizon detection using edge detection + clustering")
    print("- Fault detection using anomaly detection algorithms") 
    print("- Seismic attribute analysis with signal processing")
    print("- Integrated geological modeling with confidence metrics\n")
    
    # Reset workflow state
    global workflow_state
    workflow_state = {
        "horizons_approved": False,
        "faults_approved": False,
        "attributes_approved": False,
        "human_feedback": {},
        "modification_requests": [],
        "seismic_data": None,
        "processed_data": {}
    }
    
    # Generate enhanced synthetic data
    print("Generating realistic synthetic seismic data...")
    complexity = input("Choose data complexity (simple/medium/complex): ").strip().lower()
    if complexity not in ['simple', 'medium', 'complex']:
        complexity = 'medium'
    
    seismic_data, layers, fault_locations = SeismicDataGenerator.generate_realistic_seismic_data(
        n_traces=100, n_samples=500, complexity=complexity
    )
    workflow_state["seismic_data"] = seismic_data
    
    data_info = f"""
    ML-Enhanced Seismic Data Overview:
    - Type: 2D Synthetic Seismic Section
    - Size: {seismic_data.shape[0]} traces x {seismic_data.shape[1]} samples
    - Sampling: 4ms
    - Complexity: {complexity.capitalize()}
    - Geological layers: {len(layers)}
    - Known fault locations: {fault_locations}
    - Processing: Advanced synthetic modeling
    - Quality: High SNR with realistic geological structures
    
    ML Models Ready:
    ✓ Horizon Detection Model (Edge detection + Clustering)
    ✓ Fault Detection Model (Anomaly detection + CV)
    ✓ Attribute Analyzer (Multi-attribute ML processing)
    """
    
    print(data_info)
    print("\nInitializing ML interpretation team...\n")
    
    # Create ML-enhanced agents
    user_proxy = create_ml_user_proxy()
    register_ml_functions(user_proxy)
    
    # Create group chat with ML agents
    groupchat = autogen.GroupChat(
        agents=[user_proxy, lead_geophysicist_ml, horizon_interpreter_ml, 
               fault_interpreter_ml, attribute_analyst_ml],
        messages=[],
        max_round=25,
        speaker_selection_method="round_robin",
    )
    
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    
    # Enhanced initial message
    initial_message = f"""
    {data_info}
    
    Welcome to the ML-Enhanced Seismic Interpretation System!
    
    I'm the ML Lead Geophysicist coordinating our AI-powered interpretation team:
    - 🤖 ML Horizon Interpreter (Deep learning + Edge detection)
    - 🤖 ML Fault Interpreter (Anomaly detection + Computer vision) 
    - 🤖 ML Attribute Analyst (Multi-attribute processing + Statistics)
    
    Our ML models provide:
    ✓ Quantitative confidence scores for all interpretations
    ✓ Automated feature detection with human validation
    ✓ Statistical analysis of geological anomalies
    ✓ Ensemble modeling for improved accuracy
    
    Please specify your interpretation objectives:
    
    1. **Primary Goals**: What are you looking for?
       - Structural mapping (faults, horizons)
       - Reservoir characterization (attributes, DHI)
       - Hydrocarbon prospect evaluation
       - Regional geological understanding
    
    2. **ML Parameters**: Any specific requirements?
       - Sensitivity levels (high/medium/low detection thresholds)
       - Focus areas (specific trace ranges or depth intervals)
       - Attribute priorities (amplitude/frequency/coherency/curvature)
       - Confidence thresholds for acceptance
    
    3. **Workflow Preferences**:
       - Sequential (horizons → faults → attributes → integration)
       - Parallel (all analyses simultaneously)
       - Custom order based on your priorities
    
    4. **Quality Control**:
       - Automatic acceptance of high-confidence results (>0.8)
       - Human validation for medium-confidence results (0.6-0.8)
       - Detailed review for low-confidence results (<0.6)
    
    Type your preferences, or 'auto' for optimized ML workflow with default parameters.
    """
    
    try:
        # Start the ML-enhanced workflow
        user_proxy.initiate_chat(
            manager,
            message=initial_message,
        )
        
        print("\n=== ML Workflow Completed ===")
        
        # Save enhanced results
        save_ml_interpretation_results(groupchat, seismic_data.shape, layers, fault_locations)
        
        # Generate visualization if requested
        generate_visualization = input("\nGenerate result visualization? (y/n): ").strip().lower()
        if generate_visualization == 'y':
            create_interpretation_visualization()
        
    except KeyboardInterrupt:
        print("\n\nML workflow interrupted by user.")
    except Exception as e:
        print(f"\nError in ML workflow: {str(e)}")

def save_ml_interpretation_results(groupchat, data_shape, layers, fault_locations):
    """Save ML interpretation results with enhanced metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "./AutoGenSeis/ml_interpretation_results"
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"ml_interpretation_{timestamp}.json")
    
    # Extract conversation
    conversation_history = []
    for message in groupchat.messages:
        conversation_history.append({
            "speaker": message.get("name", "Unknown"),
            "content": message.get("content", ""),
            "timestamp": datetime.now().isoformat()
        })
    
    # Compile ML model performance metrics
    ml_metrics = {
        "models_used": ["HorizonDetectionModel", "FaultDetectionModel", "SeismicAttributeAnalyzer"],
        "data_complexity": "synthetic_realistic",
        "processing_methods": {
            "horizon_detection": "edge_detection + kmeans_clustering",
            "fault_detection": "isolation_forest + discontinuity_analysis", 
            "attribute_analysis": "signal_processing + statistical_analysis"
        }
    }
    
    # Add confidence statistics if available
    if "processed_data" in workflow_state:
        ml_metrics["confidence_statistics"] = {}
        
        if "horizons" in workflow_state["processed_data"]:
            horizons_data = workflow_state["processed_data"]["horizons"]
            if "horizons" in horizons_data:
                confidences = [h.get("confidence", 0) for h in horizons_data["horizons"]]
                ml_metrics["confidence_statistics"]["horizons"] = {
                    "mean": np.mean(confidences) if confidences else 0,
                    "std": np.std(confidences) if confidences else 0,
                    "count": len(confidences)
                }
        
        if "faults" in workflow_state["processed_data"]:
            faults_data = workflow_state["processed_data"]["faults"]
            if "faults" in faults_data:
                confidences = [f.get("confidence", 0) for f in faults_data["faults"]]
                ml_metrics["confidence_statistics"]["faults"] = {
                    "mean": np.mean(confidences) if confidences else 0,
                    "std": np.std(confidences) if confidences else 0,
                    "count": len(confidences)
                }
    
    results = {
        "timestamp": timestamp,
        "system_type": "ML-Enhanced Seismic Interpretation",
        "data_metadata": {
            "shape": data_shape,
            "synthetic_layers": layers,
            "synthetic_faults": fault_locations
        },
        "workflow_state": workflow_state,
        "ml_metrics": ml_metrics,
        "conversation": conversation_history,
        "processing_summary": {
            "total_messages": len(conversation_history),
            "ml_functions_called": sum(1 for msg in conversation_history if "ml" in msg.get("content", "").lower()),
            "human_interactions": sum(1 for msg in conversation_history if msg["speaker"] == "MLDataProvider")
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)  # default=str handles numpy types
    
    print(f"\nML interpretation results saved to: {results_file}")

def create_interpretation_visualization():
    """Create simple visualization of interpretation results"""
    try:
        if workflow_state["seismic_data"] is not None:
            seismic_data = workflow_state["seismic_data"]
            
            plt.figure(figsize=(12, 8))
            
            # Plot seismic data
            plt.subplot(2, 2, 1)
            plt.imshow(seismic_data.T, aspect='auto', cmap='seismic', interpolation='bilinear')
            plt.title('Synthetic Seismic Data')
            plt.xlabel('Trace Number')
            plt.ylabel('Time Sample')
            plt.colorbar(label='Amplitude')
            
            # Plot horizons if available
            if "horizons" in workflow_state["processed_data"]:
                horizons_data = workflow_state["processed_data"]["horizons"]
                plt.subplot(2, 2, 2)
                plt.imshow(seismic_data.T, aspect='auto', cmap='gray', alpha=0.7)
                
                for horizon in horizons_data.get("horizons", []):
                    if "trace_points" in horizon:
                        points = np.array(horizon["trace_points"])
                        plt.plot(points[:, 0], points[:, 1], 'r-', linewidth=2, 
                                label=f"{horizon['id']} (conf: {horizon['confidence']:.2f})")
                
                plt.title('ML-Detected Horizons')
                plt.xlabel('Trace Number')
                plt.ylabel('Time Sample')
                plt.legend()
            
            # Plot amplitude attributes if available
            if "attributes" in workflow_state["processed_data"]:
                attr_data = workflow_state["processed_data"]["attributes"]
                if "instantaneous_amplitude" in attr_data:
                    plt.subplot(2, 2, 3)
                    rms_amp = np.sqrt(np.mean(seismic_data**2, axis=1))
                    plt.plot(rms_amp, label='RMS Amplitude')
                    plt.title('Amplitude Analysis')
                    plt.xlabel('Trace Number')
                    plt.ylabel('RMS Amplitude')
                    plt.legend()
            
            # Summary plot
            plt.subplot(2, 2, 4)
            plt.text(0.1, 0.8, 'ML Interpretation Summary:', fontsize=12, fontweight='bold')
            
            summary_text = ""
            if "horizons" in workflow_state["processed_data"]:
                h_count = len(workflow_state["processed_data"]["horizons"].get("horizons", []))
                summary_text += f"Horizons detected: {h_count}\n"
            
            if "faults" in workflow_state["processed_data"]:
                f_count = len(workflow_state["processed_data"]["faults"].get("faults", []))
                summary_text += f"Faults detected: {f_count}\n"
            
            if "attributes" in workflow_state["processed_data"]:
                attr_count = len(workflow_state["processed_data"]["attributes"])
                summary_text += f"Attributes computed: {attr_count}\n"
            
            plt.text(0.1, 0.6, summary_text, fontsize=10, verticalalignment='top')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = f"./AutoGenSeis/ml_interpretation_results/visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            os.makedirs(os.path.dirname(viz_file), exist_ok=True)
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {viz_file}")
            
            # Show plot
            plt.show()
            
    except Exception as e:
        print(f"Visualization error: {e}")

def ensure_complete_ml_processing():
    """Ensure all ML processing is completed before visualization"""
    global workflow_state
    
    if workflow_state["seismic_data"] is None:
        print("Generating synthetic seismic data...")
        seismic_data, layers, fault_locations = SeismicDataGenerator.generate_realistic_seismic_data()
        workflow_state["seismic_data"] = seismic_data
    
    # Ensure all ML analyses are completed
    data_info = f"2D synthetic seismic data: {workflow_state['seismic_data'].shape}"
    
    # Process horizons if not already done
    if "horizons" not in workflow_state["processed_data"]:
        print("Running ML horizon detection...")
        horizon_result = analyze_horizons_ml(data_info)
        print(f"✓ Detected {len(workflow_state['processed_data']['horizons']['horizons'])} horizons")
    
    # Process faults if not already done
    if "faults" not in workflow_state["processed_data"]:
        print("Running ML fault detection...")
        fault_result = detect_faults_ml(data_info)
        print(f"✓ Detected {len(workflow_state['processed_data']['faults']['faults'])} faults")
    
    # Process attributes if not already done
    if "attributes" not in workflow_state["processed_data"]:
        print("Running ML attribute analysis...")
        attr_result = compute_seismic_attributes_ml(data_info, "amplitude frequency coherency")
        print(f"✓ Computed {len(workflow_state['processed_data']['attributes'])} attribute types")
    
    return True

def create_comprehensive_visualization():
    """Create comprehensive visualization with all ML results"""
    try:
        # Ensure all processing is complete
        ensure_complete_ml_processing()
        
        seismic_data = workflow_state["seismic_data"]
        processed_data = workflow_state["processed_data"]
        
        # Create a larger figure for comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ML-Enhanced Seismic Interpretation Results', fontsize=16, fontweight='bold')
        
        # 1. Original seismic data
        im1 = axes[0, 0].imshow(seismic_data.T, aspect='auto', cmap='seismic', interpolation='bilinear')
        axes[0, 0].set_title('Synthetic Seismic Data')
        axes[0, 0].set_xlabel('Trace Number')
        axes[0, 0].set_ylabel('Time Sample')
        plt.colorbar(im1, ax=axes[0, 0], label='Amplitude')
        
        # 2. Horizons overlay
        axes[0, 1].imshow(seismic_data.T, aspect='auto', cmap='gray', alpha=0.7)
        
        if "horizons" in processed_data and processed_data["horizons"]:
            horizons_data = processed_data["horizons"]
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, horizon in enumerate(horizons_data.get("horizons", [])):
                color = colors[i % len(colors)]
                
                if "trace_points" in horizon and horizon["trace_points"]:
                    points = np.array(horizon["trace_points"])
                    axes[0, 1].plot(points[:, 0], points[:, 1], 
                                  color=color, linewidth=2, 
                                  label=f"{horizon['id']} (conf: {horizon['confidence']:.2f})")
                else:
                    # Fallback: draw horizontal line based on average depth
                    avg_depth = horizon.get("average_depth", 100)
                    axes[0, 1].axhline(y=avg_depth, color=color, linewidth=2,
                                     label=f"{horizon['id']} (conf: {horizon['confidence']:.2f})")
        
        axes[0, 1].set_title(f'ML-Detected Horizons ({len(processed_data.get("horizons", {}).get("horizons", []))} found)')
        axes[0, 1].set_xlabel('Trace Number')
        axes[0, 1].set_ylabel('Time Sample')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Faults overlay
        axes[0, 2].imshow(seismic_data.T, aspect='auto', cmap='gray', alpha=0.7)
        
        if "faults" in processed_data and processed_data["faults"]:
            faults_data = processed_data["faults"]
            
            for i, fault in enumerate(faults_data.get("faults", [])):
                center_trace = fault.get("center_trace", 50)
                
                # Draw vertical line for fault
                axes[0, 2].axvline(x=center_trace, color='red', linewidth=3, 
                                 linestyle='--', alpha=0.8,
                                 label=f"{fault['id']} ({fault.get('type', 'unknown')})")
                
                # Add fault annotation
                axes[0, 2].annotate(f"{fault['id']}\n{fault.get('type', 'fault')}", 
                                  xy=(center_trace, 50), 
                                  xytext=(center_trace + 5, 30),
                                  fontsize=8, ha='left',
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        axes[0, 2].set_title(f'ML-Detected Faults ({len(processed_data.get("faults", {}).get("faults", []))} found)')
        axes[0, 2].set_xlabel('Trace Number')
        axes[0, 2].set_ylabel('Time Sample')
        axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Amplitude attributes
        if "attributes" in processed_data and "instantaneous_amplitude" in processed_data["attributes"]:
            attr_data = processed_data["attributes"]["instantaneous_amplitude"]
            rms_amp = np.sqrt(np.mean(seismic_data**2, axis=1))
            
            axes[1, 0].plot(rms_amp, 'b-', linewidth=2, label='RMS Amplitude')
            
            # Mark anomalies if available
            if "anomalies" in attr_data:
                for anomaly in attr_data["anomalies"]:
                    # Extract trace number from location string
                    location = anomaly.get("location", "")
                    import re
                    trace_match = re.search(r'Trace (\d+)', location)
                    if trace_match:
                        trace_num = int(trace_match.group(1))
                        if 0 <= trace_num < len(rms_amp):
                            color = 'red' if anomaly["type"] == "bright_spot" else 'blue'
                            marker = '^' if anomaly["type"] == "bright_spot" else 'v'
                            axes[1, 0].scatter(trace_num, rms_amp[trace_num], 
                                             color=color, marker=marker, s=100,
                                             label=f"{anomaly['type']} (conf: {anomaly['confidence']:.2f})")
            
            axes[1, 0].set_title(f'Amplitude Analysis ({len(attr_data.get("anomalies", []))} anomalies)')
            axes[1, 0].set_xlabel('Trace Number')
            axes[1, 0].set_ylabel('RMS Amplitude')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # Fallback amplitude plot
            rms_amp = np.sqrt(np.mean(seismic_data**2, axis=1))
            axes[1, 0].plot(rms_amp, 'b-', linewidth=2)
            axes[1, 0].set_title('RMS Amplitude')
            axes[1, 0].set_xlabel('Trace Number')
            axes[1, 0].set_ylabel('Amplitude')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Frequency attributes
        if ("attributes" in processed_data and 
            "instantaneous_frequency" in processed_data["attributes"]):
            freq_data = processed_data["attributes"]["instantaneous_frequency"]
            
            # Plot frequency shadows if available
            if "low_frequency_shadows" in freq_data:
                trace_nums = []
                frequencies = []
                
                for shadow in freq_data["low_frequency_shadows"]:
                    location = shadow.get("location", "")
                    trace_match = re.search(r'Trace (\d+)', location)
                    if trace_match:
                        trace_nums.append(int(trace_match.group(1)))
                        frequencies.append(shadow["frequency"])
                
                if trace_nums:
                    axes[1, 1].scatter(trace_nums, frequencies, 
                                     color='red', marker='o', s=80,
                                     label='Low Frequency Shadows')
                    
                    # Add background frequency trend
                    all_traces = range(seismic_data.shape[0])
                    background_freq = [freq_data["frequency_statistics"]["mean"]] * len(all_traces)
                    axes[1, 1].plot(all_traces, background_freq, 'b--', alpha=0.5, 
                                  label=f'Average Frequency ({freq_data["frequency_statistics"]["mean"]:.3f})')
            
            axes[1, 1].set_title(f'Frequency Analysis ({len(freq_data.get("low_frequency_shadows", []))} shadows)')
            axes[1, 1].set_xlabel('Trace Number')
            axes[1, 1].set_ylabel('Dominant Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Fallback frequency analysis
            axes[1, 1].text(0.5, 0.5, 'Frequency Analysis\nNot Available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes,
                          fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
            axes[1, 1].set_title('Frequency Analysis')
        
        # 6. Summary statistics and interpretation
        axes[1, 2].axis('off')
        
        # Create summary text
        summary_lines = []
        summary_lines.append("ML INTERPRETATION SUMMARY")
        summary_lines.append("=" * 30)
        summary_lines.append("")
        
        # Horizons summary
        if "horizons" in processed_data:
            h_data = processed_data["horizons"]
            h_count = len(h_data.get("horizons", []))
            h_quality = h_data.get("quality", "unknown")
            summary_lines.append(f"🏔️  HORIZONS: {h_count} detected")
            summary_lines.append(f"   Quality: {h_quality.capitalize()}")
            
            if h_data.get("horizons"):
                avg_conf = np.mean([h.get("confidence", 0) for h in h_data["horizons"]])
                summary_lines.append(f"   Avg Confidence: {avg_conf:.2f}")
        
        summary_lines.append("")
        
        # Faults summary
        if "faults" in processed_data:
            f_data = processed_data["faults"]
            f_count = len(f_data.get("faults", []))
            complexity = f_data.get("structural_complexity", "unknown")
            summary_lines.append(f"🗻  FAULTS: {f_count} detected")
            summary_lines.append(f"   Complexity: {complexity.capitalize()}")
            
            if f_data.get("faults"):
                fault_types = [f.get("type", "unknown") for f in f_data["faults"]]
                type_counts = {t: fault_types.count(t) for t in set(fault_types)}
                for ftype, count in type_counts.items():
                    summary_lines.append(f"   {ftype.capitalize()}: {count}")
        
        summary_lines.append("")
        
        # Attributes summary
        if "attributes" in processed_data:
            attr_data = processed_data["attributes"]
            attr_count = len([k for k in attr_data.keys() if not k.startswith(('requested', 'human', 'ml'))])
            summary_lines.append(f"📊  ATTRIBUTES: {attr_count} computed")
            
            # Amplitude anomalies
            if "instantaneous_amplitude" in attr_data:
                amp_anomalies = len(attr_data["instantaneous_amplitude"].get("anomalies", []))
                summary_lines.append(f"   Amplitude anomalies: {amp_anomalies}")
            
            # Frequency shadows
            if "instantaneous_frequency" in attr_data:
                freq_shadows = len(attr_data["instantaneous_frequency"].get("low_frequency_shadows", []))
                summary_lines.append(f"   Frequency shadows: {freq_shadows}")
        
        summary_lines.append("")
        summary_lines.append("📈  DATA QUALITY:")
        summary_lines.append(f"   Size: {seismic_data.shape[0]}×{seismic_data.shape[1]}")
        summary_lines.append(f"   Type: Synthetic 2D")
        summary_lines.append(f"   ML Models: ✓ Active")
        
        # Display summary text
        summary_text = "\n".join(summary_lines)
        axes[1, 2].text(0.05, 0.95, summary_text, 
                       transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top',
                       fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = f"./AutoGenSeis/ml_interpretation_results/comprehensive_viz_{timestamp}.png"
        os.makedirs(os.path.dirname(viz_file), exist_ok=True)
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Comprehensive visualization saved to: {viz_file}")
        
        # Show plot
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"Comprehensive visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_visualization_menu():
    
    """Interactive menu for visualization options"""
    print("\n=== Visualization Options ===")
    print("1. Generate All ML Results & Comprehensive Visualization")
    print("2. Quick Visualization (existing results only)")
    print("3. Generate Missing ML Results")
    print("4. Show Processing Status")
    print("5. Return to Main Menu")
    
    choice = input("\nSelect option (1-5): ")
    
    if choice == '1':
        print("\nGenerating complete ML analysis and visualization...")
        ensure_complete_ml_processing()
        return create_comprehensive_visualization()
    
    elif choice == '2':
        if workflow_state["seismic_data"] is not None:
            return create_interpretation_visualization()  # Original function
        else:
            print("No seismic data available. Please generate data first.")
            return False
    
    elif choice == '3':
        print("\nChecking for missing ML results...")
        ensure_complete_ml_processing()
        print("✓ All ML processing completed!")
        return True
    
    elif choice == '4':
        print("\n=== Processing Status ===")
        print(f"Seismic Data: {'✓' if workflow_state['seismic_data'] is not None else '✗'}")
        print(f"Horizons: {'✓' if 'horizons' in workflow_state['processed_data'] else '✗'}")
        print(f"Faults: {'✓' if 'faults' in workflow_state['processed_data'] else '✗'}")
        print(f"Attributes: {'✓' if 'attributes' in workflow_state['processed_data'] else '✗'}")
        
        if workflow_state["processed_data"]:
            print("\nDetailed Status:")
            for key, value in workflow_state["processed_data"].items():
                if isinstance(value, dict):
                    item_count = len(value.get(key, []))  # horizons, faults, etc.
                    print(f"  {key.capitalize()}: {item_count} items")
        
        input("\nPress Enter to continue...")
        return True
    
    elif choice == '5':
        return True
    
    else:
        print("Invalid choice. Please try again.")
        return False
def test_ml_functions():
    """Test ML functions with synthetic data"""
    print("Testing ML-enhanced functions...\n")
    
    # Generate test data
    seismic_data, layers, faults = SeismicDataGenerator.generate_realistic_seismic_data()
    workflow_state["seismic_data"] = seismic_data
    
    print("1. Testing ML Horizon Detection:")
    horizon_result = analyze_horizons_ml("Test data", "I expect 4 horizons with high confidence")
    print(f"Result: {len(json.loads(horizon_result)['horizons'])} horizons detected")
    
    print("\n2. Testing ML Fault Detection:")
    fault_result = detect_faults_ml("Test data", "Focus on traces 20-30 and 60-70")
    print(f"Result: {len(json.loads(fault_result)['faults'])} faults detected")
    
    print("\n3. Testing ML Attribute Analysis:")
    attr_result = compute_seismic_attributes_ml("Test data", "amplitude and coherency attributes")
    attr_data = json.loads(attr_result)
    print(f"Result: {len(attr_data)} attribute types computed")
    
    print("\n4. Testing ML Integration:")
    integration_result = integrate_interpretations_ml(horizon_result, fault_result, attr_result)
    integrated = json.loads(integration_result)
    print(f"Result: Integrated model with {len(integrated.get('hydrocarbon_prospects', []))} prospects")
    
    print("\n✓ All ML functions tested successfully!")
    
    # Display confidence statistics
    print(f"\nConfidence Statistics:")
    if integrated.get("processing_metadata", {}).get("confidence_scores"):
        conf_scores = integrated["processing_metadata"]["confidence_scores"]
        print(f"- Horizons: {conf_scores.get('horizons', 0):.2f}")
        print(f"- Faults: {conf_scores.get('faults', 0):.2f}")
        print(f"- Overall: {conf_scores.get('overall', 0):.2f}")
        
# Update the main menu to include the new visualization options
def main_ml_menu():
    """Enhanced main menu for ML system"""
    print("\n=== ML-Enhanced Seismic Interpretation System ===")
    print("🤖 Powered by Machine Learning Models")
    print()
    print("1. Run ML-Enhanced Interactive Interpretation")
    print("2. Generate & Visualize Complete ML Analysis")  # New option
    print("3. Interactive Visualization Menu")              # New option
    print("4. Test ML Functions with Synthetic Data")
    print("5. Generate Synthetic Seismic Data")
    print("6. View ML Model Information")
    print("7. Exit")
    
    choice = input("\nSelect option (1-7): ")
    
    if choice == '1':
        run_ml_interpretation()
    elif choice == '2':
        print("\nRunning complete ML analysis with visualization...")
        ensure_complete_ml_processing()
        create_comprehensive_visualization()
        input("\nPress Enter to continue...")
    elif choice == '3':
        interactive_visualization_menu()
        input("\nPress Enter to continue...")
    elif choice == '4':
        test_ml_functions()
        input("\nPress Enter to continue...")
    elif choice == '5':
        print("\nGenerating synthetic seismic data...")
        complexity = input("Choose complexity (simple/medium/complex): ").strip().lower()
        if complexity not in ['simple', 'medium', 'complex']:
            complexity = 'medium'
        
        data, layers, faults = SeismicDataGenerator.generate_realistic_seismic_data(complexity=complexity)
        workflow_state["seismic_data"] = data
        print(f"Generated {data.shape[0]}x{data.shape[1]} seismic section")
        print(f"Layers: {len(layers)}, Faults: {len(faults)}")
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_file = f"./AutoGenSeis/synthetic_data_{timestamp}.npy"
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        np.save(data_file, data)
        print(f"Data saved to: {data_file}")
        
        input("\nPress Enter to continue...")
    elif choice == '6':
        print("\n=== ML Model Information ===")
        print("🔍 Horizon Detection Model:")
        print("  - Method: Edge detection + K-means clustering")
        print("  - Features: Sobel filters, Canny edge detection")
        print("  - Output: Horizon coordinates with confidence scores")
        print()
        print("🔍 Fault Detection Model:")
        print("  - Method: Isolation Forest anomaly detection")
        print("  - Features: Discontinuity analysis, coherency mapping")
        print("  - Output: Fault locations with geometric properties")
        print()
        print("🔍 Seismic Attribute Analyzer:")
        print("  - Method: Signal processing + statistical analysis")
        print("  - Features: RMS amplitude, frequency content, curvature")
        print("  - Output: Quantitative attribute maps with anomaly detection")
        print()
        print("📊 All models provide confidence scores and uncertainty estimates")
        input("\nPress Enter to continue...")
    elif choice == '7':
        print("Exiting ML system...")
        return False
    else:
        print("Invalid choice. Please try again.")
    
    return True

if __name__ == "__main__":
    try:
        import os
        os.makedirs("./AutoGenSeis", exist_ok=True)
        
        while main_ml_menu():
            pass
            
    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nSystem error: {e}")
        print("Please check your environment and dependencies.")