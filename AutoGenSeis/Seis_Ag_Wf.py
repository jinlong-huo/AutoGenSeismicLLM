import autogen
from typing import Dict, List, Annotated
import numpy as np
import json
from datetime import datetime

# LLM configuration
config_list = [
    {
        "model": "qwen-plus",
        "api_key": "",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
]

# config_list = [
#     {
#         "model": "deepseek-chat",
#         "api_key": "",
#         "base_url": "https://api.deepseek.com",
#     }
# ]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "timeout": 120,
}

# Global state to track workflow progress and human feedback
workflow_state = {
    "horizons_approved": False,
    "faults_approved": False,
    "attributes_approved": False,
    "human_feedback": {},
    "modification_requests": []
}

# Simple synthetic seismic data generator
def generate_synthetic_seismic_data():
    """Generate simple synthetic 2D seismic data"""
    n_traces = 50
    n_samples = 500
    
    # Create synthetic seismic with some structure
    data = np.zeros((n_traces, n_samples))
    
    # Add three horizons at different depths
    for i in range(3):
        depth = 150 + i * 100
        amplitude = 1.0 - i * 0.2
        for j in range(n_traces):
            # Add some structural variation
            depth_var = depth + 10 * np.sin(j * 0.1)
            if 0 <= int(depth_var) < n_samples:
                data[j, int(depth_var)] = amplitude
    
    # Add some noise
    data += np.random.randn(n_traces, n_samples) * 0.05
    
    return data

# Enhanced functions that consider human feedback
def analyze_horizons(data_info: Annotated[str, "Information about seismic data"], 
                    human_guidance: str = None) -> str:
    """Analyze seismic data for horizon picking with optional human guidance"""
    # Check if human provided specific guidance
    num_horizons = 3
    if human_guidance and "horizons" in human_guidance.lower():
        try:
            # Extract number if mentioned
            import re
            numbers = re.findall(r'\d+', human_guidance)
            if numbers:
                num_horizons = min(int(numbers[0]), 5)  # Cap at 5
        except:
            pass
    
    # Generate synthetic horizon picks based on guidance
    horizons = []
    for i in range(num_horizons):
        horizon = {
            "id": f"H{i+1}",
            "average_depth": 150 + i * 100,
            "confidence": 0.85 + np.random.rand() * 0.1,
            "continuity": "good" if i < 2 else "moderate",
            "interpretation": f"Horizon {i+1} appears to be a strong reflector",
            "human_guided": human_guidance is not None
        }
        horizons.append(horizon)
    
    return json.dumps({
        "horizons": horizons,
        "total_horizons": len(horizons),
        "quality": "good",
        "human_guidance_applied": human_guidance is not None
    }, indent=2)

def detect_faults(data_info: Annotated[str, "Information about seismic data"],
                 focus_area: str = None) -> str:
    """Detect faults in seismic data with optional focus area"""
    # Generate synthetic fault detections
    faults = []
    
    # Adjust based on focus area
    if focus_area and "trace" in focus_area.lower():
        # Try to extract trace numbers
        import re
        trace_numbers = re.findall(r'\d+', focus_area)
        if trace_numbers:
            # Focus on specified traces
            for i, trace in enumerate(trace_numbers[:2]):  # Max 2 faults
                fault = {
                    "id": f"F{i+1}",
                    "type": "normal" if i == 0 else "reverse",
                    "strike": 45 + i * 30,
                    "dip": 60 + i * 10,
                    "confidence": 0.85 + np.random.rand() * 0.1,
                    "location": f"Trace {trace}",
                    "human_focused": True
                }
                faults.append(fault)
    else:
        # Default behavior
        for i in range(2):
            fault = {
                "id": f"F{i+1}",
                "type": "normal" if i == 0 else "reverse",
                "strike": 45 + i * 30,
                "dip": 60 + i * 10,
                "confidence": 0.75 + np.random.rand() * 0.15,
                "location": f"Trace {20 + i*15}",
                "human_focused": False
            }
            faults.append(fault)
    
    return json.dumps({
        "faults": faults,
        "total_faults": len(faults),
        "structural_complexity": "moderate",
        "focus_area": focus_area
    }, indent=2)

def compute_seismic_attributes(data_info: Annotated[str, "Information about seismic data"],
                              attribute_types: str = None) -> str:
    """Compute various seismic attributes with optional specification"""
    # Default attributes
    attributes = {}
    
    # Parse requested attributes
    requested_attrs = ["amplitude", "frequency", "coherency", "curvature"]
    if attribute_types:
        attr_lower = attribute_types.lower()
        requested_attrs = [attr for attr in requested_attrs if attr in attr_lower]
        if not requested_attrs:  # If no valid attributes found, use defaults
            requested_attrs = ["amplitude", "frequency"]
    
    if "amplitude" in requested_attrs:
        attributes["instantaneous_amplitude"] = {
            "anomalies": [
                {"location": "Trace 15-20, Time 200-250ms", "type": "bright_spot", "confidence": 0.82},
                {"location": "Trace 35-40, Time 350-400ms", "type": "dim_spot", "confidence": 0.75}
            ],
            "interpretation": "Potential hydrocarbon indicators identified"
        }
    
    if "frequency" in requested_attrs:
        attributes["instantaneous_frequency"] = {
            "low_frequency_shadows": [
                {"location": "Below H2 horizon", "extent": "15 traces", "confidence": 0.78}
            ],
            "interpretation": "Possible gas accumulation below H2"
        }
    
    if "coherency" in requested_attrs:
        attributes["coherency"] = {
            "discontinuities": [
                {"location": "Trace 20-22", "orientation": "NE-SW", "confidence": 0.85},
                {"location": "Trace 35-37", "orientation": "N-S", "confidence": 0.80}
            ],
            "interpretation": "Fault patterns confirmed by coherency analysis"
        }
    
    if "curvature" in requested_attrs:
        attributes["curvature"] = {
            "structural_features": [
                {"type": "anticline", "location": "Trace 10-30", "confidence": 0.88},
                {"type": "syncline", "location": "Trace 40-50", "confidence": 0.82}
            ],
            "interpretation": "Structural deformation identified"
        }
    
    attributes["requested_types"] = requested_attrs
    attributes["human_specified"] = attribute_types is not None
    
    return json.dumps(attributes, indent=2)

def integrate_interpretations(horizons_json: str, faults_json: str, attributes_json: str = None) -> str:
    """Integrate horizon, fault, and attribute interpretations"""
    horizons = json.loads(horizons_json)
    faults = json.loads(faults_json)
    attributes = json.loads(attributes_json) if attributes_json else {}
    
    # Enhanced integration logic
    relationships = []
    for fault in faults.get("faults", []):
        for horizon in horizons.get("horizons", []):
            if abs(int(fault["location"].split()[-1]) - 25) < 10:
                relationships.append({
                    "fault": fault["id"],
                    "horizon": horizon["id"],
                    "interaction": "Fault appears to offset horizon",
                    "estimated_throw": "10-15m"
                })
    
    # Add attribute-based insights
    hydrocarbon_prospects = []
    if attributes:
        for anomaly in attributes.get("instantaneous_amplitude", {}).get("anomalies", []):
            if anomaly["type"] == "bright_spot":
                hydrocarbon_prospects.append({
                    "type": "potential_reservoir",
                    "location": anomaly["location"],
                    "confidence": anomaly["confidence"],
                    "supporting_evidence": ["amplitude anomaly", "structural closure"]
                })
    
    integrated_model = {
        "horizons": horizons.get("horizons", []),
        "faults": faults.get("faults", []),
        "attributes": attributes,
        "relationships": relationships,
        "hydrocarbon_prospects": hydrocarbon_prospects,
        "overall_quality": "good",
        "human_modifications": workflow_state.get("modification_requests", []),
        "recommendations": [
            "Consider running AVO analysis on bright spot anomalies",
            "Fault F1 requires detailed throw analysis",
            "Velocity model update recommended for depth conversion",
            "Well tie correlation suggested for H1 horizon validation"
        ]
    }
    
    return json.dumps(integrated_model, indent=2)

# Enhanced agents with human interaction awareness
horizon_interpreter = autogen.AssistantAgent(
    name="HorizonInterpreter",
    system_message="""You are a seismic horizon interpretation specialist.
    When given seismic data information:
    1. Call the analyze_horizons function to identify horizons
    2. Interpret the results geologically
    3. Provide insights about the depositional environment
    Always be specific and technical in your interpretations.""",
    llm_config=llm_config,
)

fault_interpreter = autogen.AssistantAgent(
    name="FaultInterpreter",
    system_message="""You are a structural geology expert specializing in fault interpretation.
    When given seismic data information:
    1. Call the detect_faults function to identify faults
    2. Classify fault types and analyze structural regime
    3. Assess fault system relationships
    Be detailed in your structural analysis.""",
    llm_config=llm_config,
)

lead_geophysicist = autogen.AssistantAgent(
    name="LeadGeophysicist",
    system_message="""You are the lead geophysicist coordinating seismic interpretation.
    Your role:
    1. Direct the interpretation workflow
    2. Ensure both horizons and faults are analyzed
    3. Call integrate_interpretations to combine results
    4. Provide final geological model and recommendations
    Always summarize findings clearly at the end.""",
    llm_config=llm_config,
)

# NEW AGENT: Seismic Attribute Analyst
attribute_analyst = autogen.AssistantAgent(
    name="AttributeAnalyst",
    system_message="""You are a seismic attribute analysis specialist.
    When given seismic data information:
    1. Call the compute_seismic_attributes function to calculate various attributes
    2. Identify amplitude anomalies, frequency shadows, and coherency patterns
    3. Correlate attribute anomalies with structural and stratigraphic features
    4. Highlight potential hydrocarbon indicators
    Focus on DHI (Direct Hydrocarbon Indicators) and reservoir characterization.""",
    llm_config=llm_config,
)
# Enhanced user proxy with decision-making prompts
def create_user_proxy():
    return autogen.UserProxyAgent(
        name="DataProvider",
        system_message="""You are the seismic data provider and interpretation reviewer.
        Your role is to:
        1. Provide guidance on interpretation priorities
        2. Review and approve/reject interpretations
        3. Request modifications when needed
        4. Make final decisions on the geological model
        
        Available commands:
        - 'approve' - Accept the current interpretation
        - 'modify' - Request changes (specify what to change)
        - 'skip' - Skip current analysis
        - 'focus on [area/feature]' - Direct attention to specific areas
        - 'exit' or 'TERMINATE' - End the workflow""",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=3,
        is_termination_msg=lambda x: x.get("content", "").rstrip().lower() in ["terminate", "exit"],
        code_execution_config=False,
    )

# Register enhanced functions
def register_all_functions(user_proxy):
    autogen.register_function(
        analyze_horizons,
        caller=horizon_interpreter,
        executor=user_proxy,
        name="analyze_horizons",
        description="Analyze seismic data to identify horizons with optional human guidance",
    )
    
    autogen.register_function(
        detect_faults,
        caller=fault_interpreter,
        executor=user_proxy,
        name="detect_faults",
        description="Detect faults with optional focus area specification",
    )
    
    autogen.register_function(
        compute_seismic_attributes,
        caller=attribute_analyst,
        executor=user_proxy,
        name="compute_seismic_attributes",
        description="Compute specified seismic attributes",
    )
    
    autogen.register_function(
        integrate_interpretations,
        caller=lead_geophysicist,
        executor=user_proxy,
        name="integrate_interpretations",
        description="Integrate all interpretations into final model",
    )

# Interactive workflow execution
def run_interactive_interpretation():
    """Run the seismic interpretation with meaningful human interaction"""
    
    print("=== Interactive Seismic Interpretation Workflow ===\n")
    print("You will guide the interpretation process at each step.\n")
    
    # Reset workflow state
    global workflow_state
    workflow_state = {
        "horizons_approved": False,
        "faults_approved": False,
        "attributes_approved": False,
        "human_feedback": {},
        "modification_requests": []
    }
    
    # Generate synthetic data
    seismic_data = generate_synthetic_seismic_data()
    data_info = f"""
    Seismic Data Overview:
    - Type: 2D Seismic Section
    - Size: {seismic_data.shape[0]} traces x {seismic_data.shape[1]} samples
    - Sampling: 4ms
    - Processing: Post-stack time migration
    - Quality: Good signal-to-noise ratio
    
    The data appears to show multiple reflectors and some structural features.
    """
    
    print(data_info)
    print("\nInitializing interpretation team...\n")
    
    # Create agents
    user_proxy = create_user_proxy()
    register_all_functions(user_proxy)
    
    # Create group chat
    groupchat = autogen.GroupChat(
        agents=[user_proxy, lead_geophysicist, horizon_interpreter, fault_interpreter, attribute_analyst],
        messages=[],
        max_round=20,  # Increased for more interaction
        speaker_selection_method="round_robin",
    )
    
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    
    # Initial message that prompts for user input
    initial_message = f"""
    {data_info}
    
    Welcome to the Interactive Seismic Interpretation System.
    
    I'm the Lead Geophysicist, and I'll coordinate our interpretation team consisting of:
    - Horizon Interpreter
    - Fault Interpreter  
    - Seismic Attribute Analyst
    
    Before we begin, please let me know:
    1. What are your main interpretation objectives? (e.g., structural mapping, reservoir identification)
    2. Are there specific features or areas you want us to focus on?
    3. What's your preferred workflow order? (horizons first, faults first, or attributes first?)
    
    Type your preferences, or type 'standard' for the default workflow.
    """
    
    try:
        # Start the interactive workflow
        user_proxy.initiate_chat(
            manager,
            message=initial_message,
        )
        
        print("\n=== Workflow Completed ===")
        
        # Save results
        save_interpretation_results(groupchat, seismic_data.shape)
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")

def save_interpretation_results(groupchat, data_shape):
    """Save the interpretation results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "./AutoGenSeis/interpretation_results"
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"interactive_interpretation_{timestamp}.json")
    
    # Extract conversation
    conversation_history = []
    for message in groupchat.messages:
        conversation_history.append({
            "speaker": message.get("name", "Unknown"),
            "content": message.get("content", "")
        })
    
    results = {
        "timestamp": timestamp,
        "data_shape": data_shape,
        "workflow_state": workflow_state,
        "conversation": conversation_history
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

# Test functions remain the same
def test_individual_functions():
    """Test individual functions with human guidance examples"""
    print("Testing functions with human guidance...\n")
    
    # Test horizon analysis with guidance
    print("1. Horizon analysis with guidance:")
    horizon_result = analyze_horizons("Test data", "I see 4 horizons in the data")
    print(horizon_result)
    
    # Test fault detection with focus area
    print("\n2. Fault detection with focus:")
    fault_result = detect_faults("Test data", "Focus on traces 10-15 and 40-45")
    print(fault_result)
    
    # Test attributes with specific types
    print("\n3. Attribute analysis with specific types:")
    attr_result = compute_seismic_attributes("Test data", "Only amplitude and coherency")
    print(attr_result)
    
    print("\n✓ All functions tested successfully!")

# Main menu
def main_menu():
    """Display enhanced main menu"""
    # while True:
    print("\n=== Seismic Interpretation System ===")
    print("1. Run Interactive Interpretation (Recommended)")
    print("2. Test Individual Functions")
    print("3. View Instructions")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ")
    
    if choice == '1':
        run_interactive_interpretation()
    elif choice == '2':
        test_individual_functions()
        input("\nPress Enter to continue...")
    elif choice == '3':
        print("\n=== Instructions ===")
        print("The Interactive Interpretation mode allows you to:")
        print("- Guide the interpretation process")
        print("- Specify areas of interest")
        print("- Approve or modify interpretations")
        print("- Choose which attributes to compute")
        print("\nCommands during interpretation:")
        print("- 'approve' - Accept current interpretation")
        print("- 'modify' - Request changes")
        print("- 'focus on [area]' - Direct attention")
        print("- 'skip' - Skip current analysis")
        print("- 'exit' - End workflow")
        input("\nPress Enter to continue...")
    elif choice == '4':
        print("Exiting...")
        # break
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()