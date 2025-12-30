import autogen

class SupervisorAgent:
    def __init__(self):
        self.agent = autogen.AssistantAgent(
            name="supervisor",
            system_message="""You are a supervisor agent responsible for coordinating segmentation tasks between two UNet models.
            Your role is to:
            1. Analyze the input data
            2. Decide which UNet model to use (circle or rectangle detection)
            3. Coordinate the segmentation process
            4. Validate the results""",
            llm_config={
                "config_list": [{"model": "gpt-3.5-turbo", "api_key": "YOUR_API_KEY"}]
            }
        )

    def coordinate_task(self, task_type, data):
        """Coordinate the segmentation task"""
        if task_type == "circle":
            return {
                "model": "circle_unet",
                "data": data,
                "message": "Process circle segmentation task"
            }
        elif task_type == "rectangle":
            return {
                "model": "rectangle_unet",
                "data": data,
                "message": "Process rectangle segmentation task"
            }
        else:
            return {
                "error": "Unknown task type"
            }

    def validate_results(self, results):
        """Validate the segmentation results"""
        if results["success"]:
            return {
                "status": "success",
                "message": "Segmentation completed successfully",
                "results": results["data"]
            }
        else:
            return {
                "status": "error",
                "message": "Segmentation failed",
                "error": results["error"]
            }