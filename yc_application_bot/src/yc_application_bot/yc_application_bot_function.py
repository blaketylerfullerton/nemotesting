import logging
from pathlib import Path

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.component_ref import LLMRef
from aiq.builder.framework_enum import LLMFrameworkEnum
import json
logger = logging.getLogger(__name__)


class YcApplicationBotFunctionConfig(FunctionBaseConfig, name="yc_application_bot"):
    """
    AIQ Toolkit function template. Please update the description.
    """
    # Add your custom configuration parameters here
    llm_name: LLMRef = Field(description="The name of the LLM to use")
    max_history: int = Field(default=10, description="The maximum number of historical messages to pass to agent")
    description: str = Field(default="A math assistant that can subtract numbers and remember results", description="The description of the agent")
    verbose: bool = Field(default=True, description="Whether to enable verbose logging")


@register_function(config_type=YcApplicationBotFunctionConfig)
async def yc_application_bot_function(
    config: YcApplicationBotFunctionConfig, builder: Builder
):
    from aiq.data_models.api_server import AIQChatRequest
    from aiq.data_models.api_server import AIQChatResponse
    from langchain_core.messages.human import HumanMessage
    from langchain_core.tools import tool
    from langgraph.graph.graph import CompiledGraph
    from aiq.agent.tool_calling_agent.agent import ToolCallAgentGraph

    @tool
    def apply_for_job(name: str) -> str:
        """Apply for a job.
    
        Returns:
            A string describing the operation and new result
        """
        # Get the path to the jobs.json file relative to this Python file
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # Go up 3 levels from src/yc_application_bot/
        jobs_file = project_root / "data" / "jobs.json"
        
        with open(jobs_file, 'r') as f:
            jobs = json.load(f)
        for job in jobs:
            if job['name'] == name:
                return f"Applying for job {job['website']} at {job['job_description']}"
        return f"No job found with name {name}"

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Create list of tools - added the new random number tools
    tools = [apply_for_job]

    try:
        llm = llm.bind_tools(tools)
    except NotImplementedError as ex:
        logger.error("Failed to bind tools: %s", ex, exc_info=True)
        raise ex
    graph: CompiledGraph = await ToolCallAgentGraph(
        llm=llm,
        tools=tools,
        detailed_logs=config.verbose,
        handle_tool_errors=True
    ).build_graph()
    # Implement your function logic here
    async def _response_fn(input_message: AIQChatRequest) -> AIQChatResponse:
        # Extract the last message content
        last_message = input_message.messages[-1].content
        
        # Prepare the input for the agent
        agent_input = {
            "messages": [HumanMessage(content=last_message)]
        }
        
        # Run the agent
        result = await graph.ainvoke(agent_input)
        
        # Get the final response
        final_message = result.get("messages", [])[-1].content if result.get("messages") else "I couldn't process your request."
        
        return AIQChatResponse.from_string(final_message)

    try:
        yield FunctionInfo.create(single_fn=_response_fn)
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up yc_application_bot workflow.")