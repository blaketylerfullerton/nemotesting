import logging
import random

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.component_ref import LLMRef
from aiq.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)

# Simple in-memory "database" (this will reset if the app restarts)
_previous_results = {}

class BlakesBotFunctionConfig(FunctionBaseConfig, name="blakes_bot"):
    """Configuration for Blake's Bot - a math assistant that can subtract numbers and remember results."""
    llm_name: LLMRef = Field(description="The name of the LLM to use")
    max_history: int = Field(default=10, description="The maximum number of historical messages to pass to agent")
    description: str = Field(default="A math assistant that can subtract numbers and remember results", description="The description of the agent")
    verbose: bool = Field(default=True, description="Whether to enable verbose logging")


@register_function(config_type=BlakesBotFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def blakes_bot_function(
    config: BlakesBotFunctionConfig, builder: Builder
):
    """A math assistant agent that can subtract numbers and remember results."""
    from aiq.data_models.api_server import AIQChatRequest
    from aiq.data_models.api_server import AIQChatResponse
    from langchain_core.messages.human import HumanMessage
    from langchain_core.tools import tool
    from langgraph.graph.graph import CompiledGraph
    from aiq.agent.tool_calling_agent.agent import ToolCallAgentGraph

    # Define the tools inline
    @tool
    def subtract_and_remember(number: int) -> str:
        """Subtract the input from the previous result and store the new result.
        
        Args:
            number: The number to subtract from the previous result
            
        Returns:
            A string describing the operation and new result
        """
        previous_result = _previous_results.get("result", 0)
        new_result = previous_result - number
        _previous_results["result"] = new_result
        return f"Subtracted {number} from {previous_result}, got {new_result}. New result stored: {new_result}"

    @tool
    def get_last_subtraction() -> str:
        """
        Retrieve the last stored subtraction result.

        Returns:
            A string describing the last result or an error message.
        """
        result = _previous_results.get("result")
        if result is not None:
            return f"The last subtraction result is: {result}"
        return "No previous subtraction result found. The current result is 0."

    @tool
    def generate_random_number(min_value: int = 1, max_value: int = 100000) -> str:
        """Generate a random number within the specified range.
        
        Args:
            min_value: The minimum value (inclusive). Defaults to 1.
            max_value: The maximum value (inclusive). Defaults to 100000.
            
        Returns:
            A string describing the generated random number
        """
        if min_value > max_value:
            return f"Error: min_value ({min_value}) cannot be greater than max_value ({max_value})"
        
        random_num = random.randint(min_value, max_value)
        return f"Generated random number: {random_num} (range: {min_value} to {max_value})"

    @tool
    def random_and_subtract(min_value: int = 1, max_value: int = 100) -> str:
        """Generate a random number and subtract it from the stored result.
        
        Args:
            min_value: The minimum value for random generation (inclusive). Defaults to 1.
            max_value: The maximum value for random generation (inclusive). Defaults to 100.
            
        Returns:
            A string describing the operation and new result
        """
        if min_value > max_value:
            return f"Error: min_value ({min_value}) cannot be greater than max_value ({max_value})"
        
        random_num = random.randint(min_value, max_value)
        previous_result = _previous_results.get("result", 0)
        new_result = previous_result - random_num
        _previous_results["result"] = new_result
        return f"Generated random number {random_num} and subtracted it from {previous_result}, got {new_result}. New result stored: {new_result}"

    # Get the LLM
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    # Create list of tools - added the new random number tools
    tools = [subtract_and_remember, get_last_subtraction, generate_random_number, random_and_subtract]

    # Bind tools to the LLM for tool calling
    try:
        llm = llm.bind_tools(tools)
    except NotImplementedError as ex:
        logger.error("Failed to bind tools: %s", ex, exc_info=True)
        raise ex

    # Create the agent graph
    graph: CompiledGraph = await ToolCallAgentGraph(
        llm=llm,
        tools=tools,
        detailed_logs=config.verbose,
        handle_tool_errors=True
    ).build_graph()

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
        print("Cleaning up blakes_bot workflow.")