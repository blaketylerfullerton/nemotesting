import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.builder.framework_enum import LLMFrameworkEnum

logger = logging.getLogger(__name__)

class BlakesBotFunctionConfig(FunctionBaseConfig, name="blakes_bot"):
    # Add your custom configuration parameters here
    tool_name: list[FunctionRef] = Field(default=[], description="List of tools to use")
    llm_name: LLMRef = Field(description="The name of the LLM to use")
    max_history: int = Field(default=10, description="The maximum number of historical messages to pass to agent")
    description: str = Field(default="", description="The description of the tool")
    

@register_function(config_type=BlakesBotFunctionConfig)
async def blakes_bot_function(
    config: BlakesBotFunctionConfig, builder: Builder
):
    # Simple function that returns Blake's favorite food
    from aiq.data_models.api_server import AIQChatRequest
    from aiq.data_models.api_server import AIQChatResponse

    async def _response_fn(input_message: AIQChatRequest) -> AIQChatResponse:
        return AIQChatResponse.from_string("Blake's favorite food is spagetti tacos!")

    try:
        yield FunctionInfo.create(single_fn=_response_fn)
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up blakes_bot workflow.")