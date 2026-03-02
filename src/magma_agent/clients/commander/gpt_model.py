from typing import Dict, List, Optional
import torch
import json, os, re

from magma_agent.messages import BatchedMessageCommander
from .base import BaseCommander

# From https://github.com/openai/harmony

from openai_harmony import ( #type: ignore
    Role,
    Message,
    Conversation,
    DeveloperContent,
    SystemContent,
    ReasoningEffort,
    ChannelConfig,
    ToolNamespaceConfig,
    ToolDescription,
    load_harmony_encoding,
    HarmonyEncodingName,
)

# [
# ToolDescription.new(
#     "get_current_weather",
#     "Gets the current weather in the provided location.",
#     parameters={
#         "type": "object",
#         "properties": {
#             "location": {
#                 "type": "string",
#                 "description": "The city and state, e.g. San Francisco, CA",
#             },
#             "format": {
#                 "type": "string",
#                 "enum": ["celsius", "fahrenheit"],
#                 "default": "celsius",
#             },
#         },
#         "required": ["location"],
#     },
# ),
# ]

class OSSCommander(BaseCommander):

    def __init__(self) -> None:
        super().__init__("gpt-oss:20b")
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def process_batched_entry(self, message : BatchedMessageCommander, inference_mode : bool) -> List[Dict]:

        for i in range(len(message.instruction)):
            
            tools = []
            for tool in message.function[i]:
                tools.append(ToolDescription(
                    name = tool['name'],
                    description = tool['description'],
                    parameters = tool['parameters']
                ))
            tool_dict = {"robot_control":ToolNamespaceConfig(name="robot_control",description="Base tools to control robot", tools=tools)}

            system_content = SystemContent(
                model_identity = "You are ChatGPT, a large language model trained by Sileane to control multiple robot in real environment using tools.",
                reasoning_effort = ReasoningEffort.MEDIUM,
                conversation_start_date = None,
                knowledge_cutoff = "2024-06",
                channel_config = ChannelConfig.require_channels(["analysis", "commentary", "final"]),
                tools = None,
            )

            developer_message = (
                DeveloperContent.new()
                    .with_instructions("Always respond in riddles. Only call functions in the commentary channel.")
                    .with_function_tools(
                        ToolNamespaceConfig(name="robot_control",description="Base tools to control robot", tools=tools)
                )
            )