"""Qwen3.5 default template"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# ChatML format for Qwen3.5 hybrid attention reasoning model.
# Non-thinking mode: the assistant prefix includes <think>\n\n</think>\n
# which tells the model to skip internal reasoning and respond directly.
# For thinking mode, use the completions API with prefix ending in <think>\n
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="qwen3_5",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="You are a helpful assistant.",
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant\n<think>\n\n</think>\n"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[248046, 248044],
    )
)
