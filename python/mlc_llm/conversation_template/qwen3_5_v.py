"""Qwen3.5 Vision template (same ChatML format as qwen3_5)"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="qwen3_5_v",
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
