"""InternVL default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

from .registry import ConvTemplateRegistry

# InternVL2.5/3.5 uses ChatML format (same as Qwen)
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="internvl2_5",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[151643, 151645],
    )
)
