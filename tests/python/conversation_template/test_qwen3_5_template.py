import pytest

from mlc_llm.conversation_template import ConvTemplateRegistry

pytestmark = [pytest.mark.runtime_unittest]


def test_qwen35_prompt():
    """Test Qwen3.5 ChatML conversation template with <think> prefix.

    The assistant role includes a non-thinking prefix (<think>\\n\\n</think>\\n)
    that tells the model to skip internal reasoning and respond directly.
    """
    conversation = ConvTemplateRegistry.get_conv_template("qwen3_5")
    system_msg = "You are a helpful assistant."
    user_msg1 = "What is 2+2?"
    assistant_msg1 = "4"
    user_msg2 = "And 3+3?"

    conversation.system_message = system_msg
    conversation.messages.append(("user", user_msg1))
    conversation.messages.append(("assistant", assistant_msg1))
    conversation.messages.append(("user", user_msg2))
    conversation.messages.append(("assistant", None))
    res = conversation.as_prompt()

    expected = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "What is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        "4<|im_end|>\n"
        "<|im_start|>user\n"
        "And 3+3?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    assert res[0] == expected


if __name__ == "__main__":
    test_qwen35_prompt()
