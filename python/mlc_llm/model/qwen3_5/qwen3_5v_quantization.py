"""Quantization config for Qwen3.5 Vision-Language model."""

from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import GroupQuantize, NoQuantize

from .qwen3_5v_model import Qwen35VConfig, Qwen35VForCausalLM


def group_quant(
    model_config: Qwen35VConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen3.5V model using group quantization."""
    model: nn.Module = Qwen35VForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    quantization.tensor_parallel_shards = model_config.tensor_parallel_shards
    model = quantization.quantize_model(model, quant_map, "")
    return model, quant_map


def no_quant(
    model_config: Qwen35VConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Qwen3.5V model without quantization."""
    model: nn.Module = Qwen35VForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map
