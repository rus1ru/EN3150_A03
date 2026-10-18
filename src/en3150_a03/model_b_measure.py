"""Forward-pass parameter/MAC accounting without optional profiling packages."""
from __future__ import annotations

import torch
from torch import nn

MAC_CONVENTION = (
    "Batch-one Conv2d and Linear only; one multiply-accumulate = one MAC "
    "(approximately two FLOPs). Bias adds, activations, pooling, memory traffic "
    "and framework overhead excluded. This is operation count, not latency."
)


def profile_model(model: nn.Module, input_shape=(1, 3, 64, 64)) -> dict:
    """Record executed leaf-layer shapes, weights and Conv/Linear MACs on CPU.

    Restores training flags and removes hooks even on failure. Call with a CPU
    model so profiling never moves or changes the training model's parameters.
    """
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise ValueError("Use a batch-one NCHW input shape")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Profile a CPU model instance")
    rows, hooks = [], []
    modes = {module: module.training for module in model.modules()}

    def capture(name):
        def hook(module, inputs, output):
            macs = 0
            if isinstance(module, nn.Conv2d):
                macs = output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            elif isinstance(module, nn.Linear):
                macs = output.numel() * module.in_features
            rows.append({
                "layer": name, "type": type(module).__name__,
                "input_shape": list(inputs[0].shape), "output_shape": list(output.shape),
                "kernel": list(module.kernel_size) if isinstance(module, nn.Conv2d) else None,
                "groups": module.groups if isinstance(module, nn.Conv2d) else None,
                "parameters": sum(p.numel() for p in module.parameters(recurse=False)),
                "macs": macs,
            })
        return hook

    try:
        for name, module in model.named_modules():
            if not list(module.children()):
                hooks.append(module.register_forward_hook(capture(name)))
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(input_shape))
    finally:
        for handle in hooks:
            handle.remove()
        for module, training in modes.items():
            module.training = training
    parameters = sum(p.numel() for p in model.parameters())
    macs = sum(row["macs"] for row in rows)
    return {
        "input_shape": list(input_shape), "total_parameters": parameters,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "estimated_float32_size_kb": parameters * 4 / 1024,
        "fp32_parameter_size_mb": parameters * 4 / 1_000_000,
        "macs": macs, "flops_conv_linear": 2 * macs,
        "mac_convention": MAC_CONVENTION, "layers": rows,
    }
