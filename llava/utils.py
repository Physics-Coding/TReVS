"""Small inference helpers shared by the retained LLaVA evaluators."""


def disable_torch_init():
    """Disable redundant default initialization while loading checkpoint weights."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
