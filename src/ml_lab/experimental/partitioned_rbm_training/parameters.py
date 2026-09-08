from __future__ import annotations

from dataclasses import replace


def initialize_global_cross_weights(model, *, mode: str) -> None:
    if mode == "random":
        return
    if mode != "zero":
        raise ValueError("cross-weight initialization must be zero or random")
    torch = __import__("torch")
    with torch.no_grad():
        model.W.zero_()


def initialize_local_from_global(local_model, global_model, block) -> None:
    """Warm-start a merged local model from the corresponding global parameter block."""
    torch = __import__("torch")
    vs = slice(block.visible_start, block.visible_end)
    hs = slice(block.hidden_start, block.hidden_end)
    with torch.no_grad():
        local_model.W.copy_(global_model.W[vs, hs])
        local_model.v_bias.copy_(global_model.v_bias[vs])
        local_model.h_bias.copy_(global_model.h_bias[hs])
        if hasattr(local_model, "log_sigma") and hasattr(global_model, "log_sigma"):
            local_model.log_sigma.copy_(global_model.log_sigma[vs])
        if hasattr(local_model, "log_nu"):
            value = block.scalar_state.get("log_nu")
            if value is not None:
                local_model.log_nu.fill_(float(value))
            elif hasattr(global_model, "log_nu"):
                local_model.log_nu.copy_(global_model.log_nu)


def write_local_to_global(local_model, global_model, block, *, final: bool = False):
    torch = __import__("torch")
    vs = slice(block.visible_start, block.visible_end)
    hs = slice(block.hidden_start, block.hidden_end)
    with torch.no_grad():
        global_model.W[vs, hs].copy_(local_model.W)
        global_model.v_bias[vs].copy_(local_model.v_bias)
        global_model.h_bias[hs].copy_(local_model.h_bias)
        if hasattr(local_model, "log_sigma") and hasattr(global_model, "log_sigma"):
            global_model.log_sigma[vs].copy_(local_model.log_sigma)
        if final and hasattr(local_model, "log_nu") and hasattr(global_model, "log_nu"):
            global_model.log_nu.copy_(local_model.log_nu)
    scalar_state: dict[str, float] = {}
    if hasattr(local_model, "log_nu"):
        scalar_state["log_nu"] = float(local_model.log_nu.detach().cpu())
    return replace(block, scalar_state=scalar_state)
