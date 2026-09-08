from __future__ import annotations

import math


def build_optimizer(model, *, learning_rate: float, momentum: float, weight_decay: float):
    torch = __import__("torch")
    decayed = [model.W]
    other = [parameter for name, parameter in model.named_parameters() if name != "W"]
    groups = [
        {"params": decayed, "weight_decay": float(weight_decay)},
        {"params": other, "weight_decay": 0.0},
    ]
    return torch.optim.SGD(groups, lr=float(learning_rate), momentum=float(momentum))


def set_optimizer_momentum(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["momentum"] = float(value)


def apply_parameter_constraints(model, *, max_weight_row_norm: float) -> None:
    torch = __import__("torch")
    with torch.no_grad():
        row_norm = model.W.norm(dim=1, keepdim=True)
        model.W.mul_(torch.clamp(float(max_weight_row_norm) / (row_norm + 1e-12), max=1.0))
        if model.family == "gaussian":
            model.log_sigma.clamp_(
                float(model.settings["log_sigma_update_min"]),
                float(model.settings["log_sigma_update_max"]),
            )
        elif model.family == "student_t_poe":
            sigma_min = max(float(model.settings["sigma_sample_min"]), 1e-12)
            sigma_max = max(float(model.settings["sigma_sample_max"]), sigma_min)
            nu_min = max(float(model.settings["nu_min"]), 1e-12)
            nu_max = max(float(model.settings["nu_max"]), nu_min)
            model.log_sigma.clamp_(math.log(sigma_min), math.log(sigma_max))
            model.log_nu.clamp_(math.log(nu_min), math.log(nu_max))


def optimizer_step(
    model,
    optimizer,
    objective,
    *,
    max_weight_row_norm: float,
) -> float:
    torch = __import__("torch")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    optimizer.step()
    apply_parameter_constraints(model, max_weight_row_norm=max_weight_row_norm)
    with torch.no_grad():
        squared = torch.zeros((), device=model.W.device, dtype=model.W.dtype)
        for previous, current in zip(before, model.parameters()):
            squared = squared + (current.detach() - previous).square().sum()
        return float(torch.sqrt(squared).detach().cpu())
