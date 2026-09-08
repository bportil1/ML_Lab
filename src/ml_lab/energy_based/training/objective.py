from __future__ import annotations


def contrastive_divergence_objective(model, positive_visible, negative_visible):
    """Return the autograd-consistent CD objective ``E_data - E_negative``.

    Negative samples are detached by the sampling phase, so gradients do not
    backpropagate through the Markov chain. This keeps the update aligned with
    each model family's actual energy definition and avoids duplicated manual
    gradient formulas that can drift away from the model implementation.
    """
    return model.energy(positive_visible).mean() - model.energy(negative_visible).mean()
