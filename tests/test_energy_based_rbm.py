from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest


def test_energy_based_discovery_does_not_import_torch():
    code = (
        "import sys; import ml_lab.energy_based as eb; "
        "assert 'torch' not in sys.modules; "
        "assert [s.id for s in eb.rbm.list_families()] == "
        "['bernoulli', 'gaussian', 'student_t_poe']"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_rbm_family_aliases_and_defaults():
    from ml_lab.energy_based.rbm import RBMConfig, canonical_family

    assert canonical_family("bern") == "bernoulli"
    assert canonical_family("gauss") == "gaussian"
    assert canonical_family("stud_t") == "student_t_poe"
    cfg = RBMConfig("gauss", 5, 3, settings={"visible_noise_scale": 0.25})
    settings = cfg.resolved_settings()
    assert settings["visible_noise_scale"] == pytest.approx(0.25)
    assert "initial_log_sigma" in settings
    with pytest.raises(ValueError, match="Unknown"):
        RBMConfig("gauss", 5, 3, settings={"not_a_setting": 1}).resolved_settings()


@pytest.mark.parametrize("family", ["bernoulli", "gaussian", "student_t_poe"])
def test_rbm_models_have_common_sampling_energy_contract(family):
    torch = pytest.importorskip("torch")
    from ml_lab.energy_based import rbm

    torch.manual_seed(17)
    model = rbm.create(family, visible_dim=4, hidden_dim=3, sharpness=0.8, dropout=0.0)
    model.eval()
    x = torch.zeros(6, 4)
    probability, hidden = model.sample_hidden(x)
    chain = rbm.run_chain(model, x, steps=2)
    energy = model.energy(x)

    assert probability.shape == (6, 3)
    assert hidden.shape == (6, 3)
    assert chain.shape == (6, 4)
    assert energy.shape == (6,)
    assert torch.isfinite(probability).all()
    assert torch.isfinite(chain).all()
    assert torch.isfinite(energy).all()
    summary = rbm.summarize(model, x)
    assert summary["config"]["family"] == family
    assert np.isfinite(summary["energy_gap"])


def test_rbm_input_is_moved_to_model_device_and_shape_checked():
    torch = pytest.importorskip("torch")
    from ml_lab.energy_based import rbm

    model = rbm.create("bern", 3, 2, device="cpu")
    energy = model.energy(np.zeros((4, 3), dtype=np.float64))
    assert energy.device.type == "cpu"
    assert energy.dtype == model.W.dtype
    with pytest.raises(ValueError, match="visible input"):
        model.energy(np.zeros((4, 4)))


def test_hidden_dropout_is_unscaled_and_disabled_in_eval():
    torch = pytest.importorskip("torch")
    from ml_lab.energy_based import rbm

    torch.manual_seed(11)
    model = rbm.create("bern", 3, 2000, sharpness=1.0, dropout=0.5)
    with torch.no_grad():
        model.W.zero_()
        model.h_bias.fill_(20.0)

    model.train()
    _, sample = model.sample_hidden(torch.zeros(1, 3))
    assert set(sample.unique().tolist()).issubset({0.0, 1.0})
    assert 0.4 < sample.mean().item() < 0.6

    model.eval()
    activation = model.hidden_activation(torch.zeros(1, 3))
    assert torch.all(activation > 0.999)


def test_energy_gap_has_one_sign_convention_across_families(monkeypatch):
    torch = pytest.importorskip("torch")
    from ml_lab.energy_based import rbm

    for family in ("bernoulli", "gaussian", "student_t_poe"):
        model = rbm.create(family, 2, 2)
        x = torch.zeros(2, 2)
        original_energy = model.energy
        def fake_gibbs(value):
            return torch.ones_like(value)
        monkeypatch.setattr(model, "gibbs_step", fake_gibbs)
        pos, neg, gap = model.compute_energy_gap(x)
        expected = float((original_energy(torch.ones_like(x)).mean() - original_energy(x).mean()).detach())
        assert gap == pytest.approx(expected)
        assert gap == pytest.approx(neg - pos)


def test_student_t_uses_consistent_log_sigma_parametrization():
    torch = pytest.importorskip("torch")
    from ml_lab.energy_based import rbm

    model = rbm.create("stud_t", 3, 2, settings={"sigma_sample_min": 1e-8, "sigma_sample_max": 1e8})
    with torch.no_grad():
        model.log_sigma.fill_(np.log(2.0))
    sigma = model._sigma()
    assert torch.allclose(sigma, torch.full_like(sigma, 2.0))

    x = torch.ones(2, 3)
    h = torch.zeros(2, 2)
    lam = model.compute_lambda(x, h)
    energy = model.energy(x)
    assert torch.isfinite(lam).all()
    assert torch.isfinite(energy).all()


def test_rbm_checkpoint_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    from ml_lab.energy_based import rbm

    model = rbm.create("gaussian", 4, 3, sharpness=0.7, dropout=0.2)
    with torch.no_grad():
        model.W.fill_(0.125)
        model.log_sigma.fill_(-0.3)
    path = rbm.save(model, tmp_path / "gaussian-rbm.pt")
    loaded = rbm.load(path)

    assert loaded.configuration() == model.configuration()
    assert torch.allclose(loaded.W, model.W)
    assert torch.allclose(loaded.log_sigma, model.log_sigma)


def test_list_rbms_cli():
    completed = subprocess.run(
        [sys.executable, "-m", "ml_lab", "list-rbms"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "bernoulli" in completed.stdout
    assert "gaussian" in completed.stdout
    assert "student_t_poe" in completed.stdout
