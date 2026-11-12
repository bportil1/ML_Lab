"""S3.2 typed model controls must match S3.1 scientific configuration contracts."""
from __future__ import annotations

import pytest

from ml_lab.experiments import (
    describe_model_method, list_model_families, list_model_methods,
    validate_model_configuration,
)
from ml_lab.ui.model_forms import (
    ModelFormError, control_placeholder, model_form_values, read_model_form,
    submitted_form_values, visible_controls,
)


class Form(dict):
    def getlist(self, key):
        value = self.get(key, [])
        return value if isinstance(value, list) else [value]


def test_all_stable_method_form_definitions_have_parseable_default_fields():
    for family in list_model_families():
        name = family["id"]
        for method in list_model_methods(name):
            description = describe_model_method(name, method["id"])
            fields = visible_controls(description)
            assert fields
            for group in fields.values():
                for field in group:
                    assert "name" in field and "type" in field
                    control_placeholder(field)
            form = Form()
            if method["id"] == "suite":
                choice = next(m["id"] for m in list_model_methods(name) if m["id"] != "suite")
                form["ctl.task_options.estimators"] = [choice]
            settings = read_model_form(form, description)
            validate_model_configuration(name, method["id"], settings, n_features=3)


@pytest.mark.parametrize("family,method,form,expected", [
    ("classification", "ridge", {"ctl.config.cv_folds": "4", "ctl.config.refit": "false", "ctl.task_options.test_size": "0.25"},
     {"config": {"cv_folds": 4, "refit": False}, "task_options": {"test_size": .25}}),
    ("representation", "pca", {"ctl.config.n_components": "0.9"}, {"config": {"n_components": .9}}),
    ("representation", "mlp_autoencoder", {"ctl.model_config.hidden_dims": "48, 24", "ctl.model_config.latent_dim": "4"},
     {"model_config": {"hidden_dims": [48, 24], "latent_dim": 4}}),
    ("rbm", "gaussian", {"ctl.task_options.hidden_dim": "12", "ctl.model_config.settings.initial_log_sigma": "0.4",
                          "ctl.training_config.model_selection.enabled": "true"},
     {"task_options": {"hidden_dim": 12}, "model_config": {"settings": {"initial_log_sigma": .4}},
      "training_config": {"model_selection": {"enabled": True}}}),
    ("gan", "gan", {"ctl.model_config.generator_type": "transformer", "ctl.model_config.discriminator_type": "mlp",
                    "ctl.model_config.generator_hidden_dims": "[16, 32]"},
     {"model_config": {"generator_type": "transformer", "discriminator_type": "mlp", "generator_hidden_dims": [16, 32]}}),
])
def test_typed_controls_support_nested_arrays_and_boolean_without_json_workbench(family, method, form, expected):
    schema = describe_model_method(family, method)
    raw = read_model_form(Form(form), schema)
    # task_options are UI-only grouping, and map to root task settings.
    for group, fields in expected.items():
        if group == "task_options":
            for key, value in fields.items():
                assert raw[key] == value
        else:
            assert raw[group] == fields
    validate_model_configuration(family, method, raw, n_features=3)


def test_suite_multi_select_and_roundtrip():
    schema = describe_model_method("classification", "suite")
    settings = read_model_form(Form({"ctl.task_options.estimators": ["ridge", "rf"]}), schema)
    assert settings == {"estimators": ["ridge", "rf"]}
    canonical = validate_model_configuration("classification", "suite", settings)
    values = model_form_values(canonical)
    assert values["task_options.estimators"] == ["ridge", "rf"]
    assert read_model_form(Form({"ctl.task_options.estimators": values["task_options.estimators"]}), schema) == settings


@pytest.mark.parametrize("method,form", [
    ("ridge", {"ctl.config.cv_folds": "1.0"}),
    ("ridge", {"ctl.config.cv_folds": "True"}),
    ("ridge", {"ctl.config.refit": "on"}),
    ("ridge", {"ctl.config.cv_folds": "99999999999999999x"}),
    ("ridge", {"ctl.config.error_score": "nan"}),
    ("ridge", {"ctl.config.cv_folds": "2", "ctl.config.unknown_field": "3"}),
    ("suite", {"ctl.task_options.estimators": ["ridge", "ridge"]}),
    ("suite", {"ctl.task_options.estimators": ["__invalid__"]}),
])
def test_invalid_fields_rejected_early(method, form):
    with pytest.raises(ModelFormError):
        read_model_form(Form(form), describe_model_method("classification", method))


def test_gan_checkpoint_is_never_an_editable_control():
    schema = describe_model_method("gan", "gan")
    visible = visible_controls(schema)
    assert "checkpoint_path" not in {field["name"] for field in visible["training_config"]}
    with pytest.raises(ModelFormError):
        read_model_form(Form({"ctl.training_config.checkpoint_path": "../private.pt"}), schema)


def test_failed_form_input_preserved_and_invalid_syntax_cannot_be_saved():
    form = Form({"ctl.model_config.hidden_dims": "12,,32", "ctl.training_config.epochs": "7"})
    assert submitted_form_values(form)["model_config.hidden_dims"] == "12,,32"
    with pytest.raises(ModelFormError):
        read_model_form(form, describe_model_method("representation", "mlp_autoencoder"))


def test_template_renders_each_model_family_from_registry_without_flask(tmp_path):
    from types import SimpleNamespace
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    from ml_lab.experiments import ExperimentStore, select_experiment_dataset

    project = tmp_path / "project"
    project.mkdir()
    source = project / "samples.csv"
    source.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    store = ExperimentStore(project)
    experiment = select_experiment_dataset(store, source, feature_columns=["a", "b"])
    env = Environment(
        loader=FileSystemLoader("src/ml_lab/ui/templates"),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("ml_lab_ui/experiment_model.html")
    for family in list_model_families():
        name = family["id"]
        methods = list_model_methods(name)
        chosen = methods[-1]
        desc = describe_model_method(name, chosen["id"])
        page = template.render(
            request=SimpleNamespace(blueprint="test", endpoint="test.experiment_model"),
            url_for=lambda endpoint, **kwargs: "/" + endpoint,
            version="test", experiments=[experiment], experiment=experiment,
            families=list_model_families(), methods=methods, family=name,
            method=chosen["id"], method_name=chosen["name"],
            schema=desc, controls=visible_controls(desc),
            values={}, placeholder=control_placeholder,
            saved=False, data_error=None, error=None,
        )
        assert "Save model configuration" in page
        assert "ctl." in page
        assert "readonly" not in page.lower()
        assert "<select" in page
        # The five-tab workbench deliberately keeps tools independent.
        # Training is only offered after a model configuration is saved.
        assert "Tools are independent." in page
        assert "model-workflow-stages" in page
        assert "Train saved configuration" not in page
