"""Minimal example of mounting ML Lab inside a larger Flask/PAH service."""

from flask import Flask

from ml_lab.flask_adapter import create_blueprint


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(
        create_blueprint(enable_experimental=False),
        url_prefix="/pah/services/ml-lab",
    )
    return app


if __name__ == "__main__":
    create_app().run(debug=True)
