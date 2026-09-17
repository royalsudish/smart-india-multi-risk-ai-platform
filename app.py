from flask import Flask, render_template
from backend.routes.api import api_bp

def create_app():
    app = Flask(
        __name__,
        template_folder="frontend/templates",
        static_folder="frontend/static",
        static_url_path="/static",
    )
    app.config["JSON_SORT_KEYS"] = False
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/")
    def home():
        return render_template("index.html")

    return app

if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
