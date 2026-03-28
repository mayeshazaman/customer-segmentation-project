import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from flask import Flask, request, render_template
from pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/form")
def form_page():
    return render_template("home.html", results="", cluster_id="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        recency   = int(request.form.get("recency", 0))
        frequency = int(request.form.get("frequency", 0))
        monetary  = float(request.form.get("monetary", 0.0))

        pipeline = PredictPipeline()
        cluster_id, cluster_label = pipeline.predict(recency, frequency, monetary)

        return render_template("home.html",
                               results=cluster_label,
                               cluster_id=int(cluster_id))
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
