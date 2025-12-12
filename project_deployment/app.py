from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
slr_model = pickle.load(open("SLR_model.pkl", "rb"))
mlr_model = pickle.load(open("MLR_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", page="home")

@app.route("/slr", methods=["GET", "POST"])
def slr():
    prediction = None
    if request.method == "POST":
        try:
            years = float(request.form["years"])
            prediction = slr_model.predict([[years]])[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", page="slr", prediction=prediction)

@app.route("/mlr", methods=["GET", "POST"])
def mlr():
    prediction = None
    if request.method == "POST":
        try:
            rnd = float(request.form["rnd"])
            admin = float(request.form["admin"])
            marketing = float(request.form["marketing"])
            state = int(request.form["state"])
            features = np.array([[rnd, admin, marketing, state]])
            prediction = mlr_model.predict(features)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", page="mlr", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
