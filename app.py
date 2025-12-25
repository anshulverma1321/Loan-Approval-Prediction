from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

classifier = joblib.load("model/loan_approval_classifier.pkl")
regressor = joblib.load("model/loan_amount_regressor.pkl")
clf_features = joblib.load("model/classifier_features.pkl")
reg_features = joblib.load("model/regressor_features.pkl")
threshold = joblib.load("model/approval_threshold.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        data = {
            "Gender": int(request.form["Gender"]),
            "Married": int(request.form["Married"]),
            "Dependents": int(request.form["Dependents"]),
            "Education": int(request.form["Education"]),
            "Self_Employed": int(request.form["Self_Employed"]),
            "ApplicantIncome": float(request.form["ApplicantIncome"]),
            "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
            "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
            "Credit_History": float(request.form["Credit_History"]),
            "Property_Area_Semiurban": int(request.form["Property_Area_Semiurban"]),
            "Property_Area_Urban": int(request.form["Property_Area_Urban"]),
        }

        df_input = pd.DataFrame([data])
        prob = classifier.predict_proba(df_input[clf_features])[0][1]

        if prob >= threshold:
            amt = regressor.predict(df_input[reg_features])[0] * 1000
            result = f"✅ Approved | Probability: {prob:.2f} | Amount: ₹{amt:,.0f}"
        else:
            result = f"❌ Rejected | Probability: {prob:.2f}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
