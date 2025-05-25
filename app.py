from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetching all 24 features from the form
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        sg = float(request.form['sg'])
        al = float(request.form['al'])
        su = float(request.form['su'])
        rbc = int(request.form['rbc'])
        pc = int(request.form['pc'])
        pcc = int(request.form['pcc'])
        ba = int(request.form['ba'])
        bgr = float(request.form['bgr'])
        bu = float(request.form['bu'])
        sc = float(request.form['sc'])
        sod = float(request.form['sod'])
        pot = float(request.form['pot'])
        hemo = float(request.form['hemo'])
        pcv = float(request.form['pcv'])
        wc = float(request.form['wc'])
        rc = float(request.form['rc'])
        htn = int(request.form['htn'])
        dm = int(request.form['dm'])
        cad = int(request.form['cad'])
        appet = int(request.form['appet'])
        pe = int(request.form['pe'])
        ane = int(request.form['ane'])

        # Combine into one list for prediction
        inputs = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc,
                  sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]

        print("Form submitted successfully")
        print("Inputs:", inputs)

        features = np.array([inputs])
        prediction = model.predict(features)[0]

        result_text = "CKD Detected" if prediction == 1 else "No CKD Detected"
        return render_template('result.html', prediction=result_text)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
