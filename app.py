from flask import Flask, request
from joblib import load
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
with open('model.pkl', 'rb') as file:
    app.model = load(file)


@app.route('/predict_single')
def predict_single():

    age = request.args.get('age')
    sex = request.args.get('sex')
    pclass = request.args.get('pclass')
    relatives = request.args.get('relatives')
    x = np.array([age, sex, pclass, relatives])
    pred = app.model.predict(x.reshape(1, -1))
    return np.array_str(pred)


@app.route('/predict_many', methods=['POST'])
def predict_multi():

    if request.is_json:
        x = request.get_json()
        df = pd.read_json(x)
        pred = app.model.predict(df)
        return str(pred)

    else:
        return 'no json'


if __name__ == '__main__':
    app.run()
