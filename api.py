from flask import Flask,request,jsonify
from flask_cors import CORS
from preprocessing import preprocess
from sklearn.externals import joblib

app = Flask(__name__)
CORS(app)

MODEL_FILE = "model.pkl"
mdl = joblib.load(MODEL_FILE)

@app.route("/predict",methods=['POST'])
def predict():
    doc = request.json["document"]
    q = request.json["question"]
    data, sentences = preprocess(doc, q)
    result = mdl.predict(data)
    result = result.tolist()[0]-1
    return sentences[result]

if __name__ == "__main__":
    app.run('0.0.0.0',port=5000)