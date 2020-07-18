from flask import Flask,render_template,request
import pickle
import numpy as np
model = pickle.load(open('model.pickle','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('result.html', prediction_text='Flower Class is {}'.format(prediction))

app.run(debug=True)


