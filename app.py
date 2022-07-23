from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET' :

        print("Inside GET method")
        return render_template("predict.html")


    if request.method == 'POST' :

        array_features=[float(x) for x in request.form.values()]
        np_array = np.array(array_features)
        input_data_reshaped = np_array.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)
        if (prediction[0]== 0):
            return render_template('predict.html', 
                                positive_result = 'The patient is not likely to have heart disease!')
        else:
            return render_template('predict.html', 
                                negative_result = 'The patient is likely to have heart disease!')

if __name__ == '__main__':
    app.run(debug=True)