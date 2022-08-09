import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle


app = Flask(__name__)
model1 = pickle.load(open('decision_tree.pkl','rb'))
model2 = pickle.load(open('linearregression.pkl','rb')) 

#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp1 = float(request.args.get('exp1'))
    exp2 = float(request.args.get('exp2'))

    Model = (request.args.get('Model'))

    if Model=="Random Forest Classifier":
      prediction = model1.predict([[exp1,exp2]])
      return render_template('index.html', prediction_text='Model  has predicted Food Demand : {}'.format(prediction))

    elif Model=="Linear Classifier":
      prediction = model1.predict([[exp1,exp2]])
      return render_template('index.html', prediction_text='Model  has predicted Food Demand : {}'.format(prediction))
      
    elif Model=="Desion Forest Classifier":
      prediction = model1.predict([[exp1,exp2]])
      return render_template('index.html', prediction_text='Model  has predicted Food Demand : {}'.format(prediction))

    elif Model=="NLP Classifier":
      prediction = model1.predict([[exp1,exp2]])
      return render_template('index.html', prediction_text='Model  has predicted Food Demand : {}'.format(prediction))

    elif Model=="KNN Classifier":
      prediction = model1.predict([[exp1,exp2]])
      return render_template('index.html', prediction_text='Model  has predicted Food Demand : {}'.format(prediction))

    

if __name__ == "__main__":
    app.run(debug=True)    
        
    


