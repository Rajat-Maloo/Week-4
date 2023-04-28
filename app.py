import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def result():

    Weight= float(request.form['item_weight'])
    Fat_Content=float(request.form['item_fat_content'])
    Item_Visibility= float(request.form['item_visibility'])
    Item_Type= float(request.form['item_type'])
    MRP = float(request.form['item_mrp'])
    Year= float(request.form['outlet_establishment_year'])
    Outlet_Size= float(request.form['outlet_size'])
    Location= float(request.form['outlet_location_type'])
    Outlet_Type= float(request.form['outlet_type'])

    X= np.array([[ Weight,Fat_Content,Item_Visibility,Item_Type,MRP, Year,Outlet_Size,Location,Outlet_Type]])

    model_path=r'/Users/rajatmaloo/Documents/Internship/Week 4/models/xg.sav'

    model= joblib.load(model_path)

    Y_pred=model.predict(X)


    return jsonify({'Prediction of Sales': float(Y_pred)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)