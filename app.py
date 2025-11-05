from flask import Flask,request, url_for, redirect, render_template
import joblib
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("home.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features= [float(x) for x in request.form.values()]
    print(int_features,len(int_features))
    final4=[np.array(int_features)]
    model = joblib.load('model_rf.sav')
    predict = model.predict(final4)
    output = model.predict_proba(final4)

    if predict[0] == 1:
        return render_template('result.html',pred=f'The Transaction is Fraud!.\nProbability of Fraud is {output[0][1]}')
    else:
        return render_template('result.html',pred=f'The Transaction is Non Fraud!.\n Probability of Fraud is {output[0][1]}')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/notebook')
def notebook():
	return render_template('notebook.html')

if __name__ == '__main__':
    app.run(debug=True)
