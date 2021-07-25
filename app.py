from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
import numpy as np
import Linear 
import multi
import dtree
import RandFor


app = Flask(__name__)




## Select model
@app.route('/', methods = ['GET'])
def model():
    return render_template('model.html')


##Linear Regression
@app.route('/reg', methods = ['GET','POST'])
def predict():
    if request.method == 'POST':

        file_name = request.files['Files']
        file_path = './static/' + file_name.filename
        file_name.save(file_path)

        X = request.form['X']
        dat = file_name.filename

        C = Linear.predict(float(X),dat)
        
        return render_template('linearAlt.html', result = C)

    else:
        return render_template('linearAlt.html')
    

##Multiple Regression
@app.route("/multi", methods = ['GET','POST'])
def multi_reg_():
    if request.method == 'POST':

        file_name = request.files['Files']
        file_path = './static/' + file_name.filename
        file_name.save(file_path)

        values = [x for x in request.form.values()]
        user_input = [np.array(values)]
        dat = file_name.filename
        
        C = multi.predict(user_input,dat)

        return render_template('multiAlt.html',  result = C)
        
    else:
        return render_template('multiAlt.html')


##Decision Tree
@app.route('/decTree', methods = ['GET','POST'])
def decision():
    if request.method == 'POST':
        file_name = request.files['Files']
        file_path = './static/' + file_name.filename
        file_name.save(file_path)

        values = [x for x in request.form.values()]
        user_input = [np.array(values)]
        dat = file_name.filename

        C = dtree.dec_predict(user_input,dat)

        return render_template('decisionTreeAlt.html', result = C)

    else:
        return render_template('decisionTreeAlt.html')


#Random Forest
@app.route('/rand', methods = ['GET','POST'])
def rforest():
    if request.method == 'POST':

        file_name = request.files['Files']
        file_path = './static/' + file_name.filename
        file_name.save(file_path)

        values = [x for x in request.form.values()]
        user_input = [np.array(values)]
        dat = file_name.filename

        C = RandFor.random_predict(user_input,dat)
    
        return render_template('random_forest.html', result = C)

    else:
        return render_template('random_forest.html')
    

if __name__ == '__main__' :
    app.run(port = 3000, debug = True)
