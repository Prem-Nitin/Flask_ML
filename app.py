from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
import numpy as np
import Linear 
import multi
import dtree
import RandFor


app = Flask(__name__)




@app.route('/', methods = ['GET'])
def model():
    return render_template('model.html')


@app.route('/reg', methods = ['GET','POST'])
def predict():
    if request.method == 'POST':

        file_name = request.files['Files']
        file_path = './static/' + file_name.filename
        file_name.save(file_path)

        
        print(str(file_name))
        data = pd.read_csv(file_path)
        print(data)
        return redirect(url_for("linear", X = request.form['X'], dat = file_name.filename))

    else:
        return render_template('index.html')
    


@app.route('/linear<X>/<dat>', methods = ['GET'])
def linear(X,dat):
    print('X = ', X)
    C = Linear.predict(float(X), dat)

    return render_template('predict_linear.html', result = C)


@app.route("/multi", methods = ['GET','POST'])
def multi_reg_():
    if request.method == 'POST':
        file_name = request.files['Files']
        file_path = './static/' + file_name.filename
        file_name.save(file_path)

        print(str(file_name))
        data = pd.read_csv(file_path)
        print(data)

        x1 = request.form['x1']
        x2 = request.form['x2']
        x3 = request.form['x3']
        print(x1,x2,x3)

        return redirect(url_for("multi_cal",x1 = x1,x2 = x2,x3 = x3,dat = file_name.filename))

    else:
        return render_template('multi.html')

@app.route("/multi/<x1>/<x2>/<x3>/<dat>")
def multi_cal(x1, x2, x3, dat):
    print(x1,x2,x3,dat)
    C = multi.predict(x1,x2,x3,dat)
    return render_template('predict_multi.html',  result = C)

@app.route('/decTree', methods = ['GET','POST'])
def decision():
    if request.method == 'POST':
        file_name = request.files['file']
        file_path = './static/' + file_name.filename
        file_name.save(file_path)

        x1 = request.form['x1']
        x2 = request.form['x2']
        x3 = request.form['x3']
        x4 = request.form['x4']
        x5 = request.form['x5']
        x6 = request.form['x6']
        x7 = request.form['x7']
        x8 = request.form['x8']
        x9 = request.form['x9']
        x10 =request.form['x10']
        x11 =request.form['x11']


        data = pd.read_csv(file_path)
        print(data)

        return redirect(url_for('dec_predict', x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5, x6 = x6,
                                x7 = x7, x8 = x8, x9 = x9, x10 = x10, x11 = x11, dat = file_name.filename ))

    else:
        return render_template('DecisionTree.html')

@app.route('/dec_pred/<x1>/<x2>/<x3>/<x4>/<x5>/<x6>/<x7>/<x8>/<x9>/<x10>/<x11>/<dat>')
def dec_predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,dat):
    C = dtree.dec_predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,dat)
    return render_template('DecisionTree.html', result = C)



@app.route('/rand', methods = ['GET','POST'])
def rforest():
    if request.method == 'POST':
        file_name = request.files['file']
        file_path = './static/' + file_name.filename
        file_name.save(file_path)

        x1 = request.form['x1']
        x2 = request.form['x2']
        x3 = request.form['x3']
        x4 = request.form['x4']
        x5 = request.form['x5']
        x6 = request.form['x6']
        x7 = request.form['x7']
        x8 = request.form['x8']
        x9 = request.form['x9']
        x10 =request.form['x10']
        x11 =request.form['x11']


        data = pd.read_csv(file_path)
        print(data)

        return redirect(url_for('rand_predict', x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5, x6 = x6,
                                x7 = x7, x8 = x8, x9 = x9, x10 = x10, x11 = x11, dat = file_name.filename ))

    else:
        return render_template('random_forest.html')

@app.route('/rand/<x1>/<x2>/<x3>/<x4>/<x5>/<x6>/<x7>/<x8>/<x9>/<x10>/<x11>/<dat>')
def rand_predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,dat):
    C = RandFor.random_predict(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,dat)
    
    return render_template('random_forest.html', result = C)
    


if __name__ == '__main__' :
    app.run(port = 3000, debug = True)
