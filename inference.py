from flask import Flask, request, render_template
import pickle
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import os

app = Flask(__name__, template_folder='template')

infile = open('lr_clf.pkl', 'rb')
model = pickle.load(infile)
infile.close()

infile = open('vectorizer.pkl', 'rb')
vectorizer = pickle.load(infile)
infile.close()

c = make_pipeline(vectorizer, model)

class_names = {0: 'non-disaster', 1:'disaster'}
LIME_explainer = LimeTextExplainer(class_names=class_names)


@app.route('/')
def my_index():
    # open('static/interpret.html', 'w').close()
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict_tweet():
    exp = None
    input_text = request.form['input_text']
    text = list([input_text])
    LIME_exp = LIME_explainer.explain_instance(text[0], c.predict_proba)
    exp = LIME_exp.as_html()

    pred = round(100 * c.predict_proba(text)[0][1], 2)

    if pred >= 50:
        return render_template('index.html', exp=exp, prediction=f'Emergency; confidence ({pred}%)')
    else:
        return render_template('index.html', exp=exp, prediction=f'Non-emergency; confidence ({100-pred}%)')


if __name__ == '__main__':
    port = os.environ.get('PORT')

    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()
