import numpy as np
from scipy.special import softmax
from tensorflow.keras import optimizers, losses
from transformers import BertTokenizer, TFBertForSequenceClassification
from flask import Flask, request, render_template
from lime.lime_text import LimeTextExplainer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained("bert_model", local_files_only=True)
opt = optimizers.Adam(learning_rate=3e-5)
scce = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=opt, loss=scce, metrics='accuracy')
input_names = ['input_ids', 'token_type_ids', 'attention_mask']

class_names = {0: 'non-disaster', 1: 'disaster'}
LIME_explainer = LimeTextExplainer(class_names=class_names)

app = Flask(__name__, template_folder='template')


@app.route('/')
def my_index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict_tweet():
    """
    This function takes arguments from the URL bar, creates an array to predict on,
    and lastly uses the pickled model to make a prediction which is returned to the client
    :return: string of prediction ('0' or '1')
    """
    input_text = request.form['input_text']
    tokenized_tweet = tokenizer(input_text)
    logits = model.predict({k: np.array(tokenized_tweet[k])[None] for k in input_names})[0]
    scores = softmax(logits)
    pred = round(100 * scores.flatten()[1], 2)
    # return render_template('index.html', prediction=pred)

    exp = LIME_explainer.explain_instance(input_text, predictor, num_features=len(input_text.split()),
                                          top_labels=1, num_samples=100).as_html()

    if pred >= 50:
        return render_template('index.html', exp=exp, prediction=f'Emergency; confidence ({pred}%)')
    else:
        return render_template('index.html', exp=exp, prediction=f'Non-emergency; confidence ({100 - pred}%)')


def predictor(text):
    examples = []
    for example in text:
        examples.append(tokenizer(example, max_length=128, padding='max_length', truncation=True))

    results = []
    for example in examples:
        outputs = model.predict({k: np.array(example[k])[None] for k in input_names})
        logits = outputs[0]
        logits = softmax(logits)
        results.append(logits[0])

    return np.array(results)


if __name__ == '__main__':
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()