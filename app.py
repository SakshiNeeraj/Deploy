from flask import Flask, render_template, url_for, request
import pickle
import re
import string

filename = 'nlp_model.pkl'
ensemble = pickle.load(open(filename, 'rb'))
vectorization=pickle.load(open('transform.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        news = news.lower()
        news = re.sub('\[.*?\]', '', news)
        news = re.sub("\\W", " ", news)
        news = re.sub('https?://\S+|www\.\S+', '', news)
        news = re.sub('<.*?>+', '', news)
        news = re.sub('[%s]' % re.escape(string.punctuation), '', news)
        news = re.sub('\n', '', news)
        news = re.sub('\w*\d\w*', '', news)
        data = [news]
        vect = vectorization.transform(data).toarray()
        my_prediction = ensemble.predict(vect)
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
        app.run(debug=True)

