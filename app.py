from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

app = Flask(__name__)
ps = PorterStemmer()

# Load the saved models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('spam_detector_final.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in text]
    return " ".join(y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        # 1. Preprocess
        transformed_sms = transform_text(message)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        
        return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000, debug=True)