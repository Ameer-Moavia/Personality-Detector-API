from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle as pkl
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

personality_type = ['IE', 'NS','FT','JP']
useless_words = stopwords.words("english")
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]
b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]
lemmatiser = WordNetLemmatizer()

# Load trained models
models = {}
for personality in personality_type:
    model_filename = f"{personality}.pkl"
    with open(model_filename, 'rb') as f:
        models[personality] = pkl.load(f)

# Load CountVectorizer and TF-IDF Transformer
with open('count_vectorizer.pkl', 'rb') as f:
    count_vectorizer = pkl.load(f)

with open('tfidf_transformer.pkl', 'rb') as f:
    tfidf_transformer = pkl.load(f)

def translate_back(personality):
    # transform binary vector to mbti personality
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s
# Define function to preprocess text
def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
    list_personality = []
    list_posts = []
    
    for index, row in data.iterrows():
        posts = row['posts']

        # Remove URL links 
        temp = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)

        # Remove Non-words - keep only words
        temp = re.sub("[^a-zA-Z]", " ", temp)

        # Remove multiple spaces
        temp = re.sub(' +', ' ', temp).lower()

        # Remove repeating characters
        temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

        # Remove stop words
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split() if w not in useless_words])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split()])

        # Remove MBTI personality words from posts
        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t, "")

        list_posts.append(temp)
    return np.array(list_posts)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_text = data['text']
    mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [user_text]})
    
    # Preprocess user text
    preprocessed_text = pre_process_text(mydata, remove_stop_words=True, remove_mbti_profiles=True)
    
    # Transform preprocessed text
    X_cnt = count_vectorizer.transform(preprocessed_text)
    X_tfidf = tfidf_transformer.transform(X_cnt).toarray()
    
    # Make predictions and predict probabilities
    results = []
    max_probas = []
    sum_prob = 0.0
    for personality, model in models.items():
        prediction_proba = model.predict_proba(X_tfidf)  # Get prediction probabilities
        max_proba = max(prediction_proba[0])  # Extract maximum probability
        sum_prob += max_proba  # Accumulate sum of probabilities
        max_probas.append({personality: float(max_proba)})  # Append max probability to list
        prediction = model.predict(X_tfidf)
        results.append(int(prediction[0]))
    
    # Calculate average probability
    average_prob = sum_prob / len(models)
    
    return jsonify(translate_back(results), max_probas, (average_prob*100))

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
