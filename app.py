import streamlit as st
import sklearn
import pandas as pd
st.title('spam detecter interface')
user_input = st.text_area('enter the email to classify')

df = pd.read_csv("spam.csv", encoding = 'ISO-8859-1')
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.rename(columns={'v1':'Target','v2':'Features'},inplace=True)
from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
df['Target'] = Encoder.fit_transform(df['Target'])
X = df['Features']
y = df['Target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer()
X_train_vec = Vectorizer.fit_transform(X_train)
X_test_vec = Vectorizer.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
spam_detector = MultinomialNB()
spam_detector.fit(X_train_vec,y_train)

def prediction(email):
    vectorized_email = Vectorizer.transform([email]) # Vectorizing The Email Content
    prediction = spam_detector.predict(vectorized_email) # Making Predictions From The Model
    return prediction[0]

if st.button('predict'):
    result = prediction(user_input)
    if result == 1:
        st.error('this email is classified as spam')
    else:
        st.success('this email is classified as ham.')
        

