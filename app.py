import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
import nltk

ps=PorterStemmer()

#1 preprocessing

def text_transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
            
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))

            
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.title('Email/Spam Classification')

input_msg=st.text_area('Enter the message')

if st.button("predict"):
    #preprocessing
    tranformed_sms=text_transform(input_msg)

    #vectorized

    vertorized_text=tfidf.transform([tranformed_sms])


    # model

    result=model.predict(vertorized_text)[0]


    # display

    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')
