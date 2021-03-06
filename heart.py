import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
st.write("HEART DISEASE PREDICTION USING MACHINE LEARNING!")
image = Image.open('hands_holding_animated_heart.jpg')

st.image(image,caption='ML',use_column_width=True)

df = pd.read_csv('Cleveland, Hungary, Switzerland, and Long Beach.csv')
x= df.iloc[:,:-1] 
y = df.iloc[:,13]
x_train,x_test,y_train,y_test =train_test_split(x,y,random_state=42, test_size=0.3,shuffle=True)
# sex,cp,trestbps,chol,fbs,restecg,
# thalach,exang,oldpeak,slope,ca,thal,target
def get_user_input():
    age = st.slider('Age:-',20,130)
    sex = st.slider('Sex:-',1,0)
    cp = st.slider('Cp:-',0,3)
    trestbps = st.slider('Trestbps:-',0,190)
    chol = st.slider('Chol:-',100,400)
    fbs = st.slider('Fbs:-',0,1)
    restecg = st.slider('Restecg:-',0,2)
    thalach = st.slider('Thalach:-',0,200)
    exang = st.slider('exang:-',0,1)
    oldpeak = st.slider('Oldpeak:-',0.0,5.0)
    slope = st.slider('slope:-',0,2)
    ca = st.slider('ca:-',0,4)
    thal = st.slider('thal:-',0,3)
    
    user_data = {'age':age,
                'sex':sex,
                'cp':cp,
                'trestbps':trestbps,
                'chol':chol,
                'fbs':fbs,
                'restecg':restecg,
                'thalach':thalach,
                'exang':exang,
                'oldpeak':oldpeak,
                'slope':slope,
                'ca':ca,
                'thal':thal
    }

    fertures = pd.DataFrame(user_data, index=[0])
    return fertures
user_input = get_user_input()
st.subheader('user input:')
st.write(user_input)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy',random_state=142,splitter='random',max_features='auto')
dt.fit(x_train,y_train)

predict = dt.predict(user_input)

st.subheader('Classification:')
st.write(predict)

if predict==0:
    st.write('You Do not Have Heart Disease')
else:
    st.write("You Have Heart Disease")
