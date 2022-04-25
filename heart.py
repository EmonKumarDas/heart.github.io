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
image = Image.open('https://media.istockphoto.com/photos/3d-render-medical-heart-rate-icon-doctor-or-cardiologist-cartoon-hand-picture-id1303694638?k=20&m=1303694638&s=612x612&w=0&h=PAwph0igLyAl2JeX4Z2Rr5bejgETUL2Zks-u1b_xpl8=')

st.image(image,caption='ML',use_column_width=True)

df = pd.read_csv('E:/python/heart_disease/Cleveland, Hungary, Switzerland, and Long Beach (1).csv')
st.subheader('Data Information:')
st.dataframe(df)
st.write(df.describe())
barchart = st.bar_chart(df)
x= df.iloc[:,:-1] 
y = df.iloc[:,13]
x_train,x_test,y_train,y_test =train_test_split(x,y,random_state=42, test_size=0.3,shuffle=True)
# sex,cp,trestbps,chol,fbs,restecg,
# thalach,exang,oldpeak,slope,ca,thal,target
def get_user_input():
    age = st.text_input('Age:-')
    sex = st.slider('Sex:-',0,1)
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

st.subheader('MODEL TEST ACCURACY SCORE FOR DECISION TREE:-')

st.write(str(accuracy_score(y_test,dt.predict(x_test)) * 100)+'%')

predict = dt.predict(user_input)

st.subheader('Classification:')
st.write(predict)

if predict==0:
    st.write('You Do not Have Heart Disease')
else:
    st.write("You Have Heart Disease")
