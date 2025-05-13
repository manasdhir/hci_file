import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Binarizer, StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction App")

feature_labels={"age":"Age (years)","sex":"Sex (Male=1, Female=0)","bmi":"Body Mass Index (BMI)","bp":"Blood Pressure","s1":"Total Serum Cholesterol","s2":"Low-Density Lipoproteins (LDL)","s3":"High-Density Lipoproteins (HDL)","s4":"Thyroid Stimulating Hormone (TSH)","s5":"Blood Sugar Level (Serum)","s6":"Insulin Level"}
feature_ranges={"age":(0,100),"sex":(0,1),"bmi":(15,40),"bp":(60,180),"s1":(100,300),"s2":(50,200),"s3":(30,100),"s4":(0.3,5.0),"s5":(70,200),"s6":(2,30)}

@st.cache_data
def load_data():
 d=load_diabetes()
 X=pd.DataFrame(d.data,columns=d.feature_names)
 y=(Binarizer(threshold=np.median(d.target)).fit_transform(d.target.reshape(-1,1))).ravel()
 return X,y,d.feature_names

X,y,features=load_data()

st.sidebar.header("Model Settings")
test_size=st.sidebar.slider("Test Size (%)",10,50,20,5)/100
kernel=st.sidebar.selectbox("SVM Kernel",["linear","rbf","poly"])
C_val=st.sidebar.slider("Regularization (C)",0.1,10.0,1.0,0.1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=0)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=SVC(kernel=kernel,C=C_val,probability=True)
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)
score=model.score(X_test_scaled,y_test)

st.success(f"Model Accuracy: {score*100:.2f}%")
st.write("### Confusion Matrix")
cm=confusion_matrix(y_test,y_pred)
st.dataframe(pd.DataFrame(cm,
                         index=["Actual: No Diabetes", "Actual: Diabetes"],
                         columns=["Predicted: No Diabetes", "Predicted: Diabetes"]))

st.write("---")
st.write("## 🔍 Predict Diabetes")

cols=[]
for i in range(0,len(features),2):
 cols.append(st.columns(2))

input_data={}
for i,f in enumerate(features):
 c=cols[i//2][i%2]
 min_val,max_val=feature_ranges.get(f,(0.0,1.0))
 
 # Special handling for sex variable - should be either 0 or 1, not a range
 if f == "sex":
     label=f"{feature_labels.get(f,f)} (0=Female, 1=Male)"
     input_data[f]=c.selectbox(label, options=[0, 1], index=0)
 else:
     min_val_float = float(min_val)
     max_val_float = float(max_val)
     default_val = (min_val_float + max_val_float) / 2
     
     label=f"{feature_labels.get(f,f)} (Range: {min_val:.2f} to {max_val:.2f})"
     input_data[f]=c.number_input(label,
                                min_value=min_val_float,
                                max_value=max_val_float,
                                value=default_val)

if st.button("Predict"):
 df=pd.DataFrame([input_data])
 df_scaled=scaler.transform(df)
 prob=model.predict_proba(df_scaled)[0][1]
 st.write("### Risk Level")
 st.progress(prob)
 st.write(f"Predicted Risk: **{prob*100:.1f}%**")
