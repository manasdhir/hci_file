import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import kagglehub
import plotly.express as px
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")

# --- Data Loading & Preprocessing ---
@st.cache_data
def load_preprocess():
    path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
    for root, dirs, files in __import__('os').walk(path):
        for file in files:
            if file.endswith('.csv'):
                df = pd.read_csv(f"{root}/{file}")
                break
    # map/clean
    df = df.dropna().astype({
        'sex':'int','cp':'int','fbs':'int','restecg':'int','exang':'int','target':'int'
    })
    return df

df = load_preprocess()
X = df.drop('target', axis=1)
y = df['target']

# --- Feature names mapping ---
feature_names = {
    'age': 'Age',
    'sex': 'Sex', 
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Cholesterol Level',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Rest ECG Results',
    'thalach': 'Max Heart Rate',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression',
    'slope': 'Slope of ST Segment',
    'ca': 'Number of Major Vessels',
    'thal': 'Thalassemia'
}

# --- Sidebar for Model Training ---
st.sidebar.header("🔧 Model Training Settings")
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5) / 100
n_estimators = st.sidebar.slider("Number of Trees", min_value=50, max_value=500, value=100, step=50)
max_depth = st.sidebar.slider("Max Tree Depth", min_value=3, max_value=20, value=10, step=1)
min_samples_split = st.sidebar.slider("Min Samples to Split", min_value=2, max_value=10, value=2, step=1)

# Train button in sidebar
if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        acc = model.score(X_test_s, y_test)
        
        # Calculate feature importance
        fi = pd.DataFrame({
            'Feature': [feature_names.get(col, col) for col in X.columns],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save in session state
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["accuracy"] = acc
        st.session_state["feature_importance"] = fi
        st.session_state["confusion_matrix"] = cm
        
        st.success(f"Model Trained Successfully! Accuracy: {acc*100:.2f}%")

# --- Main Area - Results and Prediction ---
# Show results if model exists in session state
if "model" in st.session_state:
    # Add toggles for visualizations
    with st.expander("📊 Model Analysis", expanded=True):
        viz_tabs = st.tabs(["Feature Importance", "Confusion Matrix"])
        
        with viz_tabs[0]:  # Feature Importance tab
            # Feature Importance
            fig = px.bar(
                st.session_state["feature_importance"], 
                x='Importance', 
                y='Feature', 
                orientation='h',
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:  # Confusion Matrix tab
            cm = st.session_state["confusion_matrix"]
            
            # Improved confusion matrix visualization
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted: Healthy', 'Predicted: Heart Disease'],
                y=['Actual: Healthy', 'Actual: Heart Disease'],
                hoverongaps=False,
                colorscale='RdBu',
                showscale=False
            ))
            
            # Add text annotations
            annotations = []
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    annotations.append(dict(
                        x=j, 
                        y=i,
                        text=str(cm[i, j]),
                        showarrow=False,
                        font=dict(color='white' if cm[i, j] > cm.max()/2 else 'black', size=14)
                     ))
            
            # Fix: Properly call the update_layout method
            fig.update_layout(
                annotations=annotations,
                xaxis=dict(title='Predicted'),
                yaxis=dict(title='Actual'),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Section
    st.write("---")
    st.subheader("🔮 Predict Heart Disease Risk")
    
    # User Inputs with descriptive names
    cols = st.columns(2)
    inputs = {}
    
    # Define ranges and descriptions for features
    ranges = {
        'age': (29, 77, "Age in years"),
        'trestbps': (94, 200, "Resting blood pressure (mm Hg)"),
        'chol': (126, 564, "Serum cholesterol (mg/dl)"),
        'thalach': (71, 202, "Maximum heart rate achieved"),
        'oldpeak': (0.0, 6.2, "ST depression induced by exercise"),
        'ca': (0, 3, "Number of major vessels")
    }
    
    maps = {
        'sex': ['Female (0)', 'Male (1)'],
        'cp': ['Typical Angina (0)', 'Atypical Angina (1)', 'Non-anginal Pain (2)', 'Asymptomatic (3)'],
        'fbs': ['< 120 mg/dl (0)', '> 120 mg/dl (1)'],
        'restecg': ['Normal (0)', 'ST-T Abnormality (1)', 'Left Ventricular Hypertrophy (2)'],
        'exang': ['No (0)', 'Yes (1)'],
        'slope': ['Upsloping (0)', 'Flat (1)', 'Downsloping (2)'],
        'thal': ['Normal (1)', 'Fixed Defect (2)', 'Reversible Defect (3)']
    }
    
    for i, col in enumerate(X.columns):
        c = cols[i % 2]
        
        # Display with descriptive name
        display_name = feature_names.get(col, col)
        
        if col in ranges:
            lo, hi, desc = ranges[col]
            with c:
                st.write(f"**{display_name}** ({desc})")
                
                # Consistent type handling
                lo_float = float(lo)
                hi_float = float(hi)
                default = (lo_float + hi_float) / 2
                
                if isinstance(lo, int) and isinstance(hi, int) and col != 'oldpeak':
                    inputs[col] = st.slider(
                        f"Range: {lo} to {hi}", 
                        int(lo), int(hi), int(default), 
                        step=1, key=f"slider_{col}"
                    )
                else:
                    inputs[col] = st.slider(
                        f"Range: {lo:.1f} to {hi:.1f}", 
                        lo_float, hi_float, default, 
                        step=0.1, key=f"slider_{col}"
                    )
        else:
            choices = maps[col]
            with c:
                st.write(f"**{display_name}**")
                inputs[col] = st.selectbox(
                    "Select value", 
                    range(len(choices)), 
                    format_func=lambda x: choices[x],
                    key=f"select_{col}"
                )
    
    # Prediction button
    if st.button("Predict Risk"):
        user_df = pd.DataFrame([inputs])
        user_s = st.session_state["scaler"].transform(user_df)
        model = st.session_state["model"]
        proba = model.predict_proba(user_s)[0][1]
        
        # Risk visualization
        st.subheader("Heart Disease Risk Assessment")
        
        # Create risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        if proba < 0.5:
            st.success("✅ Low Risk of Heart Disease")
        else:
            st.error("⚠️ High Risk of Heart Disease")
            
else:
    # If model not trained yet
    st.info("👈 Please configure and train the model using the sidebar to begin.")
