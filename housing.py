import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="California House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction (California Housing)")

# Load California housing dataset
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    return df, housing.feature_names

df, feature_names = load_data()
st.write("### 📊 Sample of the Dataset", df.head())

# Sidebar controls
st.sidebar.header("🔧 Model Training Settings")
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5) / 100
fit_intercept = st.sidebar.selectbox("Fit Intercept?", options=[True, False])

# Train button
if st.sidebar.button("Train Model"):
    X = df.drop("PRICE", axis=1)
    y = df["PRICE"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.success("✅ Model Trained Successfully!")
    st.write(f"**Mean Squared Error on Test Data:** {mse:.2f}")

    st.session_state["model"] = model
    st.session_state["features"] = feature_names
    st.session_state["price_range"] = (min(y), max(y))
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred

# Prediction section
st.write("---")
st.write("## 🔮 Predict House Price")

# Feature input descriptions
feature_info = {
    "MedInc": ("Median Income (in 10k USD)", 0.0, 20.0),
    "HouseAge": ("Median House Age", 1, 60),
    "AveRooms": ("Average Rooms per Household", 1.0, 15.0),
    "AveBedrms": ("Average Bedrooms per Household", 0.5, 5.0),
    "Population": ("Block Population", 1, 5000),
    "AveOccup": ("Average Household Occupancy", 1.0, 10.0),
    "Latitude": ("Latitude", 32.0, 42.0),
    "Longitude": ("Longitude", -124.0, -114.0)
}

if st.sidebar.checkbox("Show Data Statistics"):
    st.write("### All Feature Ranges")
    
    # Create columns for feature ranges display
    range_cols = st.columns(2)
    
    for i, feature in enumerate(feature_names):
        col_idx = i % 2  # Alternate between the two columns
        min_val = df[feature].min()
        max_val = df[feature].max()
        with range_cols[col_idx]:
            st.write(f"**{feature}:** {min_val:.2f} to {max_val:.2f}")

    # Show Actual vs Predicted plot if model trained
    if "y_test" in st.session_state and "y_pred" in st.session_state:
        st.write("### 📈 Actual vs Predicted Prices (Test Set)")
        mse = mean_squared_error(st.session_state["y_test"], st.session_state["y_pred"])
        st.write(f"**Mean Squared Error on Test Data:** {mse:.2f}")
        fig, ax = plt.subplots()
        ax.scatter(st.session_state["y_test"], st.session_state["y_pred"], alpha=0.6, color="purple")
        ax.plot([st.session_state["y_test"].min(), st.session_state["y_test"].max()],
                [st.session_state["y_test"].min(), st.session_state["y_test"].max()],
                'r--', lw=2)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Actual vs Predicted House Prices")
        st.pyplot(fig)

# Input and prediction
if "model" in st.session_state:
    st.write("### 📥 Input Feature Values")
    
    # Create columns for paired inputs
    col_pairs = []
    for i in range(0, len(st.session_state["features"]), 2):
        col_pairs.append(st.columns(2))
    
    input_data = {}
    for i, feature in enumerate(st.session_state["features"]):
        col_idx = i // 2
        in_col_idx = i % 2
        
        label, min_val, max_val = feature_info[feature]
        min_val_float = float(min_val)
        max_val_float = float(max_val)
        default_val = (min_val_float + max_val_float) / 2
        
        with col_pairs[col_idx][in_col_idx]:
            val = st.number_input(f"{label}",
                                min_value=min_val_float,
                                max_value=max_val_float,
                                value=default_val)
            input_data[feature] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = st.session_state["model"].predict(input_df)[0]
        st.success(f"🏡 Predicted House Price: **${prediction * 100000:.2f}**")

        # Create a sleek visualization for predicted price position
        st.write("### 📉 Predicted Price Position")
        min_price, max_price = st.session_state["price_range"]
        median_price = df['PRICE'].median()
        
        # Create a more stylish figure
        fig, ax = plt.subplots(figsize=(10, 2.5), facecolor='#f9f9f9')
        
        # Create gradient background
        import matplotlib.colors as mcolors
        gradient = np.linspace(0, 1, 100)
        gradient = np.vstack((gradient, gradient))
        extent = [min_price, max_price, 0, 1]
        
        # Create a nice gradient from blue to green to red
        cmap = mcolors.LinearSegmentedColormap.from_list('price_gradient', 
                                                        ['#1E3F66', '#2E8B57', '#FFBF00', '#FF5733'])
        ax.imshow(gradient, aspect='auto', cmap=cmap, extent=extent, alpha=0.7)
        
        # Add markers for context
        ax.axvline(x=min_price, color='#444', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=max_price, color='#444', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=median_price, color='white', linestyle='-', alpha=0.9, linewidth=1.5)
        
        # Add the prediction marker
        ax.plot(prediction, 0.5, 'o', markersize=15, 
                markerfacecolor='white', markeredgecolor='#333', markeredgewidth=2)
        
        # Add labels
        ax.text(min_price, 0.8, f"Min: ${min_price*100000:.0f}", fontsize=9, color='white',
                bbox=dict(facecolor='#00000066', edgecolor='none', boxstyle='round,pad=0.3'))
        ax.text(max_price, 0.8, f"Max: ${max_price*100000:.0f}", fontsize=9, color='white',
                ha='right', bbox=dict(facecolor='#00000066', edgecolor='none', boxstyle='round,pad=0.3'))
        ax.text(median_price, 0.8, f"Median: ${median_price*100000:.0f}", fontsize=9, color='white',
                ha='center', bbox=dict(facecolor='#00000066', edgecolor='none', boxstyle='round,pad=0.3'))
        ax.text(prediction, 0.2, f"Prediction: ${prediction*100000:.0f}", fontsize=10, 
                ha='center', fontweight='bold', color='#333',
                bbox=dict(facecolor='white', edgecolor='#333', boxstyle='round,pad=0.3'))
        
        # Style the chart
        ax.set_yticks([])
        ax.set_xlim(min_price - 0.1, max_price + 0.1)
        ax.set_ylim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333333')
        ax.set_xlabel("House Price (in $100,000s)", fontsize=10, color='#333333')
        ax.set_title("Your Prediction in Context", fontsize=12, color='#333333', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.info("⚠️ Please train the model using the sidebar before making predictions.")
