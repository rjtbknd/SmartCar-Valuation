import subprocess
import sys

# Install core dependencies if needed
def install_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)

# Install required packages (XGBoost removed)
install_package("scikit-learn", "sklearn")
install_package("plotly")
install_package("streamlit")
install_package("seaborn")
install_package("scipy")

# Import necessary modules
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              StackingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from scipy.stats import zscore

# Configure Streamlit
st.set_page_config(page_title="Used Car Analyzer", layout="wide")
st.title(':car: SmartCar Valuation :dollar:')
st.header("Data-Driven Used Car Price Analysis & Prediction")
st.markdown("---")
st.image('car_header.jpg', use_container_width=True)
st.markdown("---")

########################################
# Child Code Functions (Price Prediction)
########################################

def load_and_preprocess_data(csv_file):
    # Load the data
    df = pd.read_csv(csv_file)
    # Drop columns not needed and rows missing crucial values
    df = df.drop(columns=['AdditionInfo', 'PostedDate'])
    df = df.dropna(subset=['kmDriven', 'AskPrice'])
    
    # Remove outliers using the IQR method for Age, AskPrice, and kmDriven
    Q1 = df[['Age', 'AskPrice', 'kmDriven']].quantile(0.25)
    Q3 = df[['Age', 'AskPrice', 'kmDriven']].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[['Age', 'AskPrice', 'kmDriven']] < (Q1 - 1.5 * IQR)) |
              (df[['Age', 'AskPrice', 'kmDriven']] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df

def prepare_regression_data(df):
    # Map categorical features to numerical values
    transmission_map = {'Automatic': 0, 'Manual': 1}
    fuel_map = {'Petrol': 1, 'Diesel': 2, 'Hybrid/CNG': 3, 'hybrid': 4}
    df['Transmission_Num'] = df['Transmission'].map(transmission_map)
    df['FuelType_Num'] = df['FuelType'].map(fuel_map)
    df = df.dropna(subset=['Transmission_Num', 'FuelType_Num'])
    
    features = ['Age', 'kmDriven', 'Transmission_Num', 'FuelType_Num']
    target = 'AskPrice'
    X = df[features]
    y = df[target]
    return X, y

def get_models():
    models = {
        "Linear": LinearRegression(),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
        "Ridge": Ridge(alpha=0.1, max_iter=10000),
        "Decision_Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random_Forest": RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42),
        "Gradient_Boosting": GradientBoostingRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42),
        "Kernel_Ridge_Regression": KernelRidge(alpha=0.5, kernel='rbf', gamma=0.1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, max_depth=15, random_state=42),
        "Polynomial_Ridge": Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('ridge', Ridge(alpha=1.0))
        ]),
        "Stacked_Model_Advanced": StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42)),
                ('et', ExtraTreesRegressor(n_estimators=500, max_depth=15, random_state=42)),
                ('hgb', HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, random_state=42))
            ],
            final_estimator=Ridge(alpha=0.1)
        ),
        "NN_Basic": MLPRegressor(
             hidden_layer_sizes=(150,),
             activation='relu',
             solver='adam',
             early_stopping=True,
             random_state=35,
             max_iter=500,
             validation_fraction=0.2
        ),
        "NN_Deep": MLPRegressor(
             hidden_layer_sizes=(200, 150, 100),
             activation='relu',
             solver='adam',
             learning_rate='adaptive',
             random_state=35,
             max_iter=500
        ),
        "NN_Wide": MLPRegressor(
             hidden_layer_sizes=(250, 150),
             activation='relu',
             solver='adam',
             random_state=35,
             max_iter=500,
             early_stopping=True
        )
    }
    return models

########################################
# 1. Data Loading & Preprocessing (Visualizations)
########################################
@st.cache_data
def load_data():
    df = pd.read_csv('used_cars_dataset_v2_edited.csv')
    df = df.drop(columns=['AdditionInfo', 'PostedDate'])
    df = df.dropna(subset=['kmDriven', 'AskPrice'])
    # Remove outliers using IQR method
    Q1 = df[['Age', 'AskPrice', 'kmDriven']].quantile(0.25)
    Q3 = df[['Age', 'AskPrice', 'kmDriven']].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[['Age', 'AskPrice', 'kmDriven']] < (Q1 - 1.5 * IQR)) | 
              (df[['Age', 'AskPrice', 'kmDriven']] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df

df = load_data()

########################################
# 2. Sidebar Filters
########################################
st.sidebar.header("Data Filters")
price_range = st.sidebar.slider(
    "Price Range (₹)",
    int(df['AskPrice'].min()),
    int(df['AskPrice'].max()),
    (int(df['AskPrice'].quantile(0.25)), int(df['AskPrice'].quantile(0.75)))
)
transmission_options = sorted(df['Transmission'].unique())
fuel_options = sorted(df['FuelType'].unique())
owner_options = sorted(df['Owner'].unique())

transmission_filter = st.sidebar.multiselect("Transmission Type", options=transmission_options, default=transmission_options)
fuel_filter = st.sidebar.multiselect("Fuel Type", options=fuel_options, default=fuel_options)
owner_filter = st.sidebar.multiselect("Ownership History", options=owner_options, default=owner_options)

df_filtered = df[
    (df['AskPrice'] >= price_range[0]) & (df['AskPrice'] <= price_range[1]) &
    (df['Transmission'].isin(transmission_filter)) &
    (df['FuelType'].isin(fuel_filter)) &
    (df['Owner'].isin(owner_filter))
]

########################################
# 3. Overview Metrics
########################################
st.subheader("Overview Metrics (Based on Selected Filters)")
total_brands = df_filtered['Brand'].nunique()
total_models = df_filtered['model'].nunique()  # assuming "model" column exists
avg_age = df_filtered['Age'].mean() if not df_filtered.empty else np.nan
avg_km = df_filtered['kmDriven'].mean() if not df_filtered.empty else np.nan
avg_price = df_filtered['AskPrice'].mean() if not df_filtered.empty else np.nan

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Brands", total_brands)
col2.metric("Total Models", total_models)
col3.metric("Average Age (Yrs)", f"{avg_age:.1f}" if not np.isnan(avg_age) else "N/A")
col4.metric("Average km Driven", f"{avg_km:,.0f}" if not np.isnan(avg_km) else "N/A")
col5.metric("Average AskPrice (₹)", f"{avg_price:,.0f}" if not np.isnan(avg_price) else "N/A")
st.markdown("---")

########################################
# 4. Sub-Cards by Category
########################################
def display_sub_cards(category, df_data):
    st.subheader(f"Total Cars by {category}")
    unique_vals = sorted(df_data[category].unique())
    cols = st.columns(len(unique_vals))
    for idx, val in enumerate(unique_vals):
        cols[idx].metric(label=val, value=df_data[df_data[category] == val].shape[0])

display_sub_cards("Transmission", df_filtered)
st.markdown("---")
display_sub_cards("Owner", df_filtered)
st.markdown("---")
display_sub_cards("FuelType", df_filtered)
st.markdown("---")

########################################
# 5. Main Visualizations
########################################
st.subheader("Price Analysis by Vehicle Features")
feature = st.selectbox("Select Feature for Analysis", ['Transmission', 'FuelType', 'Owner', 'Brand'])
feature_price = df_filtered.groupby(feature)['AskPrice'].mean().reset_index()
if feature == 'Brand':
    feature_price = feature_price.sort_values(by='AskPrice', ascending=False)
fig_feature = px.bar(feature_price, x=feature, y='AskPrice', color=feature,
                      title=f'Average AskPrice by {feature}')
st.plotly_chart(fig_feature, use_container_width=True)

########################################
# 6. Additional Visualizations
########################################
st.subheader("Additional Visualizations")
avg_price_year_trans = df_filtered.groupby(['Year', 'Transmission'])['AskPrice'].mean().reset_index()
fig1 = px.bar(avg_price_year_trans, x='Year', y='AskPrice', color='Transmission',
              barmode='group', title='Average AskPrice vs Year by Transmission')
st.plotly_chart(fig1, use_container_width=True)

df_filtered['kmDriven_bin'] = pd.cut(df_filtered['kmDriven'], bins=10).astype(str)
avg_price_km_fuel = df_filtered.groupby(['kmDriven_bin', 'FuelType'])['AskPrice'].mean().reset_index()
fig2 = px.bar(avg_price_km_fuel, x='kmDriven_bin', y='AskPrice', color='FuelType',
              barmode='group', title='Average AskPrice vs kmDriven (Binned) by FuelType')
st.plotly_chart(fig2, use_container_width=True)

########################################
# 7. Price Prediction Section (Updated)
########################################
st.markdown("---")
st.subheader("Price Predictor")
st.markdown("""
This section uses enhanced preprocessing:
- Outlier removal
- Feature scaling
- Advanced regression models for improved accuracy
- Cross-validated performance metrics
""")

# Use child code functions to load and prepare regression data
df_reg = load_and_preprocess_data("used_cars_dataset_v2_edited.csv")
X, y = prepare_regression_data(df_reg)

# Standardize numerical features
scaler = StandardScaler()
num_features = ['Age', 'kmDriven']
X[num_features] = scaler.fit_transform(X[num_features])

# Get models from child code
models = get_models()

# Select regression method
selected_method = st.selectbox("Select Regression Method", list(models.keys()))
model = models[selected_method]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1128, random_state=42)

# Cross-validated metrics
cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))
avg_r2 = cv_results['test_r2'].mean()
avg_mse = -cv_results['test_neg_mean_squared_error'].mean()

if "Neural Network" in selected_method:
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    st.info("Neural Network Note: Training may take longer than traditional models.")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)

st.markdown("### Predict AskPrice for a New Car")
with st.form("prediction_form"):
    input_age = st.number_input("Age (Years)", min_value=0.0, value=5.0, step=0.5)
    input_km = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000)
    transmission_options = {'Automatic': 0, 'Manual': 1}
    fuel_options = {'Petrol': 1, 'Diesel': 2, 'Hybrid/CNG': 3, 'hybrid': 4}
    input_transmission = st.selectbox("Transmission", list(transmission_options.keys()))
    input_fuel = st.selectbox("Fuel Type", list(fuel_options.keys()))
    submit_prediction = st.form_submit_button("Calculate")
    
    if submit_prediction:
        new_data = pd.DataFrame({
            "Age": [input_age],
            "kmDriven": [input_km],
            "Transmission_Num": [transmission_options[input_transmission]],
            "FuelType_Num": [fuel_options[input_fuel]]
        })
        new_data[num_features] = scaler.transform(new_data[num_features])
        predicted_price = model.predict(new_data)[0]
        st.success(f"Predicted AskPrice: ₹ {predicted_price:,.0f}")
        st.info(f"Model Confidence (Cross-Val R²): {avg_r2:.4f}")
