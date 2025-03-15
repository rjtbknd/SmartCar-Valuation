import subprocess
import sys

# Try to import plotly. If not found, attempt to install it using the --user flag.
try:
    import plotly.express as px
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "plotly"], check=True)
    import plotly.express as px

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Regression models and metrics
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

# Configure Streamlit
st.set_page_config(page_title="Used Car Analyzer", layout="wide")
st.title(':car: SmartCar Valuation :dollar:')
st.header("Data-Driven Used Car Price Analysis & Prediction")
st.markdown("---")
st.image('car_header.jpg', use_column_width=True)
st.markdown("---")

########################################
# 1. Data Loading & Preprocessing
########################################

@st.cache_data
def load_data():
    df = pd.read_csv('used_cars_dataset_v2_edited.csv')
    
    # Drop columns not needed and rows with missing crucial values
    df = df.drop(columns=['AdditionInfo', 'PostedDate'])
    df = df.dropna(subset=['kmDriven', 'AskPrice'])
    
    # Remove outliers using IQR for Age, AskPrice, and kmDriven
    Q1 = df[['Age', 'AskPrice', 'kmDriven']].quantile(0.25)
    Q3 = df[['Age', 'AskPrice', 'kmDriven']].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[['Age', 'AskPrice', 'kmDriven']] < (Q1 - 1.5 * IQR)) | 
              (df[['Age', 'AskPrice', 'kmDriven']] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return df

df = load_data()

########################################
# 2. Sidebar Filters Section
########################################

st.sidebar.header("Data Filters")

# Price range filter
price_range = st.sidebar.slider(
    "Price Range (₹)",
    int(df['AskPrice'].min()), 
    int(df['AskPrice'].max()),
    (int(df['AskPrice'].quantile(0.25)), int(df['AskPrice'].quantile(0.75)))
)

# Dynamic filter options from the data
transmission_options = sorted(df['Transmission'].unique())
fuel_options = sorted(df['FuelType'].unique())
owner_options = sorted(df['Owner'].unique())

transmission_filter = st.sidebar.multiselect(
    "Transmission Type",
    options=transmission_options,
    default=transmission_options
)

fuel_filter = st.sidebar.multiselect(
    "Fuel Type",
    options=fuel_options,
    default=fuel_options
)

owner_filter = st.sidebar.multiselect(
    "Ownership History",
    options=owner_options,
    default=owner_options
)

# Apply filters to the dataset
df_filtered = df[
    (df['AskPrice'] >= price_range[0]) & (df['AskPrice'] <= price_range[1]) &
    (df['Transmission'].isin(transmission_filter)) &
    (df['FuelType'].isin(fuel_filter)) &
    (df['Owner'].isin(owner_filter))
]

########################################
# 3. Top Number Cards Section
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
col3.metric("Average Age (in Yrs)", f"{avg_age:.1f}" if not np.isnan(avg_age) else "N/A")
col4.metric("Average km Driven", f"{avg_km:,.0f}" if not np.isnan(avg_km) else "N/A")
col5.metric("Average AskPrice (₹)", f"{avg_price:,.0f}" if not np.isnan(avg_price) else "N/A")

st.markdown("---")

########################################
# 4. Sub-Cards Section for Transmission, Owner, FuelType
########################################

def display_sub_cards(category, df_data):
    st.subheader(f"Total Cars by {category}")
    unique_vals = sorted(df_data[category].unique())
    cols = st.columns(len(unique_vals))
    for idx, val in enumerate(unique_vals):
        count_val = df_data[df_data[category] == val].shape[0]
        cols[idx].metric(label=val, value=count_val)

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

fig_feature = px.bar(
    feature_price, 
    x=feature, 
    y='AskPrice',
    color=feature,
    title=f'Average AskPrice by {feature}'
)
st.plotly_chart(fig_feature, use_container_width=True)

########################################
# 6. Additional Visualizations (Uniform Design) 
########################################

st.subheader("Additional Visualizations")

# Visualization 1: AskPrice vs Year by Transmission using Plotly 
avg_price_year_trans = df_filtered.groupby(['Year', 'Transmission'])['AskPrice'].mean().reset_index()
fig1 = px.bar(
    avg_price_year_trans, 
    x='Year', 
    y='AskPrice', 
    color='Transmission', 
    barmode='group',
    title='Average AskPrice vs Year by Transmission'
)
st.plotly_chart(fig1, use_container_width=True)

# Visualization 2: AskPrice vs kmDriven by FuelType using Plotly 
df_filtered['kmDriven_bin'] = pd.cut(df_filtered['kmDriven'], bins=10).astype(str)
avg_price_km_fuel = df_filtered.groupby(['kmDriven_bin', 'FuelType'])['AskPrice'].mean().reset_index()
fig2 = px.bar(
    avg_price_km_fuel, 
    x='kmDriven_bin', 
    y='AskPrice', 
    color='FuelType', 
    barmode='group',
    title='Average AskPrice vs kmDriven (Binned) by FuelType'
)
st.plotly_chart(fig2, use_container_width=True)

######################################## 
# 7. Price Prediction Section (Enhanced) 
######################################## 
st.markdown("---") 
st.subheader("Price Predictor") 
st.markdown(""" 
This section uses enhanced preprocessing with: 
- Outlier removal in key features 
- Feature scaling for numerical values 
- Optimized model parameters 
- Cross-validated performance metrics 
""") 

# Duplicate the original dataframe for regression (unfiltered) 
df_reg = df.copy() 

# Mapping dictionaries for conversion 
transmission_map = {'Automatic': 0, 'Manual': 1} 
fuel_map = {'Petrol': 1, 'Diesel': 2, 'Hybrid/CNG': 3, 'hybrid': 4} 

# Apply mapping and clean data 
df_reg['Transmission_Num'] = df_reg['Transmission'].map(transmission_map) 
df_reg['FuelType_Num'] = df_reg['FuelType'].map(fuel_map) 
df_reg = df_reg.dropna(subset=['Transmission_Num', 'FuelType_Num'])

# Regression setup 
features = ['Age', 'kmDriven', 'Transmission_Num', 'FuelType_Num']
target = 'AskPrice' 
X = df_reg[features] 
y = df_reg[target] 

# Split data and scale features 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.11, random_state=42) 
scaler = StandardScaler() 
num_features = ['Age', 'kmDriven'] 
X_train_scaled = X_train.copy() 
X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features]) 
X_test_scaled = X_test.copy() 
X_test_scaled[num_features] = scaler.transform(X_test[num_features]) 

# Updated models with optimized parameters 
models = { 
    "Linear Regression": LinearRegression(), 
    "Lasso Regression": Lasso(alpha=0.01, max_iter=10000), 
    "Ridge Regression": Ridge(alpha=0.1, max_iter=10000), 
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42), 
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42), 
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=42) 
} 

# Model selection and evaluation 
selected_method = st.selectbox("Select Regression Method", list(models.keys())) 
model = models[selected_method] 

# Cross-validated metrics 
cv_results = cross_validate(model, X_train_scaled, y_train, cv=5, 
                              scoring=('r2', 'neg_mean_squared_error')) 
avg_r2 = cv_results['test_r2'].mean() 
avg_mse = -cv_results['test_neg_mean_squared_error'].mean() 

# Final training and test evaluation 
model.fit(X_train_scaled, y_train) 
y_pred = model.predict(X_test_scaled) 
test_r2 = r2_score(y_test, y_pred) 
test_mse = mean_squared_error(y_test, y_pred) 

# Prediction form 
st.markdown("### Predict AskPrice for a New Car") 
with st.form("prediction_form"): 
    input_age = st.number_input("Age (Years)", min_value=0.0, value=5.0, step=0.5) 
    input_km = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000) 
    input_transmission = st.selectbox("Transmission", list(transmission_map.keys())) 
    input_fuel = st.selectbox("Fuel Type", list(fuel_map.keys())) 
    submit_prediction = st.form_submit_button("Calculate")  
    
    if submit_prediction: 
        new_data = pd.DataFrame({ 
            "Age": [input_age], 
            "kmDriven": [input_km], 
            "Transmission_Num": [transmission_map[input_transmission]], 
            "FuelType_Num": [fuel_map[input_fuel]] 
        }) 
        
        # Scale numerical features 
        new_data_scaled = new_data.copy() 
        new_data_scaled[num_features] = scaler.transform(new_data[num_features]) 
        predicted_price = model.predict(new_data_scaled)[0] 
        
        st.success(f"Predicted AskPrice: ₹ {predicted_price:,.0f}") 
        st.info(f"Model Confidence (Cross-Val R²): {avg_r2:.3f}") 

########################################
# 8. Data Summary Section (Optional)
########################################

st.markdown("---")
st.subheader("Data Summary (Filtered)")
st.write(f"**Dataset Size (after filtering):** {df_filtered.shape[0]} rows")
st.write("**Columns:**", df_filtered.columns.tolist())
st.write("**Sample Data:**")
st.dataframe(df_filtered.head(10))
