import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Saudi Arabia Electricity Load Forecasting")

# Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
else:
    st.stop()

# تنظيف البيانات
if 'Indicator' in df.columns:
    df[['Region', 'Type', 'Unit']] = df['Indicator'].str.split(' - ', expand=True)
    df.drop(columns=['Unit', 'Indicator'], inplace=True)

# اختيار الأعمدة
date_col = st.selectbox("Select Date Column", df.columns)
value_col = st.selectbox("Select Load Column", df.columns)

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])
df = df.sort_values(by=date_col)

# فلترة
if "Region" in df.columns:
    region = st.selectbox("Select Region", df["Region"].unique())
    df = df[df["Region"] == region]

series = df.set_index(date_col)[value_col]

# 🔷 1. Descriptive Statistics
st.subheader("Descriptive Statistics")
st.write(series.describe())

# 🔷 2. ADF Test
st.subheader("ADF Test")

adf_result = adfuller(series.dropna())

st.write(f"ADF Statistic: {adf_result[0]}")
st.write(f"p-value: {adf_result[1]}")

if adf_result[1] < 0.05:
    st.success("Series is Stationary")
else:
    st.warning("Series is NOT Stationary")

# 🔷 3. Decomposition
st.subheader("Time Series Decomposition")

decomposition = seasonal_decompose(series, model='additive', period=12)

fig = decomposition.plot()
st.pyplot(fig)

# 🔷 4. Plot original
st.subheader("Time Series Plot")
fig2, ax2 = plt.subplots()
ax2.plot(series)
st.pyplot(fig2)

# 🔷 5. Model Comparison
st.subheader("Model Comparison (AIC)")

models = {
    "SARIMA(1,1,1)": ((1,1,1),(1,1,1,12)),
    "SARIMA(2,1,1)": ((2,1,1),(1,1,1,12)),
    "SARIMA(1,1,2)": ((1,1,2),(1,1,1,12))
}

results = {}

for name, params in models.items():
    try:
        model = SARIMAX(series,
                        order=params[0],
                        seasonal_order=params[1])
        fit = model.fit(disp=False)
        results[name] = fit.aic
    except:
        results[name] = None

results_df = pd.DataFrame(list(results.items()), columns=["Model", "AIC"])
st.write(results_df)

best_model_name = results_df.loc[results_df["AIC"].idxmin(), "Model"]
st.success(f"Best Model: {best_model_name}")

# 🔷 6. Forecast
st.subheader("Forecast")

steps = st.slider("Forecast Steps", 1, 24, 12)

if st.button("Run Forecast"):

    best_params = models[best_model_name]

    model = SARIMAX(series,
                    order=best_params[0],
                    seasonal_order=best_params[1])

    model_fit = model.fit()

    forecast_obj = model_fit.get_forecast(steps=steps)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    future_dates = pd.date_range(start=series.index[-1],
                                 periods=steps+1,
                                 freq='MS')[1:]

    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(series, label="Original")
    ax3.plot(future_dates, forecast, label="Forecast", color='red')

    ax3.fill_between(future_dates,
                     conf_int.iloc[:, 0],
                     conf_int.iloc[:, 1],
                     alpha=0.3)

    ax3.legend()
    st.pyplot(fig3)
