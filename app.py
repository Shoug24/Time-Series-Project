import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("Saudi Arabia Electricity Load Forecasting")

# 🔷 Upload Data
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
else:
    st.warning("Please upload a dataset to continue")
    st.stop()

# 🔷 فصل Indicator
if 'Indicator' in df.columns:
    df[['Region', 'Type', 'Unit']] = df['Indicator'].str.split(' - ', expand=True)
    df.drop(columns=['Unit', 'Indicator'], inplace=True)

# 🔷 عرض البيانات
st.subheader("Dataset Preview")
st.write(df.head())

# 🔷 اختيار الأعمدة
date_col = st.selectbox("Select Date Column", df.columns)
value_col = st.selectbox("Select Load Column", df.columns)

# 🔷 تنظيف التاريخ
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])
df = df.sort_values(by=date_col)

# 🔷 فلترة Region
if "Region" in df.columns:
    region = st.selectbox("Select Region", df["Region"].unique())
    df = df[df["Region"] == region]

# 🔷 رسم البيانات
st.subheader("Time Series Plot")
fig, ax = plt.subplots()
ax.plot(df[date_col], df[value_col])
ax.set_xlabel("Date")
ax.set_ylabel("Load")
st.pyplot(fig)

# 🔷 عدد الأشهر للتنبؤ
steps = st.slider("Forecast Steps (Months)", 1, 24, 12)

# 🔷 Forecast
if st.button("Run Forecast"):

    model = SARIMAX(df[value_col],
                    order=(1,1,1),
                    seasonal_order=(1,1,1,12))

    model_fit = model.fit()

    # 🔥 التنبؤ + Confidence Interval
    forecast_obj = model_fit.get_forecast(steps=steps)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    # 🔷 تواريخ مستقبلية
    future_dates = pd.date_range(start=df[date_col].iloc[-1],
                                 periods=steps+1,
                                 freq='MS')[1:]

    # 🔷 رسم احترافي
    fig2, ax2 = plt.subplots(figsize=(10,5))

    # البيانات الأصلية
    ax2.plot(df[date_col], df[value_col], label="Original", color="blue")

    # التوقع
    ax2.plot(future_dates, forecast, label="Forecast", color="red")

    # 🔥 Confidence Interval
    ax2.fill_between(future_dates,
                     conf_int.iloc[:, 0],
                     conf_int.iloc[:, 1],
                     color='pink', alpha=0.3,
                     label="Confidence Interval")

    ax2.legend()
    ax2.set_title("Electricity Load Forecast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Load")

    st.pyplot(fig2)

    # 🔷 عرض النتائج في جدول
    st.subheader("Forecast Values")

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast.values,
        "Lower Bound": conf_int.iloc[:, 0].values,
        "Upper Bound": conf_int.iloc[:, 1].values
    })

    st.dataframe(forecast_df)
