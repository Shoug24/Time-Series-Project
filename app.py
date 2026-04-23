import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("Saudi Arabia Electricity Load Forecasting")

# تحميل البيانات من GitHub أو ملف
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a dataset to continue")
    st.stop()

# عرض البيانات
st.subheader("Dataset Preview")
st.write(df.head())

# اختيار الأعمدة
date_col = st.selectbox("Select Date Column", df.columns)
value_col = st.selectbox("Select Load Column", df.columns)

# تحويل التاريخ
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.sort_values(by=date_col)

# اختيار المنطقة (إذا موجود)
if "Region" in df.columns:
    region = st.selectbox("Select Region", df["Region"].unique())
    df = df[df["Region"] == region]

# رسم البيانات
st.subheader("Time Series Plot")
fig, ax = plt.subplots()
ax.plot(df[date_col], df[value_col])
ax.set_xlabel("Date")
ax.set_ylabel("Load")
st.pyplot(fig)

# اختيار عدد الخطوات
steps = st.slider("Forecast Steps (Months)", 1, 24, 12)

# زر التنبؤ
if st.button("Run Forecast"):

    model = SARIMAX(df[value_col],
                    order=(1,1,1),
                    seasonal_order=(1,1,1,12))

    model_fit = model.fit()

    forecast = model_fit.forecast(steps=steps)

    st.subheader("Forecast Results")
    st.write(forecast)

    # رسم التوقع
    fig2, ax2 = plt.subplots()
    ax2.plot(df[date_col], df[value_col], label="Original")
    
    future_dates = pd.date_range(start=df[date_col].iloc[-1], periods=steps+1, freq='M')[1:]
    ax2.plot(future_dates, forecast, label="Forecast")

    ax2.legend()
    st.pyplot(fig2)
