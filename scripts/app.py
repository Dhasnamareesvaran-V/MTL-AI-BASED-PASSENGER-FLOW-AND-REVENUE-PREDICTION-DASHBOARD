import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import datetime
import matplotlib.pyplot as plt
import tempfile
import os
import pandas as pd

age_model = load_model(r"D:\PASSENGER_FLOW_PREDICTION\models\cv\age_regressor.keras")
gender_model = load_model(r"D:\PASSENGER_FLOW_PREDICTION\models\cv\age_gender_cnn.keras")
forecast_df = pd.read_csv(r"D:\PASSENGER_FLOW_PREDICTION\models\forecasting\forecast_revenue_prophet.csv")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def age_group(age):
    if age < 13: return "Child"
    elif age < 25: return "Young Adult"
    elif age < 45: return "Adult"
    elif age < 65: return "Middle Aged"
    else: return "Senior"

def get_today_forecast():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    today_forecast = forecast_df[forecast_df['ds'] == today]
    if not today_forecast.empty:
        predicted_revenue = today_forecast['yhat'].values[0]
    else:
        predicted_revenue = forecast_df.iloc[-1]['yhat']
    avg_fare = 25
    predicted_flow = predicted_revenue / avg_fare
    return predicted_flow, predicted_revenue

st.set_page_config(page_title="MTL Passenger Flow", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fbc2eb, #a6c1ee);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1,5])
with col1:
    st.image(r"data/assets/logo.jpg", width=70)
with col2:
    st.title("🚇 Metro Transportation Limited (MTL)")

st.markdown("**Open website to see real-time revenue and passenger in MTL (Metro Transportation Limited)**")

uploaded_files = st.file_uploader("Upload passenger images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    ages, genders = [], []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        img = preprocess_image(tmp_path)
        predicted_age = age_model.predict(img)[0][0]
        _, predicted_gender = gender_model.predict(img)
        gender_label = "Male" if np.argmax(predicted_gender[0]) == 0 else "Female"
        ages.append(predicted_age)
        genders.append(gender_label)
        os.remove(tmp_path)

    avg_age = float(np.mean(ages))
    age_grp = age_group(avg_age)
    male_count = genders.count("Male")
    female_count = genders.count("Female")
    age_groups = [age_group(a) for a in ages]
    age_group_counts = {grp: age_groups.count(grp) for grp in set(age_groups)}
    most_common_age_group = max(age_group_counts, key=age_group_counts.get)
    predicted_flow, predicted_revenue = get_today_forecast()
    day_of_week = datetime.datetime.now().strftime("%A")

    st.info(f"Today is: **{day_of_week}**")
    st.subheader("📊 Passenger Statistics (Aggregate)")
    st.success(f"Average Age: {avg_age:.1f} years ({age_grp})")
    st.success(f"Gender Distribution → Male: {male_count}, Female: {female_count}")
    st.success(f"Forecasted Passenger Flow: {predicted_flow:.0f}")
    st.success(f"Forecasted Revenue: ₹{predicted_revenue:,.0f}")
    st.success(f"Most Frequent Age Group Using Station: {most_common_age_group}")

    fig, ax = plt.subplots(figsize=(3,2))
    bars = ax.bar(["Flow","Revenue"], [float(predicted_flow), float(predicted_revenue)], color=["skyblue","lightgreen"])
    ax.set_title("Passenger Flow & Revenue", fontsize=9)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05*yval, f"{yval:,.0f}", ha="center", va="bottom", fontsize=7)
    st.pyplot(fig, use_container_width=False)

    fig_pie, ax_pie = plt.subplots(figsize=(2.5,2.5))
    ax_pie.pie([male_count, female_count], labels=["Male","Female"], autopct="%1.1f%%", colors=["lightblue","pink"], textprops={'fontsize':7})
    ax_pie.set_title("Gender Distribution", fontsize=9)
    st.pyplot(fig_pie, use_container_width=False)

    fig_age, ax_age = plt.subplots(figsize=(3,2))
    ax_age.bar(age_group_counts.keys(), age_group_counts.values(), color="orange")
    ax_age.set_title("Age Group Usage Frequency", fontsize=9)
    ax_age.set_ylabel("Number of Passengers", fontsize=7)
    st.pyplot(fig_age, use_container_width=False)

st.subheader("📈 Revenue Forecast Trend")
fig2, ax2 = plt.subplots(figsize=(4,2.5))
ax2.plot(pd.to_datetime(forecast_df['ds']), forecast_df['yhat'], label="Revenue Forecast", color="blue")
ax2.fill_between(pd.to_datetime(forecast_df['ds']), forecast_df['yhat_lower'], forecast_df['yhat_upper'], color="lightblue", alpha=0.3, label="Confidence Interval")
ax2.set_xlabel("Date", fontsize=7)
ax2.set_ylabel("Revenue (₹)", fontsize=7)
ax2.legend(fontsize=7)
st.pyplot(fig2, use_container_width=False)

st.subheader("📈 Passenger Flow Forecast Trend")
fig3, ax3 = plt.subplots(figsize=(4,2.5))
avg_fare = 25
ax3.plot(pd.to_datetime(forecast_df['ds']), forecast_df['yhat']/avg_fare, label="Passenger Flow Forecast", color="green")
ax3.set_xlabel("Date", fontsize=7)
ax3.set_ylabel("Passenger Count", fontsize=7)
ax3.legend(fontsize=7)
st.pyplot(fig3, use_container_width=False)
