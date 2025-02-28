import streamlit as st
import pandas as pd
import requests
import os
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Hugging Face API Integration
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "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")
API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1"

def query_huggingface(payload):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def generate_recipe_huggingface(plant_type, height, ph, ec):
    prompt = f"Generate a nutrient solution recipe for {plant_type} at {height}cm, pH {ph}, EC {ec}."
    output = query_huggingface({"inputs": prompt})
    if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
        return output[0]["generated_text"]
    else:
        return f"Error generating recipe: {output.get('error', 'Unknown error occurred.')}"


def diagnose_health_huggingface(plant_type, ph, ec):
    prompt = f"Diagnose potential issues for {plant_type} at pH {ph}, EC {ec}."
    output = query_huggingface({"inputs": prompt})
    if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0] and output[0]["generated_text"]:

        return output[0]["generated_text"]
    else:
        return f"Error generating diagnosis: {output.get('error', 'Unknown error occurred.')}"


# Trefle API key (replace with your own)
TREFLE_API_KEY = os.getenv("TREFLE_API_KEY", "I6mVcTAV2cprUqdiDMEu94s5SI8uBUKxmdfkm6UXEbg")
TREFLE_BASE_URL = "https://trefle.io/api/v1"

# Function to fetch all plants from Trefle API
def fetch_all_plants():
    try:
        url = f"{TREFLE_BASE_URL}/plants"
        params = {
            "token": TREFLE_API_KEY,
            "page_size": 10000,
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("data"):
            return data["data"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching plant data: {e}")
    return []

# Fetch all plants and store in session state
if "all_plants" not in st.session_state:
    st.session_state.all_plants = fetch_all_plants()

# Extract plant names for dropdown
plant_names = [plant["common_name"] for plant in st.session_state.all_plants if plant.get("common_name")]

# Sidebar for navigation
page = st.sidebar.radio("Navigate", ["Home", "Growth Report"])

# Initialize session state for growth data
if "growth_data" not in st.session_state:
    st.session_state.growth_data = pd.DataFrame(columns=["Date", "Plant Type", "Height (cm)", "Leaf Count", "Root Health", "Water Level", "Nutrient pH", "EC"])

# Home Page
if page == "Home":
    st.header("🏠 Home")
    st.write("Track your hydroponic plant growth and optimize nutrient levels.")

    st.sidebar.header("Plant Settings")
    selected_plant = st.sidebar.selectbox("Select Plant", plant_names)

    def fetch_plant_data(plant_name):
        try:
            url = f"{TREFLE_BASE_URL}/plants"
            params = {
                "token": TREFLE_API_KEY,
                "filter[common_name]": plant_name,
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("data"):
                return data["data"][0]
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching plant data: {e}")
        return None

    if selected_plant:
        plant_data = fetch_plant_data(selected_plant)
        if plant_data:
            st.sidebar.write(f"### {plant_data['common_name']} Info")
            st.sidebar.write(f"**Family:** {plant_data['family']}")
            st.sidebar.write(f"**Scientific Name:** {plant_data['scientific_name']}")
            st.sidebar.write(f"**Description:** {plant_data.get('observations', 'No description available.')}")
        else:
            st.sidebar.error("Plant data not found or an error occurred while fetching data.")

    st.subheader("📈 Plant Growth Tracking")
    with st.form("growth_form"):
        date = st.date_input("Date")
        height = st.number_input("Height (cm)", min_value=0.0)
        leaf_count = st.number_input("Leaf Count", min_value=0)
        root_health = st.selectbox("Root Health", ["Healthy", "Moderate", "Poor"])
        water_level = st.number_input("Water Level (L)", min_value=0.0)
        nutrient_ph = st.number_input("Nutrient Solution pH", min_value=0.0, max_value=14.0, value=6.0)
        ec = st.number_input("Electrical Conductivity (EC) (mS/cm)", min_value=0.0)
        submitted = st.form_submit_button("Log Growth Data")

        if submitted:
            new_entry = pd.DataFrame({
                "Date": [pd.to_datetime(date)],
                "Plant Type": [selected_plant],
                "Height (cm)": [height],
                "Leaf Count": [leaf_count],
                "Root Health": [root_health],
                "Water Level": [water_level],
                "Nutrient pH": [nutrient_ph],
                "EC": [ec],
            })
            st.session_state.growth_data = pd.concat([st.session_state.growth_data, new_entry], ignore_index=True)
            st.success("Growth data logged successfully!")

    if not st.session_state.growth_data.empty:
        st.write("### Growth History")
        st.write(st.session_state.growth_data)

# Growth Report Page
elif page == "Growth Report":
    st.header("📊 Growth Report")
    st.write("Analyze the growth patterns of your hydroponic plants.")

    if not st.session_state.growth_data.empty:
        plant_types = st.session_state.growth_data["Plant Type"].unique()
        selected_plant_report = st.selectbox("Select Plant for Report", plant_types)
        filtered_data = st.session_state.growth_data[st.session_state.growth_data["Plant Type"] == selected_plant_report].copy()

        st.write(f"### Growth Data for {selected_plant_report}")
        st.write(filtered_data)

        if not filtered_data.empty:
            filtered_data["Growth Rate (cm/day)"] = filtered_data["Height (cm)"].diff() / pd.to_datetime(filtered_data["Date"]).diff().dt.days
            average_growth_rate = filtered_data["Growth Rate (cm/day)"].mean()
            st.write(f"**Average Growth Rate:** {average_growth_rate:.2f} cm/day")

        st.subheader("📉 Nutrient Usage Trends")
        if not filtered_data.empty:
            filtered_data["Nutrient Usage"] = filtered_data["EC"] * filtered_data["Water Level"]
            fig_nutrient_usage = px.line(filtered_data, x="Date", y="Nutrient Usage", title="Nutrient Usage Over Time")
            st.plotly_chart(fig_nutrient_usage)

        st.subheader("🔮 Predictive Growth Modeling")
        if not filtered_data.empty and len(filtered_data) > 1:
            filtered_data["Days"] = (filtered_data["Date"] - filtered_data["Date"].min()).dt.days
            X = filtered_data[["Days"]]
            y = filtered_data["Height (cm)"]
            model = LinearRegression()
            model.fit(X, y)
            future_days = np.arange(filtered_data["Days"].max() + 1, filtered_data["Days"].max() + 8).reshape(-1, 1)
            future_growth = model.predict(future_days)
            future_dates = pd.date_range(start=filtered_data["Date"].max() + pd.Timedelta(days=1), periods=7)
            predictions = pd.DataFrame({
                "Date": future_dates,
                "Predicted Height (cm)": future_growth,
            })
            st.write("### Predicted Growth for the Next 7 Days")
            st.write(predictions)
            st.plotly_chart(px.line(predictions, x="Date", y="Predicted Height (cm)", title="Predicted Growth Over Time"))

        st.subheader("🎨 Customize Report Layout")
        visualization_options = st.multiselect(
            "Select Visualizations to Display",
            ["Height Over Time", "Leaf Count Over Time", "Nutrient pH Over Time", "EC Over Time", "Growth Rate Over Time", "Root Health Trends", "Nutrient Usage Trends", "Predictive Growth Modeling"],
            default=["Height Over Time", "Leaf Count Over Time", "Nutrient pH Over Time", "EC Over Time"],
        )

        st.write("### Selected Visualizations")
        if not filtered_data.empty:
            if "Height Over Time" in visualization_options:
                st.plotly_chart(px.line(filtered_data, x="Date", y="Height (cm)", title="Height (cm) Over Time"))
            if "Leaf Count Over Time" in visualization_options:
                st.plotly_chart(px.line(filtered_data, x="Date", y="Leaf Count", title="Leaf Count Over Time"))
            if "Nutrient pH Over Time" in visualization_options:
                st.plotly_chart(px.line(filtered_data, x="Date", y="Nutrient pH", title="Nutrient pH Over Time"))
            if "EC Over Time" in visualization_options:
                st.plotly_chart(px.line(filtered_data, x="Date", y="EC", title="EC Over Time"))
            if "Growth Rate Over Time" in visualization_options:
                st.plotly_chart(px.line(filtered_data, x="Date", y="Growth Rate (cm/day)", title="Growth Rate Over Time"))
            if "Root Health Trends" in visualization_options:
                st.plotly_chart(px.bar(filtered_data, x="Date", y="Root Health", title="Root Health Over Time"))
            if "Nutrient Usage Trends" in visualization_options:
                st.plotly_chart(px.line(filtered_data, x="Date", y="Nutrient Usage", title="Nutrient Usage Over Time"))
            if "Predictive Growth Modeling" in visualization_options and len(filtered_data) > 1:
                st.write("### Predicted Growth for the Next 7 Days")
                st.write(predictions)
                st.plotly_chart(px.line(predictions, x="Date", y="Predicted Height (cm)", title="Predicted Growth Over Time"))

        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Growth Data as CSV",
            data=csv,
            file_name=f"{selected_plant_report}_growth_data.csv",
            mime="text/csv",
        )
    else:
        st.warning("No growth data available. Please log data on the Home page.")

    if not st.session_state.growth_data.empty:
        if st.button("Generate Nutrient Recipe"):
            try:
                plant_data = st.session_state.growth_data.iloc[-1]
                recipe = generate_recipe_huggingface(selected_plant_report, plant_data['Height (cm)'], plant_data['Nutrient pH'], plant_data['EC'])
                st.write(recipe)
            except Exception as e:
                st.error(f"Error generating recipe: {e}")

        if st.button("Get Health Diagnostics"):
            try:
                plant_data = st.session_state.growth_data.iloc[-1]
                diagnostics = diagnose_health_huggingface(selected_plant_report, plant_data['Nutrient pH'], plant_data['EC'])
                st.write(diagnostics)
            except Exception as e:
                st.error(f"Error generating diagnostics: {e}")

# Footer
st.write("---")
