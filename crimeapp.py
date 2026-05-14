# ------------- Import Libraries -------------

import pickle
import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Chicago Crime Dashboard",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('le_crime.pkl', 'rb') as f:
        le_crime = pickle.load(f)
    with open('le_location.pkl', 'rb') as f:
        le_location = pickle.load(f)
    return model, le_crime, le_location

model, le_crime, le_location = load_models()

# ---------------- LOAD & PREPARE DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv('Crimes-2017.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df['Month'] = df['Date'].dt.month
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df = df.dropna(subset=['Latitude', 'Longitude'])
    return df

df = load_data()

# ---------------- FILTER DATA (CACHED) ----------------
@st.cache_data
def get_filtered_data(district, filter_type, month_val=None, day_val=None):
    filtered = df[df['District'] == district]
    if filter_type == 'Month' and month_val is not None:
        filtered = filtered[filtered['Month'] == month_val]
    elif filter_type == 'Day of Week' and day_val is not None:
        filtered = filtered[filtered['DayOfWeek'] == day_val]
    return filtered

# ---------------- TITLE ----------------
st.title('🚨 Chicago Crime Prediction & Heatmap Dashboard')

# =========================================================
# DICTIONARIES
# =========================================================

month_name = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

district_names = {
    1: '1 - Central', 2: '2 - Wentworth', 3: '3 - Grand Crossing',
    4: '4 - South Chicago', 5: '5 - Calumet', 6: '6 - Gresham',
    7: '7 - Englewood', 8: '8 - Chicago Lawn', 9: '9 - Deering',
    10: '10 - Ogden', 11: '11 - Harrison', 12: '12 - Near West',
    14: '14 - Shakespeare', 15: '15 - Austin', 16: '16 - Jefferson Park',
    17: '17 - Albany Park', 18: '18 - Near North', 19: '19 - Town Hall',
    20: '20 - Lincoln', 22: '22 - Morgan Park', 24: '24 - Rogers Park',
    25: '25 - Grand Central'
}

week_names = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

# =========================================================
# PREDICTION SECTION
# =========================================================

st.header("🤖 Arrest Prediction System")

col1, col2 = st.columns(2)

with col1:
    selected_month = st.selectbox('Select Month', list(month_name.keys()))
    month = month_name[selected_month]
    hour = st.slider('Select Hour', 0, 23, 12)
    day_of_week = st.selectbox('Select Day', list(week_names.keys()))
    selected_district = st.selectbox('Select District', list(district_names.values()))
    district = int(selected_district.split(' - ')[0])

with col2:
    domestic = st.selectbox('Domestic Case', ['No', 'Yes'])
    crime_type = st.selectbox('Crime Type', le_crime.classes_)
    location = st.selectbox('Location Description', le_location.classes_)

if st.button('🔍 Predict Arrest Probability'):
    day_num = list(week_names.values())[list(week_names.keys()).index(day_of_week)]
    domestic_num = 1 if domestic == 'Yes' else 0
    crime_code = le_crime.transform([crime_type])[0]
    location_code = le_location.transform([location])[0]

    X = [[month, hour, day_num, district, domestic_num, crime_code, location_code]]
    y_proba = model.predict_proba(X)[0][1]
    st.success(f'🚔 Arrest Probability: {round(y_proba * 100, 2)}%')

# =========================================================
# AREA CRIME ANALYSIS
# =========================================================

st.header("📍 Area Crime Analysis")

area_district = st.selectbox('Choose District', list(district_names.values()), key='area_select')
area_district_num = int(area_district.split(' - ')[0])

filter_type = st.radio('Filter By', ['Full Year', 'Month', 'Day of Week'])

month_val = None
day_val = None

if filter_type == 'Month':
    selected_month_filter = st.selectbox('Select Month', list(month_name.keys()), key='month_filter')
    month_val = month_name[selected_month_filter]

elif filter_type == 'Day of Week':
    selected_day = st.selectbox('Select Day For Crime Analysis', list(week_names.keys()), key='week_day_filter')
    day_val = week_names[selected_day]

# ---------------- APPLY FILTERS BUTTON ----------------
if st.button('🔎 Apply Filters & Show Analysis'):
    st.session_state['show_analysis'] = True
    st.session_state['filter_params'] = (area_district_num, filter_type, month_val, day_val)

if st.session_state.get('show_analysis'):

    params = st.session_state['filter_params']
    filtered_df = get_filtered_data(*params)

    # ---- CRIME STATISTICS ----
    st.subheader("📊 Crime Statistics")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Total Crimes", len(filtered_df))
    with col4:
        st.metric("Arrest Cases", int(filtered_df['Arrest'].sum()))
    with col5:
        st.metric("Domestic Cases", int(filtered_df['Domestic'].sum()))

    if filtered_df.empty:
        st.warning("No crime data available for the selected filters.")
    else:

        # ---- CRIME TYPE DISTRIBUTION ----
        st.subheader("📈 Crime Type Distribution")
        crime_counts = filtered_df['Primary Type'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=crime_counts.values, y=crime_counts.index, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        # ---- CRIME MAP ----
        st.subheader("🗺 Crime Heatmap")
        map_center = [filtered_df['Latitude'].mean(), filtered_df['Longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=13)

        # Limit 2000 points for performance, sample if more
        heat_data = (
            filtered_df[['Latitude', 'Longitude']]
            .dropna()
            .sample(min(2000, len(filtered_df)), random_state=42)
            .values.tolist()
        )
        HeatMap(heat_data, radius=8, blur=10).add_to(m)
        st_folium(m, width=700)

        # ---- CORRELATION HEATMAP ----
        st.subheader("🔥 Correlation Heatmap")
        corr_cols = ['Month', 'Hour', 'DayOfWeek', 'District', 'Arrest', 'Domestic']
        available_cols = [c for c in corr_cols if c in filtered_df.columns and filtered_df[c].nunique() > 1]

        if len(available_cols) < 2:
            st.warning("Not enough variation in data to render correlation heatmap.")
        else:
            corr = filtered_df[available_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            plt.close(fig)
    st.success("✅ Crime Analysis Completed Successfully")

    # ---- DOWNLOAD ----
    st.subheader("⬇ Download Crime Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Crime Data",
        data=csv,
        file_name='filtered_crime_data.csv',
        mime='text/csv'
    )