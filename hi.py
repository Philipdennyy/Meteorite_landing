import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
st.set_page_config(
    page_title="Meteorite Hunter Dashboard",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_data
def load_data(filepath):
    df = pd.read_csv('Meteorite_Landings.csv')
    
    df.dropna(subset=['reclat', 'reclong', 'year', 'mass (g)'], inplace=True)

    df = df[df['year'] <= 2024] 
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
    df = df[df['year'] >= 1700]


    df = df[(df['reclat'] >= -90) & (df['reclat'] <= 90)]
    df = df[(df['reclong'] >= -180) & (df['reclong'] <= 180)]
    
    mass_cap = df['mass (g)'].quantile(0.99) 
    df = df[df['mass (g)'] < mass_cap]

    return df

def simplify_recclass(recclass):
    recclass_lower = str(recclass).lower()
    if 'iron' in recclass_lower:
        return 'Iron'
    elif 'chondrite' in recclass_lower or any(c in recclass_lower for c in ['h', 'l', 'll']):
        return 'Stony'
    elif 'pallasite' in recclass_lower or 'mesosiderite' in recclass_lower:
        return 'Stony-Iron'
    else:
        return 'Other'

@st.cache_resource
def train_model(df):
    df['class_group'] = df['recclass'].apply(simplify_recclass)
    
    features = ['mass (g)', 'year', 'reclat', 'reclong']
    target = 'class_group'

    X = df[features]
    y = df[target]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, le, accuracy



df = load_data('Meteorite_Landings.csv')
model, le, accuracy = train_model(df.copy())

st.sidebar.title("☄️ Meteorite Hunter")
st.sidebar.markdown("""
This interactive dashboard allows you to explore the fascinating world of meteorite landings across the globe.
Use the filters to visualize landings over time and use the machine learning model to predict a meteorite's type!
""")

st.sidebar.header("🗺️ Map Filters")
min_year, max_year = int(df['year'].min()), int(df['year'].max())
selected_year_range = st.sidebar.slider(
    "Select a year range to display:",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

filtered_df = df[(df['year'] >= selected_year_range[0]) & (df['year'] <= selected_year_range[1])]

st.title("Global Meteorite Landings Dashboard")
st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.metric("Total Landings Displayed", f"{filtered_df.shape[0]:,}")
col2.metric("Average Mass (g)", f"{filtered_df['mass (g)'].mean():,.2f}")
col3.metric("Model Accuracy", f"{accuracy:.2%}")


st.subheader("Interactive Map of Meteorite Landings")
st.markdown(f"Displaying landings from **{selected_year_range[0]}** to **{selected_year_range[1]}**.")

fig = px.scatter_geo(
    filtered_df,
    lat='reclat',
    lon='reclong',
    color='recclass',
    size='mass (g)',
    hover_name='name',
    hover_data={'recclass': True, 'mass (g)': ':,', 'year': True, 'reclat':False, 'reclong':False},
    projection="natural earth",
    title="Hover over points for details. Use the legend to filter by class.",
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0},
    legend_title_text='Meteorite Class'
)
st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.subheader("🤖 Predict Meteorite Type")
st.markdown("Enter the details of a new meteorite discovery to predict its class.")

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    mass_input = st.number_input("Mass (in grams)", min_value=1.0, value=1000.0, step=100.0)
    year_input = st.number_input("Year of Discovery", min_value=860, max_value=2024, value=2010)

with pred_col2:
    lat_input = st.number_input("Latitude (-90 to 90)", min_value=-90.0, max_value=90.0, value=0.0)
    lon_input = st.number_input("Longitude (-180 to 180)", min_value=-180.0, max_value=180.0, value=0.0)

if st.button("Predict Type", use_container_width=True, type="primary"):
    input_data = pd.DataFrame({
        'mass (g)': [mass_input],
        'year': [year_input],
        'reclat': [lat_input],
        'reclong': [lon_input]
    })

    prediction_encoded = model.predict(input_data)
    
    prediction_class = le.inverse_transform(prediction_encoded)[0]

    st.success(f"**Predicted Meteorite Class:** {prediction_class}")
    st.info("This prediction is based on a Random Forest model trained on the NASA dataset.")