import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

# Load custom CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Load the saved model
model = joblib.load('random_forest_model.pkl')

# Title of the app
st.title("Customer Segment Classifier App")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    # Education - Checkbox options
    st.write("Select Education Level:")
    education = st.selectbox("Education", ["Graduation", "Post Graduate", "Undergraduate"])
    education_mapping = {"Graduation": 0, "Post Graduate": 1, "Undergraduate": 2}
    education = education_mapping[education]

    # Income - Numeric input
    income = st.number_input("Income", min_value=0)

    # Kidhome - Numeric input
    kidhome = st.number_input("Number of Kids at Home", min_value=0)

    # Teenhome - Numeric input
    teenhome = st.number_input("Number of Teens at Home", min_value=0)

    # NumDealsPurchases - Numeric input
    num_deals = st.number_input("Number of Deals Purchased", min_value=0)

with col2:
    # Age - Numeric input
    age = st.number_input("Age", min_value=0)

    # Spent - Numeric input
    spent = st.number_input("Amount Spent", min_value=0)

    # Living With - Checkbox options
    st.write("Select Living With Status:")
    living_with = st.selectbox("Living With", ["Alone", "Together"])
    living_with = 0 if living_with == "Alone" else 1

    # Children - Numeric input
    children = st.number_input("Number of Children", min_value=0)

    # Family Size - Numeric input
    family_size = st.number_input("Family Size", min_value=0)

    # Total Promos - Numeric input
    total_promos = st.number_input("Total Promotions", min_value=0)

# Predict button
if st.button("Identify Customer Segment"):
    # Prepare the feature array for prediction
    features = np.array([[education, income, kidhome, teenhome, num_deals,
                          age, spent, living_with, children, family_size,
                          total_promos]])
    # Make prediction
    prediction = model.predict(features)
    # Display the prediction
    st.markdown(f"""
    <div class="prediction-result">
        Customer Segment: {prediction[0]}
        <br>
        <small>Model accuracy: 94.5%</small>
    </div>
    """, unsafe_allow_html=True)

# Visualizations:
st.header("Visualizations")

# Load the DataFrames
agglo_data = joblib.load('agglo_data.pkl')
df_agglo = joblib.load('df_agglo.pkl')

# Create 3D scatter plot
fig1 = px.scatter_3d(
    agglo_data,
    x='pc1',
    y='pc2',
    z='pc3',
    color='labels',
    color_continuous_scale=px.colors.sequential.Viridis,
    title="3D Cluster Visualization",
    labels={'pc1': 'PC 1', 'pc2': 'PC 2', 'pc3': 'PC 3'},
)

# Display the plot in Streamlit
st.plotly_chart(fig1, use_container_width=True)

# adding new record to df_agglo
new_record = {'Spent': spent, 'Income': income, 'labels': 4}  # label 4 to add unique color in visualization
df_agglo.loc[len(df_agglo)] = new_record

# Create two columns for the remaining plots
col1, col2 = st.columns(2)

with col1:
    # scatter plot with 'income' vs 'spent' and color based on 'labels'
    fig2 = px.scatter(
        df_agglo,
        x='Income',
        y='Spent',
        color='labels', 
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Income vs Spent with Clusters",
        labels={'Income': 'Income', 'Spent': 'Spent'},
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    # Create a strip plot (similar to a swarm plot)
    fig3 = px.strip(
        df_agglo,
        x='labels',
        y='Spent',
        title="Spent Distribution by Clusters",
        color='labels', 
        color_discrete_sequence=px.colors.sequential.Magma,
        labels={'Spent': 'Spent', 'labels': 'Clusters'},
    )
    st.plotly_chart(fig3, use_container_width=True)
