import streamlit as st # Streamlit is vital here. It acts as the interface for the app, allowing users to interact with the data and methods.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from prophet import Prophet
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

# This is the Approximator app. 
# This app is built using Streamlit, a framework for building web applications in Python.
# This app is licensed under the MIT License.
# How does it work?

appExplanation = """
This app allows users to upload a CSV file containing x and y data points from univariate/multivariate time series data.
The app will then interpolate these points using a method of the user's choice.
The goal is extrapolating to predict future values.
"""

# Function to load data from a CSV file.
# Apart from loading data, this function checks if the file has two columns and if the second column contains numeric data.
# IMPORTANT: The labels (e.g. in Earth population, year) are named as 'x', while the second column (data points, e.g. in Earth population, the population itself) is named 'y'. 
# When using this app, ensure that your uploaded file has the first column as the 'x' and the second column as the 'y'.

creatorNote = """
Creator's Note:
The motivation behind this app is to provide a simple and user-friendly interface for users to perform interpolation and extrapolation on their **univariate time series data** — that is, datasets where a single variable evolves over time.

    I identified a need for a tool that allows users to easily visualize and predict future values based on historical data, without requiring deep mathematical or programming knowledge.

    This app is particularly useful in fields such as **social sciences**, where data availability is often limited, and yet the need for forecasting and trend analysis is crucial.

    - Supports both **CSV and Excel** formats, making it accessible to a wide range of users.  
    - Designed to be **intuitive** and focused on usability.  
    - Enables **prediction** even when data is sparse or spaced out (e.g., population measured every 5 years).

    In disciplines like **demography, economics, and sociology**, extrapolation and interpolation are often used to estimate future values (e.g., population growth, economic indicators) from past trends.  

    This app aims to make those techniques **more approachable**, especially for students, researchers, and policy analysts.

    Whether you're estimating future populations, projecting economic metrics, or simply exploring trends, this tool helps you do so with ease.
"""


def loadUnivariateData(file, type):
    try:
        if type == "csv":
            data = pd.read_csv(file)
        elif type == "xlsx":
            data = pd.read_excel(file)
        if data.shape[1] != 2:
            st.error("Your file must have only two columns.")
            return None
        if not pd.api.types.is_numeric_dtype(data.iloc[:, 1]):
            st.error("The second column must contain numeric data.")
            return None
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
        return x, y
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def loadMultivariateData(file, type):
    pass


# The Approximator app uses four different interpolation methods, with each method implemented as a separate function.
# For univariate time series data, the following methods are implemented:
# 1. PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
def pchip(x, y):
    pchip = interpolate.PchipInterpolator(x, y)
    return pchip


# 2. Piecewise Linear Interpolation
def piecewiseLinear(x, y):
    linear_interp = interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")
    return linear_interp


# 3. Cubic Spline Interpolation
def cubicSpline(x, y):
    spline = interpolate.CubicSpline(x, y)
    return spline


# 4. Least Squares Interpolation (with or without scaling and centering)
def leastSquares(x, y):
    mu = np.mean(x)
    sig = np.std(x)
    x_scaled = (x - mu) / sig
    coeff_scaled = np.polyfit(x_scaled, y, 5)
    def model(x_new):
        x_new_scaled = (x_new - mu) / sig
        return np.polyval(coeff_scaled, x_new_scaled)
    return model





def main():
    st.set_page_config(layout="wide")
    st.title("📈 Approximator – Social Data Time Series Interpolation")
    st.text(appExplanation)
    with st.expander("📘 Creator's Note"):
        st.markdown("""
    **The motivation behind this app** is to provide a simple and user-friendly interface for users to perform interpolation and extrapolation on their **univariate time series data** — that is, datasets where a single variable evolves over time.

    I identified a need for a tool that allows users to easily **visualize** and **predict future values** based on historical data, without requiring deep mathematical or programming knowledge.

    This app is particularly useful in fields such as **social sciences**, where data availability is often limited, and yet the need for forecasting and trend analysis is crucial.

    - Supports both **CSV and Excel** formats, making it accessible to a wide range of users.  
    - Designed to be **intuitive** and focused on usability.  
    - Enables **prediction** even when data is sparse or spaced out (e.g., population measured every 5 years).

    In disciplines like **demography, economics, and sociology**, extrapolation and interpolation are often used to estimate future values (e.g., population growth, economic indicators) from past trends.  

    This app aims to make those techniques **more approachable**, especially for students, researchers, and policy analysts.

    Whether you're estimating future populations, projecting economic metrics, or simply exploring trends, this tool helps you do so with ease.
        """)


    st.markdown("""
    📁 Upload a CSV or Excel file with two columns:
    - The first column should contain the independent variable (e.g. year).
    - The second column should contain the dependent variable (e.g. population).
    """)

    st.subheader("💬 Why do you want to do with our software?")
    motivation = st.text_area("Let us know your goal or the story behind your data:", placeholder="E.g., 'I want to forecast future population growth based on historical trends.'")

    if motivation:
        st.success("Thanks for sharing your motivation! Let's get started. 🚀")





    # Upload file and choose if data is univariate or multivariate
    uploaded_file = st.file_uploader("📁 Upload your CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
            data_type = "csv" if uploaded_file.name.endswith("csv") else "xlsx"
            result = loadUnivariateData(uploaded_file, data_type)

            if result is not None:
                x, y = result

                # Raw Data Display
                st.subheader("📊 Raw Data")
                df = pd.DataFrame({'x': x, 'y': y})
                st.dataframe(df)

                # Choose interpolation method
                method = st.selectbox("Which method will you use?", [
                    "Cubic Spline", "Piecewise Linear", "PCHIP", "Least Squares"
                ])

                # Predict a Y-value for a new X-value
                st.subheader("🔮 Predict Future Values")
                x_new = st.number_input("Enter a new x-value to predict", value=float(x[-1] + 1))

                # Interpolation/extrapolation calculation
                st.subheader("📊 Interpolation/Extrapolation Result")
                model = None
                if method == "Cubic Spline":
                    model = cubicSpline(x, y)
                elif method == "Piecewise Linear":
                    model = piecewiseLinear(x, y)
                elif method == "PCHIP":
                    model = pchip(x, y)
                elif method == "Least Squares":
                    model = leastSquares(x, y)

                y_new = model(x_new)

                st.success(f"Estimated y at x = {x_new}: {y_new:.3f}")

                # If the user wants to see the interpolation plot, it will be displayed.
                if st.checkbox("Show interpolation plot"):
                    st.subheader("📈 Interpolation Plot")
                    x_dense = np.linspace(min(x), max(x) + 5, 300)
                    y_dense = model(x_dense)
                    fig, ax = plt.subplots()
                    ax.plot(x, y, 'o', label='Original Data')
                    ax.plot(x_dense, y_dense, '-', label='Interpolation')
                    ax.plot(x_new, y_new, 'rx', label=f'Predicted Point ({x_new}, {y_new:.2f})')
                    ax.legend()
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    st.pyplot(fig)

                if st.checkbox("Show interactive plot:"):
                    st.subheader("Interactive plot")
                    fig = px.scatter(x=x, y=y, labels={'x': 'Year', 'y': 'Y-axis'}, title="Interactive Scatter Plot")
                    fig.add_scatter(x=[x_new], y=[y_new], mode='markers', marker=dict(color='red', size=10), name='Predicted Point')
                    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
    
# This code is designed to be run in a Python environment with the necessary libraries installed.


