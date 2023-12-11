import streamlit as st
import numpy as np
from joblib import load
import datetime as dt

# Load the trained models
regressor_max = load("regressor1.pkl")
regressor_min = load("regressor2.pkl")
regressor_date = load("regressor.pkl")

# Function to arrange date more accurately
def proper_arrange(x):
    integer_part, fractional_part = divmod(x, 1)
    if fractional_part > 0.5:
        return int(integer_part + 1)
    else:
        return int(integer_part)

# Function to convert day of the year to day and month
def convert_to_date(day_of_year):
    date_object = dt.datetime.strptime(str(day_of_year), "%j")
    return date_object.strftime("%B %d")  # Adjust the format as needed

# Streamlit app
def main():
    st.markdown("<h1 style='color: yellow; text-align: center;'>TEMPERATURE AND DATE PREDICTOR</h1>", unsafe_allow_html=True)

    # Create two columns layout
    col1, col2 = st.columns(2)

    # Temperature Predictor in the first column
    with col1:
        # Section for Temperature Prediction
        st.header("Temperature Predictor")

        # User input for planetary degrees and date
        planets = ["Sun", "Moon"]

        input_features_temp = []
        for i, planet in enumerate(planets):
            # Add a unique key based on the index i
            key = f"{planet}_degree_{i}"
            degree = st.number_input(f"Enter {planet} Degree:", key=key, min_value=0.0, max_value=360.0, value=180.0, step=1.0)
            input_features_temp.append(degree)

        day_temp = st.number_input("Enter Day:", min_value=1, max_value=31, value=15)
        month_temp = st.number_input("Enter Month:", min_value=1, max_value=12, value=6)
        year_temp = st.number_input("Enter Year:", min_value=1900, max_value=2100, value=2023)

        # Make temperature prediction
        input_data_temp = np.array([input_features_temp + [day_temp, month_temp, year_temp]])
        max_temp_prediction = regressor_max.predict(input_data_temp)[0]
        min_temp_prediction = regressor_min.predict(input_data_temp)[0]

        # Display temperature predictions in red font
        st.subheader("Temperature Predictions:")
        st.markdown(f"Predicted Max Temperature: <span style='color: red;'>{round(max_temp_prediction, 2)} °C</span>", unsafe_allow_html=True)
        st.markdown(f"Predicted Min Temperature: <span style='color: red;'>{round(min_temp_prediction, 2)} °C</span>", unsafe_allow_html=True)


    with col2:
        # Section for Date Prediction
        st.header("Date Predictor")

        # User input for features
        #st.subheader("Input Features:")
        feature_names_date = ['Max Temperature', 'Min Temperature', 'Sun Degree', 'Moon Degree', 'Mars Degree', 'Mercury Degree', 'Jupiter Degree', 'Venus Degree', 'Saturn Degree', 'Rahu Degree', 'Ketu Degree']  # Update with your actual feature names
        input_features_date = []

        for i, feature in enumerate(feature_names_date):
            # Add a unique key based on the index i
            key = f"{feature}_input_{i}"
            value = st.number_input(f"Enter {feature}:", key=key, min_value=0.0, max_value=360.0, value=180.0, step=1.0)
            input_features_date.append(value)

        # Make date prediction
        input_data_date = np.array([input_features_date])
        y_pred_date = regressor_date.predict(input_data_date)[0]

        # Adjust predicted date using proper_arrange function
        predicted_date = proper_arrange(y_pred_date)

        # Convert day of the year to day and month
        converted_date = convert_to_date(predicted_date)

        # Display predicted date with color
        st.subheader("Predicted Date:")
        st.markdown(f"The predicted date is <span style='color: red;'>{converted_date}</span> of the year.", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
