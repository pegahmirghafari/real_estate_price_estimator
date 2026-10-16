import requests
import streamlit as st

st.set_page_config(
    page_title="Seattle Home Value Estimator",
    page_icon="🏠",
    layout="centered",
)

# Default to the local FastAPI server
API_URL = "http://localhost:8000"



API_URL = API_URL.rstrip("/")
# Temporary debugging message

st.title("Seattle Home Value Estimator")

st.write(
    "Enter the property information below. "
    "Missing values will be estimated automatically."
)

zipcode = st.text_input(
    "Zipcode",
    placeholder="98101 to 98199",
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=0.0,
    value=None,
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=0.0,
    value=None,
)

sqft_living = st.number_input(
    "Living area (sq ft)",
    min_value=0.0,
    value=None,
)

sqft_lot = st.number_input(
    "Lot size (sq ft)",
    min_value=0.0,
    value=None,
)

floors = st.number_input(
    "Floors",
    min_value=0.0,
    value=None,
)

sqft_above = st.number_input(
    "Above-ground area (sq ft)",
    min_value=0.0,
    value=None,
)

sqft_basement = st.number_input(
    "Basement area (sq ft)",
    min_value=0.0,
    value=None,
)


if st.button("Estimate Home Value", type="primary"):
    if not zipcode:
        st.error("Zipcode is required.")
    else:
        payload = {
            "zipcode": zipcode,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement,
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10,
            )

            if response.ok:
                result = response.json()

                st.success("Estimated Home Value")

                st.metric(
                    "Predicted Price",
                    f"${result['predicted_price']:,.0f}",
                )
            else:
                st.error(
                    f"Prediction failed: {response.text}"
                )

        except requests.RequestException as e:
            st.error(
                f"Unable to connect to the prediction service: {e}"
                )
