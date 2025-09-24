
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Latin America Regression Analysis", layout="wide")

# =========================
# Sample Historical Data (Synthetic but representative)
# =========================

data = {
    "Year": list(range(1950, 2021)),
    "Brazil_Population": np.linspace(54, 213, 71) + np.random.normal(0, 3, 71),
    "Mexico_Population": np.linspace(28, 128, 71) + np.random.normal(0, 2, 71),
    "Argentina_Population": np.linspace(17, 45, 71) + np.random.normal(0, 1.5, 71),
    "Brazil_Income": np.linspace(1000, 9500, 71) + np.random.normal(0, 500, 71),
    "Mexico_Income": np.linspace(800, 8900, 71) + np.random.normal(0, 400, 71),
    "Argentina_Income": np.linspace(1200, 8700, 71) + np.random.normal(0, 450, 71)
}

df = pd.DataFrame(data)

# =========================
# UI Controls
# =========================

st.title("Latin America Historical Regression Analysis")

category = st.selectbox(
    "Select a category",
    ["Population", "Average income"]
)

countries = st.multiselect(
    "Select countries to display",
    ["Brazil", "Mexico", "Argentina"],
    default=["Brazil"]
)

poly_degree = st.slider("Select polynomial regression degree", 3, 6, 3)
interval = st.slider("Select graph interval (years)", 1, 10, 1)
future_years = st.slider("Extrapolate model into the future (years)", 0, 50, 10)

# Editable Data Table
st.subheader("Raw Historical Data")
edited_df = st.data_editor(df, num_rows="dynamic")
st.write("Editable data for transparency and exploration.")

# =========================
# Regression and Graphing
# =========================

fig, ax = plt.subplots(figsize=(10, 6))

for country in countries:
    y_col = f"{country}_{category.replace(' ', '')}"
    if y_col not in edited_df.columns:
        st.warning(f"No data for {category} in {country}.")
        continue

    X = edited_df["Year"].values.reshape(-1, 1)
    y = edited_df[y_col].values

    # Polynomial Regression
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)

    # Extrapolation
    X_future = np.arange(1950, 2020 + future_years + 1, interval).reshape(-1, 1)
    X_future_poly = poly.transform(X_future)
    y_future = model.predict(X_future_poly)

    # Plotting
    ax.scatter(X, y, label=f"{country} Data")
    ax.plot(X, y_pred, label=f"{country} Regression")
    ax.plot(X_future, y_future, linestyle="--", label=f"{country} Projection")

    # Display Equation
    coefs = model.coef_
    intercept = model.intercept_
    equation_terms = [f"{round(coefs[i], 2)}*x^{i}" for i in range(len(coefs))]
    equation = " + ".join(equation_terms) + f" + {round(intercept, 2)}"
    st.markdown(f"**{country} {category} Regression Equation:** {equation}")

    # Function Analysis (example: max/min detection)
    deriv = np.polyder(np.poly1d(np.flip(coefs)))  # flip since sklearn stores in ascending
    critical_points = deriv.r
    st.markdown(f"**{country} {category} Function Analysis:**")
    st.write(f"Critical points (possible maxima/minima): {critical_points}")

ax.set_title(f"{category} Regression Analysis")
ax.set_xlabel("Year")
ax.set_ylabel(category)
ax.legend()
st.pyplot(fig)

# =========================
# Rate of Change Calculator
# =========================
st.subheader("Average Rate of Change Calculator")
country_choice = st.selectbox("Select country", countries)
start_year = st.number_input("Start Year", min_value=1950, max_value=2100, value=1960)
end_year = st.number_input("End Year", min_value=1950, max_value=2100, value=2000)

if st.button("Calculate Rate of Change"):
    y_col = f"{country_choice}_{category.replace(' ', '')}"
    X = edited_df["Year"].values.reshape(-1, 1)
    y = edited_df[y_col].values
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    y_start = model.predict(poly.transform(np.array([[start_year]])))[0]
    y_end = model.predict(poly.transform(np.array([[end_year]])))[0]
    avg_rate = (y_end - y_start) / (end_year - start_year)

    st.success(f"Average rate of change between {start_year} and {end_year}: {avg_rate:.2f} units per year")
