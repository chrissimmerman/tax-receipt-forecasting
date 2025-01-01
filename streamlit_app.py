import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from pmdarima.arima import auto_arima
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Streamlit setup
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("Tax Receipts Forecasting Dashboard")

@st.cache_data(show_spinner="Loading Data...")
def get_df():
    df = pd.read_csv('final_data.csv')
    # Set Month-Year as index
    df['Month-Year'] = pd.to_datetime(df['Month-Year'])
    df.set_index('Month-Year', inplace=True)
    return df

@st.cache_data(show_spinner="Loading UCM Model...")
def get_ucm_model(df):
    real_gdp = df['Real GDP Interpolated']

    # Log-transform the Real GDP to stabilize variance and directly model growth rate
    #log_real_gdp = real_gdp.apply(lambda x: np.log(x))

    ucm_model = UnobservedComponents(real_gdp, level='rwdrift')
    ucm_fit = ucm_model.fit()

    return ucm_fit

@st.cache_data(show_spinner="Loading ARIMA Model...")
def get_arima_unrate_model(df):
    unemployment_rate = df['UNRATE_Proportion']
    arima_unrate_model = auto_arima(y=unemployment_rate, seasonal=False)
    return arima_unrate_model

@st.cache_data(show_spinner="Loading SARIMAX Model...")
def get_sarimax_model(df):
    SARIMAX_model = auto_arima(
        y=df[['Log Transformed Receipts']],
        X=df[['Real GDP Interpolated', 'UNRATE_Proportion']],
        m=12,
        D=1
    )

    return SARIMAX_model

def prepare_forecast_data(ucm_fit, arima_unrate_model, periods, gdp_multiplier=1.0, unemp_multiplier=1.0):
    # Create a DataFrame for future months
    forecast_df = pd.DataFrame(
        {
            "month_index": pd.date_range(df.index[-1], periods=periods, freq="MS").month,
        },
        index=pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=periods, freq="MS"),
    )

    # Forecast Exogenous Variables
    gdp_forecast = ucm_fit.get_forecast(steps=periods)
    unemp_forecast = arima_unrate_model.predict(n_periods=periods)

    # Add the forecasted values to the DataFrame
    forecast_df["Real GDP Interpolated"] = gdp_forecast.predicted_mean * gdp_multiplier
    forecast_df["UNRATE_Proportion"] = unemp_forecast * unemp_multiplier

    return forecast_df

def forecast_receipts(SARIMAX_model, forecast_df, periods):
    # Forecast tax receipts using SARIMAX
    fitted, confint = SARIMAX_model.predict(
        n_periods=periods,
        X=forecast_df[["Real GDP Interpolated", "UNRATE_Proportion"]],
        return_conf_int=True,
    )

    # Inverse log transformation
    fitted = np.exp(fitted)
    confint = np.exp(confint)

    return fitted, confint

def create_plot_data(fitted, confint, periods):
    # Full index for historical and forecasted data
    historical_index = df.index
    forecast_index = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=periods, freq="MS")

    # Historical actual receipts
    historical_actual = np.exp(df["Log Transformed Receipts"])

    # Forecasted receipts and confidence intervals
    fitted_series = pd.Series(fitted, index=forecast_index, name="Forecasted Receipts")
    lower_series = pd.Series(confint[:, 0], index=forecast_index, name="Lower Confidence Interval")
    upper_series = pd.Series(confint[:, 1], index=forecast_index, name="Upper Confidence Interval")

    # Combine historical and forecasted data
    plot_data = pd.DataFrame({
        "Actual Receipts": historical_actual,
        "Forecasted Receipts": fitted_series,
        "Lower Confidence Interval": lower_series,
        "Upper Confidence Interval": upper_series,
    })

    return plot_data


def display_forecast_plotly(plot_data, show_confidence_intervals=True):
    fig = go.Figure()

    # Add actual receipts line
    fig.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data["Actual Receipts"],
        mode='lines',
        name='Actual Receipts',
        line=dict(color='blue')
    ))

    # Add forecasted receipts line
    fig.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data["Forecasted Receipts"],
        mode='lines',
        name='Forecasted Receipts',
        line=dict(color='green', dash='solid')
    ))

    # Add confidence interval as a filled area
    if show_confidence_intervals:
        fig.add_trace(go.Scatter(
            x=list(plot_data.index) + list(plot_data.index[::-1]),
            y=list(plot_data["Upper Confidence Interval"]) + list(plot_data["Lower Confidence Interval"][::-1]),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Confidence Interval'
        ))

    # Customize layout
    fig.update_layout(
        title="Tax Receipts Forecast with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Receipts",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )

    # Display plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_exogenous_forecast(df, forecast_df):
    # Combine historical and forecasted data for Real GDP
    combined_gdp = pd.concat([
        df["Real GDP Interpolated"],
        forecast_df["Real GDP Interpolated"]
    ])
    
    # Convert the forecast start to an ordinal value
    forecast_start = df.index[-1].toordinal()

    # Adjust tick frequency (e.g., show one tick per year)
    tick_frequency = pd.date_range(combined_gdp.index.min(), combined_gdp.index.max(), freq="YS")

    # Create GDP Forecast Plot
    gdp_fig = go.Figure()
    gdp_fig.add_trace(go.Scatter(
        x=combined_gdp.index.map(lambda x: x.toordinal()),  # Convert x-axis to ordinal
        y=combined_gdp,
        mode='lines',
        name="Real GDP",
        line=dict(color='blue')
    ))
    
    # Add vertical line for forecast start
    gdp_fig.add_vline(
        x=forecast_start,
        line_dash="dash",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    gdp_fig.update_layout(
        title="Historical and Forecasted Real GDP",
        xaxis=dict(
            title=dict(text="Date"),
            type="linear",
            tickmode="array",
            tickvals=tick_frequency.map(lambda x: x.toordinal()),  # Add only year-start ticks
            ticktext=tick_frequency.strftime("%Y-%m"),
        ),
        yaxis=dict(title=dict(text="Real GDP")),
        template="plotly_white"
    )

    # Combine historical and forecasted data for Unemployment Rate
    combined_unemp = pd.concat([
        df["UNRATE_Proportion"],
        forecast_df["UNRATE_Proportion"]
    ])
    
    # Create Unemployment Rate Forecast Plot
    unemp_fig = go.Figure()
    unemp_fig.add_trace(go.Scatter(
        x=combined_unemp.index.map(lambda x: x.toordinal()),  # Convert x-axis to ordinal
        y=combined_unemp,
        mode='lines',
        name="Unemployment Rate",
        line=dict(color='green')
    ))
    
    # Add vertical line for forecast start
    unemp_fig.add_vline(
        x=forecast_start,
        line_dash="dash",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    unemp_fig.update_layout(
        title="Historical and Forecasted Unemployment Rate",
        xaxis=dict(
            title=dict(text="Date"),
            type="linear",
            tickmode="array",
            tickvals=tick_frequency.map(lambda x: x.toordinal()),  # Add only year-start ticks
            ticktext=tick_frequency.strftime("%Y-%m"),
        ),
        yaxis=dict(title=dict(text="Unemployment Rate")),
        template="plotly_white"
    )

    # Display both plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(gdp_fig, use_container_width=True)
    with col2:
        st.plotly_chart(unemp_fig, use_container_width=True)


def sarimax_forecast(SARIMAX_model, ucm_fit, arima_unrate_model, periods=24, gdp_multiplier=1.0, unemp_multiplier=1.0, show_confidence_intervals=True, start_date=None, end_date=None):
    # Prepare forecast data
    forecast_df = prepare_forecast_data(ucm_fit, arima_unrate_model, periods, gdp_multiplier, unemp_multiplier)

    # Forecast receipts
    fitted, confint = forecast_receipts(SARIMAX_model, forecast_df, periods)

    # Prepare data for plotting
    plot_data = create_plot_data(fitted, confint, periods)

    # Combine historical and forecasted data
    combined_data = pd.DataFrame({
        "Actual Receipts": np.exp(df["Log Transformed Receipts"]),
        "Forecasted Receipts": plot_data["Forecasted Receipts"],
        "Lower Confidence Interval": plot_data["Lower Confidence Interval"],
        "Upper Confidence Interval": plot_data["Upper Confidence Interval"],
    })

    # Filter the data based on the selected date range
    if start_date and end_date:
        combined_data = combined_data.loc[start_date:end_date]

    # Display forecast using Streamlit and Plotly
    display_forecast_plotly(combined_data, show_confidence_intervals)

    # Plot exogenous variables forecast
    plot_exogenous_forecast(df, forecast_df)

df = get_df()
ucm_fit = get_ucm_model(df)
arima_unrate_model = get_arima_unrate_model(df)
SARIMAX_model = get_sarimax_model(df)

# User input for the number of forecast periods
periods = st.sidebar.number_input(
    "Enter the number of periods to forecast:",
    min_value=1,
    max_value=120,
    value=24,
    step=1,
    help="Number of months for which you want the forecast."
)


show_confidence_intervals = st.sidebar.checkbox(
    "Show Confidence Intervals",
    value=True,
    help="Enable or disable confidence intervals in the forecast plot."
)

# Date range selection
st.sidebar.write("Select Date Range for Tax Receipts Graph")
min_date = df.index.min().date()
max_date = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=periods, freq="MS")[-1].date()

start_date = st.sidebar.date_input(
    "Start Date", 
    value=min_date, 
    min_value=min_date, 
    max_value=max_date, 
    help="Start date for the tax receipts graph."
)
end_date = st.sidebar.date_input(
    "End Date", 
    value=max_date, 
    min_value=min_date, 
    max_value=max_date, 
    help="End date for the tax receipts graph."
)

# Ensure start_date is before end_date
if start_date > end_date:
    st.sidebar.error("Start Date must be before End Date.")


# GDP adjustment option
gdp_adjustment = st.sidebar.selectbox(
    "Select GDP Adjustment:",
    options=["No Change", "10% Increase", "20% Increase", "30% Increase", "10% Decrease", "20% Decrease", "30% Decrease"],
    index=0,
    help="Choose a percentage adjustment for GDP to simulate policy or economic changes."
)

# GDP adjustment option
unemp_adjustment = st.sidebar.selectbox(
    "Select Unemployment Percentage Adjustment:",
    options=["No Change", "10% Increase", "20% Increase", "30% Increase", "10% Decrease", "20% Decrease", "30% Decrease"],
    index=0,
    help="Choose a percentage adjustment for Unemployment Percentage."
)

gdp_multiplier_map = {
    "No Change": 1.0,
    "10% Increase": 1.1,
    "20% Increase": 1.2,
    "30% Increase": 1.3,
    "10% Decrease": 0.9,
    "20% Decrease": 0.8,
    "30% Decrease": 0.7,
}

unemp_multiplier_map = {
    "No Change": 1.0,
    "10% Increase": 1.1,
    "20% Increase": 1.2,
    "30% Increase": 1.3,
    "10% Decrease": 0.9,
    "20% Decrease": 0.8,
    "30% Decrease": 0.7,
}

gdp_multiplier = gdp_multiplier_map[gdp_adjustment]
unemp_multiplier = unemp_multiplier_map[unemp_adjustment]

st.write("Forecasted Tax Receipts with Confidence Intervals")

if periods:
    sarimax_forecast(SARIMAX_model, ucm_fit, arima_unrate_model, periods, gdp_multiplier, unemp_multiplier, show_confidence_intervals, start_date, end_date)