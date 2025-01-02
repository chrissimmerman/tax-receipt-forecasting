import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from pmdarima.arima import auto_arima
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Streamlit setup
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

@st.cache_data(show_spinner="Loading Tax Receipt Data...")
def get_df():
    df = pd.read_csv('final_data.csv')
    # Set Month-Year as index
    df['Month-Year'] = pd.to_datetime(df['Month-Year'])
    df.set_index('Month-Year', inplace=True)
    return df

@st.cache_data(show_spinner="Loading GDP Data...")
def get_ucm_model(df):
    real_gdp = df['Real GDP Interpolated']
    ucm_model = UnobservedComponents(real_gdp, level='rwdrift')
    ucm_fit = ucm_model.fit()
    return ucm_fit

@st.cache_data(show_spinner="Loading Unemployment Data...")
def get_arima_unrate_model(df):
    unemployment_rate = df['UNRATE_Proportion']
    arima_unrate_model = auto_arima(y=unemployment_rate, seasonal=False)
    return arima_unrate_model

@st.cache_data(show_spinner="Loading Time Series Analysis...")
def get_sarimax_model(df):
    SARIMAX_model = auto_arima(
        y=df[['Log Transformed Receipts']],
        X=df[['Real GDP Interpolated', 'UNRATE_Proportion']],
        m=12,
        D=1
    )
    return SARIMAX_model

def prepare_forecast_data(ucm_fit, arima_unrate_model, periods, scenario="Baseline"):
    # Base forecasts
    gdp_forecast = ucm_fit.get_forecast(steps=periods).predicted_mean
    unemp_forecast = arima_unrate_model.predict(n_periods=periods)

    # Modify forecasts based on the selected scenario
    if scenario == "Recession":
        gdp_forecast *= np.linspace(1.0, 0.85, periods)  # Gradual GDP decline
        unemp_forecast *= np.linspace(1.0, 1.2, periods)  # Gradual unemployment rise
    elif scenario == "Boom":
        gdp_forecast *= np.linspace(1.0, 1.2, periods)  # Gradual GDP growth
        unemp_forecast *= np.linspace(1.0, 0.85, periods)  # Gradual unemployment decline
    elif scenario == "Policy Intervention":
        gdp_forecast[:periods // 2] *= 1.1  # 10% boost for first half
        unemp_forecast[periods // 2:] *= 0.9  # 10% improvement in second half

    # Return modified forecasts in a DataFrame
    forecast_df = pd.DataFrame({
        "Real GDP Interpolated": gdp_forecast,
        "UNRATE_Proportion": unemp_forecast,
    }, index=pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=periods, freq="MS"))
    
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
    historical_actual = df["Log Transformed Receipts"]

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
        line=dict(color='red'),
        customdata=combined_gdp.index,  # Store original datetime values
        hovertemplate="Date: %{customdata|%Y-%m-%d}<br>Real GDP: %{y}<extra></extra>"
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
        line=dict(color='purple'),
        customdata=combined_unemp.index,  # Store original datetime values
        hovertemplate="Date: %{customdata|%Y-%m-%d}<br>Unemployment Rate: %{y}<extra></extra>"
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

def display_forecast_plotly(plot_data, show_confidence_intervals=True):
    fig = go.Figure()

    # Add actual receipts line
    fig.add_trace(go.Scatter(
        x=plot_data.index,  # X-axis remains the same
        y=plot_data["Actual Receipts"],
        mode='lines',
        name='Actual Receipts',
        line=dict(color='blue'),
        customdata=plot_data.index,  # Store original datetime values
        hovertemplate="Date: %{customdata|%Y-%m-%d}<br>Receipts: %{y}<extra></extra>"
    ))

    # Add forecasted receipts line
    fig.add_trace(go.Scatter(
        x=plot_data.index,  # X-axis remains the same
        y=plot_data["Forecasted Receipts"],
        mode='lines',
        name='Forecasted Receipts',
        line=dict(color='green', dash='solid'),
        customdata=plot_data.index,  # Store original datetime values
        hovertemplate="Date: %{customdata|%Y-%m-%d}<br>Forecasted Receipts: %{y}<extra></extra>"
    ))

    # Add confidence interval as a filled area
    if show_confidence_intervals:
        fig.add_trace(go.Scatter(
            x=list(plot_data.index) + list(plot_data.index[::-1]),  # X-axis remains the same
            y=list(plot_data["Upper Confidence Interval"]) + list(plot_data["Lower Confidence Interval"][::-1]),
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",  # Skip hover for the confidence interval
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

def sarimax_forecast(SARIMAX_model, ucm_fit, arima_unrate_model, periods, gdp_multiplier, unemp_multiplier, show_confidence_intervals, start_date, end_date, scenario="Baseline"):
    # Prepare forecast data
    forecast_df = prepare_forecast_data(ucm_fit, arima_unrate_model, periods, scenario=scenario)

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

    # Plot GDP and Unemployment forecasts
    plot_exogenous_forecast(df, forecast_df)

# Main Code
df = get_df()
ucm_fit = get_ucm_model(df)
arima_unrate_model = get_arima_unrate_model(df)
SARIMAX_model = get_sarimax_model(df)

# Sidebar: User Inputs
st.sidebar.header("Tax Receipt Forecast Configuration")
scenario = st.sidebar.selectbox(
    label="Choose a Scenario Template",
    options=["Baseline", "Recession", "Boom", "Policy Intervention"],
    index=0,
    help="Select an economic scenario to simulate."
)

periods = st.sidebar.number_input(
    label="📅 Forecast Horizon (Months)",
    min_value=1,
    max_value=120,
    value=24,
    step=1,
    help="Specify how many months into the future you'd like to forecast tax receipts."
)

show_confidence_intervals = st.sidebar.checkbox(
    label="📊 Show Confidence Intervals",
    value=True,
    help="Enable or disable confidence intervals in the forecast plot."
)

st.title("📊 Tax Receipts Forecast Dashboard")
st.markdown(
    """
    Welcome to the Tax Receipts Forecast Dashboard! This tool allows you to:
    - **Adjust scenarios** to simulate different economic changes.
    - **Explore tax receipt forecasts** over a custom time horizon.
    - **Visualize confidence intervals** for the forecasts.
    """
)

if periods:
    sarimax_forecast(
        SARIMAX_model, 
        ucm_fit, 
        arima_unrate_model, 
        periods=periods, 
        gdp_multiplier=1.0, 
        unemp_multiplier=1.0, 
        show_confidence_intervals=show_confidence_intervals, 
        start_date=None, 
        end_date=None, 
        scenario=scenario
    )
