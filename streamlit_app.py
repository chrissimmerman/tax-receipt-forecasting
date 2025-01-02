import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from pmdarima.arima import auto_arima
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Streamlit setup
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

@st.cache_data(show_spinner="Loading Tax Receipts...")
def get_df():
    df = pd.read_csv('comp_data.csv')
    # Set Month-Year as index
    df['Month-Year'] = pd.to_datetime(df['Month-Year'])
    df.set_index('Month-Year', inplace=True)
    return df

@st.cache_data(show_spinner="Loading Corporate Profits...")
def get_ucm_model(df):
    real_cp = df['Real CP Interpolated']
    ucm_model = UnobservedComponents(real_cp, level='rwdrift')
    ucm_fit = ucm_model.fit()
    return ucm_fit

@st.cache_data(show_spinner="Loading Job Openings...")
def get_jtsjol_arima_model(df):
    jtsjol = df['JTSJOL']
    arima_jtsjol_model = auto_arima(y=jtsjol, X=df[["Real CP Interpolated"]], seasonal=False)
    return arima_jtsjol_model

@st.cache_data(show_spinner="Loading Time Series Analysis...")
def get_sarimax_model(df):
    SARIMAX_model = auto_arima(
        y=df[['Log Transformed Receipts']],
        X=df[['Real CP Interpolated', 'JTSJOL']],
        m=12,
        D=1
    )
    return SARIMAX_model

def prepare_forecast_data(ucm_fit, arima_jtsjol_model, periods, scenario="Baseline"):
    # Base forecasts
    cp_forecast = ucm_fit.get_forecast(steps=periods).predicted_mean

    # Modify forecasts based on the selected scenario
    if scenario == "Recession":
        cp_forecast *= np.linspace(1.0, 0.85, periods)
    elif scenario == "Boom":
        cp_forecast *= np.linspace(1.0, 1.2, periods)
    elif scenario == "Policy Intervention":
        cp_forecast[:periods // 2] *= 1.1
    elif scenario == "Pandemic":
        # Sharp drop in the first third, then slow recovery
        drop_periods = periods // 3
        recovery_periods = periods - drop_periods
        cp_forecast[:drop_periods] *= np.linspace(1.0, 0.7, drop_periods)
        cp_forecast[drop_periods:] *= np.linspace(0.7, 1.0, recovery_periods)

    # Turn cp_forecast into a DataFrame
    cp_forecast_df = pd.DataFrame(cp_forecast)

    jtsjol_forecast = arima_jtsjol_model.predict(n_periods=periods, X=cp_forecast_df)

    if scenario == "Recession":
        jtsjol_forecast *= np.linspace(1.0, 0.7, periods)
    elif scenario == "Boom":
        jtsjol_forecast *= np.linspace(1.0, 1.3, periods)
    elif scenario == "Policy Intervention":
        jtsjol_forecast[periods // 2:] *= 0.9
    elif scenario == "Pandemic":
        # Sharp drop in the first third, then slow recovery
        jtsjol_forecast[:drop_periods] *= np.linspace(1.0, 0.5, drop_periods)
        jtsjol_forecast[drop_periods:] *= np.linspace(0.5, 1.0, recovery_periods)

    # Return modified forecasts in a DataFrame
    forecast_df = pd.DataFrame({
        "Real CP Interpolated": cp_forecast,
        "JTSJOL": jtsjol_forecast,
    }, index=pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=periods, freq="MS"))
    
    return forecast_df

def forecast_receipts(SARIMAX_model, forecast_df, periods):
    # Forecast tax receipts using SARIMAX
    fitted, confint = SARIMAX_model.predict(
        n_periods=periods,
        X=forecast_df[['Real CP Interpolated', 'JTSJOL']],
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
    # Combine historical and forecasted data for Real CP
    combined_gdp = pd.concat([
        df["Real CP Interpolated"],
        forecast_df["Real CP Interpolated"]
    ])
    
    # Convert the forecast start to an ordinal value
    forecast_start = df.index[-1].toordinal()

    # Adjust tick frequency (e.g., show one tick per year)
    tick_frequency = pd.date_range(combined_gdp.index.min(), combined_gdp.index.max(), freq="YS")

    # Create CP Forecast Plot
    gdp_fig = go.Figure()
    gdp_fig.add_trace(go.Scatter(
        x=combined_gdp.index.map(lambda x: x.toordinal()),  # Convert x-axis to ordinal
        y=combined_gdp,
        mode='lines',
        name="Real CP",
        line=dict(color='red'),
        customdata=combined_gdp.index,  # Store original datetime values
        hovertemplate="Date: %{customdata|%Y-%m-%d}<br>Real CP: %{y}<extra></extra>"
    ))
    
    # Add vertical line for forecast start
    gdp_fig.add_vline(
        x=forecast_start,
        line_dash="dash",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    gdp_fig.update_layout(
        title="Historical and Forecasted Corporate Profits",
        xaxis=dict(
            title=dict(text="Date"),
            type="linear",
            tickmode="array",
            tickvals=tick_frequency.map(lambda x: x.toordinal()),  # Add only year-start ticks
            ticktext=tick_frequency.strftime("%Y-%m"),
        ),
        yaxis=dict(title=dict(text="Real CP")),
        template="plotly_white"
    )

    # Combine historical and forecasted data for JTSJOL
    combined_unemp = pd.concat([
        df["JTSJOL"],
        forecast_df["JTSJOL"]
    ])
    
    # Create JTSJOL Forecast Plot
    unemp_fig = go.Figure()
    unemp_fig.add_trace(go.Scatter(
        x=combined_unemp.index.map(lambda x: x.toordinal()),  # Convert x-axis to ordinal
        y=combined_unemp,
        mode='lines',
        name="JTSJOL",
        line=dict(color='purple'),
        customdata=combined_unemp.index,  # Store original datetime values
        hovertemplate="Date: %{customdata|%Y-%m-%d}<br>Job Openings: %{y}<extra></extra>"
    ))
    
    # Add vertical line for forecast start
    unemp_fig.add_vline(
        x=forecast_start,
        line_dash="dash",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    unemp_fig.update_layout(
        title="Historical and Forecasted Job Openings",
        xaxis=dict(
            title=dict(text="Date"),
            type="linear",
            tickmode="array",
            tickvals=tick_frequency.map(lambda x: x.toordinal()),  # Add only year-start ticks
            ticktext=tick_frequency.strftime("%Y-%m"),
        ),
        yaxis=dict(title=dict(text="Job Openings")),
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

def sarimax_forecast(SARIMAX_model, ucm_fit, arima_jtsjol_model, periods, show_confidence_intervals, start_date=None, end_date=None, scenario="Baseline"):
    # Prepare forecast data
    forecast_df = prepare_forecast_data(ucm_fit, arima_jtsjol_model, periods, scenario=scenario)

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
jtsjol_arima_model = get_jtsjol_arima_model(df)
SARIMAX_model = get_sarimax_model(df)

# Sidebar: User Inputs
st.sidebar.header("Tax Receipt Forecast Configuration")
scenario = st.sidebar.selectbox(
    label="Choose a Scenario Template",
    options=["Baseline", "Recession", "Boom", "Policy Intervention", "Pandemic"],
    index=0,
    help="Select an economic scenario to simulate."
)

periods = st.sidebar.number_input(
    label="📅 Forecast Horizon (Months)",
    min_value=6,
    max_value=60,
    value=24,
    step=6,
    help="Specify how many months into the future you'd like to forecast tax receipts."
)

show_confidence_intervals = st.sidebar.checkbox(
    label="📊 Show Confidence Intervals",
    value=True,
    help="Enable or disable confidence intervals in the forecast plot."
)

# Date range selection
st.sidebar.subheader("Date Range Selection")
st.sidebar.write("Choose the range of dates to display on the tax receipts graph.")
min_date = df.index.min().date()
max_date = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=periods, freq="MS")[-1].date()

start_date = st.sidebar.date_input(
    label="Start Date", 
    value=min_date, 
    min_value=min_date, 
    max_value=max_date, 
    help="Set the starting date for the tax receipts graph."
)
end_date = st.sidebar.date_input(
    label="End Date", 
    value=max_date, 
    min_value=min_date, 
    max_value=max_date, 
    help="Set the ending date for the tax receipts graph."
)

# Ensure start_date is before end_date
if start_date > end_date:
    st.sidebar.error("⚠️ Start Date must be before End Date.")




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
        jtsjol_arima_model, 
        periods=periods, 
        show_confidence_intervals=show_confidence_intervals, 
        start_date=start_date, 
        end_date=end_date, 
        scenario=scenario
    )
