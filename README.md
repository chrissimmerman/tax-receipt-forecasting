
# Tax Receipts Forecasting Dashboard

## **Overview**
The **Tax Receipts Forecasting Dashboard** is an interactive tool designed to model and forecast tax revenue under various economic scenarios. Built with **Streamlit**, this dashboard uses **SARIMAX models** with exogenous variables to generate insights into the impact of economic factors like **Corporate Profits (CP)** and **Job Openings (JTSJOL)** on tax receipts.

### Key Features:
- 📊 **Dynamic Scenario Modeling**: Simulate scenarios such as recessions, booms, policy interventions, and pandemics.
- 🔮 **Extrapolative Forecasting**: Predict tax receipts and key economic indicators over a user-defined time horizon.
- 📈 **Customizable Visualizations**: Explore trends in Corporate Profits, Job Openings, and Tax Receipts using intuitive plots.
- ⚙️ **Machine Learning Models**: Leverage **SARIMAX** and **Unobserved Components Models (UCM)** for accurate predictions.

---

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [License](#license)

---

## **Features**
- **Scenario Analysis**:
  - Recession: Gradual decline in economic activity.
  - Boom: Economic growth with increasing profits and job openings.
  - Policy Intervention: Short-term boost due to fiscal or monetary policies.
  - Pandemic: Sharp declines followed by recovery trends.

- **Forecast Customization**:
  - Adjustable forecast horizons (6-60 months).
  - Confidence intervals for forecast uncertainty.

- **Data Insights**:
  - Real-time plotting of historical and forecasted data.

---

## **Installation**
### Prerequisites:
- Python 3.8+
- Virtual environment (optional but recommended)

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/chrissimmerman/tax-receipts-forecast.git
   cd tax-receipts-forecast
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## **Usage**
1. Launch the app in your browser by following the instructions in the terminal.
2. Use the sidebar to:
   - Select a scenario to model.
   - Adjust the forecast horizon.
   - Toggle confidence intervals.
3. Explore the forecasted trends in tax receipts, Corporate Profits (CP), and Job Openings (JTSJOL).

## **Technical Details**
### Machine Learning Models:
- **SARIMAX**:
  - Predicts tax receipts with exogenous variables (`CP`, `JTSJOL`).
- **UCM (Unobserved Components Model)**:
  - Models trends in **Corporate Profits (CP)**.

### Data Sources:
- **FRED API**:
  - Corporate Profits (CP)
  - Job Openings (JTSJOL)
  - Consumer Price Index (CPIAUCSL)
- **US Treasury Deptartment API**:
  - Monthly Treasury Statement (MTS)
- Custom transformations for inflation-adjusted data.
- Linear Interpolation of Corporate Profits (CP)