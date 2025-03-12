import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Fuel Sales Analyzer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling with dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #E0E0E0;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-card {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
    }
    .model-info {
        background-color: #1E1E1E;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #E0E0E0;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Title and intro
st.markdown("<h1 class='main-header'>Fuel Sales Analysis & Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Upload your sales data to analyze trends and forecast future sales using five advanced algorithms.")

# Sidebar for data upload and configuration
with st.sidebar:
    st.header("Data & Configuration")
    
    uploaded_file = st.file_uploader("Upload your Excel file:", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is None:
        st.info("No file uploaded. Using sample data for demonstration.")
        
        # Create sample data similar to the image
        dates = pd.date_range(start='2024-03-01', periods=30)
        # Generate more realistic sample data with weekly patterns
        weekday_factor = np.array([0.8, 0.9, 1.0, 0.95, 1.1, 1.2, 0.85])  # Mon-Sun factors
        base_diesel = 700 + np.random.normal(0, 50, 30)
        base_petrol = 500 + np.random.normal(0, 70, 30)
        base_new = 200 + np.random.normal(0, 30, 30)
        
        sample_data = pd.DataFrame({
            'Date': dates,
            'Day': [d.strftime('%A') for d in dates],
            'Total Die': base_diesel * [weekday_factor[d.weekday()] for d in dates],
            'Total Petrol': base_petrol * [weekday_factor[d.weekday()] for d in dates],
            'NEW(Liters)': base_new * [weekday_factor[d.weekday()] for d in dates]
        })
        
        df = sample_data
    else:
        try:
            # Try to determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"File loaded successfully! {len(df)} records found.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()
    
    st.subheader("Available Columns")
    all_columns = df.columns.tolist()
    st.write(all_columns)
    
    # Ensure date column is properly formatted
    date_col = st.selectbox("Select Date Column:", all_columns, index=all_columns.index('Date') if 'Date' in all_columns else 0)
    
    # Try to convert the date column to datetime
    if date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            st.write(f"Date range: {min_date.date()} to {max_date.date()}")
        except:
            st.warning("Could not convert the selected column to date. Please ensure it contains date values.")
    
    # Analytics configuration
    st.subheader("Analysis Configuration")
    chart_type = st.selectbox("Chart Type:", ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot", "Heatmap"])
    
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Added option for X-axis selection
    if chart_type != "Heatmap":
        x_column = st.selectbox("Select X-axis column:", all_columns, index=all_columns.index(date_col) if date_col in all_columns else 0)
        y_columns = st.multiselect("Select Y-axis column(s):", numerical_cols, 
                                  default=numerical_cols[:2] if len(numerical_cols) >= 2 else numerical_cols[:1])
    else:
        y_columns = st.multiselect("Select columns for correlation:", numerical_cols, 
                                  default=numerical_cols)
    
    # Prediction configuration
    st.subheader("Prediction Configuration")
    target_column = st.selectbox("Target Column for Prediction:", numerical_cols)
    prediction_model = st.selectbox("Prediction Model:", 
                                  ["ARIMA", "SARIMA", "XGBoost", "RandomForest", "ElasticNet"])
    prediction_days = st.slider("Number of days to predict:", 1, 30, 7)
    
    # Advanced model parameters
    st.subheader("Advanced Model Parameters")
    with st.expander("Model Parameters"):
        if prediction_model == "ARIMA" or prediction_model == "SARIMA":
            auto_param = st.checkbox("Auto-select parameters", value=True)
            if not auto_param:
                p_value = st.slider("p (AR order):", 0, 5, 1)
                d_value = st.slider("d (Differencing):", 0, 2, 1)
                q_value = st.slider("q (MA order):", 0, 5, 1)
                if prediction_model == "SARIMA":
                    m_value = st.slider("m (Seasonal periods):", 2, 12, 7)
        
        elif prediction_model == "XGBoost":
            n_estimators = st.slider("Number of estimators:", 50, 500, 100)
            max_depth = st.slider("Max depth:", 3, 10, 6)
            learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1)
        
        elif prediction_model == "RandomForest":
            rf_n_estimators = st.slider("Number of trees:", 50, 500, 100)
            rf_max_depth = st.slider("Max depth:", 3, 20, 10)
            rf_min_samples_split = st.slider("Min samples split:", 2, 10, 2)
        
        elif prediction_model == "ElasticNet":
            alpha = st.slider("Alpha:", 0.01, 1.0, 0.5)
            l1_ratio = st.slider("L1 ratio:", 0.0, 1.0, 0.5)
            max_iter = st.slider("Max iterations:", 500, 5000, 1000)
        
        elif prediction_model == "Prophet":
            yearly_seasonality = st.selectbox("Yearly seasonality:", ["auto", True, False])
            weekly_seasonality = st.selectbox("Weekly seasonality:", ["auto", True, False])
            daily_seasonality = st.selectbox("Daily seasonality:", ["auto", True, False])
            include_holidays = st.checkbox("Include country holidays", value=False)
            if include_holidays:
                country = st.selectbox("Country for holidays:", ["US", "UK", "IN", "FR", "DE"])

# Data cleaning and preprocessing
@st.cache_data
def preprocess_data(data, date_column):
    """Preprocess the data for analysis"""
    df_proc = data.copy()
    
    # Set date as index if not already
    if date_column in df_proc.columns and not isinstance(df_proc.index, pd.DatetimeIndex):
        df_proc[date_column] = pd.to_datetime(df_proc[date_column])
        df_proc = df_proc.set_index(date_column)
    
    # Check for missing values
    if df_proc.isnull().sum().sum() > 0:
        # Fill missing values in numerical columns with mean
        for col in df_proc.select_dtypes(include=['float64', 'int64']).columns:
            df_proc[col] = df_proc[col].fillna(df_proc[col].mean())
    
    # Add some useful time-based features
    df_proc['day_of_week'] = df_proc.index.dayofweek
    df_proc['month'] = df_proc.index.month
    df_proc['quarter'] = df_proc.index.quarter
    df_proc['year'] = df_proc.index.year
    df_proc['is_weekend'] = df_proc.index.dayofweek >= 5
    
    return df_proc

# Main content area - Organized in tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "Detailed Analysis", "Predictions"])

with tab1:
    # Dashboard overview
    st.markdown("<h2 class='sub-header'>Sales Dashboard Overview</h2>", unsafe_allow_html=True)
    
    if date_col in df.columns:
        # Process data
        df_processed = preprocess_data(df, date_col)
        
        # Show some key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'Total Die' in df.columns:
                total_diesel = df['Total Die'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Diesel</h3>
                    <h2>{total_diesel:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Records</h3>
                    <h2>{len(df)}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if 'Total Petrol' in df.columns:
                total_petrol = df['Total Petrol'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Petrol</h3>
                    <h2>{total_petrol:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            elif len(numerical_cols) > 0:
                avg_value = df[numerical_cols[0]].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg {numerical_cols[0]}</h3>
                    <h2>{avg_value:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'Total Die' in df.columns and 'Total Petrol' in df.columns:
                ratio = df['Total Die'].sum() / df['Total Petrol'].sum() if df['Total Petrol'].sum() > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Diesel:Petrol Ratio</h3>
                    <h2>{ratio:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            elif len(numerical_cols) > 1:
                max_value = df[numerical_cols[1]].max()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Max {numerical_cols[1]}</h3>
                    <h2>{max_value:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            # For best day calculation, use the first available numeric column
            if len(df_processed) > 0 and len(numerical_cols) > 0:
                numeric_col = numerical_cols[0]
                try:
                    weekday_avg = df_processed.groupby('day_of_week')[numeric_col].mean()
                    best_day_idx = weekday_avg.idxmax()
                    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    best_day = days[best_day_idx]
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Best Sales Day</h3>
                        <h2>{best_day}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Data Points</h3>
                        <h2>{len(df_processed)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Main chart
        st.subheader("Sales Trends")
        
        # Reset index for plotting
        plot_df = df_processed.reset_index()
        
        # Updated plotting with dark theme
        dark_template = "plotly_dark"
        
        # Modified to use the selected x-axis column
        if chart_type == "Line Chart" and len(y_columns) > 0:
            fig = px.line(plot_df, x=x_column, y=y_columns, 
                          title="Sales Trend",
                          template=dark_template)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Bar Chart" and len(y_columns) > 0:
            fig = px.bar(plot_df, x=x_column, y=y_columns,
                         title="Sales Comparison",
                         barmode='group',
                         template=dark_template)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Area Chart" and len(y_columns) > 0:
            fig = px.area(plot_df, x=x_column, y=y_columns,
                          title="Cumulative Sales",
                          template=dark_template)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Scatter Plot" and len(y_columns) > 0:
            fig = px.scatter(plot_df, x=x_column, y=y_columns,
                             title="Sales Distribution",
                             template=dark_template)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Heatmap" and len(y_columns) > 1:
            corr_matrix = df[y_columns].corr()
            fig = px.imshow(corr_matrix, 
                            text_auto=True, 
                            color_continuous_scale='Blues',
                            title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        
        # Day of week analysis
        st.subheader("Day of Week Analysis")
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Create a new dataframe with day names
        day_analysis = df_processed.reset_index()
        if 'day_of_week' in day_analysis.columns:
            day_analysis['Day Name'] = day_analysis['day_of_week'].apply(lambda x: day_names[x])
            
            # Make sure to only select numeric columns for mean calculation
            valid_numeric_cols = []
            for col in y_columns:
                if col in day_analysis.columns and day_analysis[col].dtype in ['float64', 'int64']:
                    valid_numeric_cols.append(col)
            
            # Only proceed if we have numeric columns to analyze
            if len(valid_numeric_cols) > 0:
                # Calculate mean only for numeric columns
                day_avg = day_analysis.groupby('Day Name')[valid_numeric_cols].mean().reindex(day_names)
                fig = px.bar(day_avg.reset_index(), x='Day Name', y=valid_numeric_cols, 
                             title="Average Sales by Day of Week",
                             template=dark_template)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available for day of week analysis.")

with tab2:
    # Detailed Analysis
    st.markdown("<h2 class='sub-header'>Detailed Analysis</h2>", unsafe_allow_html=True)
    
    if date_col in df.columns:
        # Display raw data with pagination
        st.subheader("Raw Data")
        page_size = st.slider("Rows per page:", 5, 50, 10)
        total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
        page_number = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
        start_idx = (page_number - 1) * page_size
        end_idx = min(start_idx + page_size, len(df))
        
        st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
        
        # Statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Time-based analysis (if applicable)
        if isinstance(df_processed.index, pd.DatetimeIndex):
            st.subheader("Time-Based Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if len(y_columns) > 0:
                    # Monthly trends
                    monthly_data = df_processed.reset_index()
                    monthly_data['Month'] = monthly_data[date_col].dt.strftime('%b %Y')
                    
                    # Ensure we only use numeric columns for mean calculation
                    valid_y_cols = [col for col in y_columns if col in monthly_data.columns 
                                  and pd.api.types.is_numeric_dtype(monthly_data[col])]
                    
                    if valid_y_cols:
                        monthly_avg = monthly_data.groupby('Month')[valid_y_cols].mean()
                        
                        fig = px.line(monthly_avg.reset_index(), x='Month', y=valid_y_cols,
                                      title="Monthly Average Sales",
                                      markers=True,
                                      template=dark_template)
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(y_columns) > 0:
                    # Weekday vs Weekend
                    # Ensure we only use numeric columns for mean calculation
                    valid_y_cols = [col for col in y_columns if col in df_processed.columns 
                                   and pd.api.types.is_numeric_dtype(df_processed[col])]
                    
                    if valid_y_cols:
                        try:
                            # First check if 'is_weekend' is in the columns
                            if 'is_weekend' in df_processed.columns:
                                weekend_comparison = df_processed.groupby('is_weekend')[valid_y_cols].mean()
                                weekend_data = pd.DataFrame({
                                    'Day_Type': ['Weekday', 'Weekend'],
                                })
                                
                                # Add the mean values for each column
                                for col in valid_y_cols:
                                    if len(weekend_comparison) == 2:  # Both weekday and weekend exist
                                        weekend_data[col] = [weekend_comparison.loc[False, col], 
                                                            weekend_comparison.loc[True, col]]
                                    else:
                                        # Handle case where only weekday or weekend exists
                                        if True in weekend_comparison.index:
                                            weekend_data[col] = [0, weekend_comparison.loc[True, col]]
                                        else:
                                            weekend_data[col] = [weekend_comparison.loc[False, col], 0]
                                
                                # Create the plot with the properly structured DataFrame
                                fig = px.bar(weekend_data, 
                                            x='Day_Type', y=valid_y_cols,
                                            title="Weekday vs Weekend Average Sales",
                                            template=dark_template)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Cannot create weekend comparison chart: 'is_weekend' column not found.")
                        except Exception as e:
                            st.error(f"Error creating weekend comparison chart: {e}")
                            # Alternative: just show a simple bar chart of the data
                            try:
                                fig = px.bar(df_processed.reset_index(), x=date_col, y=valid_y_cols[0],
                                            title=f"{valid_y_cols[0]} by Date",
                                            template=dark_template)
                                st.plotly_chart(fig, use_container_width=True)
                            except:
                                st.warning("Could not create alternative chart.")

            # Correlation analysis
            if len(numerical_cols) > 1:
                st.subheader("Correlation Analysis")
                
                corr_matrix = df[numerical_cols].corr()
                
                # Heatmap for correlation with dark theme
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
                
                # Show strongest correlations
                st.subheader("Top Correlations")
                
                # Get the upper triangle of the correlation matrix
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                # Find the strongest correlations
                strongest_corr = upper.unstack().dropna().abs().sort_values(ascending=False)
                
                if len(strongest_corr) > 0:
                    st.dataframe(pd.DataFrame({
                        'Variables': [f"{x[0]} & {x[1]}" for x in strongest_corr.index[:5]],
                        'Correlation': strongest_corr.values[:5]
                    }))
                else:
                    st.write("No correlations to display.")
with tab3:
    # Predictions
    st.markdown("<h2 class='sub-header'>Sales Forecasting</h2>", unsafe_allow_html=True)
    
    # Declare global variables at the top
    scaler = None
    model = None
    
    # Model Descriptions
    with st.expander("About the Forecasting Models"):
        st.markdown("""
        <div class="model-info">
            <h3>ARIMA (AutoRegressive Integrated Moving Average)</h3>
            <p>A statistical model that captures trends in time series data. Best for short-term forecasts when data shows clear patterns without too much external influence.</p>
        </div>
        
        <div class="model-info">
            <h3>SARIMA (Seasonal ARIMA)</h3>
            <p>Extends ARIMA by handling seasonality. Ideal for fuel sales that show weekly, monthly, or yearly patterns like higher weekend sales.</p>
        </div>
        
        <div class="model-info">
            <h3>XGBoost (Extreme Gradient Boosting)</h3>
            <p>A machine learning model that captures complex non-linear relationships in data. Excellent when external factors like weather, events, or fuel prices affect sales.</p>
        </div>
        
        <div class="model-info">
            <h3>RandomForest</h3>
            <p>An ensemble learning method that operates by constructing multiple decision trees and outputting the average prediction. Handles non-linear relationships and is robust to outliers.</p>
        </div>
        
        <div class="model-info">
            <h3>ElasticNet</h3>
            <p>A linear regression model with L1 and L2 regularization. Good for when you suspect linear relationships between features and the target, with some feature selection capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    if date_col in df.columns and target_column in df.columns:
        # Prepare time series data
        ts_data = df.copy()
        ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        ts_data = ts_data.set_index(date_col)
        ts_data = ts_data.sort_index()
        
        # Check if we have enough data
        if len(ts_data) < 5:
            st.warning("Not enough data for meaningful prediction. Please upload a dataset with more records.")
            st.stop()
        
        # Display target column data
        st.subheader(f"Historical {target_column} Data")
        fig = px.line(ts_data, y=target_column, title=f"{target_column} Time Series", template=dark_template)
        st.plotly_chart(fig, use_container_width=True)
        
        # Features for machine learning models
        if prediction_model in ["XGBoost", "RandomForest", "ElasticNet"]:
            ts_data['day_of_week'] = ts_data.index.dayofweek
            ts_data['month'] = ts_data.index.month
            ts_data['quarter'] = ts_data.index.quarter
            ts_data['year'] = ts_data.index.year
            ts_data['is_weekend'] = ts_data.index.dayofweek >= 5
        
        # Prepare train/test split
        train_size = int(len(ts_data) * 0.8)
        train_data = ts_data.iloc[:train_size]
        test_data = ts_data.iloc[train_size:]
        
        # Add a predict button
        predict_button = st.button("Generate Prediction")
        
        if predict_button:
            with st.spinner(f"Training {prediction_model} model and generating predictions..."):
                # Forecast future values
                
                def forecast():
                    global scaler, model  # Use global scaler and model
                    
                    last_date = ts_data.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
                    
                    # Container for predictions
                    predictions = []
                    
                    if prediction_model == "ARIMA":
                        # ARIMA model
                        try:
                            if 'auto_param' in locals() and auto_param:
                                model = auto_arima(train_data[target_column], seasonal=False, trace=False,
                                                  error_action='ignore', suppress_warnings=True)
                                # Fit on the entire dataset before predicting
                                model.fit(ts_data[target_column])
                                # Forecast
                                forecast, conf_int = model.predict(n_periods=prediction_days, return_conf_int=True)
                            else:
                                if 'p_value' not in locals():
                                    p_value, d_value, q_value = 1, 1, 1  # Default values
                                model = ARIMA(ts_data[target_column], order=(p_value, d_value, q_value))
                                model_fit = model.fit()
                                # Forecast
                                forecast = model_fit.forecast(steps=prediction_days)
                                # Simple confidence interval for custom ARIMA
                                std_dev = np.std(ts_data[target_column])
                                conf_int = np.array([[val - 1.96 * std_dev, val + 1.96 * std_dev] for val in forecast])
                            
                            for i, date in enumerate(future_dates):
                                predictions.append({
                                    'Date': date,
                                    'Predicted': forecast[i],
                                    'Lower_CI': conf_int[i][0],
                                    'Upper_CI': conf_int[i][1]
                                })
                            return pd.DataFrame(predictions)
                        except Exception as e:
                            st.error(f"Error with ARIMA model: {str(e)}")
                            return pd.DataFrame(columns=['Date', 'Predicted', 'Lower_CI', 'Upper_CI'])
                    
                    elif prediction_model == "SARIMA":
                        try:
                            # Seasonal ARIMA
                            if 'auto_param' in locals() and auto_param:
                                model = auto_arima(train_data[target_column], seasonal=True, m=7,
                                                  trace=False, error_action='ignore', suppress_warnings=True)
                                # Fit on the entire dataset before predicting
                                model.fit(ts_data[target_column])
                                # Forecast
                                forecast, conf_int = model.predict(n_periods=prediction_days, return_conf_int=True)
                            else:
                                # Use specified parameters
                                if 'p_value' not in locals():
                                    p_value, d_value, q_value, m_value = 1, 1, 1, 7  # Default values
                                model = SARIMAX(ts_data[target_column], 
                                              order=(p_value, d_value, q_value), 
                                              seasonal_order=(p_value, d_value, q_value, m_value))
                                model_fit = model.fit(disp=False)
                                # Forecast
                                forecast = model_fit.forecast(steps=prediction_days)
                                # Simple confidence interval
                                std_dev = np.std(ts_data[target_column])
                                conf_int = np.array([[val - 1.96 * std_dev, val + 1.96 * std_dev] for val in forecast])
                            
                            for i, date in enumerate(future_dates):
                                predictions.append({
                                    'Date': date,
                                    'Predicted': forecast[i],
                                    'Lower_CI': conf_int[i][0],
                                    'Upper_CI': conf_int[i][1]
                                })
                            return pd.DataFrame(predictions)
                        except Exception as e:
                            st.error(f"Error with SARIMA model: {str(e)}")
                            return pd.DataFrame(columns=['Date', 'Predicted', 'Lower_CI', 'Upper_CI'])
                    
                    elif prediction_model == "XGBoost":
                        try:
                            # Prepare features
                            features = ['day_of_week', 'month', 'quarter', 'year', 'is_weekend']
                            # Add lag features
                            for lag in range(1, min(7, len(ts_data))):
                                ts_data[f'lag_{lag}'] = ts_data[target_column].shift(lag)
                                features.append(f'lag_{lag}')
                            
                            # Drop NaN from lag creation
                            ml_data = ts_data.dropna()
                            
                            # Train/test split
                            X = ml_data[features]
                            y = ml_data[target_column]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Define model with specified parameters
                            if 'n_estimators' not in locals():
                                n_estimators, max_depth, learning_rate = 100, 6, 0.1
                            model = xgb.XGBRegressor(n_estimators=n_estimators, 
                                                    max_depth=max_depth, 
                                                    learning_rate=learning_rate,
                                                    random_state=42)
                            model.fit(X_train, y_train)
                            
                            # Create feature data for future dates
                            future_features = []
                            for date in future_dates:
                                features_dict = {
                                    'day_of_week': date.dayofweek,
                                    'month': date.month,
                                    'quarter': date.quarter,
                                    'year': date.year,
                                    'is_weekend': date.dayofweek >= 5
                                }
                                
                                # Add lag features from the last available data
                                last_data = ts_data.tail(7)[target_column].values
                                for lag in range(1, min(7, len(ts_data))):
                                    if lag <= len(last_data):
                                        features_dict[f'lag_{lag}'] = last_data[-lag]
                                    else:
                                        features_dict[f'lag_{lag}'] = 0
                                
                                future_features.append(features_dict)
                            
                            future_df = pd.DataFrame(future_features)
                            # Make predictions
                            forecast = model.predict(future_df[features])
                            
                            # Calculate simple confidence intervals
                            y_pred_test = model.predict(X_test)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                            
                            for i, date in enumerate(future_dates):
                                predictions.append({
                                    'Date': date,
                                    'Predicted': forecast[i],
                                    'Lower_CI': forecast[i] - 1.96 * rmse,
                                    'Upper_CI': forecast[i] + 1.96 * rmse
                                })
                            return pd.DataFrame(predictions)
                        except Exception as e:
                            st.error(f"Error with XGBoost model: {str(e)}")
                            return pd.DataFrame(columns=['Date', 'Predicted', 'Lower_CI', 'Upper_CI'])
                    
                    elif prediction_model == "RandomForest":
                        try:
                            # Prepare features
                            features = ['day_of_week', 'month', 'quarter', 'year', 'is_weekend']
                            # Add lag features
                            for lag in range(1, min(7, len(ts_data))):
                                ts_data[f'lag_{lag}'] = ts_data[target_column].shift(lag)
                                features.append(f'lag_{lag}')
                            
                            # Drop NaN from lag creation
                            ml_data = ts_data.dropna()
                            
                            # Train/test split
                            X = ml_data[features]
                            y = ml_data[target_column]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Define model with specified parameters
                            if 'rf_n_estimators' not in locals():
                                rf_n_estimators, rf_max_depth, rf_min_samples_split = 100, 10, 2
                            model = RandomForestRegressor(n_estimators=rf_n_estimators, 
                                                         max_depth=rf_max_depth, 
                                                         min_samples_split=rf_min_samples_split,
                                                         random_state=42)
                            model.fit(X_train, y_train)
                            
                            # Create feature data for future dates
                            future_features = []
                            for date in future_dates:
                                features_dict = {
                                    'day_of_week': date.dayofweek,
                                    'month': date.month,
                                    'quarter': date.quarter,
                                    'year': date.year,
                                    'is_weekend': date.dayofweek >= 5
                                }
                                
                                # Add lag features from the last available data
                                last_data = ts_data.tail(7)[target_column].values
                                for lag in range(1, min(7, len(ts_data))):
                                    if lag <= len(last_data):
                                        features_dict[f'lag_{lag}'] = last_data[-lag]
                                    else:
                                        features_dict[f'lag_{lag}'] = 0
                                
                                future_features.append(features_dict)
                            
                            future_df = pd.DataFrame(future_features)
                            # Make predictions
                            forecast = model.predict(future_df[features])
                            
                            # Get prediction intervals from forest
                            def pred_ints(model, X, percentile=95):
                                err_down = []
                                err_up = []
                                for x in range(len(X)):
                                    preds = []
                                    for pred in model.estimators_:
                                        preds.append(pred.predict(X[x:x+1])[0])
                                    err_down.append(np.percentile(preds, (100 - percentile) / 2.))
                                    err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
                                return err_down, err_up
                            
                            err_down, err_up = pred_ints(model, future_df[features].values, percentile=95)
                            
                            for i, date in enumerate(future_dates):
                                predictions.append({
                                    'Date': date,
                                    'Predicted': forecast[i],
                                    'Lower_CI': err_down[i],
                                    'Upper_CI': err_up[i]
                                })
                            return pd.DataFrame(predictions)
                        except Exception as e:
                            st.error(f"Error with RandomForest model: {str(e)}")
                            return pd.DataFrame(columns=['Date', 'Predicted', 'Lower_CI', 'Upper_CI'])
                    
                    elif prediction_model == "ElasticNet":
                        try:
                            # Prepare features
                            features = ['day_of_week', 'month', 'quarter', 'year', 'is_weekend']
                            # Add lag features
                            for lag in range(1, min(7, len(ts_data))):
                                ts_data[f'lag_{lag}'] = ts_data[target_column].shift(lag)
                                features.append(f'lag_{lag}')
                            
                            # Drop NaN from lag creation
                            ml_data = ts_data.dropna()
                            
                            # Train/test split
                            X = ml_data[features]
                            y = ml_data[target_column]
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                            
                            # Define model with specified parameters
                            if 'alpha' not in locals():
                                alpha, l1_ratio, max_iter = 0.5, 0.5, 1000
                            model = ElasticNet(alpha=alpha, 
                                              l1_ratio=l1_ratio, 
                                              max_iter=max_iter,
                                              random_state=42)
                            model.fit(X_train, y_train)
                            
                            # Create feature data for future dates
                            future_features = []
                            for date in future_dates:
                                features_dict = {
                                    'day_of_week': date.dayofweek,
                                    'month': date.month,
                                    'quarter': date.quarter,
                                    'year': date.year,
                                    'is_weekend': date.dayofweek >= 5
                                }
                                
                                # Add lag features from the last available data
                                last_data = ts_data.tail(7)[target_column].values
                                for lag in range(1, min(7, len(ts_data))):
                                    if lag <= len(last_data):
                                        features_dict[f'lag_{lag}'] = last_data[-lag]
                                    else:
                                        features_dict[f'lag_{lag}'] = 0
                                
                                future_features.append(features_dict)
                            
                            future_df = pd.DataFrame(future_features)
                            # Scale the future features
                            future_scaled = scaler.transform(future_df[features])
                            
                            # Make predictions
                            forecast = model.predict(future_scaled)
                            
                            # Calculate confidence intervals
                            y_pred_test = model.predict(X_test)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                            
                            for i, date in enumerate(future_dates):
                                predictions.append({
                                    'Date': date,
                                    'Predicted': forecast[i],
                                    'Lower_CI': forecast[i] - 1.96 * rmse,
                                    'Upper_CI': forecast[i] + 1.96 * rmse
                                })
                            return pd.DataFrame(predictions)
                        except Exception as e:
                            st.error(f"Error with ElasticNet model: {str(e)}")
                            return pd.DataFrame(columns=['Date', 'Predicted', 'Lower_CI', 'Upper_CI'])
                
                # Call the forecast function
                prediction_df = forecast()
                
                if not prediction_df.empty:
                    # Display results
                    st.subheader("Forecast Results")
                    st.dataframe(prediction_df)
                    
                    # Visualization of forecast
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=ts_data.index, 
                        y=ts_data[target_column],
                        mode='lines',
                        name='Historical',
                        line=dict(color='#1E88E5')
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=prediction_df['Date'], 
                        y=prediction_df['Predicted'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#FFC107')
                    ))
                    
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=prediction_df['Date'].tolist() + prediction_df['Date'].tolist()[::-1],
                        y=prediction_df['Upper_CI'].tolist() + prediction_df['Lower_CI'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 193, 7, 0.2)',
                        line=dict(color='rgba(255, 193, 7, 0)'),
                        name='95% Confidence Interval'
                    ))
                    
                    # Layout
                    fig.update_layout(
                        title=f"{prediction_model} Forecast for {target_column}",
                        xaxis_title="Date",
                        yaxis_title=target_column,
                        template=dark_template,
                        showlegend=True,
                        legend=dict(x=0, y=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Model evaluation metrics for test data
                    if len(test_data) > 0:
                        st.subheader("Model Evaluation")
                        
                        # Get predictions for test data
                        test_predictions = []
                        
                        if prediction_model in ["ARIMA", "SARIMA"]:
                            # For time series models
                            test_model = None
                            if prediction_model == "ARIMA":
                                try:
                                    test_model = ARIMA(train_data[target_column], order=(p_value, d_value, q_value) if 'p_value' in locals() else (1, 1, 1))
                                    model_fit = test_model.fit()
                                    test_forecast = model_fit.forecast(steps=len(test_data))
                                    test_predictions = test_forecast
                                except Exception as e:
                                    st.warning(f"Could not evaluate ARIMA model on test data: {str(e)}")
                            else:  # SARIMA
                                try:
                                    test_model = SARIMAX(train_data[target_column], 
                                                      order=(p_value, d_value, q_value) if 'p_value' in locals() else (1, 1, 1),
                                                      seasonal_order=(p_value, d_value, q_value, m_value) if 'p_value' in locals() else (1, 1, 1, 7))
                                    model_fit = test_model.fit(disp=False)
                                    test_forecast = model_fit.forecast(steps=len(test_data))
                                    test_predictions = test_forecast
                                except Exception as e:
                                    st.warning(f"Could not evaluate SARIMA model on test data: {str(e)}")
                        
                        elif prediction_model in ["XGBoost", "RandomForest", "ElasticNet"]:
                            # For ML models, recreate features
                            ml_test = test_data.copy()
                            ml_test['day_of_week'] = ml_test.index.dayofweek
                            ml_test['month'] = ml_test.index.month
                            ml_test['quarter'] = ml_test.index.quarter
                            ml_test['year'] = ml_test.index.year
                            ml_test['is_weekend'] = ml_test.index.dayofweek >= 5
                            
                            # Add lag features from training data
                            for lag in range(1, min(7, len(train_data))):
                                combined = pd.concat([train_data[[target_column]], ml_test[[target_column]]])
                                combined[f'lag_{lag}'] = combined[target_column].shift(lag)
                                ml_test[f'lag_{lag}'] = combined[f'lag_{lag}'].values[-len(ml_test):]
                            
                            features = ['day_of_week', 'month', 'quarter', 'year', 'is_weekend'] + [f'lag_{lag}' for lag in range(1, min(7, len(train_data)))]
                            ml_test = ml_test.dropna()
                            
                            if len(ml_test) > 0:
                                if prediction_model == "ElasticNet":
                                    # Use the global scaler variable created during model training
                                    X_test_scaled = scaler.transform(ml_test[features])
                                    test_predictions = model.predict(X_test_scaled)
                                else:
                                    # Use the global model variable created during model training
                                    test_predictions = model.predict(ml_test[features])
                            else:
                                st.warning("Not enough test data for evaluation after feature creation")
                        
                        # Calculate metrics if we have predictions
                        if len(test_predictions) > 0:
                            actual = test_data[target_column].values[:len(test_predictions)]
                            
                            mae = mean_absolute_error(actual, test_predictions)
                            rmse = np.sqrt(mean_squared_error(actual, test_predictions))
                            r2 = r2_score(actual, test_predictions)
                            mape = np.mean(np.abs((actual - test_predictions) / actual)) * 100
                            
                            # Display metrics
                            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                            
                            with metrics_col1:
                                st.metric("MAE", f"{mae:.2f}")
                            
                            with metrics_col2:
                                st.metric("RMSE", f"{rmse:.2f}")
                            
                            with metrics_col3:
                                st.metric("R²", f"{r2:.4f}")
                            
                            with metrics_col4:
                                st.metric("MAPE", f"{mape:.2f}%")
                            
                            # Compare predictions vs actual for test period
                            fig = go.Figure()
                            
                            # Actual values
                            fig.add_trace(go.Scatter(
                                x=test_data.index[:len(test_predictions)],
                                y=actual,
                                mode='lines',
                                name='Actual',
                                line=dict(color='#1E88E5')
                            ))
                            
                            # Predicted values
                            fig.add_trace(go.Scatter(
                                x=test_data.index[:len(test_predictions)],
                                y=test_predictions,
                                mode='lines',
                                name='Predicted',
                                line=dict(color='#FFC107')
                            ))
                            
                            # Layout
                            fig.update_layout(
                                title="Model Performance on Test Data",
                                xaxis_title="Date",
                                yaxis_title=target_column,
                                template=dark_template,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                # Add feature importance if using applicable model
                if prediction_model in ["XGBoost", "RandomForest"] and model is not None:
                    st.subheader("Feature Importance")
                    
                    try:
                        importances = pd.DataFrame({
                            'Feature': features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(importances, x='Importance', y='Feature', 
                                    orientation='h',
                                    title="Feature Importance",
                                    template=dark_template)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not display feature importance: {str(e)}")
                
                # Model comparison if asked
                compare_models = st.checkbox("Compare with other models")
                
                if compare_models:
                    st.subheader("Model Comparison")
                    st.warning("This will train multiple models and may take a moment...")
                    
                    # Define a function to get the RMSE of each model
                    def get_model_rmse(model_name, ts_data, target_column):
                        try:
                            train_size = int(len(ts_data) * 0.8)
                            train = ts_data.iloc[:train_size]
                            test = ts_data.iloc[train_size:]
                            
                            if model_name == "ARIMA":
                                model = auto_arima(train[target_column], seasonal=False, suppress_warnings=True)
                                pred = model.predict(n_periods=len(test))
                                return np.sqrt(mean_squared_error(test[target_column].values[:len(pred)], pred))
                            
                            elif model_name == "SARIMA":
                                model = auto_arima(train[target_column], seasonal=True, m=7, suppress_warnings=True)
                                pred = model.predict(n_periods=len(test))
                                return np.sqrt(mean_squared_error(test[target_column].values[:len(pred)], pred))
                            
                            else:  # ML models
                                # Add features
                                data = ts_data.copy()
                                data['day_of_week'] = data.index.dayofweek
                                data['month'] = data.index.month
                                data['year'] = data.index.year
                                data['is_weekend'] = data.index.dayofweek >= 5
                                
                                # Add lag features
                                for lag in range(1, min(7, len(data))):
                                    data[f'lag_{lag}'] = data[target_column].shift(lag)
                                
                                features = ['day_of_week', 'month', 'year', 'is_weekend'] + [f'lag_{lag}' for lag in range(1, min(7, len(data)))]
                                data = data.dropna()
                                
                                # Split
                                train_idx = int(len(data) * 0.8)
                                train_data = data.iloc[:train_idx]
                                test_data = data.iloc[train_idx:]
                                
                                X_train = train_data[features]
                                y_train = train_data[target_column]
                                X_test = test_data[features]
                                y_test = test_data[target_column]
                                
                                if model_name == "XGBoost":
                                    model = xgb.XGBRegressor(random_state=42)
                                    model.fit(X_train, y_train)
                                    pred = model.predict(X_test)
                                    return np.sqrt(mean_squared_error(y_test, pred))
                                
                                elif model_name == "RandomForest":
                                    model = RandomForestRegressor(random_state=42)
                                    model.fit(X_train, y_train)
                                    pred = model.predict(X_test)
                                    return np.sqrt(mean_squared_error(y_test, pred))
                                
                                elif model_name == "ElasticNet":
                                    # Scale
                                    local_scaler = StandardScaler()
                                    X_train_scaled = local_scaler.fit_transform(X_train)
                                    X_test_scaled = local_scaler.transform(X_test)
                                    
                                    model = ElasticNet(random_state=42)
                                    model.fit(X_train_scaled, y_train)
                                    pred = model.predict(X_test_scaled)
                                    return np.sqrt(mean_squared_error(y_test, pred))
                            
                            return float('inf')  # Return infinity if something fails
                        except Exception as e:
                            st.error(f"Error evaluating {model_name}: {str(e)}")
                            return float('inf')
                    
                    # Get RMSE for each model
                    models = ["ARIMA", "SARIMA", "XGBoost", "RandomForest", "ElasticNet"]
                    results = []
                    
                    for model_name in models:
                        with st.spinner(f"Evaluating {model_name}..."):
                            rmse = get_model_rmse(model_name, ts_data, target_column)
                            results.append({"Model": model_name, "RMSE": rmse})
                    
                    # Display results
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values('RMSE')
                    
                    # Bar chart of results
                    fig = px.bar(results_df, x='Model', y='RMSE', 
                                title="Model Comparison (Lower RMSE is better)",
                                template=dark_template)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendation
                    best_model = results_df.iloc[0]["Model"]
                    st.success(f"Based on this comparison, the {best_model} model performs best for your data.")

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #555;">
    <p style="color: #888;">Fuel Sales Analyzer Dashboard | v1.0</p>
</div>
""", unsafe_allow_html=True)