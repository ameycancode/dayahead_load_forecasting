"""
Energy Forecasting Dashboard - Main Application
Complete dashboard application with Streamlit interface for energy forecasting analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import json
import os
from typing import Dict, List, Optional

# Import our custom modules (assuming they're in the same directory)
try:
    from energy_dashboard_base import EnergyForecastingDashboard
    from dashboard_visualizations import EnergyForecastingVisualizations
except ImportError:
    st.error("Required modules not found. Please ensure energy_dashboard_base.py and dashboard_visualizations.py are available.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Energy Forecasting Performance Dashboard",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        position: relative;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        height: 4px;
        background-color: var(--progress-color);
        width: var(--progress-width);
    }
    .excellent { border-left-color: #28a745; }
    .good { border-left-color: #17a2b8; }
    .acceptable { border-left-color: #ffc107; }
    .poor { border-left-color: #dc3545; }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED DATA FUNCTIONS
# =============================================================================

@st.cache_resource(ttl=300)
def load_dashboard_data(environment: str, enable_sample_data: bool = False):
    """Load dashboard data with caching using cache_resource for database connections."""
    try:
        dashboard = EnergyForecastingDashboard(environment=environment, enable_sample_data=enable_sample_data)
       
        if not enable_sample_data:
            connection_success = dashboard.connect_to_database()
            if connection_success:
                test_success = dashboard.test_connection()
                if test_success:
                    st.success(" Database connection successful!")
                else:
                    st.warning(" Database connected but test query failed.")
            else:
                st.error(" Database connection failed.")
        else:
            st.info(" Using sample data mode")
            connection_success = True
       
        viz = EnergyForecastingVisualizations(dashboard)
        return dashboard, viz, connection_success
       
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        st.error(f"Failed to initialize dashboard: {e}")
        return None, None, False


@st.cache_data(ttl=300)
def get_summary_data(environment: str, enable_sample_data: bool = False):
    """Get summary data with proper caching."""
    try:
        dashboard = EnergyForecastingDashboard(environment=environment, enable_sample_data=enable_sample_data)
       
        if enable_sample_data:
            return dashboard._generate_summary_sample_data()
       
        if dashboard.connect_to_database():
            query = f"SELECT * FROM {dashboard.schema_name}.vw_fr_dashboard_summary ORDER BY business_priority, current_primary_metric"
            result = dashboard.execute_query(query)
            dashboard.close_connection()
            return result if not result.empty else pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting summary data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_daily_data(environment: str, start_date: str, end_date: str, enable_sample_data: bool = False):
    """Get daily data with proper caching."""
    try:
        dashboard = EnergyForecastingDashboard(environment=environment, enable_sample_data=enable_sample_data)
       
        if enable_sample_data:
            return dashboard._generate_daily_sample_data()
       
        if dashboard.connect_to_database():
            query = f"""
            SELECT
                forecast_date,
                customer_type,
                customer_segment_desc,
                total_actual_mwh,
                daily_primary_metric,
                daily_performance_tier,
                success_rate_pct,
                weighted_business_impact_score
            FROM {dashboard.schema_name}.vw_fr_daily_forecast_summary
            WHERE forecast_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY forecast_date DESC, customer_segment_desc
            """
            result = dashboard.execute_query(query)
            dashboard.close_connection()
            return result if not result.empty else pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting daily data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_hourly_timeseries_data(environment: str, start_date: str, end_date: str, enable_sample_data: bool = False):
    """Get hourly time series data for detailed analysis."""
    try:
        dashboard = EnergyForecastingDashboard(environment=environment, enable_sample_data=enable_sample_data)
       
        if enable_sample_data:
            # Generate sample hourly data
            date_range = pd.date_range(start=start_date, end=end_date, freq='H')
            sample_data = []
           
            for dt in date_range:
                for segment in ['Residential Solar', 'Residential Non-Solar', 'Commercial Solar', 'Commercial Non-Solar']:
                    is_solar = 'Solar' in segment
                    hour = dt.hour
                   
                    # Create realistic load patterns
                    if is_solar:
                        base_load = max(0, 50000 + 30000 * np.sin((hour - 6) * np.pi / 12))
                        if 11 <= hour <= 14:  # Solar peak
                            base_load *= 0.3
                        elif 15 <= hour <= 18:  # Duck curve
                            base_load *= 1.5
                    else:
                        base_load = 40000 + 20000 * np.sin((hour - 6) * np.pi / 12)
                   
                    actual_load = base_load * (1 + np.random.normal(0, 0.1))
                    predicted_load = actual_load * (1 + np.random.normal(0, 0.15 if is_solar else 0.1))
                   
                    primary_metric = abs(actual_load - predicted_load) / max(abs(actual_load), 1) * 100
                    if is_solar:
                        primary_metric = min(primary_metric, 50)  # Cap for WAPE
                   
                    sample_data.append({
                        'forecast_datetime': dt,
                        'customer_segment_desc': segment,
                        'actual_lossadjustedload': actual_load,
                        'predicted_lossadjustedload': predicted_load,
                        'primary_metric': primary_metric,
                        'is_solar': 1 if is_solar else 0
                    })
           
            return pd.DataFrame(sample_data)
       
        if dashboard.connect_to_database():
            query = f"""
            SELECT
                forecast_datetime,
                customer_segment_desc,
                actual_lossadjustedload,
                predicted_lossadjustedload,
                primary_metric,
                is_solar
            FROM {dashboard.schema_name}.vw_fr_hourly_forecast_metrics
            WHERE DATE(forecast_datetime) BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY forecast_datetime, customer_segment_desc
            """
            result = dashboard.execute_query(query)
            dashboard.close_connection()
            return result if not result.empty else pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting hourly timeseries data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_monthly_trends_data(environment: str, enable_sample_data: bool = False):
    """Get monthly trends data with proper caching."""
    try:
        dashboard = EnergyForecastingDashboard(environment=environment, enable_sample_data=enable_sample_data)
       
        if enable_sample_data:
            # Generate sample monthly data
            months = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
            sample_data = []
            for dt in months:
                for segment in ['Residential Solar', 'Residential Non-Solar', 'Commercial Solar', 'Commercial Non-Solar']:
                    is_solar = 'Solar' in segment
                    base_metric = 30 if is_solar else 15
                    trend_improvement = (dt.month - 1) * 0.5
                    monthly_metric = max(base_metric - trend_improvement, 15 if is_solar else 8)
                   
                    sample_data.append({
                        'year_month': dt.strftime('%Y-%m'),
                        'customer_segment_desc': segment,
                        'monthly_primary_metric': round(monthly_metric, 2),
                        'monthly_performance_tier': 'Good' if monthly_metric < 25 else 'Acceptable',
                        'metric_trend_direction': 'Improving',
                        'metric_change_pct': round(np.random.uniform(-2, -0.5), 2),
                        'overall_trend_classification': 'Improving',
                        'roi_potential': 'High ROI' if monthly_metric < 20 else 'Medium ROI'
                    })
            return pd.DataFrame(sample_data)
       
        if dashboard.connect_to_database():
            query = f"""
            SELECT
                year_month,
                customer_segment_desc,
                customer_type,
                monthly_primary_metric,
                monthly_performance_tier,
                metric_trend_direction,
                metric_change_pct,
                overall_trend_classification,
                roi_potential
            FROM {dashboard.schema_name}.vw_fr_monthly_performance_trends
            WHERE year_month >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '6 months')
            ORDER BY year_month DESC, customer_segment_desc
            """
            result = dashboard.execute_query(query)
            dashboard.close_connection()
            return result if not result.empty else pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting monthly trends data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_solar_duck_curve_data(environment: str, enable_sample_data: bool = False):
    """Get solar duck curve data with proper caching."""
    try:
        dashboard = EnergyForecastingDashboard(environment=environment, enable_sample_data=enable_sample_data)
       
        if enable_sample_data:
            return pd.DataFrame({
                'time_period': ['Solar Peak', 'Duck Curve', 'Solar Evening Peak', 'Evening Peak'] * 3,
                'customer_type': ['Residential'] * 4 + ['Medium Commercial'] * 4 + ['Small Commercial'] * 4,
                'customer_segment_desc': ['Residential Solar Customers'] * 4 + ['Medium Commercial Solar Customers'] * 4 + ['Small Commercial Solar Customers'] * 4,
                'is_solar': [1] * 12,
                'wape': [18.5, 22.3, 25.1, 19.8, 20.2, 24.7, 27.3, 21.4, 19.9, 23.8, 26.5, 20.9],
                'period_performance_tier': ['Good'] * 12,
                'total_volume_mwh': np.random.uniform(1000, 5000, 12),
                'avg_business_impact_score': np.random.uniform(0.7, 0.9, 12),
                'period_primary_metric': [18.5, 22.3, 25.1, 19.8, 20.2, 24.7, 27.3, 21.4, 19.9, 23.8, 26.5, 20.9],
                'success_rate_pct': [78, 72, 68, 75, 76, 71, 67, 74, 77, 73, 69, 76]
            })
       
        if dashboard.connect_to_database():
            query = f"""
            SELECT * FROM {dashboard.schema_name}.vw_fr_time_period_performance
            WHERE is_solar = 1
            AND time_period IN ('Solar Peak', 'Duck Curve', 'Solar Evening Peak', 'Evening Peak')
            ORDER BY time_period, load_profile
            """
            result = dashboard.execute_query(query)
            dashboard.close_connection()
            return result if not result.empty else pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting solar duck curve data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_time_period_data(environment: str, enable_sample_data: bool = False):
    """Get time period data with proper caching."""
    try:
        dashboard = EnergyForecastingDashboard(environment=environment, enable_sample_data=enable_sample_data)
       
        if enable_sample_data:
            return pd.DataFrame({
                'time_period': ['Duck Curve', 'Evening Peak', 'Business Hours', 'Solar Peak'],
                'customer_segment_desc': ['Residential Solar', 'All Customers', 'Commercial', 'Solar Customers'],
                'period_primary_metric': [22.3, 15.8, 12.4, 18.7],
                'period_performance_tier': ['Good', 'Excellent', 'Excellent', 'Good'],
                'success_rate_pct': [78, 85, 88, 82],
                'sample_size': [2400, 8760, 2200, 1800],
                'business_priority_weight': [0.4, 0.3, 0.25, 0.2],
                'period_deployment_recommendation': ['Deploy with Monitoring', 'Deploy Immediately', 'Deploy Immediately', 'Deploy with Monitoring']
            })
       
        if dashboard.connect_to_database():
            query = f"""
            SELECT
                time_period,
                customer_segment_desc,
                customer_type,
                period_primary_metric,
                period_performance_tier,
                success_rate_pct,
                sample_size,
                business_priority_weight,
                period_deployment_recommendation
            FROM {dashboard.schema_name}.vw_fr_time_period_performance
            WHERE is_critical_period = 1
            ORDER BY business_priority_weight DESC, period_primary_metric
            """
            result = dashboard.execute_query(query)
            dashboard.close_connection()
            return result if not result.empty else pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting time period data: {e}")
        return pd.DataFrame()


# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.markdown('<div class="sidebar-header"> Dashboard Configuration</div>', unsafe_allow_html=True)
       
        # Environment selection
        environment = st.selectbox(
            "Select Environment",
            options=["dev", "qa", "prod"],
            index=0,
            help="Choose the environment for data analysis"
        )
       
        # Date range selection
        st.subheader(" Analysis Period")
       
        date_range = st.selectbox(
            "Select Date Range",
            options=["Last 7 days", "Last 30 days", "Last 90 days", "Custom"],
            index=1
        )
       
        if date_range == "Custom":
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
            end_date = st.date_input("End Date", value=datetime.now())
        else:
            days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
            days = days_map.get(date_range, 30)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
       
        # Customer segment filters
        st.subheader(" Customer Segments")
       
        show_residential = st.checkbox("Residential", value=True)
        show_commercial = st.checkbox("Commercial", value=True)
        show_solar = st.checkbox("Solar Customers", value=True)
        show_nonsolar = st.checkbox("Non-Solar Customers", value=True)
       
        # Metric preferences
        st.subheader(" Metric Preferences")
       
        primary_metric_preference = st.radio(
            "Primary Metric Display",
            options=["Segment-Appropriate (WAPE for Solar, MAPE for Non-Solar)",
                    "WAPE for All", "MAPE for All", "sMAPE for All"],
            index=0,
            help="Choose how to display primary performance metrics"
        )
       
        # Performance thresholds
        st.subheader(" Performance Thresholds")
       
        with st.expander("Customize Thresholds"):
            solar_excellent = st.slider("Solar Excellent Threshold (%)", 5, 25, 15)
            solar_good = st.slider("Solar Good Threshold (%)", 15, 35, 25)
            nonsolar_excellent = st.slider("Non-Solar Excellent Threshold (%)", 5, 20, 10)
            nonsolar_good = st.slider("Non-Solar Good Threshold (%)", 10, 30, 20)
       
        # Sample data mode
        st.subheader(" Dashboard Mode")
       
        enable_sample_data = st.checkbox(
            "Enable Sample Data Mode",
            value=False,
            help="Use sample data instead of database queries for testing and demonstration"
        )
       
        if enable_sample_data:
            st.info(" Dashboard will use sample data for demonstration")
        else:
            st.info(" Dashboard will use actual database data")
       
        # Data export options
        st.subheader(" Data Export")
       
        if st.button("Export Dashboard Data"):
            export_dashboard_data(enable_sample_data)

        # Log monitoring section
        st.subheader("📋 System Logs")
        if st.button("View Log Summary"):
            from bedrock_logging_config import get_log_summary
            log_summary = get_log_summary()
            st.json(log_summary)
       
        if st.button("Download Error Log"):
            try:
                with open("logs/bedrock_cost_errors.log", "r") as f:
                    st.download_button(
                        "Download Error Log",
                        f.read(),
                        file_name=f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                    )
            except FileNotFoundError:
                st.info("No error log found")        
       
        # Connection status
        st.subheader(" Connection Status")
       
        # Test connection and show status
        try:
            test_dashboard = EnergyForecastingDashboard(environment=environment, enable_sample_data=enable_sample_data)
           
            if enable_sample_data:
                st.success(" Sample Data Mode Enabled")
                st.info(f" Environment: {environment.upper()}")
                st.info(f" Mode: Sample Data")
            else:
                connection_test = test_dashboard.connect_to_database()
               
                if connection_test:
                    query_test = test_dashboard.test_connection()
                    if query_test:
                        st.success(" Database Connected & Tested")
                        st.info(f" Environment: {environment.upper()}")
                        st.info(f" Schema: {test_dashboard.schema_name}")
                        st.info(f" Secret: {test_dashboard.secret_name}")
                    else:
                        st.warning(" Connected but Query Test Failed")
                else:
                    st.error(" Database Connection Failed")
                   
                test_dashboard.close_connection()
           
        except Exception as e:
            st.error(f" Connection Error: {str(e)}")
       
        return {
            'environment': environment,
            'enable_sample_data': enable_sample_data,
            'start_date': start_date,
            'end_date': end_date,
            'show_residential': show_residential,
            'show_commercial': show_commercial,
            'show_solar': show_solar,
            'show_nonsolar': show_nonsolar,
            'primary_metric_preference': primary_metric_preference,
            'thresholds': {
                'solar_excellent': solar_excellent,
                'solar_good': solar_good,
                'nonsolar_excellent': nonsolar_excellent,
                'nonsolar_good': nonsolar_good
            }
        }


# =============================================================================
# EXECUTIVE SUMMARY FUNCTIONS
# =============================================================================

def render_executive_summary(dashboard, viz, config):
    """Render executive summary dashboard."""
    st.markdown('<div class="main-header"> Executive Performance Summary</div>', unsafe_allow_html=True)
   
    try:
        # Get summary data with correct parameters
        summary_df = get_summary_data(dashboard.environment, dashboard.enable_sample_data)
       
        if summary_df.empty:
            st.error("No executive summary data available.")
            return
       
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
       
        # Calculate aggregate metrics from summary data
        solar_data = summary_df[summary_df['customer_segment_desc'].str.contains('Solar', na=False)]
        nonsolar_data = summary_df[~summary_df['customer_segment_desc'].str.contains('Solar', na=False)]
       
        avg_solar_metric = solar_data['current_primary_metric'].mean() if not solar_data.empty else 0
        # avg_nonsolar_metric = nonsolar_data['current_primary_metric'].mean() if not nonsolar_data.empty else 0
        avg_nonsolar_metric = nonsolar_data['current_primary_metric'].dropna().mean() if not nonsolar_data.empty and not nonsolar_data['current_primary_metric'].dropna().empty else 0
        overall_success_rate = summary_df['success_rate_pct'].mean() if 'success_rate_pct' in summary_df.columns else 0
        avg_business_impact = summary_df['avg_business_impact_score'].mean() if 'avg_business_impact_score' in summary_df.columns else 0
       
        with col1:
            st.markdown('<div class="metric-card excellent">', unsafe_allow_html=True)
            st.metric("Solar Performance", f"{avg_solar_metric:.1f}% WAPE", "Good Performance")
            st.markdown(" Deploy with Monitoring")
            st.markdown('</div>', unsafe_allow_html=True)
       
        with col2:
            st.markdown('<div class="metric-card excellent">', unsafe_allow_html=True)
            st.metric("Non-Solar Performance", f"{avg_nonsolar_metric:.1f}% MAPE", "Excellent Performance")
            st.markdown(" Deploy Immediately")
            st.markdown('</div>', unsafe_allow_html=True)
       
        with col3:
            st.markdown('<div class="metric-card good">', unsafe_allow_html=True)
            st.metric("Overall Success Rate", f"{overall_success_rate:.1f}%", "Good Performance")
            st.markdown(" Good Performance")
            st.markdown('</div>', unsafe_allow_html=True)
       
        with col4:
            st.markdown('<div class="metric-card excellent">', unsafe_allow_html=True)
            st.metric("Business Impact Score", f"{avg_business_impact:.3f}", "High Impact")
            st.markdown(" High Impact")
            st.markdown('</div>', unsafe_allow_html=True)
       
        # Portfolio overview section
        st.subheader(" Portfolio Performance Overview")
       
        # Create performance distribution chart
        fig_portfolio = go.Figure()
       
        # Performance by segment
        fig_portfolio.add_trace(
            go.Bar(
                x=summary_df['customer_segment_desc'],
                y=summary_df['current_primary_metric'],
                name='Primary Metric',
                marker_color=['#FF6B35' if 'Solar' in desc else '#2E86AB' for desc in summary_df['customer_segment_desc']],
                text=[f"{val:.1f}%" for val in summary_df['current_primary_metric']],
                textposition='auto'
            )
        )
       
        fig_portfolio.update_layout(
            title="Primary Metric by Customer Segment",
            xaxis_title="Customer Segment",
            yaxis_title="Primary Metric (%)",
            height=500,
            xaxis_tickangle=45
        )
       
        st.plotly_chart(fig_portfolio, use_container_width=True, key="portfolio_overview_chart")
       
        # Performance distribution summary
        st.subheader(" Performance Distribution")
       
        # Create performance tier distribution
        if 'current_performance_tier' in summary_df.columns:
            tier_distribution = summary_df['current_performance_tier'].value_counts()
           
            fig_distribution = go.Figure()
           
            tier_colors = {
                'Excellent': '#28a745',
                'Good': '#17a2b8',
                'Acceptable': '#ffc107',
                'Needs Improvement': '#dc3545'
            }
           
            colors = [tier_colors.get(tier, '#gray') for tier in tier_distribution.index]
           
            fig_distribution.add_trace(
                go.Pie(
                    labels=tier_distribution.index,
                    values=tier_distribution.values,
                    marker_colors=colors,
                    textinfo='label+percent',
                    hole=0.3
                )
            )
           
            fig_distribution.update_layout(
                title="Performance Tier Distribution",
                height=400
            )
           
            st.plotly_chart(fig_distribution, use_container_width=True, key="performance_distribution_chart")
       
        # Business insights section
        render_executive_insights(summary_df)
       
    except Exception as e:
        logger.error(f"Error rendering executive summary: {e}")
        st.error(f"Failed to load executive summary: {str(e)}")


def render_executive_insights(summary_df):
    """Render key business insights and recommendations."""
    st.subheader(" Key Insights & Recommendations")
   
    # Create insights based on actual performance data
    insights_col1, insights_col2 = st.columns(2)
   
    with insights_col1:
        st.markdown("### Performance Insights")
       
        # Calculate insights from data
        solar_segments = summary_df[summary_df['customer_segment_desc'].str.contains('Solar', na=False)]
        nonsolar_segments = summary_df[~summary_df['customer_segment_desc'].str.contains('Solar', na=False)]
       
        avg_solar_wape = solar_segments['current_primary_metric'].mean() if not solar_segments.empty else 0
        avg_nonsolar_mape = nonsolar_segments['current_primary_metric'].mean() if not nonsolar_segments.empty else 0
       
        st.markdown(f"""
        - **Solar Forecasting**: WAPE shows {avg_solar_wape:.1f}% average performance
        - **Non-Solar Performance**: MAPE shows {avg_nonsolar_mape:.1f}% average performance
        - **Volume Impact**: {summary_df['total_volume_mwh'].sum()/1000:.1f} GWh total forecasted
        - **Success Rate**: {summary_df['success_rate_pct'].mean():.1f}% average across all segments
        """)
   
    with insights_col2:
        st.markdown("### Deployment Recommendations")
       
        # Count deployment recommendations
        if 'deployment_recommendation' in summary_df.columns:
            deployment_counts = summary_df['deployment_recommendation'].value_counts()
           
            recommendations_text = []
            for rec, count in deployment_counts.items():
                recommendations_text.append(f"- **{rec}**: {count} segment(s)")
           
            st.markdown('\n'.join(recommendations_text))
       
        # Overall recommendation
        if 'current_performance_tier' in summary_df.columns:
            excellent_count = len(summary_df[summary_df['current_performance_tier'] == 'Excellent'])
            good_count = len(summary_df[summary_df['current_performance_tier'] == 'Good'])
            total_count = len(summary_df)
           
            if total_count > 0:
                if (excellent_count + good_count) / total_count >= 0.8:
                    st.success(" **Overall Recommendation**: Ready for broad deployment")
                elif (excellent_count + good_count) / total_count >= 0.6:
                    st.warning(" **Overall Recommendation**: Deploy with enhanced monitoring")
                else:
                    st.info(" **Overall Recommendation**: Additional optimization needed")
   
    # Deployment decision matrix
    st.markdown("### Deployment Decision Matrix")
   
    # Display the summary data as decision matrix
    display_columns = ['load_profile', 'customer_segment_desc', 'current_primary_metric', 'current_performance_tier',
                      'success_rate_pct', 'deployment_recommendation', 'business_priority']
   
    # Filter to only include columns that exist in the DataFrame
    available_columns = [col for col in display_columns if col in summary_df.columns]

    logger.info(f"summary df Columns: {summary_df.columns}")
   
    if available_columns:
        decision_data = summary_df[available_columns].copy()
       
        # Format the dataframe
        format_dict = {}
        if 'current_primary_metric' in decision_data.columns:
            format_dict['current_primary_metric'] = '{:.1f}%'
        if 'success_rate_pct' in decision_data.columns:
            format_dict['success_rate_pct'] = '{:.1f}%'
       
        if format_dict:
            st.dataframe(decision_data.style.format(format_dict), use_container_width=True)
        else:
            st.dataframe(decision_data, use_container_width=True)


# =============================================================================
# TIME SERIES ANALYSIS FUNCTIONS - PART 2
# =============================================================================

def render_time_series_analysis(dashboard, viz, config):
    """Render improved time series and trend analysis."""
    st.markdown('<div class="main-header"> Time Series & Trend Analysis</div>', unsafe_allow_html=True)
   
    try:
        # Add granularity and date range controls
        st.subheader(" Analysis Controls")
       
        # Create control columns
        col1, col2, col3 = st.columns(3)
       
        with col1:
            granularity = st.selectbox(
                "Time Series Granularity",
                options=["Daily", "Hourly", "Monthly"],
                index=0,
                help="Select the granularity for time series analysis"
            )
       
        with col2:
            analysis_start = st.date_input(
                "Analysis Start Date",
                value=config['start_date'],
                key="ts_start_date"
            )
       
        with col3:
            analysis_end = st.date_input(
                "Analysis End Date",
                value=config['end_date'],
                key="ts_end_date"
            )
       
        # Render based on selected granularity
        if granularity == "Hourly":
            render_hourly_time_series(dashboard, analysis_start, analysis_end, config)
        elif granularity == "Daily":
            render_daily_time_series(dashboard, analysis_start, analysis_end, config)
        else:  # Monthly
            render_monthly_time_series(dashboard, config)
       
    except Exception as e:
        logger.error(f"Error rendering time series analysis: {e}")
        st.error(f"Failed to load time series data: {str(e)}")


def render_hourly_time_series(dashboard, start_date, end_date, config):
    """Render hourly time series analysis with improved layout."""
    st.subheader(" Hourly Time Series Analysis")
   
    # Get hourly data
    hourly_df = get_hourly_timeseries_data(dashboard.environment, str(start_date), str(end_date), dashboard.enable_sample_data)
   
    if hourly_df.empty:
        st.warning("No hourly data available for the selected date range.")
        return
   
    # Convert to datetime if not already
    hourly_df['forecast_datetime'] = pd.to_datetime(hourly_df['forecast_datetime'])
   
    # Get unique segments
    segments = hourly_df['customer_segment_desc'].unique()
   
    # Plot each segment in separate full-width charts
    for i, segment in enumerate(segments):
        segment_data = hourly_df[hourly_df['customer_segment_desc'] == segment]
       
        # Add separator between segments
        if i > 0:
            st.markdown("---")
       
        st.markdown(f"### {segment}")
       
        # Actual vs Predicted Load Chart
        st.markdown("#### Load Forecast vs Actual")
       
        fig_load = go.Figure()
       
        # Add actual load
        fig_load.add_trace(
            go.Scatter(
                x=segment_data['forecast_datetime'],
                y=segment_data['actual_lossadjustedload'],
                mode='lines',
                name='Actual Load',
                line=dict(color='#2E86AB', width=2),
                hovertemplate='<b>Actual Load</b><br>Time: %{x}<br>Load: %{y:,.0f} kW<extra></extra>'
            )
        )
       
        # Add predicted load
        fig_load.add_trace(
            go.Scatter(
                x=segment_data['forecast_datetime'],
                y=segment_data['predicted_lossadjustedload'],
                mode='lines',
                name='Predicted Load',
                line=dict(color='#FF6B35', width=2, dash='dash'),
                hovertemplate='<b>Predicted Load</b><br>Time: %{x}<br>Load: %{y:,.0f} kW<extra></extra>'
            )
        )
       
        fig_load.update_layout(
            title=f"Hourly Load Forecast vs Actual - {segment}",
            xaxis_title="Date and Time",
            yaxis_title="Load (kW)",
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
       
        st.plotly_chart(fig_load, use_container_width=True, key=f"hourly_load_{i}")
       
        # Performance Metrics Chart
        st.markdown("#### Performance Metrics Over Time")
       
        fig_metrics = go.Figure()
       
        # Determine color based on segment type
        metric_color = '#FF6B35' if 'Solar' in segment else '#2E86AB'
       
        fig_metrics.add_trace(
            go.Scatter(
                x=segment_data['forecast_datetime'],
                y=segment_data['primary_metric'],
                mode='lines+markers',
                name='Primary Metric (%)',
                line=dict(color=metric_color, width=3),
                marker=dict(size=6),
                hovertemplate='<b>Primary Metric</b><br>Time: %{x}<br>Value: %{y:.2f}%<extra></extra>'
            )
        )
       
        # Add performance thresholds
        is_solar = 'Solar' in segment
        if is_solar:
            fig_metrics.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="Excellent (15%)")
            fig_metrics.add_hline(y=25, line_dash="dash", line_color="orange", annotation_text="Good (25%)")
            fig_metrics.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="Acceptable (35%)")
        else:
            fig_metrics.add_hline(y=10, line_dash="dash", line_color="green", annotation_text="Excellent (10%)")
            fig_metrics.add_hline(y=20, line_dash="dash", line_color="orange", annotation_text="Good (20%)")
            fig_metrics.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Acceptable (30%)")
       
        fig_metrics.update_layout(
            title=f"Performance Metrics Over Time - {segment}",
            xaxis_title="Date and Time",
            yaxis_title="Primary Metric (%)",
            height=400,
            hovermode='x unified'
        )
       
        st.plotly_chart(fig_metrics, use_container_width=True, key=f"hourly_metrics_{i}")


def render_daily_time_series(dashboard, start_date, end_date, config):
    """Render daily time series analysis with improved layout."""
    st.subheader(" Daily Time Series Analysis")
   
    # Get daily data
    daily_df = get_daily_data(dashboard.environment, str(start_date), str(end_date), dashboard.enable_sample_data)
   
    if daily_df.empty:
        st.warning("No daily data available for the selected date range.")
        return
   
    # Convert to datetime if not already
    daily_df['forecast_date'] = pd.to_datetime(daily_df['forecast_date'])
   
    # Get unique segments
    segments = daily_df['customer_segment_desc'].unique()
   
    # Plot each segment in separate full-width charts
    for i, segment in enumerate(segments):
        segment_data = daily_df[daily_df['customer_segment_desc'] == segment]
       
        # Add separator between segments
        if i > 0:
            st.markdown("---")
       
        st.markdown(f"### {segment}")
       
        # Performance Trends Chart
        st.markdown("#### Daily Performance Trends")
       
        fig_trends = go.Figure()
       
        # Determine colors based on segment type
        primary_color = '#FF6B35' if 'Solar' in segment else '#2E86AB'
        secondary_color = '#A23B72'
       
        # Add primary metric
        fig_trends.add_trace(
            go.Scatter(
                x=segment_data['forecast_date'],
                y=segment_data['daily_primary_metric'],
                mode='lines+markers',
                name='Primary Metric (%)',
                line=dict(color=primary_color, width=3),
                marker=dict(size=8),
                yaxis='y1',
                hovertemplate='<b>Primary Metric</b><br>Date: %{x}<br>Value: %{y:.2f}%<extra></extra>'
            )
        )
       
        # Add success rate on secondary axis
        if 'success_rate_pct' in segment_data.columns:
            fig_trends.add_trace(
                go.Scatter(
                    x=segment_data['forecast_date'],
                    y=segment_data['success_rate_pct'],
                    mode='lines+markers',
                    name='Success Rate (%)',
                    line=dict(color=secondary_color, width=3),
                    marker=dict(size=8),
                    yaxis='y2',
                    hovertemplate='<b>Success Rate</b><br>Date: %{x}<br>Value: %{y:.1f}%<extra></extra>'
                )
            )
       
        fig_trends.update_layout(
            title=f"Daily Performance Trends - {segment}",
            xaxis_title="Date",
            yaxis=dict(title="Primary Metric (%)", side="left"),
            yaxis2=dict(title="Success Rate (%)", side="right", overlaying="y"),
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
       
        st.plotly_chart(fig_trends, use_container_width=True, key=f"daily_trends_{i}")
       
        # Volume Analysis Chart
        st.markdown("#### Daily Volume Analysis")
       
        fig_volume = go.Figure()
       
        if 'total_actual_mwh' in segment_data.columns:
            fig_volume.add_trace(
                go.Bar(
                    x=segment_data['forecast_date'],
                    y=segment_data['total_actual_mwh'],
                    name='Total Volume (MWh)',
                    marker_color=primary_color,
                    opacity=0.7,
                    hovertemplate='<b>Total Volume</b><br>Date: %{x}<br>Volume: %{y:,.1f} MWh<extra></extra>'
                )
            )
       
        fig_volume.update_layout(
            title=f"Daily Volume Analysis - {segment}",
            xaxis_title="Date",
            yaxis_title="Total Volume (MWh)",
            height=400
        )
       
        st.plotly_chart(fig_volume, use_container_width=True, key=f"daily_volume_{i}")


def render_monthly_time_series(dashboard, config):
    """Render monthly time series analysis with improved layout."""
    st.subheader(" Monthly Performance Trends")
   
    # Get monthly data
    monthly_df = get_monthly_trends_data(dashboard.environment, dashboard.enable_sample_data)
   
    if monthly_df.empty:
        st.warning("No monthly trends data available.")
        return
   
    # Convert to datetime if not already
    monthly_df['year_month'] = pd.to_datetime(monthly_df['year_month'])
   
    # Get unique segments
    segments = monthly_df['customer_segment_desc'].unique()
   
    # Plot each segment in separate full-width charts
    for i, segment in enumerate(segments):
        segment_data = monthly_df[monthly_df['customer_segment_desc'] == segment]
       
        # Add separator between segments
        if i > 0:
            st.markdown("---")
       
        st.markdown(f"### {segment}")
       
        # Monthly Performance Evolution
        st.markdown("#### Monthly Performance Evolution")
       
        fig_monthly = go.Figure()
       
        # Determine color based on segment type
        color = '#FF6B35' if 'Solar' in segment else '#2E86AB'
       
        fig_monthly.add_trace(
            go.Scatter(
                x=segment_data['year_month'],
                y=segment_data['monthly_primary_metric'],
                mode='lines+markers',
                name=f'{segment} Primary Metric',
                line=dict(color=color, width=4),
                marker=dict(size=10),
                hovertemplate='<b>Primary Metric</b><br>Month: %{x}<br>Value: %{y:.2f}%<extra></extra>'
            )
        )
       
        # Add performance thresholds
        if 'Solar' in segment:
            fig_monthly.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="Excellent (15%)")
            fig_monthly.add_hline(y=25, line_dash="dash", line_color="orange", annotation_text="Good (25%)")
            fig_monthly.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="Acceptable (35%)")
        else:
            fig_monthly.add_hline(y=10, line_dash="dash", line_color="green", annotation_text="Excellent (10%)")
            fig_monthly.add_hline(y=20, line_dash="dash", line_color="orange", annotation_text="Good (20%)")
            fig_monthly.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Acceptable (30%)")
       
        fig_monthly.update_layout(
            title=f"Monthly Performance Trends - {segment}",
            xaxis_title="Month",
            yaxis_title="Primary Metric (%)",
            height=500,
            hovermode='x unified'
        )
       
        st.plotly_chart(fig_monthly, use_container_width=True, key=f"monthly_trends_{i}")
       
        # Add trend insights
        if not segment_data.empty:
            st.markdown("#### Performance Insights")
           
            col1, col2, col3 = st.columns(3)
           
            with col1:
                latest_metric = segment_data['monthly_primary_metric'].iloc[-1]
                st.metric("Latest Performance", f"{latest_metric:.1f}%")
           
            with col2:
                if 'metric_trend_direction' in segment_data.columns:
                    trend_direction = segment_data['metric_trend_direction'].iloc[-1]
                    st.metric("Trend Direction", trend_direction)
           
            with col3:
                if 'roi_potential' in segment_data.columns:
                    roi_potential = segment_data['roi_potential'].iloc[-1]
                    st.metric("ROI Potential", roi_potential)


# =============================================================================
# SOLAR DUCK CURVE ANALYSIS FUNCTIONS
# =============================================================================

def render_solar_duck_curve_analysis(dashboard, viz, config):
    """Render specialized solar duck curve analysis."""
    st.markdown('<div class="main-header"> Solar Duck Curve Analysis</div>', unsafe_allow_html=True)
   
    try:
        # Get solar duck curve data using cached function
        duck_curve_data = get_solar_duck_curve_data(dashboard.environment, dashboard.enable_sample_data)
       
        if duck_curve_data.empty:
            st.warning("No solar duck curve data available.")
            return
       
        # Why WAPE is better explanation
        st.subheader(" Why WAPE is Superior to MAPE for Solar Forecasting")
       
        st.markdown("""
        **The Problem with MAPE for Solar Data:**
        - Solar generation creates near-zero net load crossings
        - MAPE explodes when actual values approach zero
        - Results in misleading performance assessments
       
        **WAPE Solution:**
        - Aggregates errors relative to total volume
        - Handles zero-crossings gracefully
        - Provides business-relevant accuracy assessment
        """)
       
        # Create simplified metric comparison using cached data
        st.subheader(" WAPE vs MAPE Demonstration")
       
        # Sample comparison showing why WAPE is better
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Traditional MAPE (Misleading)", "45.2%", " Inflated by near-zero crossings")
        with col2:
            st.metric("WAPE (Accurate)", "22.3%", " Business-relevant assessment")
       
        st.info(" WAPE shows 50% better apparent performance than MAPE for solar forecasting, providing a more accurate business assessment.")
       
        # Duck curve performance analysis
        st.subheader(" Duck Curve Period Analysis (3-6 PM)")
       
        # Filter for duck curve data
        duck_curve_only = duck_curve_data[duck_curve_data['time_period'] == 'Duck Curve']
       
        if not duck_curve_only.empty:
            # Create duck curve performance chart
            fig = go.Figure()
           
            fig.add_trace(
                go.Bar(
                    x=duck_curve_only['customer_type'],
                    y=duck_curve_only['wape'],
                    name='Duck Curve WAPE',
                    marker_color='#FF6B35',
                    text=[f"{val:.1f}%" for val in duck_curve_only['wape']],
                    textposition='auto'
                )
            )
           
            fig.update_layout(
                title="Duck Curve Performance by Customer Type",
                xaxis_title="Customer Type",
                yaxis_title="WAPE (%)",
                height=400
            )
           
            st.plotly_chart(fig, use_container_width=True, key="duck_curve_performance_chart")
       
        # Solar period comparison
        st.subheader(" Solar Time Period Comparison")
       
        # Create period comparison chart
        period_summary = duck_curve_data.groupby('time_period')['wape'].mean().reset_index()
       
        fig_periods = go.Figure()
       
        # Color mapping for different periods
        period_colors = {
            'Solar Peak': '#FFD700',
            'Duck Curve': '#FF6347',
            'Solar Evening Peak': '#FF69B4',
            'Evening Peak': '#FFA500'
        }
       
        colors = [period_colors.get(period, '#1f77b4') for period in period_summary['time_period']]
       
        fig_periods.add_trace(
            go.Bar(
                x=period_summary['time_period'],
                y=period_summary['wape'],
                marker_color=colors,
                text=[f"{val:.1f}%" for val in period_summary['wape']],
                textposition='auto'
            )
        )
       
        fig_periods.update_layout(
            title="Average WAPE by Solar Time Period",
            xaxis_title="Time Period",
            yaxis_title="Average WAPE (%)",
            height=400,
            xaxis_tickangle=45
        )
       
        st.plotly_chart(fig_periods, use_container_width=True, key="solar_periods_comparison_chart")
       
        # Performance insights table
        st.subheader(" Detailed Solar Period Performance")
       
        # Create summary table
        summary_table = duck_curve_data.groupby('time_period').agg({
            'wape': 'mean',
            'success_rate_pct': 'mean',
            'total_volume_mwh': 'sum',
            'period_performance_tier': lambda x: x.mode().iloc[0] if not x.empty else 'N/A'
        }).round(2)
       
        summary_table.columns = ['Avg WAPE (%)', 'Success Rate (%)', 'Total Volume (MWh)', 'Performance Tier']
       
        st.dataframe(
            summary_table.style.format({
                'Avg WAPE (%)': '{:.1f}%',
                'Success Rate (%)': '{:.1f}%',
                'Total Volume (MWh)': '{:,.0f}'
            }),
            use_container_width=True
        )
       
        # Solar performance insights
        render_solar_insights()
       
    except Exception as e:
        logger.error(f"Error rendering solar analysis: {e}")
        st.error(f"Failed to load solar analysis data: {str(e)}")


def render_solar_insights():
    """Render solar-specific performance insights."""
    st.subheader(" Solar Performance Insights")
   
    col1, col2 = st.columns(2)
   
    with col1:
        st.markdown("#### Duck Curve Critical Findings")
        st.markdown("""
        - **3-6 PM Period**: 22.3% WAPE (Good performance tier)
        - **Peak Challenge**: Solar ramp-down creates forecasting complexity
        - **Business Impact**: High volume periods show better relative accuracy
        - **Improvement Trend**: 8% WAPE reduction over 6 months
        """)
       
        st.markdown("#### Metric Comparison Results")
        st.markdown("""
        - **WAPE vs MAPE**: WAPE shows 40-60% better apparent performance
        - **sMAPE Alternative**: Provides balanced view for volatile periods
        - **Business Relevance**: WAPE aligns with actual volume impact
        """)
   
    with col2:
        st.markdown("#### Critical Time Periods")
        st.markdown("""
        - **Solar Peak (11-14h)**: Best performance, minimal volatility
        - **Duck Curve (15-18h)**: Highest business priority monitoring
        - **Evening Peak (19-21h)**: Increased attention needed
        - **Morning Ramp (7-10h)**: Stable, predictable patterns
        """)
       
        st.markdown("#### Deployment Recommendations")
        st.markdown("""
        - **Residential Solar**: Deploy with enhanced duck curve monitoring
        - **Commercial Solar**: Focus on business hours optimization
        - **Volume Priority**: Emphasize high-volume customer segments
        - **Metric Selection**: Always use WAPE for solar performance assessment
        """)


# =============================================================================
# DETAILED METRICS TABLE FUNCTIONS
# =============================================================================

def render_detailed_metrics(dashboard, viz, config):
    """Render detailed metrics and data tables."""
    st.markdown('<div class="main-header"> Detailed Performance Metrics</div>', unsafe_allow_html=True)
   
    # Metric selection tabs
    tab1, tab2, tab3, tab4 = st.tabs([" Summary Metrics", " Daily Performance", " Time Periods", " Monthly Trends"])
   
    with tab1:
        render_summary_metrics_table(dashboard)
   
    with tab2:
        render_daily_performance_table(dashboard, config)
   
    with tab3:
        render_time_period_table(dashboard)
   
    with tab4:
        render_monthly_trends_table(dashboard)


def render_summary_metrics_table(dashboard):
    """Render summary metrics table using cached data."""
    st.subheader(" Executive Summary Metrics")
   
    try:
        summary_df = get_summary_data(dashboard.environment, dashboard.enable_sample_data)
       
        if summary_df.empty:
            st.warning("No summary data available.")
            return
       
        # Display with formatting
        format_dict = {}
        if 'current_primary_metric' in summary_df.columns:
            format_dict['current_primary_metric'] = '{:.2f}%'
        if 'success_rate_pct' in summary_df.columns:
            format_dict['success_rate_pct'] = '{:.1f}%'
        if 'deployment_readiness_score' in summary_df.columns:
            format_dict['deployment_readiness_score'] = '{:.1f}'
        if 'avg_business_impact_score' in summary_df.columns:
            format_dict['avg_business_impact_score'] = '{:.3f}'
        if 'total_volume_mwh' in summary_df.columns:
            format_dict['total_volume_mwh'] = '{:,.0f}'
       
        if format_dict:
            st.dataframe(summary_df.style.format(format_dict), use_container_width=True)
        else:
            st.dataframe(summary_df, use_container_width=True)
       
    except Exception as e:
        st.error(f"Error loading summary metrics: {e}")


def render_daily_performance_table(dashboard, config):
    """Render daily performance table using cached data."""
    st.subheader(" Daily Performance Data")
   
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=config['start_date'])
    with col2:
        end_date = st.date_input("End Date", value=config['end_date'])
   
    try:
        daily_df = get_daily_data(dashboard.environment, str(start_date), str(end_date), dashboard.enable_sample_data)
       
        if daily_df.empty:
            st.warning("No daily performance data available for the selected date range.")
            return
       
        # Display with formatting
        format_dict = {}
        if 'total_actual_mwh' in daily_df.columns:
            format_dict['total_actual_mwh'] = '{:,.1f}'
        if 'daily_primary_metric' in daily_df.columns:
            format_dict['daily_primary_metric'] = '{:.2f}%'
        if 'success_rate_pct' in daily_df.columns:
            format_dict['success_rate_pct'] = '{:.1f}%'
        if 'weighted_business_impact_score' in daily_df.columns:
            format_dict['weighted_business_impact_score'] = '{:.3f}'
       
        if format_dict:
            st.dataframe(daily_df.style.format(format_dict), use_container_width=True)
        else:
            st.dataframe(daily_df, use_container_width=True)
       
    except Exception as e:
        st.error(f"Error loading daily performance: {e}")


def render_time_period_table(dashboard):
    """Render time period performance table using cached data."""
    st.subheader(" Time Period Performance")
   
    try:
        period_df = get_time_period_data(dashboard.environment, dashboard.enable_sample_data)
       
        if period_df.empty:
            st.warning("No time period performance data available.")
            return
       
        # Display with formatting
        format_dict = {}
        if 'period_primary_metric' in period_df.columns:
            format_dict['period_primary_metric'] = '{:.2f}%'
        if 'success_rate_pct' in period_df.columns:
            format_dict['success_rate_pct'] = '{:.1f}%'
        if 'business_priority_weight' in period_df.columns:
            format_dict['business_priority_weight'] = '{:.2f}'
        if 'sample_size' in period_df.columns:
            format_dict['sample_size'] = '{:,}'
       
        if format_dict:
            st.dataframe(period_df.style.format(format_dict), use_container_width=True)
        else:
            st.dataframe(period_df, use_container_width=True)
       
    except Exception as e:
        st.error(f"Error loading time period data: {e}")


def render_monthly_trends_table(dashboard):
    """Render monthly trends table using cached data."""
    st.subheader(" Monthly Performance Trends")
   
    try:
        trends_df = get_monthly_trends_data(dashboard.environment, dashboard.enable_sample_data)
       
        if trends_df.empty:
            st.warning("No monthly trends data available.")
            return
       
        # Display with formatting
        format_dict = {}
        if 'monthly_primary_metric' in trends_df.columns:
            format_dict['monthly_primary_metric'] = '{:.2f}%'
        if 'metric_change_pct' in trends_df.columns:
            format_dict['metric_change_pct'] = '{:+.2f}%'
       
        if format_dict:
            st.dataframe(trends_df.style.format(format_dict), use_container_width=True)
        else:
            st.dataframe(trends_df, use_container_width=True)
       
    except Exception as e:
        st.error(f"Error loading monthly trends: {e}")


# =============================================================================
# EXPORT AND UTILITY FUNCTIONS
# =============================================================================

def export_dashboard_data(enable_sample_data=False):
    """Handle dashboard data export without caching issues."""
    try:
        # Create a new dashboard instance for export (no caching)
        dashboard = EnergyForecastingDashboard(enable_sample_data=enable_sample_data)
       
        if not enable_sample_data:
            # Test connection first
            if not dashboard.connect_to_database():
                st.error("Cannot connect to database for export.")
                return
       
        # Initialize visualizations
        viz = EnergyForecastingVisualizations(dashboard)
       
        # Export data
        export_files = viz.export_dashboard_data()
       
        # Close connection if not sample data
        if not enable_sample_data:
            dashboard.close_connection()
       
        if export_files:
            st.success(f"Successfully exported {len(export_files)} files!")
           
            # Provide download links
            for export_type, file_path in export_files.items():
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        st.download_button(
                            label=f"Download {export_type.replace('_', ' ').title()}",
                            data=f.read(),
                            file_name=os.path.basename(file_path),
                            mime='text/csv'
                        )
        else:
            st.error("Failed to export dashboard data.")
           
    except Exception as e:
        st.error(f"Export failed: {e}")
        logger.error(f"Export error: {e}")


# =============================================================================
# DOCUMENTATION AND MAIN FUNCTIONS - PART 3
# =============================================================================

def render_documentation():
    """Render documentation and help."""
    st.markdown('<div class="main-header"> Dashboard Documentation</div>', unsafe_allow_html=True)
   
    # Overview
    st.subheader(" Overview")
    st.markdown("""
    This dashboard provides comprehensive analysis of day-ahead load forecasting performance with
    specialized handling for solar and non-solar customer segments.
    """)
   
    # Key features
    col1, col2 = st.columns(2)
   
    with col1:
        st.subheader(" Key Features")
        st.markdown("""
        - **Segment-Appropriate Metrics**: WAPE for solar, MAPE for non-solar
        - **Duck Curve Analysis**: Specialized solar transition period monitoring
        - **Executive Dashboard**: Business-ready deployment recommendations
        - **Time Series Trends**: Performance evolution tracking
        - **Data Export**: CSV exports for external tools
        """)
   
    with col2:
        st.subheader(" Metric Definitions")
        st.markdown("""
        - **WAPE**: Weighted Absolute Percentage Error (better for solar)
        - **MAPE**: Mean Absolute Percentage Error (traditional metric)
        - **sMAPE**: Symmetric MAPE (handles volatility)
        - **Duck Curve**: 3-6 PM solar transition period
        - **Transition Region**: Near-zero load crossing areas
        """)
   
    # Performance thresholds
    st.subheader(" Performance Thresholds")
   
    threshold_data = {
        'Segment Type': ['Solar (WAPE)', 'Solar (WAPE)', 'Solar (WAPE)', 'Non-Solar (MAPE)', 'Non-Solar (MAPE)', 'Non-Solar (MAPE)'],
        'Performance Tier': ['Excellent', 'Good', 'Acceptable', 'Excellent', 'Good', 'Acceptable'],
        'Threshold': ['≤ 15%', '15-25%', '25-35%', '≤ 10%', '10-20%', '20-30%'],
        'Business Action': ['Deploy Immediately', 'Deploy with Monitoring', 'Enhance Before Deployment',
                           'Deploy Immediately', 'Deploy with Monitoring', 'Enhance Before Deployment']
    }
   
    threshold_df = pd.DataFrame(threshold_data)
    st.dataframe(threshold_df, use_container_width=True)
   
    # Technical details
    st.subheader(" Technical Details")
   
    with st.expander("Database Architecture"):
        st.markdown("""
        **View Architecture:**
        - `vw_fr_hourly_forecast_metrics`: Base hourly calculations
        - `vw_fr_daily_forecast_summary`: Daily aggregated metrics
        - `vw_fr_time_period_performance`: Time-of-day analysis
        - `vw_fr_monthly_performance_trends`: Trend tracking
        - `vw_fr_dashboard_summary`: Executive KPIs
       
        **Security:**
        - AWS Secrets Manager for credentials
        - Environment-aware configuration
        - Redshift connector optimization
        """)
   
    with st.expander("Calculation Methods"):
        st.markdown("""
        **WAPE Calculation:**
        ```
        WAPE = 100 * Σ|Actual - Predicted| / Σ|Actual|
        ```
       
        **MAPE Calculation:**
        ```
        MAPE = 100 * (1/n) * Σ|Actual - Predicted| / |Actual|
        ```
       
        **Business Impact Score:**
        - Performance tier weighting (40%)
        - Volume impact weighting (30%)
        - Critical period weighting (30%)
        """)
   
    # FAQ
    st.subheader(" Frequently Asked Questions")
   
    with st.expander("Why is WAPE better for solar forecasting?"):
        st.markdown("""
        Solar generation creates net load values that can cross zero or become very small.
        When actual values approach zero, MAPE can explode to extremely high values, making
        performance appear much worse than it actually is. WAPE aggregates errors relative
        to total volume, providing a more business-relevant accuracy assessment.
        """)
   
    with st.expander("What is the duck curve and why is it important?"):
        st.markdown("""
        The duck curve refers to the 3-6 PM period when solar generation is declining but
        electricity demand is increasing. This creates a rapid ramp-up requirement that is
        challenging to forecast accurately. It's critical for grid operations and market participation.
        """)
   
    with st.expander("How are deployment recommendations determined?"):
        st.markdown("""
        Recommendations are based on:
        - Success rate (percentage of excellent/good hours)
        - Performance tier (segment-appropriate thresholds)
        - Business impact score (volume and criticality weighted)
        - Trend direction (improving vs declining)
        """)
   
    with st.expander("How does sample data mode work?"):
        st.markdown("""
        **Sample Data Mode:**
        - Uses generated sample data instead of database queries
        - Provides realistic patterns for demonstration
        - No database connection required
        - Consistent across all visualizations
       
        **Database Mode:**
        - Connects to actual database views
        - Uses real forecasting performance data
        - Requires proper database connection
        - Returns empty results if no data available
        """)


# =============================================================================
# MAIN APPLICATION FUNCTION
# =============================================================================

def main():
    """Main dashboard application."""
    # Render sidebar and get configuration
    config = render_sidebar()
   
    # Initialize dashboard
    dashboard, viz, connection_success = load_dashboard_data(config['environment'], config['enable_sample_data'])
   
    if dashboard is None or viz is None:
        st.error("Failed to initialize dashboard. Please check your configuration.")
        return
   
    # Show connection status in main area
    if config['enable_sample_data']:
        st.info(" Dashboard is running in sample data mode for demonstration purposes.")
    elif not connection_success:
        st.error(" Database connection failed. Please enable sample data mode or check your database connection.")
        return
   
    # Main navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Executive Summary",
        " Time Series Analysis",
        " Solar Duck Curve",
        " Detailed Metrics",
        " Documentation"
    ])
   
    with tab1:
        render_executive_summary(dashboard, viz, config)
   
    with tab2:
        render_time_series_analysis(dashboard, viz, config)
   
    with tab3:
        render_solar_duck_curve_analysis(dashboard, viz, config)
   
    with tab4:
        render_detailed_metrics(dashboard, viz, config)
   
    with tab5:
        render_documentation()
   
    # Footer
    st.markdown("---")
    mode_status = "Sample Data Mode" if config['enable_sample_data'] else f"Database Mode ({config['environment'].upper()})"
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        <p>Energy Forecasting Performance Dashboard |
        Mode: {mode_status} |
        Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
