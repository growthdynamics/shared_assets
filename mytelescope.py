"""
MyTelescope API Python Client

Usage:
    from mytelescope import MyTelescopeClient, fetch_all_states_data, create_us_map_sos
    
    # Single dashboard
    client = MyTelescopeClient(api_key="your_key", dashboard_id="your_id")
    df = client.get_share_of_search()
    
    # Multiple states
    DASHBOARDS = {"California": "abc123", "New York": "xyz789"}
    df_all = fetch_all_states_data(api_key="your_key", dashboards=DASHBOARDS)
    
    # Visualize
    metrics = calculate_yoy_metrics(df_all, brand="Brilliant Earth")
    fig = create_us_map_sos(metrics, brand="Brilliant Earth")
    fig.show()
"""

import requests
import pandas as pd
from io import StringIO
from dataclasses import dataclass
from typing import Optional, Dict, Literal

# =============================================================================
# STATE ABBREVIATION MAPPING
# =============================================================================

STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC"
}


# =============================================================================
# CLIENT
# =============================================================================

@dataclass
class MyTelescopeClient:
    """Client for MyTelescope API"""
    
    api_key: str
    dashboard_id: str
    base_url: str = "https://us-central1-mytelescope-prod.cloudfunctions.net/mytelescope-api/v1"
    
    def _fetch_csv(self, url: str) -> pd.DataFrame:
        """Fetch CSV data and return as DataFrame"""
        response = requests.get(url)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith("Metadata") or line.startswith("[message]"):
                data_start = i + 1
            else:
                break
        
        csv_text = '\n'.join(lines[data_start:]) if data_start > 0 else response.text
        return pd.read_csv(StringIO(csv_text))
    
    def get_share_of_search(
        self,
        chart_type: Literal["line-chart", "pie-chart"] = "line-chart",
        date_range_type: str = "all",
        date_range_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get Share of Search data
        
        Default columns for line-chart: date, topic, volume, percentage
        """
        url = f"{self.base_url}/widget/search-share/{chart_type}/csv?dashboardId={self.dashboard_id}&apiKey={self.api_key}&dateRangeType={date_range_type}"
        if date_range_id:
            url += f"&dateRangeId={date_range_id}"
        return self._fetch_csv(url)
    
    def get_volume_index(
        self,
        chart_type: Literal["line-chart", "pie-chart"] = "line-chart",
        date_range_type: str = "all",
        date_range_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get MyTelescope Index (volume) data
        
        Default columns for line-chart: topic, date, volume, isForecasted
        """
        url = f"{self.base_url}/widget/volume-share/{chart_type}/csv?dashboardId={self.dashboard_id}&apiKey={self.api_key}&dateRangeType={date_range_type}"
        if date_range_id:
            url += f"&dateRangeId={date_range_id}"
        return self._fetch_csv(url)
    
    def get_top_keywords(self, date_range_type: str = "all") -> pd.DataFrame:
        """Get top keywords by volume"""
        url = f"{self.base_url}/widget/top-keywords/list-chart/csv?dashboardId={self.dashboard_id}&apiKey={self.api_key}&dateRangeType={date_range_type}"
        return self._fetch_csv(url)
    
    def get_trending_keywords(self, date_range_type: str = "all") -> pd.DataFrame:
        """Get trending keywords"""
        url = f"{self.base_url}/widget/trending-keywords/list-chart/csv?dashboardId={self.dashboard_id}&apiKey={self.api_key}&dateRangeType={date_range_type}"
        return self._fetch_csv(url)
    
    def get_summary(
        self,
        summary_type: Literal["ai", "custom"] = "ai",
        date_range_type: str = "all",
        date_range_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get AI-generated or custom summary"""
        url = f"{self.base_url}/widget/insights/{summary_type}/csv?dashboardId={self.dashboard_id}&apiKey={self.api_key}&dateRangeType={date_range_type}"
        if date_range_id:
            url += f"&dateRangeId={date_range_id}"
        return self._fetch_csv(url)
    
    def export_keywords(self, date_from: str, date_to: str) -> pd.DataFrame:
        """
        Export all dashboard tracker keywords in TSV format
        
        Args:
            date_from: Start date in YYYY-MM format (e.g., "2024-09")
            date_to: End date in YYYY-MM format (e.g., "2024-11")
        """
        url = f"{self.base_url}/export-dashboard-keywords/tsv?dashboardId={self.dashboard_id}&apiKey={self.api_key}&dateFrom={date_from}&dateTo={date_to}"
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), sep='\t')


# =============================================================================
# MULTI-STATE DATA FETCHING
# =============================================================================

def fetch_all_states_data(api_key: str, dashboards: Dict[str, str]) -> pd.DataFrame:
    """
    Fetch Share of Search data for all states in the dashboard dictionary.
    
    Args:
        api_key: Your MyTelescope API key
        dashboards: Dict mapping state names to dashboard IDs
                   e.g., {"California": "abc123", "New York": "xyz789"}
    
    Returns:
        Combined DataFrame with state and state_abbrev columns added
    """
    all_data = []
    
    for state, dashboard_id in dashboards.items():
        print(f"Fetching {state}...", end=" ")
        
        try:
            client = MyTelescopeClient(api_key=api_key, dashboard_id=dashboard_id)
            df = client.get_share_of_search()
            df['state'] = state
            df['state_abbrev'] = STATE_ABBREV.get(state, state)
            all_data.append(df)
            print(f"✓ ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if not all_data:
        raise ValueError("No data fetched from any state")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    
    return combined


def fetch_all_states_volume(api_key: str, dashboards: Dict[str, str]) -> pd.DataFrame:
    """
    Fetch Volume Index data for all states in the dashboard dictionary.
    """
    all_data = []
    
    for state, dashboard_id in dashboards.items():
        print(f"Fetching volume {state}...", end=" ")
        
        try:
            client = MyTelescopeClient(api_key=api_key, dashboard_id=dashboard_id)
            df = client.get_volume_index()
            df['state'] = state
            df['state_abbrev'] = STATE_ABBREV.get(state, state)
            all_data.append(df)
            print(f"✓ ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if not all_data:
        raise ValueError("No data fetched from any state")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    
    return combined


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_yoy_metrics(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    Calculate YoY change in Share of Search for a specific brand.
    
    Args:
        df: Combined DataFrame from fetch_all_states_data()
        brand: Brand name to filter (matches if topic contains this string)
    
    Returns:
        DataFrame with state-level YoY metrics:
        - current_sos: Average SoS over last 12 months
        - prior_sos: Average SoS 12-24 months ago
        - sos_yoy_change: Percentage point change
        - sos_yoy_pct: Percent change
    """
    # Filter for the brand
    df_brand = df[df['topic'].str.contains(brand, case=False, na=False)].copy()
    
    if df_brand.empty:
        print(f"Warning: No data found for brand '{brand}'")
        print(f"Available topics: {df['topic'].unique()[:10]}")
        return pd.DataFrame()
    
    # Define time periods
    max_date = df_brand['date'].max()
    one_year_ago = max_date - pd.DateOffset(months=12)
    two_years_ago = max_date - pd.DateOffset(months=24)
    
    # Current period (last 12 months)
    current = df_brand[(df_brand['date'] > one_year_ago) & (df_brand['date'] <= max_date)]
    
    # Prior period (12-24 months ago)
    prior = df_brand[(df_brand['date'] > two_years_ago) & (df_brand['date'] <= one_year_ago)]
    
    # Aggregate by state
    current_agg = current.groupby(['state', 'state_abbrev']).agg({
        'percentage': 'mean',
        'volume': 'sum'
    }).rename(columns={'percentage': 'current_sos', 'volume': 'current_volume'})
    
    prior_agg = prior.groupby(['state', 'state_abbrev']).agg({
        'percentage': 'mean',
        'volume': 'sum'
    }).rename(columns={'percentage': 'prior_sos', 'volume': 'prior_volume'})
    
    # Merge and calculate YoY
    metrics = current_agg.join(prior_agg, how='left').reset_index()
    
    metrics['sos_yoy_change'] = metrics['current_sos'] - metrics['prior_sos']
    metrics['sos_yoy_pct'] = ((metrics['current_sos'] - metrics['prior_sos']) / metrics['prior_sos'] * 100).round(1)
    metrics['volume_yoy_pct'] = ((metrics['current_volume'] - metrics['prior_volume']) / metrics['prior_volume'] * 100).round(1)
    
    return metrics


def calculate_total_market_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total market volume and YoY change per state (all brands combined).
    """
    max_date = df['date'].max()
    one_year_ago = max_date - pd.DateOffset(months=12)
    two_years_ago = max_date - pd.DateOffset(months=24)
    
    current = df[(df['date'] > one_year_ago) & (df['date'] <= max_date)]
    prior = df[(df['date'] > two_years_ago) & (df['date'] <= one_year_ago)]
    
    current_vol = current.groupby(['state', 'state_abbrev'])['volume'].sum().rename('current_volume')
    prior_vol = prior.groupby(['state', 'state_abbrev'])['volume'].sum().rename('prior_volume')
    
    metrics = pd.concat([current_vol, prior_vol], axis=1).reset_index()
    metrics['volume_yoy_pct'] = ((metrics['current_volume'] - metrics['prior_volume']) / metrics['prior_volume'] * 100).round(1)
    
    return metrics


# =============================================================================
# STATE COORDINATES FOR LABELS
# =============================================================================

STATE_COORDS = {
    "AL": (32.7, -86.7), "AK": (64.0, -153.0), "AZ": (34.2, -111.6), "AR": (34.8, -92.2),
    "CA": (37.2, -119.4), "CO": (39.0, -105.5), "CT": (41.6, -72.7), "DE": (39.0, -75.5),
    "FL": (28.6, -82.4), "GA": (32.6, -83.4), "HI": (20.8, -156.3), "ID": (44.4, -114.6),
    "IL": (40.0, -89.2), "IN": (39.9, -86.3), "IA": (42.0, -93.5), "KS": (38.5, -98.4),
    "KY": (37.8, -85.7), "LA": (31.0, -92.0), "ME": (45.3, -69.0), "MD": (39.0, -76.8),
    "MA": (42.2, -71.5), "MI": (44.3, -85.4), "MN": (46.3, -94.3), "MS": (32.7, -89.7),
    "MO": (38.3, -92.4), "MT": (47.0, -110.0), "NE": (41.5, -99.8), "NV": (39.3, -116.6),
    "NH": (43.6, -71.5), "NJ": (40.1, -74.7), "NM": (34.4, -106.1), "NY": (42.9, -75.5),
    "NC": (35.5, -79.8), "ND": (47.4, -100.3), "OH": (40.4, -82.8), "OK": (35.6, -97.5),
    "OR": (44.0, -120.5), "PA": (40.9, -77.8), "RI": (41.7, -71.5), "SC": (33.9, -80.9),
    "SD": (44.4, -100.2), "TN": (35.8, -86.3), "TX": (31.5, -99.4), "UT": (39.3, -111.7),
    "VT": (44.0, -72.7), "VA": (37.5, -78.8), "WA": (47.4, -120.5), "WV": (38.9, -80.4),
    "WI": (44.6, -89.7), "WY": (43.0, -107.5), "DC": (38.9, -77.0)
}


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_us_map_with_labels(
    metrics: pd.DataFrame, 
    brand: str = "",
    color_by: str = "sos_yoy_change"
) -> "plotly.graph_objects.Figure":
    """
    Create US choropleth map with text labels showing both YoY metrics on each state.
    
    Args:
        metrics: DataFrame from calculate_yoy_metrics() - must have columns:
                 state_abbrev, sos_yoy_change, volume_yoy_pct
        brand: Brand name for title
        color_by: Column to use for coloring states ('sos_yoy_change' or 'volume_yoy_pct')
    
    Returns:
        Plotly Figure object with state labels
    """
    import plotly.graph_objects as go
    
    # Create choropleth
    fig = go.Figure()
    
    # Round the metrics
    metrics = metrics.copy()
    metrics['sos_yoy_change'] = metrics['sos_yoy_change'].round(1)
    metrics['volume_yoy_pct'] = metrics['volume_yoy_pct'].round(1)
    
    # Add choropleth layer
    fig.add_trace(go.Choropleth(
        locations=metrics['state_abbrev'],
        z=metrics[color_by],
        locationmode='USA-states',
        colorscale='RdYlGn',
        zmid=0,
        text=metrics['state'],
        hovertemplate=(
            '<b>%{text}</b><br>'
            'SoS YoY: %{customdata[0]:+.1f}pp<br>'
            'Vol YoY: %{customdata[1]:+.1f}%<extra></extra>'
        ),
        customdata=metrics[['sos_yoy_change', 'volume_yoy_pct']].values,
        colorbar=dict(
            title='SoS YoY (pp)' if color_by == 'sos_yoy_change' else 'Vol YoY (%)',
            x=1.0
        ),
    ))
    
    # Add text labels for each state
    for _, row in metrics.iterrows():
        abbrev = row['state_abbrev']
        if abbrev in STATE_COORDS:
            lat, lon = STATE_COORDS[abbrev]
            
            # Format the label text - bold and rounded
            sos_change = row['sos_yoy_change']
            vol_change = row['volume_yoy_pct']
            
            label_text = f"<b>{abbrev}<br>SoS: {sos_change:+.1f}pp<br>Vol: {vol_change:+.1f}%</b>"
            
            fig.add_trace(go.Scattergeo(
                lon=[lon],
                lat=[lat],
                text=[label_text],
                mode='text',
                textfont=dict(size=14, color='black', family='Arial Black'),
                showlegend=False,
                hoverinfo='skip',
            ))
    
    fig.update_geos(
        scope='usa',
        bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_layout(
        title=dict(
            text='YoY Search Volume and Share of Search by State',
            x=0.5
        ),
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    return fig


def create_us_map_sos(metrics: pd.DataFrame, brand: str = "") -> "plotly.graph_objects.Figure":
    """
    Create US choropleth map showing YoY Share of Search change by state.
    
    Args:
        metrics: DataFrame from calculate_yoy_metrics()
        brand: Brand name for title
    
    Returns:
        Plotly Figure object
    """
    import plotly.express as px
    
    fig = px.choropleth(
        metrics,
        locations='state_abbrev',
        locationmode='USA-states',
        color='sos_yoy_change',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        scope='usa',
        hover_name='state',
        hover_data={
            'state_abbrev': False,
            'current_sos': ':.1f',
            'prior_sos': ':.1f',
            'sos_yoy_change': ':.1f',
        },
        labels={
            'sos_yoy_change': 'SoS YoY Change (pp)',
            'current_sos': 'Current SoS (%)',
            'prior_sos': 'Prior SoS (%)',
        },
        title=f'{brand} - Share of Search YoY Change by State' if brand else 'Share of Search YoY Change by State'
    )
    
    fig.update_layout(
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        title_x=0.5,
    )
    
    return fig


def create_us_map_volume(metrics: pd.DataFrame) -> "plotly.graph_objects.Figure":
    """
    Create US choropleth map showing total market volume by state.
    
    Args:
        metrics: DataFrame from calculate_total_market_metrics()
    
    Returns:
        Plotly Figure object
    """
    import plotly.express as px
    
    fig = px.choropleth(
        metrics,
        locations='state_abbrev',
        locationmode='USA-states',
        color='current_volume',
        color_continuous_scale='Blues',
        scope='usa',
        hover_name='state',
        hover_data={
            'state_abbrev': False,
            'current_volume': ':,.0f',
            'volume_yoy_pct': ':.1f',
        },
        labels={
            'current_volume': 'Total Volume (12mo)',
            'volume_yoy_pct': 'Volume YoY (%)',
        },
        title='Total Market Search Volume by State (Last 12 Months)'
    )
    
    fig.update_layout(
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        title_x=0.5,
    )
    
    return fig


def create_combined_dashboard(
    sos_metrics: pd.DataFrame, 
    volume_metrics: pd.DataFrame, 
    brand: str = ""
) -> "plotly.graph_objects.Figure":
    """
    Create a combined view with SoS and Volume maps side by side.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
        subplot_titles=[f'{brand} SoS YoY Change (pp)', 'Total Market Volume (12mo)']
    )
    
    # SoS map
    fig.add_trace(
        go.Choropleth(
            locations=sos_metrics['state_abbrev'],
            z=sos_metrics['sos_yoy_change'],
            locationmode='USA-states',
            colorscale='RdYlGn',
            zmid=0,
            text=sos_metrics['state'],
            hovertemplate='<b>%{text}</b><br>SoS YoY: %{z:.1f}pp<extra></extra>',
            colorbar=dict(title='SoS Δ (pp)', x=0.45),
        ),
        row=1, col=1
    )
    
    # Volume map
    fig.add_trace(
        go.Choropleth(
            locations=volume_metrics['state_abbrev'],
            z=volume_metrics['current_volume'],
            locationmode='USA-states',
            colorscale='Blues',
            text=volume_metrics['state'],
            hovertemplate='<b>%{text}</b><br>Volume: %{z:,.0f}<extra></extra>',
            colorbar=dict(title='Volume', x=1.0),
        ),
        row=1, col=2
    )
    
    fig.update_geos(scope='usa')
    fig.update_layout(
        title_text='Lab Diamond Retailer Market Analysis by State',
        title_x=0.5,
        height=500,
    )
    
    return fig


# =============================================================================
# TIME SERIES CHARTS
# =============================================================================

def plot_volume_by_state(
    df: pd.DataFrame, 
    state: str,
    title: str = None
) -> "plotly.graph_objects.Figure":
    """
    Create a time series line chart of search volume by brand for a specific state.
    
    Args:
        df: Combined DataFrame from fetch_all_states_data()
        state: State name to filter (e.g., "California")
        title: Optional custom title
    
    Returns:
        Plotly Figure object
    """
    import plotly.express as px
    
    df_state = df[df['state'] == state].copy()
    
    if df_state.empty:
        raise ValueError(f"No data found for state: {state}")
    
    # Extract brand name from topic (remove state suffix)
    df_state['brand'] = df_state['topic'].str.replace(f' - {state}.*', '', regex=True)
    
    fig = px.line(
        df_state,
        x='date',
        y='volume',
        color='brand',
        title=title or f'MyTelescope Index - {state}',
        labels={'volume': 'Search Volume', 'date': '', 'brand': 'Brand'}
    )
    
    fig.update_layout(
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=450,
    )
    
    return fig


def plot_sos_by_state(
    df: pd.DataFrame, 
    state: str,
    title: str = None
) -> "plotly.graph_objects.Figure":
    """
    Create a time series line chart of share of search by brand for a specific state.
    
    Args:
        df: Combined DataFrame from fetch_all_states_data()
        state: State name to filter (e.g., "California")
        title: Optional custom title
    
    Returns:
        Plotly Figure object
    """
    import plotly.express as px
    
    df_state = df[df['state'] == state].copy()
    
    if df_state.empty:
        raise ValueError(f"No data found for state: {state}")
    
    # Extract brand name from topic (remove state suffix)
    df_state['brand'] = df_state['topic'].str.replace(f' - {state}.*', '', regex=True)
    
    fig = px.line(
        df_state,
        x='date',
        y='percentage',
        color='brand',
        title=title or f'MyTelescope Share of Search - {state}',
        labels={'percentage': 'Share of Search (%)', 'date': '', 'brand': 'Brand'}
    )
    
    fig.update_layout(
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=450,
        yaxis=dict(ticksuffix='%'),
    )
    
    return fig


def plot_all_states_volume(
    df: pd.DataFrame,
    states: list = None
) -> dict:
    """
    Create volume time series charts for multiple states.
    
    Args:
        df: Combined DataFrame from fetch_all_states_data()
        states: List of states to plot. If None, plots all available states.
    
    Returns:
        Dictionary of {state: figure}
    """
    if states is None:
        states = df['state'].unique()
    
    figures = {}
    for state in states:
        try:
            figures[state] = plot_volume_by_state(df, state)
        except ValueError as e:
            print(f"Skipping {state}: {e}")
    
    return figures


def plot_all_states_sos(
    df: pd.DataFrame,
    states: list = None
) -> dict:
    """
    Create share of search time series charts for multiple states.
    
    Args:
        df: Combined DataFrame from fetch_all_states_data()
        states: List of states to plot. If None, plots all available states.
    
    Returns:
        Dictionary of {state: figure}
    """
    if states is None:
        states = df['state'].unique()
    
    figures = {}
    for state in states:
        try:
            figures[state] = plot_sos_by_state(df, state)
        except ValueError as e:
            print(f"Skipping {state}: {e}")
    
    return figures


def generate_state_report(
    df: pd.DataFrame,
    output_path: str = "state_report.pdf",
    states: list = None
):
    """
    Generate a PDF report with Volume and Share of Search charts for each state.
    
    Charts are organized by state: each state gets Volume chart followed by SoS chart.
    
    Args:
        df: Combined DataFrame from fetch_all_states_data()
        output_path: Path for the output PDF file
        states: List of states to include. If None, includes all available states.
    
    Returns:
        Path to the generated PDF
    """
    from plotly.io import write_image
    from PIL import Image
    import io
    import os
    
    if states is None:
        states = sorted(df['state'].unique())
    
    print(f"Generating report for {len(states)} states...")
    
    # Create temporary directory for images
    temp_dir = "/tmp/state_charts"
    os.makedirs(temp_dir, exist_ok=True)
    
    images = []
    
    for i, state in enumerate(states):
        print(f"  [{i+1}/{len(states)}] {state}...", end=" ")
        
        try:
            # Volume chart
            vol_fig = plot_volume_by_state(df, state)
            vol_fig.update_layout(width=1200, height=500)
            vol_path = f"{temp_dir}/{state}_volume.png"
            vol_fig.write_image(vol_path, scale=2)
            images.append(Image.open(vol_path).convert('RGB'))
            
            # SoS chart
            sos_fig = plot_sos_by_state(df, state)
            sos_fig.update_layout(width=1200, height=500)
            sos_path = f"{temp_dir}/{state}_sos.png"
            sos_fig.write_image(sos_path, scale=2)
            images.append(Image.open(sos_path).convert('RGB'))
            
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if not images:
        raise ValueError("No charts generated")
    
    # Save as PDF
    print(f"\nSaving PDF to {output_path}...")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        resolution=150
    )
    
    # Cleanup temp files
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    
    print(f"Done! Report saved to: {output_path}")
    return output_path


def generate_state_report_html(
    df: pd.DataFrame,
    output_path: str = "state_report.html",
    states: list = None
):
    """
    Generate an HTML report with Volume and Share of Search charts for each state.
    
    Charts are organized by state: each state gets Volume chart followed by SoS chart.
    Interactive charts (can zoom, hover, etc.)
    
    Args:
        df: Combined DataFrame from fetch_all_states_data()
        output_path: Path for the output HTML file
        states: List of states to include. If None, includes all available states.
    
    Returns:
        Path to the generated HTML
    """
    import plotly.io as pio
    
    if states is None:
        states = sorted(df['state'].unique())
    
    print(f"Generating HTML report for {len(states)} states...")
    
    html_parts = ["""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Share of Search Report by State</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }
            h2 { color: #666; margin-top: 40px; }
            .chart-container { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .state-section { margin-bottom: 60px; }
            .toc { background: white; padding: 20px; border-radius: 8px; margin-bottom: 40px; }
            .toc a { margin-right: 15px; text-decoration: none; color: #0066cc; }
            .toc a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>Share of Search Report by State</h1>
        <div class="toc">
            <strong>Jump to:</strong><br><br>
    """]
    
    # Add table of contents
    for state in states:
        state_id = state.replace(" ", "_")
        html_parts.append(f'<a href="#{state_id}">{state}</a>')
    
    html_parts.append("</div>")
    
    for i, state in enumerate(states):
        print(f"  [{i+1}/{len(states)}] {state}...", end=" ")
        state_id = state.replace(" ", "_")
        
        try:
            html_parts.append(f'<div class="state-section" id="{state_id}">')
            html_parts.append(f'<h2>{state}</h2>')
            
            # Volume chart
            vol_fig = plot_volume_by_state(df, state)
            vol_fig.update_layout(height=450)
            vol_html = pio.to_html(vol_fig, full_html=False, include_plotlyjs=False)
            html_parts.append(f'<div class="chart-container">{vol_html}</div>')
            
            # SoS chart
            sos_fig = plot_sos_by_state(df, state)
            sos_fig.update_layout(height=450)
            sos_html = pio.to_html(sos_fig, full_html=False, include_plotlyjs=False)
            html_parts.append(f'<div class="chart-container">{sos_html}</div>')
            
            html_parts.append('</div>')
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    html_parts.append("</body></html>")
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write('\n'.join(html_parts))
    
    print(f"\nDone! Report saved to: {output_path}")
    return output_path