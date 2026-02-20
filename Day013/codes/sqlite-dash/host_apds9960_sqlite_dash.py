import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import sqlite3
import paramiko
import os

# --- CONFIGURATION ---
RPI_IP = "192.168.1.106"  # CHANGE THIS to your RPi's IP
RPI_USER = "embai"  # Default RPi username
RPI_PASS = "abc12345"  # CHANGE THIS to your RPi's password
REMOTE_DB_PATH = "/home/embai/sensor_data.db"
LOCAL_DB_COPY = "local_copy.db"


# --- SSH / SFTP FUNCTION ---
def download_db_from_rpi():
    """Connects to RPi via SSH and downloads the latest DB file."""
    print("Fetching database from RPi...", end="")
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(RPI_IP, username=RPI_USER, password=RPI_PASS)

        sftp = ssh.open_sftp()
        sftp.get(REMOTE_DB_PATH, LOCAL_DB_COPY)
        sftp.close()
        ssh.close()
        print(" Done.")
        return True
    except Exception as e:
        print(f" Failed: {e}")
        return False


# --- DASH APP SETUP ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Remote RPi APDS9960 Data", style={'textAlign': 'center'}),

    html.Div([
        html.Button('Refresh Data', id='btn-refresh', n_clicks=0),
        html.Span(id='status-msg', style={'marginLeft': '15px', 'color': 'gray'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    dcc.Graph(id='sensor-graph'),

    # Auto-refresh every 5 seconds (5000ms)
    dcc.Interval(id='interval-timer', interval=5000, n_intervals=0)
])


@app.callback(
    [Output('sensor-graph', 'figure'),
     Output('status-msg', 'children')],
    [Input('interval-timer', 'n_intervals'),
     Input('btn-refresh', 'n_clicks')]
)
def update_graph(n, n_clicks):
    # 1. Download the latest DB from RPi
    success = download_db_from_rpi()
    status = "Last Updated: Just now" if success else "Update Failed (Check Connection)"

    if not os.path.exists(LOCAL_DB_COPY):
        return go.Figure(), "Waiting for data..."

    # 2. Read the LOCAL copy of the database
    try:
        conn = sqlite3.connect(LOCAL_DB_COPY)
        df = pd.read_sql_query("SELECT * FROM readings ORDER BY timestamp DESC LIMIT 100", conn)
        conn.close()

        if df.empty:
            return go.Figure(), status

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 3. Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['clear'], name='Clear', line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['red'], name='Red', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['green'], name='Green', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['blue'], name='Blue', line=dict(color='blue')))

        fig.update_layout(title="Light Sensor History (Last 100 Points)", template="plotly_white")

        return fig, status

    except Exception as e:
        return go.Figure(), f"Error processing data: {e}"


if __name__ == '__main__':
    # Turn off reloader to prevent double-downloading on startup
    app.run(debug=True, use_reloader=False)