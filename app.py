import dash
from dash import dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.sql import text  # Make sure this is imported
import dash_bootstrap_components as dbc
import dash_daq as daq
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import datetime
from datetime import timedelta
import plotly.graph_objects as go
from model_training import load_xgboost_model, forecast_future_failures_xgboost
from data_utils import all_data

# Load the trained XGBoost model
model = load_xgboost_model()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.8.1/font/bootstrap-icons.min.css"])
server = app.server  # For deployment purposes if needed

# Database connection using SQLAlchemy
def combine_data():
    engine = create_engine('postgresql://motor_user:Admin@localhost:5432/motor_db')
    
    # Fetch predictions and acknowledgment status from the database
    query_predictions = """
        SELECT timestamp, motor_id, failure_probability, maintenance_tasks, acknowledged
        FROM motor_failure_predictions
        ORDER BY timestamp DESC
    """
    df_predictions = pd.read_sql(query_predictions, engine)
    
    # Fetch sensor data (temperature, vibration, current)
    query_sensor_data = """
        SELECT timestamp, motor_id, temperature, vibration, current
        FROM motor_sensor_data
        ORDER BY timestamp DESC
    """
    df_sensor = pd.read_sql(query_sensor_data, engine)
    
    # Merge predictions with sensor data on timestamp and motor_id
    df_combined = pd.merge(df_predictions, df_sensor, on=["timestamp", "motor_id"], how="inner")

    # Ensure acknowledgment column exists in combined data
    if 'acknowledged' not in df_combined.columns:
        df_combined['acknowledged'] = False
    
    return df_combined

# Function to get the failure predictions from the database
def fetch_failure_predictions():
    engine = create_engine('postgresql://motor_user:Admin@localhost:5432/motor_db')
    query = """
    SELECT timestamp, failure_probability, motor_id
    FROM motor_failure_predictions
    ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, engine)
    return df

# Cache to store the last checked timestamp
last_checked_timestamp = None

# Define layout with cards, alerts table, and predictions
app.layout = dbc.Container([
   dcc.Store(id='pause-resume-store', data={'paused': False}),
   dcc.Store(id='graph-click-store', data=None),
   dbc.Row([
    dbc.Col(
        html.Div(
            [
                # You can add an icon or logo here if needed
                html.H3("Electrical Motor Predictive Maintainance Dashboard - Powered By Machine Learning", className="display-6"),
                html.P(
                    "Real-time monitoring and failure predictions for motors",
                    className="lead",
                    style={"font-size": "13px","margin-bottom": "5px"}
                ),
                html.Small(
                    "Prince Mashinini",
                    className="text-muted",  # Optional class for muted, smaller text
                    style={"font-size": "11px"}  # You can adjust the size
                )
            ],
            style={
                "textAlign": "center",
                "background": "linear-gradient(145deg, #e0e0e0, #ffffff)",  # Soft gradient
                "border-radius": "10px",
                "margin-bottom": "15px",
                "margin-top": "15px",
                "padding": "20px",
                "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)",  # Subtle shadow for depth
                "display": "block",  # Stack elements vertically
                "align-items": "center",
                "justify-content": "center"
            }
        ),
        width=12
    )
    ]),

    dbc.Row([
        # High-Priority Alerts Table
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.P(
                    id="alerts_total", 
                    className="lead",
                    style={
                        "font-size": "16px",
                        "background": "linear-gradient(145deg, #e0e0e0, #ffffff)",  # Soft gradient
                        "color": "#721c24",  # Dark red text for contrast
                        "padding": "10px 20px",  # Padding for spacing
                        "border-radius": "8px",  # Rounded corners
                        "border": "1px solid #f5c6cb",  # Border matching the background
                        "margin-top": "10px",  # Add space above the paragraph
                        "text-align": "center",  # Center align the text
                        "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)"  # Subtle shadow for depth
                    }
                ),

                dash_table.DataTable(
                    id='high_priority_alerts_table',
                    columns=[
                        {'name': 'Motor ID', 'id': 'motor_id'},
                        {'name': 'Timestamp', 'id': 'timestamp'},
                        {'name': 'Failure Probability', 'id': 'failure_probability'},
                        {'name': 'Temperature (°C)', 'id': 'temperature'},
                        {'name': 'Vibration', 'id': 'vibration'},
                        {'name': 'Current (A)', 'id': 'current'},
                        {'name': 'Acknowledged', 'id': 'acknowledged', 'presentation': 'dropdown'},
                    ],
                    data=[],  # Data will be populated via callback
                    editable=True,
                    dropdown={
                        'acknowledged': {
                            'options': [
                                {'label': 'Yes', 'value': 'Yes'},
                                {'label': 'No', 'value': 'No'}
                            ]
                        }
                    },
                    style_cell={
                        'textAlign': 'center',
                        'backgroundColor': '#f9f9f9',
                        'color': '#333333',
                        'padding': '10px'
                    },
                    style_header={
                        'backgroundColor': '#333333',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'border': '1px solid #dddddd'
                    },
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': '#eeeeee'}
                    ],
                    page_size=5
                )
                ,
                dcc.Interval(id="interval-component-alerts", interval=5* 1000, n_intervals=0)  # 60 seconds interval
            ])
        ], color="danger", inverse=True, className="mb-2"), width=12),
    ]),
    # Add the button here
    html.Div(
                    dbc.Button(
                        "Pause Updates", 
                        id="pause-resume-button", 
                        color="dark",
                        style={
                            "background-color": "#343a40",  # Dark background
                            "color": "white",  # White text
                            "border": "none",  # No border
                            "padding": "10px 20px",  # Padding
                            "border-radius": "5px",  # Rounded corners
                            "cursor": "pointer",  # Pointer cursor
                            "text-align": "center",
                            "font-size": "14px"
                        }
                    ),
                    style={"text-align": "right", "margin-bottom": "10px"}
                ),
    # Modal for detailed alert information
    dbc.Modal([
        dbc.ModalHeader("Alert Details"),
        dbc.ModalBody([
            html.Div(id="alert_modal_body"),
            dbc.Button("Acknowledge", id="acknowledge_button", color="warning", className="mt-2 mx-2"),  # Add Acknowledge button
            dbc.Button("Print Job Card", id="print_job_card", color="success", className="mt-2")
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close_modal", className="ml-auto")
        ),
    ], id="alert_modal", is_open=False),

    # ML Predictions and Suggested Maintenance Cards
    dbc.Row([
        # ML Predictions Card
        # Card for ML predictions
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H5("Current Motor Health Predictions", className="card-title", style={'font-weight': 'bold', 'font-size': '24px'}),
            html.P("Monitor real-time predictions from the ML model", className="card-subtitle", style={'margin-bottom': '15px'}),

            # Prediction Confidence
            html.Div([
                dbc.Row([
                    dbc.Col(html.I(className="bi bi-graph-up"), width=1),  # Bootstrap icon for prediction confidence
                    dbc.Col(html.Span(id="prediction_confidence", className="prediction-text"), width=11),
                ], align="center"),
            ], style={"padding": "10px", "border-bottom": "1px solid #ccc"}),

            # Time to Failure
            html.Div([
                dbc.Row([
                    dbc.Col(html.I(className="bi bi-hourglass-split"), width=1),  # Bootstrap icon for time to failure
                    dbc.Col(html.Span(id="time_to_failure", className="time-to-failure-text"), width=11),
                ], align="center"),
            ], style={"padding": "10px", "border-bottom": "1px solid #ccc"}),

            # Trending Metrics
            html.Div([
                dbc.Row([
                    dbc.Col(html.I(className="bi bi-hourglass-split"), width=1),  # Bootstrap icon for trending metrics
                    dbc.Col(html.Span(id="trending_metrics", className="trending-text"), width=11),
                ], align="center"),
            ], style={"padding": "10px", "border-bottom": "1px solid #ccc"}),

            # Failure Contributors
            html.Div([
                dbc.Row([
                    dbc.Col(html.I(className="bi bi-list-check"), width=1),  # Bootstrap icon for failure contributors
                    dbc.Col(html.Span(id="failure_contributors", className="failure-contributors-text"), width=11),
                ], align="center"),
            ], style={"padding": "10px", "border-bottom": "1px solid #ccc"}),

            # Failure Type Prediction
            html.Div([
                dbc.Row([
                    dbc.Col(html.I(className="bi bi-exclamation-triangle"), width=1),  # Bootstrap icon for failure type prediction
                    dbc.Col(html.Span(id="failure_type_prediction", className="failure-type-prediction-text"), width=11),
                ], align="center"),
            ], style={"padding": "10px"}),
        ])
    ], style={
        "background": "linear-gradient(145deg, #ffffff, #e6e6e6)",  # Soft gradient background
        "box-shadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",  # Subtle box-shadow for depth
        "border-radius": "10px",
        "padding": "20px",
        "margin-bottom": "20px",
        "color": "#333333"  # Darker font color for readability
    }), width=6),

        # Suggested Maintenance and Motor Health cards stacked vertically within 6 columns
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Suggested Actions", className="card-title", style={'font-weight': 'bold', 'font-size': '24px'}),
                    html.Div(id="suggested_actions", style={"padding": "10px"})
                ])
            ], style={
                "background": "linear-gradient(145deg, #ffffff, #e6e6e6)",
                "box-shadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
                "border-radius": "10px",
                "padding": "20px",
                "margin-bottom": "20px",
                "color": "#333333"
            }),
            
            # Motor Health and RUL Card below the Suggested Actions card
            dbc.Card([
                dbc.CardBody([
                    html.H5("Motor Health & Remaining Useful Life", className="card-title", style={'font-weight': 'bold', 'font-size': '24px'}),

                    # Motor Health Score
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.I(className="bi bi-heart-pulse-fill"), width=1),  # Bootstrap icon for health score
                            dbc.Col(html.Span(id="motor_health_score", className="motor-health-score-text"), width=11),
                        ], align="center"),
                    ], style={"padding": "10px", "border-bottom": "1px solid #ccc"}),

                    # Remaining Useful Life (RUL)
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.I(className="bi bi-hourglass-bottom"), width=1),  # Bootstrap icon for RUL
                            dbc.Col(html.Span(id="rul_estimate", className="rul-text"), width=11),
                        ], align="center"),
                    ], style={"padding": "10px", "border-bottom": "1px solid #ccc"}),
                ])
            ], style={
                "background": "linear-gradient(145deg, #ffffff, #e6e6e6)",  # Soft gradient background
                "box-shadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",  # Subtle box-shadow for depth
                "border-radius": "10px",
                "padding": "20px",
                "margin-bottom": "20px",
                "color": "#333333"
            })
        ], width=6)
    ]),
    dbc.Row(
        dbc.Col(
            dcc.Dropdown(
                id='timestamp-dropdown',
                options=[
                    {'label': '5 Minutes', 'value': '5_minutes'},
                    {'label': '10 Minutes', 'value': '10_minutes'},
                    {'label': 'Last Hour', 'value': 'last_hour'},
                    {'label': 'Last Day', 'value': 'last_day'},
                    {'label': 'Last Week', 'value': 'last_week'},
                ],
                value='5_minutes',  # Default value
                clearable=False,
                className="mb-3"  # Bootstrap styling
            ),
            width=6  # Adjust the width as needed
        )
    ),

    # Failure Prediction Graph with real-time updates
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("Failure Probability Forecast", className="card-title", style={"textAlign": "center"})),
            dbc.CardBody([
                dcc.Graph(
                    id="failure_prediction_graph",
                    config={"displayModeBar": True},
                    style={"height": "400px"}  # Adjust the height for better visualization
                ),
                dcc.Graph(
                    id="failure_probability_heatmap",
                    config={"displayModeBar": False},  # Disable unnecessary controls
                    style={"height": "300px"}  # Adjust the height for heatmap visualization
                ),
            ]),
            dbc.CardFooter(
                html.P("This graph shows the historical and predicted failure probabilities.", className="text-muted", style={"textAlign": "center"})
            )
        ], color="light", className="mb-4"), width=12)
    ]),

    dcc.Interval(id="interval-component", interval=5 * 1000, n_intervals=0)  # 60 seconds interval
], fluid=True)


# Callback to toggle the pause/resume state
@app.callback(
    Output('pause-resume-store', 'data'),
    Input('pause-resume-button', 'n_clicks'),
    State('pause-resume-store', 'data'),
    prevent_initial_call=True
)
def toggle_pause_resume(n_clicks, store_data):
    # Toggle the paused state
    is_paused = not store_data['paused']
    return {'paused': is_paused}

# Callback to update the button text based on the pause state
@app.callback(
    Output('pause-resume-button', 'children'),
    Input('pause-resume-store', 'data')
)
def update_button_text(store_data):
    if store_data['paused']:
        return "Resume Updates"
    else:
        return "Pause Updates"
    
# Callback to store click data from the graph
@app.callback(
    Output('graph-click-store', 'data'),
    Input('failure_prediction_graph', 'clickData'),
    prevent_initial_call=True
)
def store_click_data(click_data):
    if click_data:
        return click_data['points'][0]['x']  # Return the timestamp of the clicked point
    return None

# Function to create a well-formatted PDF job card
def create_pdf(job_card_data):
    # Define the directory where job cards will be saved
    directory = "job_cards"
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate a unique file name for the job card (e.g., MotorID_Timestamp.pdf)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{directory}/job_card_{job_card_data['motor_id']}_{timestamp}.pdf"
    
    # Create the PDF for the job card
    c = canvas.Canvas(file_name, pagesize=letter)
    c.setTitle("Job Card")

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 750, "Job Card")
    
    # Add a line separator
    c.setStrokeColor(colors.black)
    c.setLineWidth(2)
    c.line(50, 740, 550, 740)

    # Job Card Information
    c.setFont("Helvetica", 12)
    y_position = 720  # Starting y position for text
    
    # Equipment ID
    c.drawString(100, y_position, f"Equipment ID: {job_card_data['motor_id']}")
    y_position -= 20  # Move down for the next line
    
    # Priority Level
    c.drawString(100, y_position, "Priority Level: High")
    y_position -= 20
    
    # Failure Probability
    c.drawString(100, y_position, f"Failure Probability: {job_card_data['failure_probability']:.2f}")
    y_position -= 20
    
    # Suggested Completion Date
    suggested_date = datetime.date.today() + datetime.timedelta(days=7)
    c.drawString(100, y_position, f"Suggested Completion Date: {suggested_date}")
    y_position -= 20
    
    # Maintenance Tasks
    maintenance_tasks = job_card_data['maintenance_tasks']
    c.drawString(100, y_position, "Maintenance Tasks:")
    y_position -= 15  # Add a bit of space before listing tasks
    for task in maintenance_tasks.split(', '):  # Assuming tasks are comma-separated
        c.drawString(120, y_position, f"- {task}")
        y_position -= 15  # Move down for each task
    
    # Finish the PDF
    c.save()

    return file_name

# Callback for handling modal display and printing job card
@app.callback(
    Output("alert_modal", "is_open"),
    Output("alert_modal_body", "children"),
    Input("high_priority_alerts_table", "active_cell"),
    Input("close_modal", "n_clicks"),
    Input("print_job_card", "n_clicks"),
    Input("acknowledge_button", "n_clicks"),  # Acknowledge button click event
    State("alert_modal", "is_open"),
    State("high_priority_alerts_table", "data"),
)
def handle_modal(active_cell, close_n_clicks, print_n_clicks, acknowledge_n_clicks, is_open, table_data):
    ctx = dash.callback_context
    # Get the name of the button triggering the callback
    button_triggered = ctx.triggered[0]['prop_id'].split('.')[0]

    # Handle opening modal when a row is clicked
    if active_cell and not is_open:
        row = table_data[active_cell['row']]
        
        # Prepare the details to display in the modal
        details = [
            html.P(f"Motor ID: {row['motor_id']}"),
            html.P(f"Timestamp: {row['timestamp']}"),
            html.P(f"Failure Probability: {row['failure_probability']}"),
            html.P(f"Temperature: {row['temperature']} °C"),
            html.P(f"Vibration: {row['vibration']}"),
            html.P(f"Current: {row['current']} A"),
            html.P(f"Maintenance Tasks: {row['maintenance_tasks']}"),
            html.P(f"Acknowledged: {'Yes' if row['acknowledged'] else 'No'}")
        ]
        
        return True, details

    # Handle acknowledgment logic
    if button_triggered == 'acknowledge_button' and active_cell:
        row = table_data[active_cell['row']]
        engine = create_engine('postgresql://motor_user:Admin@localhost:5432/motor_db')
        update_query = text("""
            UPDATE motor_failure_predictions
            SET acknowledged = TRUE
            WHERE timestamp = :timestamp AND motor_id = :motor_id
        """)
        
        with engine.connect() as conn:
            try:
                timestamp = pd.to_datetime(row['timestamp'])
                motor_id = int(row['motor_id'])

                conn.execute(update_query, {
                    'timestamp': timestamp,
                    'motor_id': motor_id
                })
                conn.commit()
                success_message = f"Acknowledged alert for Motor ID {motor_id} at {timestamp}."
                print(success_message)
            except Exception as e:
                success_message = f"Failed to acknowledge alert for Motor ID {motor_id}: {e}"
                print(success_message)

        # Display success or failure message in modal
        return is_open, html.Div([html.P(success_message), html.P("Click Close to exit.")])
    
    # Handle printing the job card
    if button_triggered == 'print_job_card' and active_cell:
        row = table_data[active_cell['row']]
        job_card_data = {
            'motor_id': row['motor_id'],
            'failure_probability': row['failure_probability'],
            'maintenance_tasks': row['maintenance_tasks']
        }
        pdf_file = create_pdf(job_card_data)  # This now saves the file in the 'job_cards' folder with a unique name
        success_message = f"Job card generated and saved as: {pdf_file}"
        print(success_message)
        
        # Display success message in modal
        return is_open, html.Div([html.P(success_message), html.P("Click Close to exit.")])

    # Handle closing the modal
    if close_n_clicks:
        return False, ""

    return is_open, ""


# Initial commissioning values (estimated RUL and health score)
INITIAL_RUL_HOURS = 1000  # For example, 1000 hours
INITIAL_HEALTH_SCORE = 100  # 100% health

# Global variables to track current health and RUL
current_health = INITIAL_HEALTH_SCORE  # Initially set to 100% health
current_rul = INITIAL_RUL_HOURS  # Initially set to 1000 hours

# Global variable to track the last timestamp of the received data
last_timestamp = None

# Function to calculate motor health and remaining useful life
@app.callback(
    [Output("motor_health_score", "children"),
     Output("rul_estimate", "children")],
    Input('interval-component', 'n_intervals'),
    Input('pause-resume-store', 'data')
)
def update_health_and_rul(n_intervals,store_data):
            # Check if updates are paused
    if store_data['paused']:
        return dash.no_update  # Do not update the graph if paused
    global current_health, current_rul, last_timestamp  # Use global variables to persist changes

    # Fetch the latest failure predictions
    df = fetch_failure_predictions()

    # If there's no data, return default placeholder values
    if df.empty:
        return (html.Span("Motor Health: No Data", style={"color": "#333"}), 
                html.Span("Remaining Useful Life: No Data", style={"color": "#333"}))

    # Get the latest timestamp
    latest_timestamp = df['timestamp'].max()

    # Check if there's new data (i.e., new timestamp)
    if last_timestamp == latest_timestamp:
        # No new data, so we keep the current health and RUL without any updates
        return (html.Span(f"Motor Health: {current_health:.0f}/100"), 
                f"Remaining Useful Life: {current_rul:.0f} hours")

    # Update the last timestamp to the latest one
    last_timestamp = latest_timestamp

    # Calculate degradation based on historical predictions
    latest_prediction = df.iloc[-1]  # Get the latest prediction
    previous_prediction = df.iloc[-2] if len(df) > 1 else latest_prediction  # Get previous if available

    # Current failure probability
    prediction_prob = float(latest_prediction['failure_probability'])

    # Degradation logic: adjust health and RUL based on failure probability trend
    # We reduce the health and RUL more drastically if the probability of failure is higher
    if prediction_prob > 0.8:
        # Critical state, rapid degradation
        health_delta = -5  # Lose 5% health
        rul_delta = -50  # Lose 50 hours of RUL
    elif 0.5 < prediction_prob <= 0.8:
        # Moderate degradation
        health_delta = -3  # Lose 3% health
        rul_delta = -30  # Lose 30 hours of RUL
    elif 0.2 < prediction_prob <= 0.5:
        # Slower degradation
        health_delta = -1  # Lose 1% health
        rul_delta = -10  # Lose 10 hours of RUL
    else:
        # Minimal degradation, if failure probability is very low
        health_delta = 0  # No health loss
        rul_delta = 0  # No RUL loss

    # Update the global variables
    current_health = max(current_health + health_delta, 0)  # Ensure health doesn't go negative
    current_rul = max(current_rul + rul_delta, 0)  # Ensure RUL doesn't go negative

    # Color logic for motor health score
    if current_health >= 80:
        health_status = "Good"
        color = "#28a745"  # Green for Good health
    elif 50 <= current_health < 80:
        health_status = "Moderate Risk"
        color = "#ffc107"  # Yellow for Moderate Risk
    else:
        health_status = "High Risk"
        color = "#dc3545"  # Red for High Risk

    motor_health_score_text = html.Span(f"Motor Health: {current_health:.0f}/100 ({health_status})", style={"color": color})
    rul_estimate_text = f"Remaining Useful Life: {current_rul:.0f} hours"

    return motor_health_score_text, rul_estimate_text


@app.callback(
    Output('failure_prediction_graph', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('timestamp-dropdown', 'value'),
    Input('pause-resume-store', 'data')
)
def update_failure_prediction_graph(n_intervals, selected_timestamp,store_data):
        # Check if updates are paused
    if store_data['paused']:
        return dash.no_update  # Do not update the graph if paused
    # Step 1: Fetch all failure predictions
    df = fetch_failure_predictions()

    # Step 2: If there's no data, return an empty figure
    if df.empty:
        return {}

    # Step 3: Filter data based on selected timestamp
    if selected_timestamp == '5_minutes':
        time_filter = pd.Timestamp.now() - pd.Timedelta(minutes=5)
    elif selected_timestamp == '10_minutes':
        time_filter = pd.Timestamp.now() - pd.Timedelta(minutes=10)
    elif selected_timestamp == 'last_hour':
        time_filter = pd.Timestamp.now() - pd.Timedelta(hours=1)
    elif selected_timestamp == 'last_day':
        time_filter = pd.Timestamp.now() - pd.Timedelta(days=1)
    elif selected_timestamp == 'last_week':
        time_filter = pd.Timestamp.now() - pd.Timedelta(weeks=1)
    else:
        time_filter = pd.Timestamp.now() - pd.Timedelta(minutes=5)

    # Filter DataFrame based on the timestamp
    df = df[df['timestamp'] >= time_filter]
    df_forecast = all_data()

    # Step 4: Use the XGBoost model to forecast future failures
    forecast_df = forecast_future_failures_xgboost(df_forecast, model, forecast_period_seconds=60)

    # Combine the original and forecasted data
    combined_df = pd.concat([df, forecast_df], ignore_index=True)

    # Step 5: Create the original line for historical data
    fig = px.line(
        df,
        x='timestamp',
        y='failure_probability',
        color='motor_id',
        title="Failure Probability and Future Forecast",
        markers=True,
    )

    # Step 6: Add a separate trace for forecasted data using go.Scatter
    forecast_trace = go.Scatter(
        x=forecast_df['timestamp'],
        y=forecast_df['failure_probability'],
        mode='lines',
        line=dict(dash='dash', color='orange'),
        name="Forecasted Data"
    )

    fig.add_trace(forecast_trace)

    # Update layout for better aesthetics
    fig.update_layout(
        title="Failure Probability with 60-Second Forecast",
        xaxis_title="Time",
        yaxis_title="Failure Probability",
        template="plotly_white"
    )

    return fig


@app.callback(
    Output('failure_probability_heatmap', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('timestamp-dropdown', 'value'),  # Add the timestamp dropdown input
    Input('pause-resume-store', 'data')
)
def update_failure_heatmap(n_intervals, selected_timestamp, store_data):
    # Check if updates are paused
    if store_data['paused']:
        return dash.no_update  # Do not update the graph if paused
    # Fetch the latest failure predictions
    df = fetch_failure_predictions()

    # If there's no data, return an empty figure
    if df.empty:
        return go.Figure()

    # Convert timestamp to datetime and filter data based on the selected timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter data based on the selected timestamp
    if selected_timestamp == '5_minutes':
        time_filter = pd.Timestamp.now() - pd.Timedelta(minutes=5)
    elif selected_timestamp == '10_minutes':
        time_filter = pd.Timestamp.now() - pd.Timedelta(minutes=10)
    elif selected_timestamp == 'last_hour':
        time_filter = pd.Timestamp.now() - pd.Timedelta(hours=1)
    elif selected_timestamp == 'last_day':
        time_filter = pd.Timestamp.now() - pd.Timedelta(days=1)
    elif selected_timestamp == 'last_week':
        time_filter = pd.Timestamp.now() - pd.Timedelta(weeks=1)
    else:
        time_filter = pd.Timestamp.now() - pd.Timedelta(minutes=5)  # Default to 5 minutes

    # Filter the DataFrame based on the timestamp
    df = df[df['timestamp'] >= time_filter]

    # Prepare data for the heatmap
    df['time_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Convert to string for the heatmap

    # Create the heatmap trace with yellow to red color scale
    heatmap = go.Heatmap(
        x=df['time_str'],  # Timestamps on the x-axis
        y=df['motor_id'],  # Motor IDs on the y-axis
        z=df['failure_probability'],  # Failure probabilities as the heatmap colors
        colorscale=[  # Custom color scale: yellow for low, red for high
            [0, 'yellow'],   # Lowest failure probability (0) is yellow
            [1, 'red']       # Highest failure probability (1) is red
        ],
        colorbar=dict(title="Failure Probability", titleside='right')
    )

    # Create the heatmap figure
    fig = go.Figure(data=[heatmap])

    # Update layout for the heatmap to match dashboard styling
    fig.update_layout(
        title="Real-Time Failure Probability Heatmap",
        xaxis_title="Timestamp",
        yaxis_title="Motor ID",
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_nticks=10,  # Limit the number of x-ticks for readability
    )

    return fig

def match_timestamp_with_tolerance(df, clicked_time, tolerance='1ms'):
    # Convert the clicked timestamp to a pandas Timestamp
    clicked_time = pd.to_datetime(clicked_time)
    
    # Convert timestamps in the DataFrame to pandas Timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Find rows where the timestamp is within the tolerance range
    mask = (df['timestamp'] >= clicked_time - pd.Timedelta(tolerance)) & (df['timestamp'] <= clicked_time + pd.Timedelta(tolerance))
    df_clicked = df[mask]

    return df_clicked

@app.callback(
    [Output("prediction_confidence", "children"),
     Output("time_to_failure", "children"),
     Output("trending_metrics", "children"),
     Output("failure_contributors", "children"),
     Output("failure_type_prediction", "children"),
     Output("suggested_actions", "children")],
    Input('interval-component', 'n_intervals'),
    Input('graph-click-store', 'data'),
    Input('pause-resume-store', 'data')
)

# Prince Mashinini
def update_cards(n_intervals, clicked_data, pause_resume_state):
    # Fetch the latest failure predictions
    df = all_data()

    # If there's no data, return default placeholder values
    if df.empty:
        return ("No Prediction Data Available", 
                "No Data Available", 
                "No Trend Data Available", 
                "No Failure Contributors Available", 
                "No Health Data Available", 
                "No Failure Type Prediction Available",
                html.Ul([html.Li("No immediate actions required.")]))  # For suggested actions
    
    # Check if the dashboard is paused and there's clicked data
    if pause_resume_state['paused'] and clicked_data:
        df_clicked = match_timestamp_with_tolerance(df, clicked_data)
        latest_prediction = df_clicked.iloc[-1]

        # Prediction Probability and Confidence
        try:
            prediction_prob = float(latest_prediction['failure_probability'])
            confidence_interval = (prediction_prob - 0.05, prediction_prob + 0.05)  # Static confidence interval
            prediction_confidence_text = (f"Prediction: {prediction_prob:.2f} "
                                        f"(95% Confidence Interval: {confidence_interval[0]:.2f} - {confidence_interval[1]:.2f})")
        except (KeyError, ValueError):
            prediction_confidence_text = "Invalid Prediction Data"

        # Time to Failure Estimate based on prediction probability
        if prediction_prob > 0.8:
            time_to_failure_text = "Estimated Time to Failure: Immediate to a Few hours"
        elif 0.5 < prediction_prob <= 0.8:
            time_to_failure_text = "Estimated Time to Failure: Within a few hours to 1 day"
        elif 0.2 < prediction_prob <= 0.5:
            time_to_failure_text = "Estimated Time to Failure: 48 to 72 hours"
        else:
            time_to_failure_text = "Estimated Time to Failure: More than 3 days, possibly weeks"

        # Trending Metrics (change in failure probability over time)
        if len(df) > 1:
            previous_prediction = df.iloc[-2]['failure_probability']

            if previous_prediction == 0:
                # Handle the case where the previous prediction is 0 to avoid division by zero
                trend_percentage_change = "N/A (Previous probability was 0)"
                trending_metrics_text = "No meaningful trend data (Previous probability was 0)"
            else:
                trend_percentage_change = ((prediction_prob - previous_prediction) / previous_prediction) * 100
                trending_metrics_text = f"Failure Probability Change: {trend_percentage_change:.2f}% from previous prediction"
        else:
            trending_metrics_text = "No Trend Data Available"

        # Failure Contributors (key factors contributing to the failure prediction)
        failure_contributors_text = []
        if latest_prediction.get('vibration', None) and latest_prediction['vibration'] > 3.0:
            failure_contributors_text.append("High Vibration")
        if latest_prediction.get('temperature', None) and latest_prediction['temperature'] > 95:
            failure_contributors_text.append("High Temperature")
        if latest_prediction.get('current', None):
            if latest_prediction['current'] > 10:
                failure_contributors_text.append("High Current")
            elif latest_prediction['current'] < 3:
                failure_contributors_text.append("Low Current")
        
        if failure_contributors_text:
            failure_contributors = ", ".join(failure_contributors_text)
            failure_contributors_text = f"Key Contributors: {failure_contributors}"
        else:
            failure_contributors_text = "No Significant Failure Contributors Identified"

        # Failure Type Prediction (based on contributors or trends)
        if "High Temperature" in failure_contributors_text:
            failure_type_prediction_text = "Likely Failure Type: Overheating , Bearing Failure ,Stator Winding Damage, Rotor Bar Damage, Cooling System Malfunction."
        elif "High Vibration" in failure_contributors_text:
            failure_type_prediction_text = "Likely Failure Type: Bearing Failure, Rotor Imbalance, Misalignment, Loose Components, Foundation Issues."
        elif "High Current" in failure_contributors_text:
            failure_type_prediction_text = "Likely Failure Type: Electrical Overload, Short Circuit, Rotor Bar Damage, Insulation Failure, Power Supply Issues."
        else:
            failure_type_prediction_text = "No Specific Failure Type Predicted"

        # Suggested Maintenance Tasks
        maintenance_tasks = latest_prediction.get('maintenance_tasks', '')
        sensor_vibration = latest_prediction.get('vibration', None)
        sensor_temperature = latest_prediction.get('temperature', None)
        sensor_current = latest_prediction.get('current', None)

        suggested_actions = []

        # High Failure Probability (> 0.8) => Immediate attention
        if prediction_prob > 0.8:
            suggested_actions.append(html.P("Condition: The motor is in a critical state, with significant operational anomalies indicating an imminent failure."))
            suggested_actions.append(html.P("Recommended Actions:"))
            suggested_actions.append(html.Li("Immediate shutdown, emergency maintenance, root cause analysis, prepare replacement motor."))
        
        if 0.5 < prediction_prob <= 0.8:
            suggested_actions.append(html.P("Condition: The motor is under significant stress, and failure is likely within the next few hours or by the end of the day."))
            suggested_actions.append(html.P("Recommended Actions:"))
            suggested_actions.append(html.Li("Immediate maintenance, reduce load, continuous monitoring, check for imbalances, prepare for repair or replacement."))  

        if 0.2 < prediction_prob <= 0.5:
            suggested_actions.append(html.P("Condition: The motor shows signs of moderate wear or stress. Some operational values may be trending higher but are still within acceptable ranges."))
            suggested_actions.append(html.P("Recommended Actions:"))
            suggested_actions.append(html.Li("Schedule short-term preventive maintenance, inspect bearings, perform diagnostic tests, monitor trends."))  

        # Vibration Issues
        if sensor_vibration is not None and sensor_vibration > 3.0:
            suggested_actions.append(html.Li("High Vibration Levels: Inspect motor for possible misalignment or worn bearings due to high vibration."))

        # Temperature Issues
        if sensor_temperature is not None and sensor_temperature > 95:
            suggested_actions.append(html.Li("High Temperature Levels: Check the motor cooling system and ensure proper airflow to prevent overheating."))

        # Current Issues
        if sensor_current is not None:
            if sensor_current < 3:
                suggested_actions.append(html.Li("Low Current Levels: Investigate electrical system for possible loose connections or power supply issues (Low Current)."))
            elif sensor_current > 10:
                suggested_actions.append(html.Li("High Current Levels: Check motor for overloading or electrical imbalance (High Current)."))

        if not suggested_actions:
            suggested_actions.append(html.Li("Continue routine maintenance, inspect for minor issues, optimize operation, monitor data trends."))

        suggested_actions_text = html.Ul(suggested_actions)

        return (prediction_confidence_text, 
                time_to_failure_text, 
                trending_metrics_text, 
                failure_contributors_text, 
                failure_type_prediction_text,
                suggested_actions_text)      
        

    # If the dashboard is not paused, proceed with the regular real-time update
    if not pause_resume_state['paused']:
        # Get the latest prediction
        latest_prediction = df.iloc[-1]

        # Prediction Probability and Confidence
        try:
            prediction_prob = float(latest_prediction['failure_probability'])
            confidence_interval = (prediction_prob - 0.05, prediction_prob + 0.05)  # Static confidence interval
            prediction_confidence_text = (f"Prediction: {prediction_prob:.2f} "
                                        f"(95% Confidence Interval: {confidence_interval[0]:.2f} - {confidence_interval[1]:.2f})")
        except (KeyError, ValueError):
            prediction_confidence_text = "Invalid Prediction Data"

        # Time to Failure Estimate based on prediction probability
        if prediction_prob > 0.8:
            time_to_failure_text = "Estimated Time to Failure: Immediate to a Few hours"
        elif 0.5 < prediction_prob <= 0.8:
            time_to_failure_text = "Estimated Time to Failure: Within a few hours to 1 day"
        elif 0.2 < prediction_prob <= 0.5:
            time_to_failure_text = "Estimated Time to Failure: 48 to 72 hours"
        else:
            time_to_failure_text = "Estimated Time to Failure: More than 3 days, possibly weeks"

        # Trending Metrics (change in failure probability over time)
        if len(df) > 1:
            previous_prediction = df.iloc[-2]['failure_probability']

            if previous_prediction == 0:
                # Handle the case where the previous prediction is 0 to avoid division by zero
                trend_percentage_change = "N/A (Previous probability was 0)"
                trending_metrics_text = "No meaningful trend data (Previous probability was 0)"
            else:
                trend_percentage_change = ((prediction_prob - previous_prediction) / previous_prediction) * 100
                trending_metrics_text = f"Failure Probability Change: {trend_percentage_change:.2f}% from previous prediction"
        else:
            trending_metrics_text = "No Trend Data Available"

        # Failure Contributors (key factors contributing to the failure prediction)
        failure_contributors_text = []
        if latest_prediction.get('vibration', None) and latest_prediction['vibration'] > 3.0:
            failure_contributors_text.append("High Vibration")
        if latest_prediction.get('temperature', None) and latest_prediction['temperature'] > 95:
            failure_contributors_text.append("High Temperature")
        if latest_prediction.get('current', None):
            if latest_prediction['current'] > 10:
                failure_contributors_text.append("High Current")
            elif latest_prediction['current'] < 3:
                failure_contributors_text.append("Low Current")
        
        if failure_contributors_text:
            failure_contributors = ", ".join(failure_contributors_text)
            failure_contributors_text = f"Key Contributors: {failure_contributors}"
        else:
            failure_contributors_text = "No Significant Failure Contributors Identified"

        # Failure Type Prediction (based on contributors or trends)
        if "High Temperature" in failure_contributors_text:
            failure_type_prediction_text = "Likely Failure Type: Overheating , Bearing Failure ,Stator Winding Damage, Rotor Bar Damage, Cooling System Malfunction."
        elif "High Vibration" in failure_contributors_text:
            failure_type_prediction_text = "Likely Failure Type: Bearing Failure, Rotor Imbalance, Misalignment, Loose Components, Foundation Issues."
        elif "High Current" in failure_contributors_text:
            failure_type_prediction_text = "Likely Failure Type: Electrical Overload, Short Circuit, Rotor Bar Damage, Insulation Failure, Power Supply Issues."
        else:
            failure_type_prediction_text = "No Specific Failure Type Predicted"

        # Suggested Maintenance Tasks
        maintenance_tasks = latest_prediction.get('maintenance_tasks', '')
        sensor_vibration = latest_prediction.get('vibration', None)
        sensor_temperature = latest_prediction.get('temperature', None)
        sensor_current = latest_prediction.get('current', None)

        suggested_actions = []

        # High Failure Probability (> 0.8) => Immediate attention
        if prediction_prob > 0.8:
            suggested_actions.append(html.P("Condition: The motor is in a critical state, with significant operational anomalies indicating an imminent failure."))
            suggested_actions.append(html.P("Recommended Actions:"))
            suggested_actions.append(html.Li("Immediate shutdown, emergency maintenance, root cause analysis, prepare replacement motor."))
        
        if 0.5 < prediction_prob <= 0.8:
            suggested_actions.append(html.P("Condition: The motor is under significant stress, and failure is likely within the next few hours or by the end of the day."))
            suggested_actions.append(html.P("Recommended Actions:"))
            suggested_actions.append(html.Li("Immediate maintenance, reduce load, continuous monitoring, check for imbalances, prepare for repair or replacement."))  

        if 0.2 < prediction_prob <= 0.5:
            suggested_actions.append(html.P("Condition: The motor shows signs of moderate wear or stress. Some operational values may be trending higher but are still within acceptable ranges."))
            suggested_actions.append(html.P("Recommended Actions:"))
            suggested_actions.append(html.Li("Schedule short-term preventive maintenance, inspect bearings, perform diagnostic tests, monitor trends."))  

        # Vibration Issues
        if sensor_vibration is not None and sensor_vibration > 3.0:
            suggested_actions.append(html.Li("High Vibration Levels: Inspect motor for possible misalignment or worn bearings due to high vibration."))

        # Temperature Issues
        if sensor_temperature is not None and sensor_temperature > 95:
            suggested_actions.append(html.Li("High Temperature Levels: Check the motor cooling system and ensure proper airflow to prevent overheating."))

        # Current Issues
        if sensor_current is not None:
            if sensor_current < 3:
                suggested_actions.append(html.Li("Low Current Levels: Investigate electrical system for possible loose connections or power supply issues (Low Current)."))
            elif sensor_current > 10:
                suggested_actions.append(html.Li("High Current Levels: Check motor for overloading or electrical imbalance (High Current)."))

        if not suggested_actions:
            suggested_actions.append(html.Li("Continue routine maintenance, inspect for minor issues, optimize operation, monitor data trends."))

        suggested_actions_text = html.Ul(suggested_actions)

        return (prediction_confidence_text, 
                time_to_failure_text, 
                trending_metrics_text, 
                failure_contributors_text, 
                failure_type_prediction_text,
                suggested_actions_text)
    
    # If paused but no clicked data, maintain the current state of the cards (no update)
    return dash.no_update


@app.callback(
    Output("high_priority_alerts_table", "data"),
    Output("alerts_total", "children"),
    Input("interval-component-alerts", "n_intervals"),
    Input("timestamp-dropdown", "value"),  # Keep the dropdown for timestamp filtering
    Input("high_priority_alerts_table", "data_timestamp"),  # Triggers when user interacts with the table
    State("high_priority_alerts_table", "data")  # Keeps track of the table's data state
)
def update_dashboard(n_intervals, selected_timestamp, data_timestamp, existing_data):
    # Fetch updated data
    df = combine_data()
    # Filter high-priority alerts where failure probability > 0.8
   # Filter high-priority alerts where failure probability > 0.8 and acknowledged is False
    high_priority = df[(df['failure_probability'] > 0.5) & (df['acknowledged'] == False)]

    # Filter data based on selected timestamp
    if selected_timestamp == '5_minute':
        time_filter = pd.Timestamp.now() - pd.Timedelta(minutes=5)
    elif selected_timestamp == '10_minutes':
        time_filter = pd.Timestamp.now() - pd.Timedelta(minutes=10)
    elif selected_timestamp == 'last_hour':
        time_filter = pd.Timestamp.now() - pd.Timedelta(hours=1)
    elif selected_timestamp == 'last_day':
        time_filter = pd.Timestamp.now() - pd.Timedelta(days=1)
    elif selected_timestamp == 'last_week':
        time_filter = pd.Timestamp.now() - pd.Timedelta(weeks=1)
    else:
        time_filter = pd.Timestamp.now() - pd.Timedelta(minutes=5)  # Default to last day

    # Filter DataFrame based on the timestamp
    high_priority = high_priority[high_priority['timestamp'] >= time_filter]

    # Initialize all acknowledgment statuses to "No" in the dropdown
    high_priority['acknowledged'] = 'No'  # Always start with "No"

    # If the user has manually updated the acknowledgment, prevent overwriting
    if existing_data:
        existing_df = pd.DataFrame(existing_data)
        for index, row in existing_df.iterrows():
            if row['acknowledged'] == 'Yes':  # User manually selected 'Yes'
                # Update the 'acknowledged' status to 'Yes' in the new data
                high_priority.loc[
                    (high_priority['motor_id'] == row['motor_id']) &
                    (high_priority['timestamp'] == row['timestamp']),
                    'acknowledged'
                ] = 'Yes'

    # If user changed 'acknowledged' to 'Yes', update the database
    if existing_data:
        engine = create_engine('postgresql://motor_user:Admin@localhost:5432/motor_db')
        update_query = text("""
            UPDATE motor_failure_predictions
            SET acknowledged = TRUE
            WHERE timestamp = :timestamp AND motor_id = :motor_id
        """)

        with engine.connect() as conn:
            for row in existing_data:
                if row['acknowledged'] == 'Yes':  # Only update for rows marked as 'Yes'
                    try:
                        # Ensure correct data types for query parameters
                        timestamp = pd.to_datetime(row['timestamp'])  # Ensure correct timestamp format
                        motor_id = int(row['motor_id'])

                        conn.execute(update_query, {
                            'timestamp': timestamp,
                            'motor_id': motor_id
                        })
                        conn.commit()  # Commit the transaction to ensure the update is applied
                        print(f"Updated acknowledgment status for motor_id {motor_id} and timestamp {timestamp}.")
                    except Exception as e:
                        print(f"Failed to update acknowledgment status for motor_id {motor_id}: {e}")

    # Prepare data for DataTable
    data_table = high_priority.to_dict('records')

    # Total number of high-priority alerts
    total_alerts = len(high_priority)
    alerts_total_text = f"Total High Risk Alerts: {total_alerts}"

    return data_table, alerts_total_text

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
