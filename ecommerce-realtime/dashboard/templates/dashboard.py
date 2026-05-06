import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import requests
from datetime import datetime
import json

API_URL = "http://backend:8000"

app = dash.Dash(__name__, external_stylesheets=['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'])

app.layout = html.Div([
    html.H1("📊 E-commerce Realtime Dashboard", style={'text-align': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        html.Div([
            html.H3("💰 Total Revenue Today"),
            html.H2(id="total-revenue", children="$0", style={'color': '#27ae60'})
        ], className="stat-card"),
        
        html.Div([
            html.H3("📦 Total Orders Today"),
            html.H2(id="total-orders", children="0", style={'color': '#2980b9'})
        ], className="stat-card"),
        
        html.Div([
            html.H3("⭐ Average Order Value"),
            html.H2(id="avg-order-value", children="$0", style={'color': '#e67e22'})
        ], className="stat-card"),
    ], style={'display': 'flex', 'justify-content': 'space-around', 'padding': '20px'}),
    
    html.Div([
        dcc.Graph(id="sales-by-category"),
        dcc.Graph(id="hourly-trend")
    ], style={'display': 'flex', 'flex-direction': 'column'}),
    
    html.Div([
        dcc.Graph(id="top-products", style={'width': '50%', 'display': 'inline-block'}),
        html.Div(id="recent-orders", style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'})
    ]),
    
    dcc.Interval(id="interval", interval=5000),  # Update every 5 seconds
    
    html.Div([
        html.Button("Generate Sample Data", id="generate-btn", n_clicks=0,
                   style={'background': '#3498db', 'color': 'white', 'padding': '10px 20px',
                          'border': 'none', 'border-radius': '5px', 'cursor': 'pointer'})
    ], style={'text-align': 'center', 'padding': '20px'})
])

@app.callback(
    [Output("total-revenue", "children"),
     Output("total-orders", "children"),
     Output("avg-order-value", "children"),
     Output("sales-by-category", "figure"),
     Output("hourly-trend", "figure"),
     Output("top-products", "figure"),
     Output("recent-orders", "children")],
    [Input("interval", "n_intervals"),
     Input("generate-btn", "n_clicks")]
)
def update_dashboard(n_intervals, n_clicks):
    # Fetch data from API
    try:
        stats = requests.get(f"{API_URL}/api/stats").json()
        
        # Calculate totals
        total_revenue = sum(stat['total_sales'] for stat in stats['sales_by_category'])
        total_orders = sum(stat['order_count'] for stat in stats['sales_by_category'])
        avg_order = total_revenue / total_orders if total_orders > 0 else 0
        
        # Sales by category pie chart
        categories = [stat['category'] for stat in stats['sales_by_category']]
        sales = [float(stat['total_sales']) for stat in stats['sales_by_category']]
        pie_fig = go.Figure(data=[go.Pie(labels=categories, values=sales, hole=0.3)])
        pie_fig.update_layout(title="Sales Distribution by Category")
        
        # Hourly trend line chart
        hourly = stats['hourly_stats']
        hours = [stat['hour'] for stat in hourly]
        revenue = [float(stat['total_revenue']) for stat in hourly]
        line_fig = go.Figure(data=[go.Scatter(x=hours, y=revenue, mode='lines+markers')])
        line_fig.update_layout(title="Revenue Trend (Last 24 Hours)", xaxis_title="Hour", yaxis_title="Revenue")
        
        # Top products bar chart
        products = stats['top_products']
        product_names = [p['product_name'] for p in products[:5]]
        quantities = [p['total_quantity'] for p in products[:5]]
        bar_fig = go.Figure(data=[go.Bar(x=product_names, y=quantities)])
        bar_fig.update_layout(title="Top 5 Products", xaxis_title="Product", yaxis_title="Quantity Sold")
        
        # Recent orders table
        orders = stats['recent_orders'][:10]
        orders_html = html.Div([
            html.H3("Recent Orders", style={'text-align': 'center'}),
            html.Table([
                html.Thead(html.Tr([html.Th("Order ID"), html.Th("Product"), html.Th("Amount"), html.Th("Status")])),
                html.Tbody([
                    html.Tr([
                        html.Td(order['order_id'][:8] + "..."),
                        html.Td(order['product_name']),
                        html.Td(f"${float(order['total_amount']):.2f}"),
                        html.Td(order['order_status'], style={'color': 'green' if order['order_status'] == 'COMPLETED' else 'orange'})
                    ]) for order in orders
                ])
            ], style={'width': '100%', 'border-collapse': 'collapse'})
        ])
        
        return f"${total_revenue:,.2f}", f"{total_orders}", f"${avg_order:.2f}", pie_fig, line_fig, bar_fig, orders_html
    
    except Exception as e:
        return "$0", "0", "$0", go.Figure(), go.Figure(), go.Figure(), html.Div("Error loading data")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
