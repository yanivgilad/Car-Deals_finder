import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

# Import the scraper modules
from scraper import VehicleScraper
import yad2_parser

# For web visualization
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from deal_classifier import classify_deal, get_simple_label, get_color_for_deal

from dash.exceptions import PreventUpdate

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vehicle Price Analyzer')
    parser.add_argument('--output-dir', type=str, default='scraped_vehicles',
                        help='Directory to save scraped data')
    parser.add_argument('--manufacturer', type=int, default=12,
                        help='Manufacturer ID to scrape')
    parser.add_argument('--model', type=int, default=10154,
                        help='Model ID to scrape')
    parser.add_argument('--max-pages', type=int, default=25,
                        help='Maximum number of pages to scrape')
    parser.add_argument('--skip-scrape', action='store_true',
                        help='Skip scraping and use existing data')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port to run the web server on')
    return parser.parse_args()

def scrape_data(output_dir, manufacturer, model, max_pages):
    """Run the scraper to collect vehicle data"""
    print(f"Scraping data for manufacturer={manufacturer}, model={model}...")
    scraper = VehicleScraper(output_dir, manufacturer, model)
    scraper.scrape_pages(max_page=max_pages)
    
def process_data(output_dir):
    """Process the scraped HTML files into a CSV"""
    print("Processing scraped HTML files...")
    dir_name = Path(output_dir).name
    yad2_parser.process_directory(output_dir)
    output_file = f"{dir_name}_summary.csv"
    output_path = os.path.join(output_dir, output_file)
    
    # Check if the CSV file exists
    if not os.path.exists(output_path):
        print(f"Error: Could not find processed data at {output_path}")
        sys.exit(1)
        
    return output_path

def load_data(csv_path):
    """Load and prepare the CSV data for visualization"""
    try:
        df = pd.read_csv(csv_path)
        
        # הסרת כפילויות (אם יש)
        df.drop_duplicates(subset=['adNumber'], inplace=True)
        
        # סינון רכבים ללא מחיר או עם מחיר 0
        df = df[df['price'] > 0]
        
        # המרת תאריך הייצור ל-datetime
        df['productionDate'] = pd.to_datetime(df['productionDate'], errors='coerce')
        df.dropna(subset=['productionDate'], inplace=True)
        
        # הוספת productionYear
        df['productionYear'] = df['productionDate'].dt.year

        # === כאן מתחיל הקטע של סיווג העסקה ===
        def classify_row(description):
            if isinstance(description, str) and description.strip() != "":
                full_label, score, _ = classify_deal(description)
                simple_label = get_simple_label(full_label)
                
                # קודם כל, אם המודל מחזיר "עסקה גרועה" ואילו score >= 0.8 – שמור על "עסקה גרועה" (אדום)
                if simple_label == "עסקה גרועה" and score >= 0.8:
                    simple_label = "עסקה גרועה"
                # אחר כך, אם המודל מחזיר "עסקה מצויינת" ואילו score >= 0.9 – שמור על "עסקה מצויינת" (ירוק)
                elif simple_label == "עסקה מצויינת" and score >= 0.9:
                    simple_label = "עסקה מצויינת"
                # בכל שאר המקרים – סווג כ"בינונית" (צהוב)
                else:
                    simple_label = "עסקה בינונית"
                
                return simple_label, score
            else:
                return "N/A", 0.0




        df[['deal_category', 'deal_score']] = df['description'].apply(lambda d: pd.Series(classify_row(d)))

        def map_deal_color(category):
            if category == "עסקה מצויינת":
                return "green"
            elif category == "עסקה בינונית":
                return "yellow"
            elif category == "עסקה גרועה":
                return "red"
            else:
                return "gray"

        df['deal_color'] = df['deal_category'].apply(map_deal_color)
        # === כאן מסתיים קטע הסיווג ===

        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)


def create_dashboard(df, port=8050):
    """Create and run an interactive Dash app for visualizing the data"""
    external_stylesheets = [
        {
            'href': 'https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap',
            'rel': 'stylesheet'
        }
    ]
    app = dash.Dash(
        __name__, 
        title="Vehicle Price Analyzer",
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=True
    )
    
    # Get unique values for filters
    km_ranges = [
        {'label': 'All', 'value': 'all'},
        {'label': '≤ 10,000 km/year', 'value': '0-10000'},
        {'label': '≤ 15,000 km/year', 'value': '0-15000'},
        {'label': '≤ 20,000 km/year', 'value': '0-20000'},
        {'label': '≤ 25,000 km/year', 'value': '0-25000'},
        {'label': '> 25,000 km/year', 'value': '25000-999999'}
    ]
    
    hands = [{'label': 'All Hands', 'value': 'all'}] + [
        {'label': f'Hand ≤ {h}', 'value': f'0-{h}'} for h in sorted(df['hand'].unique()) if h > 0
    ]
    
    sub_models = [{'label': 'All Sub-models', 'value': 'all'}] + [
        {'label': sm, 'value': sm} for sm in sorted(df['subModel'].unique())
    ]
    
    models = [{'label': m, 'value': m} for m in sorted(df['model'].unique())]
    
    ad_types = [{'label': 'All', 'value': 'all'}] + [
        {'label': at, 'value': at} for at in sorted(df['listingType'].unique())
    ]
    
    # Define CSS styles
    styles = {
        'container': {
            'font-family': 'Roboto, sans-serif',
            'max-width': '1200px',
            'margin': '0 auto',
            'padding': '20px',
            'background-color': '#f9f9f9',
            'border-radius': '8px',
            'box-shadow': '0 4px 8px rgba(0,0,0,0.1)'
        },
        'header': {
            'background-color': '#2c3e50',
            'color': 'white',
            'padding': '15px 20px',
            'margin-bottom': '20px',
            'border-radius': '5px',
            'text-align': 'center'
        },
        'filter_container': {
            'display': 'flex',
            'flex-wrap': 'wrap',
            'gap': '15px',
            'background-color': 'white',
            'padding': '15px',
            'border-radius': '5px',
            'box-shadow': '0 2px 4px rgba(0,0,0,0.05)',
            'margin-bottom': '20px'
        },
        'filter': {
            'width': '23%',
            'min-width': '200px',
            'padding': '10px'
        },
        'label': {
            'font-weight': 'bold',
            'margin-bottom': '5px',
            'color': '#2c3e50'
        },
        'graph': {
            'background-color': 'white',
            'padding': '15px',
            'border-radius': '5px',
            'box-shadow': '0 2px 4px rgba(0,0,0,0.05)',
            'margin-bottom': '20px'
        },
        'summary': {
            'background-color': 'white',
            'padding': '15px',
            'border-radius': '5px',
            'box-shadow': '0 2px 4px rgba(0,0,0,0.05)',
            'margin-bottom': '20px'
        },
        'summary_header': {
            'color': '#2c3e50',
            'border-bottom': '2px solid #3498db',
            'padding-bottom': '10px',
            'margin-bottom': '15px'
        },
        'button': {
            'background-color': '#2c3e50',
            'color': 'white',
            'border': 'none',
            'padding': '10px 20px',
            'border-radius': '5px',
            'cursor': 'pointer',
            'font-weight': 'bold',
            'margin-top': '10px',
            'width': '100%'
        },
        'clear_button': {
            'background-color': '#e74c3c',
            'color': 'white',
            'border': 'none',
            'padding': '10px 20px',
            'border-radius': '5px',
            'cursor': 'pointer',
            'font-weight': 'bold',
            'margin-top': '10px',
            'width': '100%'
        },
        'click_instruction': {
            'text-align': 'center',
            'font-style': 'italic',
            'color': '#3498db',
            'margin': '10px 0',
            'padding': '8px',
            'background-color': '#f0f7ff',
            'border-radius': '5px',
            'border-left': '3px solid #3498db'
        }
    }
    
    app.layout = html.Div([
        html.Div([
            html.H1("Vehicle Price Analysis Dashboard", style={'margin': '0'})
        ], style=styles['header']),
        
        html.Div([
            html.Div([
                html.Label("Filter by km/year:", style=styles['label']),
                dcc.Dropdown(
                    id='km-filter',
                    options=km_ranges,
                    value='all',
                    clearable=False
                ),
            ], style=styles['filter']),
            
            html.Div([
                html.Label("Filter by owner hand:", style=styles['label']),
                dcc.Dropdown(
                    id='hand-filter',
                    options=hands,
                    value='all',
                    clearable=False
                ),
            ], style=styles['filter']),
            
            html.Div([
                html.Label("Filter by model:", style=styles['label']),
                dcc.Dropdown(
                    id='model-filter',
                    options=models,
                    value=[],
                    multi=True,
                    placeholder="Select model(s)"
                ),
            ], style=styles['filter']),
            
            html.Div([
                html.Label("Filter by listing type:", style=styles['label']),
                dcc.Dropdown(
                    id='adtype-filter',
                    options=ad_types,
                    value='all',
                    clearable=False
                ),
            ], style=styles['filter']),

            html.Div([
                html.Label("Filter by sub-model:", style=styles['label']),
                html.Div([
                    dcc.Checklist(
                        id='submodel-checklist',
                        options=[],
                        value=[],
                        labelStyle={'display': 'block', 'margin-bottom': '8px', 'cursor': 'pointer'},
                        style={'max-height': '200px', 'overflow-y': 'auto', 'padding': '10px', 'background-color': '#f5f9ff', 'border-radius': '5px'}
                    ),
                ]),
                html.Div([
                    html.Button(
                        'Apply Filters', 
                        id='apply-submodel-button', 
                        style=styles['button']
                    ),
                    html.Button(
                        'Clear Selection', 
                        id='clear-submodel-button', 
                        style=styles['clear_button']
                    ),
                ], style={'display': 'flex', 'gap': '10px'}),
            ], style={'width': '23%', 'min-width': '200px', 'padding': '10px', 'flex-grow': '1'}),
            
        ], style=styles['filter_container']),
        
        html.Div([
            html.P("👆 לחץ על נקודה בתרשים להצגת תיאור המודעה")
        ], style=styles['click_instruction']),
        
        html.Div([
            dcc.Graph(id='price-date-scatter')
        ], style=styles['graph']),
        
        # Description container (RTL for Hebrew)
        html.Div(id='description-container', style={
            'direction': 'rtl',
            'text-align': 'right',
            'border': '1px solid #ccc',
            'padding': '10px',
            'margin': '10px 0'
        }),
        
        # Summary section - overall
        html.Div([
            html.H3("Data Summary", style=styles['summary_header']),
            html.Div(id='summary-stats')
        ], style=styles['summary']),
        
        # Second summary row for the selected year's stats
        html.Div([
            html.H3("Yearly Summary", style=styles['summary_header']),
            html.Div(id='summary-stats-yearly')
        ], style=styles['summary']),
    ], style=styles['container'])
    
    # ~~~~~~~~~~~~~ Callbacks ~~~~~~~~~~~~~

    @app.callback(
        Output('submodel-checklist', 'options'),
        Input('model-filter', 'value'),
    )
    def update_submodel_options(selected_models):
        if not selected_models:
            # show all submodels
            submodel_options = []
            for sm in sorted(df['subModel'].unique()):
                models_for_submodel = df[df['subModel'] == sm]['model'].unique()
                if len(models_for_submodel) == 1:
                    label = f"[{models_for_submodel[0]}] {sm}"
                else:
                    label = f"[{models_for_submodel[0]}+] {sm}"
                submodel_options.append({'label': label, 'value': sm})
        else:
            filtered_df = df[df['model'].isin(selected_models)]
            submodel_options = []
            for sm in sorted(filtered_df['subModel'].unique()):
                models_for_submodel = filtered_df[filtered_df['subModel'] == sm]['model'].unique()
                if len(models_for_submodel) == 1:
                    label = f" {sm} [{models_for_submodel[0]}]"
                else:
                    models_str = '+'.join(models_for_submodel)
                    label = f" {sm} [{models_str}]"
                submodel_options.append({'label': label, 'value': sm})
        return sorted(submodel_options, key=lambda x: x['label'])
    
    @app.callback(
        Output('submodel-checklist', 'value'),
        Input('clear-submodel-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_submodel_selection(n_clicks):
        return []
    
    @app.callback(
        [Output('price-date-scatter', 'figure'),
         Output('summary-stats', 'children')],
        [Input('km-filter', 'value'),
         Input('hand-filter', 'value'),
         Input('model-filter', 'value'),
         Input('apply-submodel-button', 'n_clicks'),
         Input('adtype-filter', 'value')],
        [State('submodel-checklist', 'value')]
    )
    def update_graph(km_range, hand, models, submodel_btn_clicks, adtype, submodel_list):
        filtered_df = df.copy()
        
        # סינונים
        if km_range != 'all':
            min_km, max_km = map(int, km_range.split('-'))
            filtered_df = filtered_df[filtered_df['km_per_year'] <= max_km]
            if min_km > 0:
                filtered_df = filtered_df[filtered_df['km_per_year'] > min_km]
        
        if hand != 'all':
            min_hand, max_hand = map(int, hand.split('-'))
            filtered_df = filtered_df[filtered_df['hand'] <= max_hand]
        
        if models:
            filtered_df = filtered_df[filtered_df['model'].isin(models)]
            
        if submodel_list:
            filtered_df = filtered_df[filtered_df['subModel'].isin(submodel_list)]
            
        if adtype != 'all':
            filtered_df = filtered_df[filtered_df['listingType'] == adtype]
        
        # חישוב days_since_newest לצורך עקומת הדעיכה
        if not filtered_df.empty:
            newest_date = filtered_df['productionDate'].max()
            filtered_df['days_since_newest'] = (newest_date - filtered_df['productionDate']).dt.days
        else:
            # אם אין מודעות, נמנע מבעיות
            filtered_df['days_since_newest'] = np.nan
        
        # ציור הנקודות – השתמשו בתאריך הייצור האמיתי כ-X
        color_discrete_map = {
            "עסקה מצויינת": "green",
            "עסקה בינונית": "yellow",
            "עסקה גרועה": "red",
            "N/A": "gray"
        }

        fig = px.scatter(
            filtered_df,
            x='productionDate', 
            y='price',
            color='deal_category',  # השתמשו בעמודת הסיווג
            color_discrete_map=color_discrete_map,
            size_max=8,
            hover_data=['model', 'subModel', 'hand', 'km', 'city', 'productionDate', 'link', 'description', 'deal_category', 'deal_score'],
            labels={
                'productionDate': 'Production Date',
                'price': 'Price (₪)',
                'deal_category': 'Deal Quality'
            },
            title=f'Vehicle Prices by Production Date ({len(filtered_df)} vehicles)'
        )

                
        
        # 2) הכנת customData ל-hover
        custom_data = np.column_stack((
            filtered_df['model'], 
            filtered_df['subModel'], 
            filtered_df['hand'], 
            filtered_df['km'], 
            filtered_df['city'],
            filtered_df['productionDate'],
            filtered_df['link'],
            filtered_df['description'],
            filtered_df['deal_score']
        ))


        
        fig.update_traces(
            marker=dict(
                size=8,
                opacity=0.8,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            customdata=custom_data,
            hovertemplate=(
                '<b>%{customdata[0]} %{customdata[1]}</b><br>'
                'Price: ₪%{y:,.0f}<br>'
                'Production Date: %{customdata[5]}<br>'
                'Hand: %{customdata[2]}<br>'
                'KM: %{customdata[3]:,.0f}<br>'
                'City: %{customdata[4]}<br>'
                'Score: %{customdata[8]:.2f}<br>'  # מציג את ה־score עם שתי ספרות אחרי הנקודה
                '<b>👆 לחץ להצגת תיאור</b>'
            )
        )

            
        
        # 3) עקומת דעיכה אקספוננציאלית (על days_since_newest)
        if len(filtered_df) > 1:
            # מיון לפי גיל (days_since_newest)
            sorted_df = filtered_df.sort_values('days_since_newest')
            x = sorted_df['days_since_newest'].values
            y = sorted_df['price'].values
            valid_indices = ~np.isnan(x) & ~np.isnan(y)
            x = x[valid_indices]
            y = y[valid_indices]
            
            if len(x) > 1:
                try:
                    from scipy import optimize
                    
                    def exp_decay_with_offset(x, a, b, c):
                        return a * np.exp(-b * x) + c
                    
                    max_price = np.max(y)
                    min_price = np.min(y)
                    mean_price = np.mean(y)
                    
                    p0 = [max_price - min_price, 0.001, min_price]
                    bounds = ([0, 0.0001, 0], [2 * max_price, 0.1, mean_price])
                    
                    try:
                        params, _ = optimize.curve_fit(
                            exp_decay_with_offset, x, y,
                            p0=p0, bounds=bounds,
                            method='trf', maxfev=10000
                        )
                        a, b, c = params
                    except RuntimeError:
                        # fallback to simpler model
                        def exp_decay(x, a, b):
                            return a * np.exp(-b * x)
                        p0_simple = [max_price, 0.001]
                        bounds_simple = ([0, 0.0001], [2 * max_price, 0.1])
                        params, _ = optimize.curve_fit(
                            exp_decay, x, y,
                            p0=p0_simple, bounds=bounds_simple,
                            method='trf', maxfev=10000
                        )
                        a, b = params
                        c = 0
                    
                    # משרטטים 200 נקודות חלקות בין 0 לערך מקסימלי
                    x_curve = np.linspace(0, x.max(), 200)
                    
                    # חישוב ערכי המחיר
                    def model(t):
                        return a * np.exp(-b * t) + c
                    
                    y_curve = model(x_curve)
                    
                    # המרת x_curve (גיל) חזרה לתאריכים אמיתיים
                    newest_date = sorted_df['productionDate'].max()
                    curve_dates = newest_date - pd.to_timedelta(x_curve, unit='D')
                    
                    fig.add_trace(go.Scatter(
                        x=curve_dates,
                        y=y_curve,
                        mode='lines',
                        name='Exponential Trend',
                        line=dict(color='red', width=3, dash='solid'),
                        hoverinfo='none'
                    ))
                except Exception as e:
                    print(f"Error fitting exponential curve: {str(e)}")
                    # ניתן להוסיף fallback ל-linear וכו'
        
        # 4) הפיכת ציר ה־X כדי שהחדש ביותר יהיה משמאל
        #fig.update_layout(
        #    xaxis=dict(autorange='reversed')
        #)
        
        # 5) סיכום ראשי
        summary_style = {
            'container': {
                'display': 'flex',
                'flex-wrap': 'wrap',
                'gap': '20px'
            },
            'card': {
                'flex': '1',
                'min-width': '180px',
                'padding': '15px',
                'border-radius': '5px',
                'background-color': '#f5f9ff',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.05)',
                'text-align': 'center'
            },
            'value': {
                'font-size': '20px',
                'font-weight': 'bold',
                'color': '#2c3e50',
                'margin': '10px 0'
            },
            'label': {
                'font-size': '14px',
                'color': '#7f8c8d',
                'margin': '0'
            }
        }
        
        summary = html.Div([
            html.Div([
                html.P("Number of Vehicles", style=summary_style['label']),
                html.P(f"{len(filtered_df)}", style=summary_style['value'])
            ], style=summary_style['card']),
            
            html.Div([
                html.P("Average Price", style=summary_style['label']),
                html.P(f"₪{filtered_df['price'].mean():,.0f}" if len(filtered_df) else "—", 
                       style=summary_style['value'])
            ], style=summary_style['card']),
            
            html.Div([
                html.P("Price Range", style=summary_style['label']),
                (html.P(f"₪{filtered_df['price'].min():,.0f} - ₪{filtered_df['price'].max():,.0f}", 
                        style=summary_style['value'])
                 if len(filtered_df) else html.P("—", style=summary_style['value']))
            ], style=summary_style['card']),
            
            html.Div([
                html.P("Average km/year", style=summary_style['label']),
                html.P(f"{filtered_df['km_per_year'].mean():,.0f}" if len(filtered_df) else "—", 
                       style=summary_style['value'])
            ], style=summary_style['card']),
            
            html.Div([
                html.P("Average Vehicle Age", style=summary_style['label']),
                html.P(f"{filtered_df['number_of_years'].mean():.1f} years" if len(filtered_df) else "—", 
                       style=summary_style['value'])
            ], style=summary_style['card']),
        ], style=summary_style['container'])
        
        return fig, summary
    
    @app.callback(
        Output('description-container', 'children'),
        Input('price-date-scatter', 'clickData')
    )
    def display_description(clickData):
        if not clickData or 'points' not in clickData or len(clickData['points']) == 0:
            return ""
        point = clickData['points'][0]
        description = point['customdata'][7]
        link = point['customdata'][6]
        return html.Div([
            html.P(description, style={'whiteSpace': 'pre-wrap', 'margin': '0 0 10px 0'}),
            html.A("לעבור להודעה", href=link, target="_blank", style={
                'display': 'inline-block',
                'padding': '10px 20px',
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'textDecoration': 'none',
                'borderRadius': '5px'
            })
        ])

    # סיכום לפי השנה של הרכב שנבחר
    @app.callback(
        Output('summary-stats-yearly', 'children'),
        [Input('price-date-scatter', 'clickData'),
         Input('km-filter', 'value'),
         Input('hand-filter', 'value'),
         Input('model-filter', 'value'),
         Input('apply-submodel-button', 'n_clicks'),
         Input('adtype-filter', 'value')],
        [State('submodel-checklist', 'value')]
    )
    def update_yearly_summary(clickData, km_range, hand, models, submodel_btn_clicks, adtype, submodel_list):
        if not clickData or 'points' not in clickData or len(clickData['points']) == 0:
            return ""
        
        # שליפת השנה מה-productionDate
        point = clickData['points'][0]
        production_date_str = point['customdata'][5]  # זהו ה-productionDate
        if not production_date_str:
            return ""
        try:
            year_clicked = pd.to_datetime(production_date_str).year
        except:
            return ""
        
        # מסננים בדיוק כמו בפונקציית update_graph
        filtered_df = df.copy()
        
        if km_range != 'all':
            min_km, max_km = map(int, km_range.split('-'))
            filtered_df = filtered_df[filtered_df['km_per_year'] <= max_km]
            if min_km > 0:
                filtered_df = filtered_df[filtered_df['km_per_year'] > min_km]
        
        if hand != 'all':
            min_hand, max_hand = map(int, hand.split('-'))
            filtered_df = filtered_df[filtered_df['hand'] <= max_hand]
        
        if models:
            filtered_df = filtered_df[filtered_df['model'].isin(models)]
            
        if submodel_list:
            filtered_df = filtered_df[filtered_df['subModel'].isin(submodel_list)]
            
        if adtype != 'all':
            filtered_df = filtered_df[filtered_df['listingType'] == adtype]
        
        # כאן נסנן לשנה הספציפית
        filtered_df = filtered_df[filtered_df['productionYear'] == year_clicked]
        
        # אם אין רכבים בשנה זו תחת הפילטרים, נחזיר כלום
        if filtered_df.empty:
            return ""
        
        summary_style = {
            'container': {
                'display': 'flex',
                'flex-wrap': 'wrap',
                'gap': '20px'
            },
            'card': {
                'flex': '1',
                'min-width': '180px',
                'padding': '15px',
                'border-radius': '5px',
                'background-color': '#f5f9ff',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.05)',
                'text-align': 'center'
            },
            'value': {
                'font-size': '20px',
                'font-weight': 'bold',
                'color': '#2c3e50',
                'margin': '10px 0'
            },
            'label': {
                'font-size': '14px',
                'color': '#7f8c8d',
                'margin': '0'
            }
        }
        
        summary = html.Div([
            html.Div([
                html.P(f"Vehicles in {year_clicked}", style=summary_style['label']),
                html.P(f"{len(filtered_df)}", style=summary_style['value'])
            ], style=summary_style['card']),
            
            html.Div([
                html.P("Average Price", style=summary_style['label']),
                html.P(f"₪{filtered_df['price'].mean():,.0f}", style=summary_style['value'])
            ], style=summary_style['card']),
            
            html.Div([
                html.P("Price Range", style=summary_style['label']),
                html.P(f"₪{filtered_df['price'].min():,.0f} - ₪{filtered_df['price'].max():,.0f}", 
                       style=summary_style['value'])
            ], style=summary_style['card']),
            
            html.Div([
                html.P("Average km/year", style=summary_style['label']),
                html.P(f"{filtered_df['km_per_year'].mean():,.0f}", style=summary_style['value'])
            ], style=summary_style['card']),
            
            html.Div([
                html.P("Average Vehicle Age", style=summary_style['label']),
                html.P(f"{filtered_df['number_of_years'].mean():.1f} years", style=summary_style['value'])
            ], style=summary_style['card']),
        ], style=summary_style['container'])
        
        return summary

    print(f"Starting dashboard on http://127.0.0.1:{port}/")
    app.run(debug=False, port=port)

def main():
    args = parse_arguments()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_scrape:
        scrape_data(args.output_dir, args.manufacturer, args.model, args.max_pages)
    
    csv_path = process_data(args.output_dir)
    df = load_data(csv_path)
    
    # (לא חובה למחוק את ה-CSV, תלוי בצרכים שלכם)
    # os.unlink(csv_path)
    
    create_dashboard(df, args.port)

if __name__ == "__main__":
    main()
