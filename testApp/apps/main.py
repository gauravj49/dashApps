# import os
# import dash_core_components as dcc
# import dash_html_components as html
# import pandas as pd
# import plotly.graph_objs as go
# from dash.dependencies import Input, Output
# from app import app
# import umap
# import numpy as np
# from sklearn.datasets import load_iris, load_digits
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# sns.set(context="paper", style="white")

# iris = load_iris()

# if 'DYNO' in os.environ:
#     app_name = os.environ['DASH_APP_NAME']
# else:
#     app_name = 'dash-testApp'

# layout = html.Div(
#     [html.Div([html.H1(" Iris UMAP")], className="row", style={'textAlign': "center"}),
#      html.Div(
#          [dcc.Dropdown(id="selected-type", options=[{"label": i, "value": i} for i in iris.target_names.unique()],
#                        value='',
#                        style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"})],
#          className="row"),
#      html.Div([dcc.Graph(id="my-graph", style={"margin-right": "auto", "margin-left": "auto", "width": "80%"})],
#               className="row")
#      ], className="container")

# @app.callback(
#     Output("my-graph", "figure"),
#     [Input("selected-type", "value")])
# def update_figure(selected):
#     dff = df[df["Indicator Name"] == selected]
#     trace = go.Heatmap(x=dff.columns.values[3:], y=dff['Country Name'].unique(), z=dff[
#         ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
#          '2014']].values, colorscale='Electric', colorbar={"title": "Percentage"}, showscale=True)
#     return {"data": [trace],
#             "layout": go.Layout(width=800, height=750, title=f"{selected.title()} vs Year", xaxis={"title": "Year"},
#                                 yaxis={"title": "Country", "tickmode": "array",
#                                        "tickvals": dff['Country Name'].unique(),
#                                        "ticktext": ['Afghanistan', 'Arab World', 'Australia', 'Belgium', 'Bangladesh',
#                                                     'Brazil', 'Canada', 'Colombia', 'Germany', 'East Asia & Pacific',
#                                                     'Europe &<br>Central Asia', 'India', 'Japan',
#                                                     'Latin America &<br>Caribbean', 'Middle East &<br>North Africa',
#                                                     'Mexico', 'North America', 'Saudi Arabia', 'Singapore',
#                                                     'Virgin Islands (US)', 'South Africa', 'Zimbabwe'],
#                                        "tickfont": {"size": 8}, "tickangle": -20}, )}


# © 2019 GitHub, Inc.
# reducer = umap.UMAP()
# embedding = reducer.fit_transform(iris.data)
# embedding.shape


import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly
import plotly.graph_objs as go


#Define Dash App
app=dash.Dash()

if 'DYNO' in os.environ:
    app_name = os.environ['DASH_APP_NAME']
else:
    app_name = 'dash-testApp'

df =  pd.DataFrame({
    'Range1': [1, 2, 3, 4, 5, 6],
    'Range2': [6, 7, 8, 9, 10, 11],
})


app.layout = html.Div([
    html.Div([
        html.H2('Sliders'),

        html.H4('Range1'),
        dcc.RangeSlider(
                        id='rangeslider_range1',
                        min=df['Range1'].min(),
                        max=df['Range1'].max(),
                        marks={str(range1): str(range1) for range1 in df['Range1'].unique()},
                        value = [df['Range1'].min(),df['Range1'].max()]
                        ),

        html.H4('Range2'),
        dcc.RangeSlider(
                        id='rangeslider_range2',
                        min=df['Range2'].min(),
                        max=df['Range2'].max(),
                        marks={str(range2): str(range2) for range2 in df['Range2'].unique()},
                        value = [df['Range2'].min(),df['Range2'].max()]
                        ),


        ],style={'width': '30%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='graph_test'),
        ],
        style={'width': '60%', 'display': 'inline-block', 'float': 'right'})
    ]
)

@app.callback(
     dash.dependencies.Output('graph_test', 'figure'),
    [dash.dependencies.Input('rangeslider_range1', 'value'),
     dash.dependencies.Input('rangeslider_range2', 'value')#,

     ])

def update_graph(
                 rangeslider_range1,
                 rangeslider_range2
                 ):

    bln0 = ((df.loc[:, "Range1"] == rangeslider_range1[0]) | (df.loc[:, "Range1"] == rangeslider_range1[1]))
    bln1 = ((df.loc[:, "Range2"] == rangeslider_range2[0]) | (df.loc[:, "Range2"] == rangeslider_range2[1]))
    filtered_data = df.loc[bln0 & bln1, :]
    # filtered_data = df[df['Range1'] == rangeslider_range1 and df['Range2'] == rangeslider_range2]

    return {
        'data': [go.Scatter(
            x=filtered_data['Range1'],
            y=filtered_data['Range2'],
            mode='markers',
        )],
        'layout': go.Layout(
            xaxis={
                'title': 'Range1',
            },
            yaxis={
                'title': 'Range2',
            },
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    # app.run_server(port = 8000)
    app.run_server(debug=True)