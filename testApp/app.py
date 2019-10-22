#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
***********************************************
- PROGRAM: app.py
- CONTACT: Gaurav Jain (gaurav.jain@tum.de)
- LICENSE: Copyright (C) <year>  <name of author>
           This program is free software: you can redistribute it and/or modify
           it under the terms of the GNU General Public License as published by
           the Free Software Foundation, either version 3 of the License, or
           (at your option) any later version.
           This program is distributed in the hope that it will be useful,
           but WITHOUT ANY WARRANTY; without even the implied warranty of
           MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
           GNU General Public License for more details.
           You should have received a copy of the GNU General Public License
           along with this program.  If not, see <http://www.gnu.org/licenses/>.
***********************************************
__author__     = "Gaurav Jain"
__copyright__  = "Copyright 2019"
__credits__    = ["Gaurav Jain"]
__license__    = "GPL3+"
__maintainer__ = "Gaurav Jain"
__email__      = "gaurav.jain near tum.de"
__status__     = "Development"
***********************************************
"""
print (__doc__)

# Dash specific
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import plotly
import plotly.graph_objs as go

# project specific
import os
import pandas as pd
import umap
import numpy as np
import itertools
from sklearn.datasets import load_iris, load_digits
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define Dash App
app=dash.Dash()

#Stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Import the data 
iris              = load_iris()
irisDF            = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
irisDF['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
featuresList      = irisDF.columns.tolist()

iris = load_iris()
irisDF = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
irisDF['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Set random state
np.random.seed(2105)

reducer = umap.UMAP(random_state = 2105, transform_seed = 2105, n_components=3)
embedding = reducer.fit_transform(iris.data)
# embDF = pd.DataFrame(embedding, index=irisDF['species'].values.tolist(), columns=['UM1','UM2'])
embDF = pd.DataFrame(embedding, columns=['UM1','UM2','UM3'])

# Define a nice colour map for gene expression
colors2     = plt.cm.Reds(np.linspace(0, 1, 128))
colors3     = plt.cm.Greys_r(np.linspace(0.7,0.8,20))
colorsComb  = np.vstack([colors3, colors2])
expColorMap = LinearSegmentedColormap.from_list('ExpressionColorMap', colorsComb)
fvals = irisDF['sepal length (cm)']
markerDict = dict(  color=fvals,
                    colorscale='viridis', opacity=0.9,
                    showscale=True,
                    colorbar=dict(  thickness=20, title="Test", titleside='right'))
# print(markerDict)

app.layout = html.Div([
    # Dropdown
    html.Div([
        html.H2('Genes to plot'),
        dcc.Dropdown(id='featureslistDropdown',
            options = [
                {'label': i, 'value': i} for i in featuresList
            ], multi=True, placeholder='Select genes...'
        )],
    style={'width': '20%', 'display': 'inline-block'}),

    html.Div([
        # Graph
        dcc.Graph(id='umapScatterPlot')],
    style={'width': '70%', 'display': 'inline-block', 'float': 'right'}),

    html.Div([
        # Graph
        dcc.Graph(id='umapScatterPlot3D')],
    style={'width': '70%', 'display': 'inline-block', 'float': 'right'})
])

@app.callback(
    Output(component_id='umapScatterPlot',component_property='figure'),
    [Input(component_id='featureslistDropdown', component_property='value')]
)

def update_2dScatter(dropdown_value):
    print(dropdown_value)
    fvals = irisDF[dropdown_value].values.tolist()
    fvals = list(itertools.chain(*fvals))
    # print(fvals)
    markerDict = dict(color=fvals,
                        colorscale='Viridis', opacity=0.9,
                        showscale=True,
                        colorbar=dict(thickness=20,
                                        title="".join(dropdown_value),
                                        titleside='right'))
    # print(markerDict)

    return {
            'data':[ 
                go.Scatter(
                    x=embDF['UM1'], 
                    y=embDF['UM2'], 
                    mode='markers',
                    hoverinfo='text', text=fvals, marker = markerDict)], 
            'layout': go.Layout(
                        xaxis={'title': 'Umap1'},
                        yaxis={'title': 'Umap2'},
                        hovermode='closest',
                        title="".join(dropdown_value))
            }

@app.callback(
    Output(component_id='umapScatterPlot3D',component_property='figure'),
    [Input(component_id='featureslistDropdown', component_property='value')]
)

def update_3dScatter(dropdown_value):
    print(dropdown_value)
    fvals = irisDF[dropdown_value].values.tolist()
    fvals = list(itertools.chain(*fvals))
    print(fvals)
    markerDict = dict(color=fvals,
                        colorscale='Viridis', opacity=0.9,
                        showscale=True,
                        colorbar=dict(thickness=20,
                                        title="".join(dropdown_value),
                                        titleside='right'))
    print(markerDict)

    return {
            'data':[ 
                go.Scatter3d(
                    x=embDF['UM1'], 
                    y=embDF['UM2'], 
                    z=embDF['UM3'], 
                    mode='markers',
                    hoverinfo='text', text=fvals, marker = markerDict)], 
            'layout': go.Layout(
                        scene = {
                            'xaxis':{'title': 'Umap1'},
                            'yaxis':{'title': 'Umap2'},
                            'zaxis':{'title': 'Umap3'},
                            'hovermode':'closest',
                            'aspectmode': "cube"
                        })
            }



if __name__ == '__main__':
    # app.run_server(port = 8000)
    app.run_server()