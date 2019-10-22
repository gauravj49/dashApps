#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
***********************************************
- PROGRAM: dashUMAP-main.py
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

# Methods for creating components in the layout code
def Card(children, **kwargs):
    return html.Section(children, className="card-style")

def create_layout(app):
    # Actual layout of the app
    return html.Div(
        className="row",
        style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
        children=[
            # Header
            html.Div(
                className="row header",
                id="app-header",
                style={"background-color": "#f9f9f9"},
                children=[
                    html.Div(
                        [
                            html.Img(
                                src=app.get_asset_url("dash-logo.png"),
                                className="logo",
                                id="plotly-image",
                            )
                        ],
                        className="three columns header_img",
                    ),
                    html.Div(
                        [
                            html.H3(
                                "UMAP Explorer",
                                className="header_title",
                                id="app-title",
                            )
                        ],
                        className="nine columns header_title_container",
                    ),
                ],
            ),
            # Body
            html.Div(
                className="row background",
                style={"padding": "10px"},
                children=[
                    html.Div(
                        className="three columns",
                        style={"width": '30%'},
                        children=[
                            Card(
                                    [
                                        dcc.Dropdown(id='featureslistDropdown',
                                        options = [
                                            {'label': i, 'value': i} for i in featuresList
                                        ], multi=True, placeholder='Select genes...'
                                    ),
                                    NamedSlider(
                                        name="Number Of neighbors",
                                        short="nNeighbors",
                                        min=2,
                                        max=100,
                                        step=None,
                                        val=15,
                                        marks={
                                            i: str(i) for i in [2, 5, 10, 15, 20, 25, 50, 100]
                                        },
                                    ),
                                    NamedSlider(
                                        name="Minimum distance",
                                        short="min_dist",
                                        min=0.0,
                                        max=0.99,
                                        step=None,
                                        val=0.1,
                                        marks={i: str(i) for i in [0.0, 0.1, 0.25, 0.5, 0.75, 0.99]},
                                    ),
                                    NamedSlider(
                                        name="Number of components",
                                        short="nComponents",
                                        min=1,
                                        max=50,
                                        step=None,
                                        val=3,
                                        marks={i: str(i) for i in [1, 2, 3, 5, 10, 50]},
                                    ),
                                    html.Div(
                                        id="div-graph-type",
                                        style={"width": '33%'},
                                        children=[
                                            NamedInlineRadioItems(
                                                name="Display Mode",
                                                short="graph-type-display-mode",
                                                options=[
                                                        {
                                                            "label": " 2D",
                                                            "value": "2D",
                                                        },
                                                        {
                                                            "label": " 3D",
                                                            "value": "3D",
                                                        },
                                                ],
                                                val="2D",
                                            )
                                        ],
                                    ),
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="six columns",
                        style={"width": '70%', 'float': 'right', 'display': 'inline-block'},
                        children=[
                            dcc.Graph(id="graph-plot-umap", style={"height": "50vh"})
                        ],
                    ),
                ],
            ),
        ],
    )

def main_callbacks(app):
    def generate_figure_image(groups, layout):
        data = []

        for idx, val in groups:
            scatter = go.Scatter3d(
                name=idx,
                x=val["x"],
                y=val["y"],
                z=val["z"],
                text=[idx for _ in range(val["x"].shape[0])],
                textposition="top center",
                mode="markers",
                marker=dict(size=3, symbol="circle"),
            )
            data.append(scatter)

        figure = go.Figure(data=data, layout=layout)

        return figure


def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )


def NamedInlineRadioItems(name, short, options, val, **kwargs):
    return html.Div(
        id=f"div-{short}",
        style={"display": "inline-block"},
        children=[
            f"{name}:",
            dcc.RadioItems(
                id=f"radio-{short}",
                options=options,
                value=val,
                labelStyle={"display": "inline-block", "margin-right": "7px"},
                style={"display": "inline-block", "margin-left": "7px"},
            ),
        ],
    )

