from typing import Sequence

import pandas as pd
import numpy as np
import geopandas

import yaml
from yaml.loader import SafeLoader

import plotly_express as px
import plotly.graph_objects as go

from dash import Dash, Input, Output
from dash import dcc
from dash import html
from pynput.mouse import Controller
import dash_bootstrap_components as dbc


def load_travel_to_work_datasets(
    metric_path: str,
    file_names: Sequence[str],
) -> pd.DataFrame:
    """Loads the timeseries for travel-to-work datasets.

    Parameters
    ----------
    metric_path
        Folder containing the travel-to-work datasets.
    file_names
        Names of the travel-to-work datasets to load.

    Returns
    -------
    DataFrame containing timeseries for all loaded datasets.
    """
    loaded_dfs = {}
    for file in file_names:
        loaded_dfs[file] = pd.read_csv(f"{metric_path}/{file}")

    df = pd.concat(loaded_dfs, ignore_index=True)

    return df


def load_shapefile(
    shapefile_path: str,
) -> pd.DataFrame:
    """Load the shapefile to be used for plotting map.

    Parameters
    ----------
    shapefile_path
        Path to the shapefile.

    Returns
    -------
    DataFrame containing geography data.
    """
    # Load shapefile for map plotting
    df_shapefile = geopandas.read_file(shapefile_path)
    # This line makes choropleth super fast by simplifying polygons
    df_shapefile["geometry"] = (
        df_shapefile.to_crs(df_shapefile.estimate_utm_crs()).simplify(1000).to_crs(df_shapefile.crs))
    df_shapefile.to_crs(epsg=4326, inplace=True)
    df_shapefile = df_shapefile.rename(columns={
        'LAD21CD': 'AREACD',
    })
    df_shapefile = df_shapefile.loc[:, ['AREACD', 'geometry']]

    return df_shapefile


def initialise_choropleth(
    df: pd.DataFrame,
    df_shapefile: pd.DataFrame,
):
    df_shapefile.loc[:, 'colour'] = "not_selected"
    df_unique = df.loc[:, ['AREACD', 'AREANM']].drop_duplicates()
    df_shapefile = df_shapefile.merge(df_unique, on="AREACD", how="inner")

    df_shapefile = df_shapefile.set_index("AREANM")
    fig = px.choropleth_mapbox(
        df_shapefile,
        geojson=df_shapefile.geometry,
        locations=df_shapefile.index,
        color=df_shapefile["colour"],
        hover_name = "AREACD",
        color_discrete_map={
            "not_selected": "rgb(245, 245, 246)",
            "selected": "rbg(188, 188, 189)",
        },
        center={"lat": 55.09621, "lon": -4.0286298},
        mapbox_style="carto-positron",
        zoom=5,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    

    return fig


def add_regions(
    df: pd.DataFrame,
    region_file_path: str,
) -> pd.DataFrame:
    """Adds regions to loaded data.

    Parameters
    ----------
    df
        Contains the loaded data.
    region_file_path
        Path to localauthority-to-region lookup file.

    Returns
    -------
    DataFrame with added region column.
    """
    df_region_lookup = pd.read_csv(region_file_path)
    df_region_lookup = df_region_lookup.rename(columns={
        "LAD21CD": "AREACD",
    })
    df = df.merge(df_region_lookup, how="inner", on="AREACD")

    return df


def add_coastal_information(
    df: pd.DataFrame,
    coastal_file_path: str,
) -> pd.DataFrame:
    """Adds coastal town to the loaded data.

    Parameters
    ----------
    df
        Contains the loaded data.
    coastal_file_path
        Path to the coastal town lookup file.

    Returns
    -------
    DataFrame with coastal towns added.
    """
    df_coastal_lookup = pd.read_csv(coastal_file_path)
    df_coastal_lookup = df_coastal_lookup.rename(columns={
        "Local Authority Code": "AREACD",
    })
    df_coastal_lookup = df_coastal_lookup.loc[:, ['AREACD', 'Coastal towns']]
    df_coastal_lookup.loc[df_coastal_lookup['Coastal towns'] == "Y", 'Coastal towns'] = "Coastal"
    df_coastal_lookup.loc[df_coastal_lookup['Coastal towns'] == "N", 'Coastal towns'] = "Non-coastal"

    df = df.merge(df_coastal_lookup, how="inner", on="AREACD")

    return df


def filter_df(
    df: pd.DataFrame,
    selected_period: str,
    selected_areas: Sequence[str],
    selected_metric: str,
    selected_coastal: str,
) -> pd.DataFrame:
    """Creates subsets using various dashboard filters.

    Parameters
    ----------
    df
        Contains the loaded data.
    selected_period
        Period to filter by.
    selected_areas
        Areas that have been chosen in dropdown menu.
    selected_metric
        Metric to filter the data by.
    selected_coastal
        Coastal or non-coastal filter.

    Returns
    -------
    DataFrame with filters applied.
    """
    df_subset = df.loc[(
            (df['Period'] == selected_period) &
            (df['Indicator'] == selected_metric) &
            (df['RGN21NM'].isin(selected_areas))
    )]

    if selected_coastal != "blank":
        df_subset = df_subset.loc[(
                df['Coastal towns'] == selected_coastal
        )]

    return df_subset


def load_metadata(
    metadata_path: str,
) -> str:
    """Loads the metadata from a yaml config.

    Parameters
    ----------
    metadata_path
        Path to the yaml file containing the metadata.

    Returns
    -------
    List of Strings containing formatted metadata.
    """
    with open(metadata_path, encoding="utf-8") as f:
        loaded_config = yaml.load(f, Loader=SafeLoader)

    contact_details = loaded_config['contact_details']
    information = f"{loaded_config['description']}. The data is provided by {loaded_config['data_owner']}."
    geographies = f"These datasets use the following geography boundaries: {loaded_config['geographies']}."

    return [contact_details, information, geographies]


app = Dash(__name__)
server = app.server


# Load travel to work datasets
# project_path = "C:/Users/tobyh/Desktop/testing plotly/Dashboard_April_Demo"
# metric_folder = "C:/Users/tobyh/Desktop/testing plotly/Dashboard_April_Demo/data"
project_path = ""
metric_folder = "data"
travel_to_work_metric_files = [
    "5_2023_Public transport or walk to employment centre with 500 to 4999 jobs.csv",
    "5_2023_Drive to employment centre with 500 to 4999 jobs.csv",
    "5_2023_Cycle to employment centre with 500 to 4999 jobs.csv",
]

metadata = load_metadata(metadata_path=f"assets/metric_metadata.yaml")

df = load_travel_to_work_datasets(metric_folder, travel_to_work_metric_files)
df_shapefile = load_shapefile(shapefile_path=f"data/lad_shapefile/LAD_DEC_2021_GB_BGC.shp")
df_copy = df_shapefile.copy(deep=True)

# Add region column for filtering use
df = add_regions(df, region_file_path=f"data/lad_to_region_2021_england.csv")
df = add_coastal_information(df, coastal_file_path=f"data/coastal_lookup.csv")
unique_regions = df.loc[:, 'RGN21NM'].unique()
unique_periods = sorted(df.loc[:, 'Period'].unique())

subset_df = df.copy(deep=True)

fig = initialise_choropleth(df, 
                            df_shapefile=df_copy,
                            )
area_map = dcc.Graph(id="map", figure=fig, style={"marginLeft": "100px", "z-index": 1})

# Initialise dcc components
region_dropdown = dcc.Dropdown(
    id="region-dropdown",
    options=[{"label": x, "value": x} for x in unique_regions],
    value=None,
    multi=True,
)

# Travel to work metrics
metric_dropdown = dcc.Dropdown(
    id="metric-dropdown",
    options=[
        {"label": "Public transport or walk to employment centre", "value": 0},
        {"label": "Drive to employment centre", "value": 1},
        {"label": "Cycle to employment centre", "value": 2},
    ],
    value=0,
    multi=False,
)
coastal_type_dropdown = dcc.Dropdown(
    id="coastal-type-dropdown",
    options=[
        {"label": "Coastal", "value": 0},
        {"label": "Non-coastal", "value": 1},
    ]
)

# Timeseries slider
slider_marks = {str(x): str(unique_periods[x]) for x in range(len(unique_periods))}
timeseries_slider = dcc.Slider(
    id="timeseries-slider",
    marks=slider_marks,
    value=0,
    step=None,
)

introduction_text = "This is an explanation of what the dashboard tool does."

app.layout = html.Div(children=[

    # Divs for storing values
    html.Div(id="test-div"),
    html.Div(id="slider-value", style={"display": "none"}),
    html.Div(id="selected-areas", style={"display": "none"}),
    html.Div(id="selected-metric", style={"display": "none"}),
    html.Div(id="selected-coastal", style={"display": "none"}),

    html.Div([
        html.Img(src=app.get_asset_url('logo.png')),
        html.H1(id="heading", children="Towns and Cities Selector Tool", className="title")],
    className = "banner"),

    dcc.Tabs(id="tab-selector", children=[
        dcc.Tab(id="map-tab", label="Map", children=[
            html.Div(id="introduction-text", children=introduction_text),
            html.Div(children=[
                html.Div(id="filter-div", children=[
                    html.H2(id="filter-h2", children="Filter"),
                    html.H3(children="Region"),
                    region_dropdown,
                    html.H3(children="Metric"),
                    metric_dropdown,
                    html.H3(children="Coastal type"),
                    coastal_type_dropdown,
                    html.H3(children="Time period"),
                    timeseries_slider,

                ], style={"width": "400px"}, className = "filter_div" ),

                html.Div(id="map-div", children=[
                    html.H2(id="map-h2", children="Map"),
                    area_map, 
                ], className = "map_div"),
            ], style={'display': 'flex', 'flex-direction': 'row'}),
        ]),

        dcc.Tab(id="information-tab", label="Information", children=[
            html.Div([
                # html.H6("Change the value in the text box to see callbacks in action!"),
                # html.Div([
                #     "Input: ",
                #     dcc.Input(id='my-input', value='initial value', type='text')
                # ]),
                # html.Br(),
                # html.Div(id='my-output'),
                html.H4("Contact details"),
                html.Hr(),
                html.Div(id="contact-details-div", children=metadata[0]),
                html.H4("Important information", style={"padding-top": "20px"}),
                html.Hr(),
                html.Div(id="important-information-div", children=metadata[1]),
                html.H4("Geography information", style={"padding-top": "20px"}),
                html.Hr(),
                html.Div(id="geography-information-div", children=metadata[2]),
            ], style={"white-space": "pre-wrap"})
        ]),
    ])
])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    return f'this is the test for the inputed value: {input_value}'


@app.callback(
    Output("slider-value", component_property="children"),
    Input("timeseries-slider", component_property="value"),
)
def update_period(value):
    return unique_periods[value]


@app.callback(
    Output("selected-areas", component_property="children"),
    Input("region-dropdown", component_property="value"),
)
def update_selected_regions(value):
    if not isinstance(value, list):
        value = [value]

    return value


@app.callback(
    Output("selected-coastal", component_property="children"),
    Input("coastal-type-dropdown", component_property="value"),
)
def update_coastal_type(value):
    coastal_type = ""
    if value == 0:
        coastal_type = "Coastal"
    elif value == 1:
        coastal_type = "Non-coastal"
    elif value is None:
        coastal_type = "blank"

    return coastal_type


@app.callback(
    Output("selected-metric", component_property="children"),
    Input("metric-dropdown", component_property="value"),
)
def update_selected_metric(value):
    metric = ""
    if value == 0:
        metric = "Public transport or walk to employment centre with 500 to 4999 jobs"
    elif value == 1:
        metric = "Drive to employment centre with 500 to 4999 jobs"
    elif value == 2:
        metric = "Cycle to employment centre with 500 to 4999 jobs"

    return metric


@app.callback(
    Output(area_map, component_property="figure"),
    Input("slider-value", component_property="children"),
    Input("selected-areas", component_property="children"),
    Input("selected-metric", component_property="children"),
    Input("selected-coastal", component_property="children"),
)
def update_graph(time_period, selected_areas, selected_metric, selected_coastal):
    df_subset = filter_df(
        df,
        selected_period=time_period,
        selected_areas=selected_areas,
        selected_metric=selected_metric,
        selected_coastal=selected_coastal,
    )

    x = df_shapefile.copy()
    x.loc[:, 'colour'] = 'not_selected'
    subset_codes = df_subset.loc[:, 'AREACD'].unique()
    x.loc[x['AREACD'].isin(subset_codes), 'colour'] = 'selected'
    df_unique = df.loc[:, ['AREACD', 'AREANM']].drop_duplicates()

    # Find AREACDs that aren't in df_subset and add as empty row to ensure whole map is plotted.
    missing_codes = set(df_unique.loc[:, 'AREACD'].unique()) - set(subset_codes)
    df_unique.loc[df_unique['AREACD'].isin(missing_codes), 'Value'] = -1
    df_unique = df_unique.loc[df_unique['AREACD'].isin(missing_codes)]
    df_unique = pd.concat([df_unique, df_subset.loc[:, ['AREACD', 'AREANM', 'Value']]])

    x = x.merge(df_unique, on="AREACD", how="inner")
    x["location_tag"] = x["AREANM"]
    x = x.set_index("AREANM")

    x['Value'] = pd.to_numeric(x['Value'])

    fig = px.choropleth_mapbox(
        x,
        geojson=x.geometry,
        locations=x["location_tag"],
        color=x['Value'],
        color_continuous_scale="Blues",
        # color=x["colour"],
        # color_discrete_map={
        #     "not_selected": "rgb(245, 245, 246)",
        #     "selected": "rgb(39, 160, 204)",
        #     "": "rgb(245, 245, 246)",
        # },

        center={"lat": 55.09621, "lon": -4.0286298},
        hover_name = x.index,
        hover_data={'colour': False, "location_tag": False, "Value": False},
        mapbox_style="carto-positron",
        zoom=5,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      hoverlabel=dict(
                        bgcolor="YELLOW",
                        font_size=16,
                        font_family="Rockwell"
                      ),
                      uirevision="cheese",
    )
    fig.update_traces(hoverinfo = "skip")

    return fig


@app.callback(
    Output("test-div", component_property="children"),
    Output("test-div", component_property="style"),
    Input("map", component_property="clickData"),
    Input("slider-value", component_property="children"),
    Input("selected-areas", component_property="children"),
    Input("selected-metric", component_property="children"),
    Input("selected-coastal", component_property="children"),
)
def show_tooltip(click_data, time_period, selected_areas, selected_metric, selected_coastal):
    x_coordinate = 0
    y_coordinate = 0

    if click_data is not None:
        location = click_data['points'][0]['location']
        click_data = click_data['points'][0]['bbox']
        x_coordinate = click_data['x0']
        y_coordinate = click_data['y0']

    df_subset = filter_df(
        df,
        selected_period=time_period,
        selected_areas=selected_areas,
        selected_metric=selected_metric,
        selected_coastal=selected_coastal,
    )

    value = None
    if (not df_subset.empty and (location in df_subset['AREANM'].values.tolist())):
        value = df_subset.loc[df_subset['AREANM'] == location, 'Value'].values[0]

    output_text = ""
    if value is not None:
        output_text = f"Value: {value}"

    # mouse = Controller()
    div_style = {
        # "top": mouse.position[1],
        # "left": mouse.position[0],
        "top": y_coordinate - 50,
        "left": x_coordinate,
        "position": "absolute",
        "border": "1px solid black",
        "z-index": "5",
        "background-color": "yellow",
        "font-family": "Rockwell",
    }

    return output_text, div_style


if __name__ == '__main__':
    app.run_server(debug=False)
