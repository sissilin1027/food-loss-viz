import dash  # (version 1.12.0)
import numpy as np
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from urllib.request import urlopen
import json
import base64
from skimage import io
import plotly.graph_objects as go
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.io as pio
import plotly


# Create an app
app = dash.Dash(__name__)
server = app.server

##### Loading and preparing data #####

# Choropleth data
df = pd.read_csv("for_choropleth.csv")
##### Visualizations #####

# Choropleth figure
# min year in the dataset
year = 1961

data_slider = []
for year in df['Year'].unique():
    df_segmented = df[df['Year'] == year]

    for col in df_segmented.columns:
        df_segmented[col] = df_segmented[col].astype(str)
    df_segmented['text'] = df_segmented['Area']  # + \
    # ' Loss to Production Ratio: ' + df_segmented['Ratio_percent']

    data_each_yr = dict(
        type='choropleth',
        locations=df_segmented['country_code'],
        z=df_segmented['Ratio_percent'].astype(float),
        zmin=min(df_segmented['Ratio_percent'].astype(float)),
        zmax=max(df_segmented['Ratio_percent'].astype(float)),
        zauto=True,
        locationmode='ISO-3',
        colorscale='RdBu',
        text=df_segmented['text'],
        autocolorscale=False,
        marker=dict(line=dict(color="rgb(255, 255, 255)", width=1)),
        colorbar={'title': 'Loss to Production Ratio (%)'})

    data_slider.append(data_each_yr)

steps = []
for i in range(len(data_slider)):
    step = dict(method='restyle',
                args=['visible', [False] * len(data_slider)],
                label='{}'.format(i + 1961 - 1))
    step['args'][1][i] = True
    steps.append(step)

sliders = [dict(active=0, pad={"t": 1}, steps=steps)]

layout = dict(#title='Loss to Production Ratio Per Country <br> 1961-2018</br>',
              geo=dict(scope='world',
                       projection={'type': 'equirectangular'}),
              sliders=sliders,
              height=750, margin={"r": 20, "t": 50, "l": 20, "b": 20})


fig = go.Figure(data=data_slider)
fig.update_layout(layout)
# fig.show()

# tree map viz
# Read Data from csv
df_tree = pd.read_csv('country_year_perishableornp_ratios_tibble.csv')

dfy = list(df_tree.groupby("Year"))

first_title = '<b>Loss to Production Ratio (%) in <b> ' + dfy[0][0]
traces = []
buttons = []
for i, d in enumerate(dfy):
    visible = [False] * len(dfy)
    visible[i] = True
    name = d[0]
    traces.append(
        px.treemap(d[1],
                   path=['Year','Country','Non_Perishable'],
                   values='Loss_To_Production_Ratio(%)').update_traces(visible=True if i==0 else False).data[0]
    )
    buttons.append(dict(label=name,
                        method="update",
                        args=[{"visible": visible},
                              {"title": '<b>Loss To Production Ratio (%) in </b>' + f"{name}"}]))

updatemenus = [{'active': 0, "buttons": buttons}]

fig_treemap = go.Figure(data=traces,
                        layout=dict(updatemenus=updatemenus))
# Update the layout
fig_treemap.update_layout(title=first_title, title_x=0.5, title_font_size=20,
                          title_font_family='Arial')

# timeseries img & fsc
img_tsah = io.imread('tsa_high5.png')
img_tsal = io.imread('tsa_low5.png')


fig_tsal = px.imshow(img_tsal)
fig_tsah = px.imshow(img_tsah)

fig_tsal.update_layout(coloraxis_showscale=False, width=600, height=600, margin=dict(l=10, r=10, t=10, b=10))
fig_tsal.update_xaxes(showticklabels=False)
fig_tsal.update_yaxes(showticklabels=False)

fig_tsah.update_layout(coloraxis_showscale=False, width=600, height=600, margin=dict(l=10, r=10, t=10, b=10))
fig_tsah.update_xaxes(showticklabels=False)
fig_tsah.update_yaxes(showticklabels=False)

# FSC per food classification bar chart viz
df_fsc = pd.read_csv('FSC.csv')

barchart = px.bar(
    data_frame=df_fsc,
    x="Food Value Chain Stage",
    y="percentage_loss_of_quantity",
    color="Food Classification",               
    opacity=0.8,                  
    orientation="v",              
    barmode='relative', 

    labels={"percentage_loss_of_quantity":"<b>Percentage Loss of Quantity</b>",
    "Food Classification":"<b>Food Classification</b>",
    "Food Value Chain Stage":"<b>Food Value Chain Stage</b>"},          
    title='<b>Food Supply Chain Loss per Food Classification</b>', 
    width=1300,                   
    height=800,                   
    template='gridon',)


# App layout
app.layout = html.Div([
    # Title
    html.Br(),
    html.H1("Combat Food Loss: Eat Responsibly", style={
            'text-align': 'center', 'font-family': 'arial', 'font-size': '60px'}),
    html.P("“If food loss and waste were a country, it would be the third biggest source of greenhouse gas emissions” - Inger Anderson, United Nations Environment Program",
           style={'text-align': 'center', 'font-family': 'courier new', 'margin-bottom': '50px', 'font-size': '16pt','width': '60%','padding-left': '20%','padding-right':'20%'}),
    html.Br(),

    html.H1(children='Loss to Production Ratio (%) Per Country 1961-2018', 
             style={
                    "font-family":"arial", 
                    "padding-top":"30px",
                    "padding-bottom":"30px",
                    "padding-left":"20px",
                    "padding-right":"20px",
                    "width":"70%", 
                    "background-color":" #a6bddb",
                    "border-radius": "10pt",
                    "color":"white",
                    "text-shadow": "2px 2px 4px #000000",
                    "text-align":"center",
                    "margin-top":"50px",
                    "margin-left":"auto",
                    "margin-right":"auto",
                   }),

    html.P("Loss to Production Ratio (%): This ratio captures the loss amount to the production amount and is expressed as a percentage. Having a lower ratio is better as it indicates a smaller loss.  ",
           style={
                    "font-family":"arial", 
                    "font-size": "18px",
                    "padding-top":"30px",
                    "padding-bottom":"30px",
                    "padding-left":"20px",
                    "padding-right":"20px",
                    "width":"70%", 
                    "background-color":" rgb(205, 223, 247, 0.5)",
                    "border-radius": "10pt",
                    "color":"black",
                    "text-align":"left",
                    "margin-top":"50px",
                    "margin-left":"auto",
                    "margin-right":"auto",
                   }),
    html.Div(dcc.Graph(id='choropleth', figure=fig), style={
             'width': '80%', 'padding-left': '10%', 'padding-right': '20%'}),

    html.Br(),
    
    html.H1(children='Treemap with breakdown for Perishable and Non-perishable items', 
             style={
                    "font-family":"arial", 
                    "padding-top":"30px",
                    "padding-bottom":"30px",
                    "padding-left":"20px",
                    "padding-right":"20px",
                    "width":"70%", 
                    "background-color":" #a6bddb",
                    "border-radius": "10pt",
                    "color":"white",
                    "text-shadow": "2px 2px 4px #000000",
                    "text-align":"center",
                    "margin-top":"50px",
                    "margin-left":"auto",
                    "margin-right":"auto",
                   }),
    
    
    html.Div(dcc.Graph(id='treemap', figure=fig_treemap), style={
             'width': '90%', 'height': '80%','padding-left': '0%', 'padding-right': '10%'}),

    html.Br(),

    ### Cluster analysis part
    html.H1(children='Cluster Analysis', 
             style={
                    "font-family":"arial", 
                    "padding-top":"30px",
                    "padding-bottom":"30px",
                    "padding-left":"20px",
                    "padding-right":"20px",
                    "width":"70%", 
                    "background-color":" #a6bddb",
                    "border-radius": "10pt",
                    "color":"white",
                    "text-shadow": "2px 2px 4px #000000",
                    "text-align":"center",
                    "margin-top":"50px",
                    "margin-left":"auto",
                    "margin-right":"auto",
                   }),
    
    dcc.RadioItems(
        id='cluster',
        options=[
            {'label': '10 countries with the highest food loss to production ratio (%)', 'value': 'Cluster assignments for 10 countries with the highest food loss to production ratio (%)'},
            {'label': '10 countries with the lowest food loss to production ratio (%) ', 'value': 'Cluster assignments for 10 countries with the lowest food loss to production ratio (%)'},

        ],
        value='Cluster assignments for 10 countries with the highest food loss to production ratio (%)',
        labelStyle={'display': 'inline-block','padding':'10px', "font-family": "arial", "font-size":"large",
                   "padding-bottom":"30px"},
        style = {'text-align':'center'}
    ),

    html.Div(id='cluster_analysis', style={
                    "font-family":"arial", 
                    "padding-top":"30px",
                    "padding-bottom":"30px",
                    "padding-left":"20px",
                    "padding-right":"20px",
                    "width":"70%", 
                    "background-color":" rgb(205, 223, 247, 0.5)",
                    "border-radius": "10pt",
                    "color":"black",
                    "text-align":"left",
                    "margin-top":"50px",
                    "margin-left":"auto",
                    "margin-right":"auto",
                   }),

    html.Br(),


    ### Time Series Forecasting part
    html.H1(children='Time Series Forecasting Analysis', 
             style={
                    "font-family":"arial", 
                    "padding-top":"30px",
                    "padding-bottom":"30px",
                    "padding-left":"20px",
                    "padding-right":"20px",
                    "width":"70%", 
                    "background-color":" #a6bddb",
                    "border-radius": "10pt",
                    "color":"white",
                    "text-shadow": "2px 2px 4px #000000",
                    "text-align":"center",
                    "margin-top":"50px",
                    "margin-left":"auto",
                    "margin-right":"auto",
                   }),

    dcc.RadioItems(
        id='tsa',
        options=[
            {'label': '5 countries with the highest food loss to production ratio (%)', 'value': '5 countries with the highest food loss to production ratio (%)'},
            {'label': '5 countries with the lowest food loss to production ratio (%)', 'value': '5 countries with the lowest food loss to production ratio (%)'},

        ],
        value='5 countries with the highest food loss to production ratio (%)',
        labelStyle={'display': 'inline-block','padding':'10px', "font-family": "arial", "font-size":"large",
                   "padding-bottom":"30px"},
        style = {'text-align':'center'},
        persistence= False
        
    ),

 #   html.Div(dcc.Graph(id='tsa_low', figure=fig_tsal), style={
  #          'width': '80%', 'padding-left': '10%', 'padding-right': '20%', "text-align":"center"}),
    html.Div(id='tsa_analysis', style={
                    "font-family":"arial", 
                    "padding-top":"30px",
                    "padding-bottom":"30px",
                    "padding-left":"20px",
                    "padding-right":"20px",
                    "width":"70%", 
                    "background-color":" rgb(205, 223, 247, 0.5)",
                    "border-radius": "10pt",
                    "color":"black",
                    "text-align":"left",
                    "margin-top":"50px",
                    "margin-left":"auto",
                    "margin-right":"auto",
                   }),

    html.Br(),



    ### FSC loss
    html.H1(children='Food Supply Chain Loss Analysis', 
             style={
                    "font-family":"arial", 
                    "padding-top":"30px",
                    "padding-bottom":"30px",
                    "padding-left":"20px",
                    "padding-right":"20px",
                    "width":"70%", 
                    "background-color":" #a6bddb",
                    "border-radius": "10pt",
                    "color":"white",
                    "text-shadow": "2px 2px 4px #000000",
                    "text-align":"center",
                    "margin-top":"50px",
                    "margin-left":"auto",
                    "margin-right":"auto",
                   }),

     html.P("This  considers  food  losses  as  occurring  along  the  food  supply  chain  from various supply chain stages before it reaches the original consumer. Also all crops are been classified as Vegetables, Fruits, Milk, Egg, Meat and Dry cereals. The chart shows the percentage loss quantity for different crops per food supply chain stages.  ",
           style={
                    "font-family":"arial", 
                    "font-size": "18px",
                    "padding-top":"30px",
                    "padding-bottom":"30px",
                    "padding-left":"20px",
                    "padding-right":"20px",
                    "width":"70%", 
                    "background-color":" rgb(205, 223, 247, 0.5)",
                    "border-radius": "10pt",
                    "color":"black",
                    "text-align":"left",
                    "margin-top":"50px",
                    "margin-left":"auto",
                    "margin-right":"auto",
                   }),

        
    html.Div(dcc.Graph(id='barchart', figure=barchart), style={
             'width': '60%', 'height': '70%','padding-left': '10%', 'padding-right': '20%', 'padding-bottom':'20%', 'display': 'inline-block'}),
    

])




# Callback for cluster analysis option    
@app.callback(
    Output(component_id='cluster_analysis', component_property='children'),
    [Input(component_id='cluster', component_property='value')]
)  
def update_cluster(topic):
    
    # Information about the highest top 10 countries
    text_high = html.Div([
        html.H3(topic),
        html.Table([
            html.Tr([html.Th('Countries'),
                     html.Th('Perishable '),
                     html.Th('Non-Perishable '),
                     html.Th('Total ')],style={"background-color": "white"}),
            html.Tr([html.Td('Ghana'),
                     html.Td('2'),
                     html.Td('1'),
                     html.Td('1')], style={"background-color": "white"}),
            html.Tr([html.Td('Algeria'),
                     html.Td('1'),
                     html.Td('1'), 
                     html.Td('1')],style={"background-color": "white"}),
            html.Tr([html.Td('Namibia'),
                     html.Td('1'),
                     html.Td('1'), 
                     html.Td('1')],style={"background-color": "white"}),
            html.Tr([html.Td('Cameroon'),
                     html.Td('2'),
                     html.Td('1'),
                     html.Td('1')],style={"background-color": "white"}),
            html.Tr([html.Td('United Arab Emirates'),
                     html.Td('1'),
                     html.Td('0'),
                     html.Td('2')],style={"background-color": "white"}),
            html.Tr([html.Td('China, Hong Kong SAR'),
                     html.Td('0'),
                     html.Td('1'), 
                     html.Td('2')],style={"background-color": "white"}),
            html.Tr([html.Td('Gabon'),
                     html.Td('0'),
                     html.Td('0'),
                     html.Td('2')],style={"background-color": "white"}),
            html.Tr([html.Td('Kuwait'),
                     html.Td('1'),
                     html.Td('0'), 
                     html.Td('2')],style={"background-color": "white"}),
            html.Tr([html.Td('Saint Lucia'),
                     html.Td('2'),
                     html.Td('2'),
                     html.Td('2')],style={"background-color": "white"}),
            html.Tr([html.Td('Dominica'),
                     html.Td('1'),
                     html.Td('2'),
                     html.Td('2')],style={"background-color": "white"})    
        ], style={"font-family": "arial", "font-size": "large", 'text_align':"center"}),
        
        html.P("0 refers to cluster with low food loss to production ratio (%) ", style={"font-style":"italic"}),
        html.P("1 refers to cluster with medium food loss to production ratio (%) ", style={"font-style":"italic"}),
        html.P("2 refers to cluster with high food loss to production ratio (%) ", style={"font-style":"italic"}),
        html.P("From our analysis we can see that: "),

        html.Ul([
            html.Li("China, Hong Kong SAR and Gabon seem to have low loss to production ratio (%) for perishables, but medium loss to production ratio (%) for non-perishables.  "),
            html.Li("Ghana seems to have high loss to production ratio (%) for perishables but medium loss to production ratio (%) for non-perishables. "),
            html.Li("Saint Lucia is the only country here that seems to have high loss to production ratio (%) for both perishables and non-perishables. "),
            html.Li("All the other countries seem to have a mix of both low to medium loss to production ratio (%) under perishable category and low to medium loss to production ratio (%) under non-perishable category.  "),
        ]),
        html.P("Since clustering was performed separately for perishables, non-perishables, and the combined data, we see the 10 countries with highest loss to production ratios belong to different clusters for the perishables and non-perishables. "),
        html.P("However, looking at the Total Clusters column we can conclude the following: "),
        html.Ul([
            html.Li("United Arab Emirates, China, Hong Kong SAR, Gabon, Kuwait, Saint Lucia and Dominica belong to the cluster with high loss to production ratio (%)    "),
            html.Li("Ghana, Algeria and Namibia belong to the cluster with medium loss to production ratio (%). "),
        ]),
        html.P("This is mainly due to the use of the soft-DTW k-means method which allows non-linear alignments between two time series and accommodates sequences that are similar and cluster them irrespective of the timeline in question. ")

    ])
    
    # Information about the lowest top 10 countries    
    text_low = html.Div([
        html.H3(topic),
        html.Table([
            html.Tr([html.Th('Countries'),
                     html.Th('Perishable '),
                     html.Th('Non-Perishable '),
                     html.Th('Total ')],style={"background-color": "white"}),
            html.Tr([html.Td('Argentina'),
                     html.Td('0'),
                     html.Td('2'),
                     html.Td('0')], style={"background-color": "white"}),
            html.Tr([html.Td('Hungary'),
                     html.Td('0'),
                     html.Td('2'), 
                     html.Td('0')],style={"background-color": "white"}),
            html.Tr([html.Td('Romania'),
                     html.Td('0'),
                     html.Td('0'),
                     html.Td('0')],style={"background-color": "white"}),
            html.Tr([html.Td('Russian Federation'),
                     html.Td('2'),
                     html.Td('0'), 
                     html.Td('0')],style={"background-color": "white"}),
            html.Tr([html.Td('Malaysia'),
                     html.Td('0'),
                     html.Td('0'), 
                     html.Td('0')],style={"background-color": "white"}),
            html.Tr([html.Td('Panama'),
                     html.Td('0'),
                     html.Td('2'), 
                     html.Td('0')],style={"background-color": "white"}),
            html.Tr([html.Td('Ireland'),
                     html.Td('0'),
                     html.Td('2'),
                     html.Td('0')],style={"background-color": "white"}),
            html.Tr([html.Td('Latvia'),
                     html.Td('2'),
                     html.Td('1'),
                     html.Td('1')],style={"background-color": "white"}),
            html.Tr([html.Td('Estonia'),
                     html.Td('1'),
                     html.Td('1'),
                     html.Td('1')],style={"background-color": "white"}),
            html.Tr([html.Td('New Zealand'),
                     html.Td('1'),
                     html.Td('0'),
                     html.Td('2')],style={"background-color": "white"}),    
        ], style={"font-family": "arial", "font-size": "large", 'text_align':"center"}),
        
        html.P("0 refers to cluster with low food loss to production ratio (%) ", style={"font-style":"italic"}),
        html.P("1 refers to cluster with medium food loss to production ratio (%) ", style={"font-style":"italic"}),
        html.P("2 refers to cluster with high food loss to production ratio (%) ", style={"font-style":"italic"}),
        html.P("From our analysis we can see that: "),
        html.Ul([
            html.Li("Argentina, Hungary, Romania, Malaysia, Panama and Ireland have low food loss to production ratio (%) for perishables.    "),
            html.Li("Romania, Russian Federation, Malaysia and New Zealand have low food loss to production ratio (%) for non-perishables.   "),
        ]),
        html.P("Since clustering was performed separately for perishables, non-perishables, and the combined data, we see the 10 countries with highest loss to production ratios belong to different clusters for the perishables and non-perishables.  "),
        html.P("However, looking at the Total Clusters column we can conclude the following:   "),

        html.Ul([
            html.Li("Argentina, Hungary, Romania, Russian Federation, Malaysia, Panama and Ireland belong to the low food loss to production ratio (%) cluster   "),
            html.Li("Latvia, Estonia and New Zealand seem to be between medium to high loss to production ratio (%) cluster.  "),
        ]),
        html.P("This is mainly due to the use of the soft-DTW k-means method which allows non-linear alignments between two time series and accommodates sequences that are similar and cluster them irrespective of the timeline in question.   "),

    ])


    
    # Return corresponding text based on selection
    if topic == 'Cluster assignments for 10 countries with the highest food loss to production ratio (%)':
        return text_high
    elif topic == 'Cluster assignments for 10 countries with the lowest food loss to production ratio (%)':
        return text_low


    return topic

@app.callback(
    Output(component_id='tsa_analysis', component_property='children'),
    [Input(component_id='tsa', component_property='value')]
)  

def update_tsa_analysis(topic):
    tsa_high = html.Div([
        html.H3(topic, style={"text-align":"center"}),
        dcc.Graph(id='tsa_high', figure=fig_tsah, style ={
            'height': '600px', 'width': '100%', 'text-align': 'center', 'margin-top': '0px', 'margin-bottom': '20px','margin-left': '180px', 'margin-right': '200px'}),
        
    ])

    tsa_low = html.Div([
        html.H3(topic, style={"text-align": "center"}),
        dcc.Graph(id='tsa_low', figure=fig_tsal, style ={
            'height': '600px', 'width': '100%', 'text-align': 'center', 'margin-top': '0px', 'margin-bottom': '20px', 'margin-left': '180px', 'margin-right': '200px'}),
        
    ])


    # Return corresponding text based on selection
    if topic == '5 countries with the highest food loss to production ratio (%)':
        return tsa_high
    elif topic == '5 countries with the lowest food loss to production ratio (%)':
        return tsa_low
    return topic

if __name__ == '__main__':
    app.run_server(port=1038, debug=False)


