import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import numpy as np
import plotly
import plotly.io as pio

# Read Data from csv
df = pd.read_csv('FSC.csv')


barchart = px.bar(
    data_frame=df,
    x="Food Value Chain Stage",
    y="percentage_loss_of_quantity",
    color="Food Classification",               
    opacity=0.8,                  
    orientation="v",              
    barmode='relative', 

    labels={"percentage_loss_of_quantity":"Percentage loss of quantity",
    "Food Classification":"Food Classification"},           # map the labels of the figure
    title='Food Supply Chain Loss per Food Classification', # figure title
    width=1400,                   # figure width in pixels
    height=720,                   # figure height in pixels
    template='gridon',)

pio.show(barchart)