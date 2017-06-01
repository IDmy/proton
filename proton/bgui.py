# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:14:59 2017

@author: DIhnatov
"""

#============================Limitations========================================
'''PROBLEMS: the lower bound of widget time is not changing with capacity.
   EXAMPLE: if we have capacity 200 and minimum time for calculation is 3 hour, then if the user will set 2 hour the program will not calculate the result '''

# Import
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import TextInput, Button, Paragraph, Div, RadioButtonGroup, Slider, DataTable, DateFormatter, TableColumn
from bokeh.layouts import layout, row, column, widgetbox

from random import randint

from optimizer import *
import pandas as pd
import os

#Download data
df = pd.read_csv('data/PayoffMatrix.txt', delim_whitespace=True, header=None)
BED = df.values
number_of_patients = BED.shape[0]
min_LP_time = math.ceil(number_of_patients/6) # 12 * t[hrs] >= 2 *number_of_patients --> t >= number_of_patients/6
# Titles of paragraphs
p = Div(text="""<font size="+1", color="#0c701c", face="Times New Roman"><b></b></font>""", width=1, height=1)
p0 = Div(text="""<div style="background-color:#00b33c;color:white;padding:20px;"><span><center><font size="+5", color="#ffffff", face="Times New Roman"><b>Allocation of proton radiotherapy over patients </b></font></center></span></div>""", width=1350, height=90)
p1 = Div(text="""<font size="+3", color="#154696"><b>Settings: </b></font>""", width=600, height=40)
p2 = Div(text="""<font size="+3", color="#154696"><b>Results: </b></font>""", width=600, height=50)

# Titles of widgets
RadioButton_title = Div(text="""<font size="-0.5">Select the type of model: </font>""", width=600, height=15)

#Widgets
RadioButton = RadioButtonGroup(labels=["Linear", "Heuristic", "Automatic"], active=2)
capacity = TextInput(value = "100", title="Set the capacity:")
time = Slider(start = min_LP_time, end=20, value=min_LP_time, step=1, title = "Set the time for calculation in hours")
calculation_button = Button(label = 'Calculate', button_type="success")

# Initialization of table object
data = dict(patients=[],
            fractions=[])
source = ColumnDataSource(data)
columns = [TableColumn(field="patients", title="Patient ID"),
           TableColumn(field="fractions", title="Number of fractions")]
results_table = DataTable(source=source, columns=columns, width=700, height=550)

#============================FUNCTIONS==========================================
#Model selection
def model_selection(attr, old, new):
    if RadioButton.active == 1:
        time.title = "1"
        time.end = 1
        time.step=2
        time.value=0
    else:
        time.title = 'Set the time for calculation'
        time.end = 10
        time.step=1
        time.value=1

# Table with results
def show_table():
    t = time.value
    c = int(capacity.value)
    if RadioButton.active == 0:
        opt = SmartOptimizer().build(BED, capacity=c, max_time=t * 60, force_linear = True)
    elif RadioButton.active == 1:
        opt = HeuristicOptimizer().build(BED, capacity = c)
    else:
        opt = SmartOptimizer().build(BED, capacity = c, max_time = t*60)

    output = opt.get_optimum()
    results = dict(patients=list(output.keys()),
                fractions=list(output.values()))
    source.data = results

# Visualization function
def l(p0, inputs, table):
    curdoc().add_root(column(p0,row(inputs, table, width = 1200)))

#Interactions
RadioButton.on_change("active", model_selection)
calculation_button.on_click(show_table)
#===============================================================================

# Webpage visualization
title = p0
inputs = widgetbox(p1, RadioButton_title, RadioButton, capacity, time, calculation_button, width=550, height=550)
table = widgetbox(p2, results_table, width=700, height=500)
l(p0, inputs, table)
os.system('bokeh serve --show bgui.py')
