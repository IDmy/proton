# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:14:59 2017

@author: DIhnatov
"""
# Import
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import TextInput, Button, Paragraph, Div, RadioButtonGroup, Slider, DataTable, DateFormatter, TableColumn
from bokeh.layouts import layout, row, column, widgetbox

from random import randint

# create widgets
button = Button(label = 'Calculate')

# Titles of paragraphs
p = Div(text="""<font size="+1", color="#0c701c", face="Times New Roman"><b></b></font>""", width=1, height=1)
p0 = Div(text="""<font size="+5", color="#0c701c", face="Times New Roman"><b>Allocation of proton radiotherapy over patients </b></font>""", width=1300, height=50)
p1 = Div(text="""<font size="+3", color="#154696"><b>Settings: </b></font>""", width=600, height=30)
p2 = Div(text="""<font size="+3", color="#154696"><b>Results: </b></font>""", width=600, height=30)

# Titles of widgets
RadioButton_title = Div(text="""<font size="-1">Select the type of model: </font>""", width=600, height=20)
#capacity_title = Div(text="""<font size="+0">Set the capacity: </font>""", width=300, height=20)
#slider1_title = Div(text="""<font size="+0">Set the number of points for interpolation: </font>""", width=300, height=20)
#slider2_title = Div(text="""<font size="+0">Set the time for calculation: </font>""", width=300, height=20)
#slider3_title = Div(text="""<font size="+0">Set the accuracy: </font>""", width=300, height=20)



#Widgets
RadioButton = RadioButtonGroup(labels=["Linear", "Heuristic", "Automatic"])


capacity = TextInput(value = '100', title="Set the capacity:")
#slider1 = Slider(start=0, end=10, value=1, step=1, title = "Set the granularity")
time = Slider(start=0, end=10, value=1, step=1, title = "Set the time for calculation")
#slider3 = Slider(start=0, end=10, value=1, step=1, title = "Set the accuracy")

button = Button(label = 'Calculate', button_type="success")


table = p
inputs = widgetbox(p1, RadioButton_title, RadioButton, capacity, time, button, width=550, height=550)
#Model selection
def model_selection(attr, old, new):
    if RadioButton.active == 1:
        time.title = "1"
        time.end = 1
        time.step=2
    else:
        time.title = 'Set the time for calculation'
        time.end = 10
        time.step=1
#function that can run optimization
#def calculate():
#        output.text = 'Hello, ' + text_input.value

#button.on_click(calculate)

# Table with results
def show_table():
    t = time.value
    c = capacity.value
#    r = RadioButton.value

    data = dict(patients=[randint(0, 100) for i in range(100)],
                fractions=[randint(0, 100) for i in range(100)])

    source = ColumnDataSource(data)

    columns = [TableColumn(field="patients", title="Patient ID"),
               TableColumn(field="fractions", title="Number of fractions")]

    results_table = DataTable(source=source, columns=columns, width=700, height=550)
    table_r = widgetbox(p2, results_table, width=700, height=500)
    p = table_r
#    source.data = dict(x=x, y=y)

def l(p0, inputs, table):
#    column(p0,row(inputs, table, width = 1200))
    curdoc().add_root(column(p0,row(inputs, table, width = 1200)))


RadioButton.on_change("active", model_selection)
button.on_click(show_table)
#lay_out = layout([[p0],
#                [p1],
#                [RadioButton_title, RadioButton],
#                [capacity_title, capacity],
#                [slider1_title, slider1],
#                [slider2_title, slider2],
#                [slider3_title, slider3],
#                [button],
#                [p2],
#                [results_table]])

#, sizing_mode="scale_both"
#curdoc().add_root(lay_out)

title = p0
l(p0, inputs, table)

#curdoc().add_root(l(p0, inputs, table))
