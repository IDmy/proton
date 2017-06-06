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
from bokeh.models.callbacks import CustomJS
from random import randint

from optimizer import *
from estimator import BEDPredictorUpperBoundCorrect
import pandas as pd
import os


#Download data
df = pd.read_csv('data/PayoffMatrix.txt', delim_whitespace=True, header=None)
BED = df.values
#constants
NUM_PATIENTS = BED.shape[0]
MAX_FRACTIONS = BED.shape[1] - 1
MIN_LP_TIME = math.ceil(NUM_PATIENTS / 6) # 12 * t[hrs] >= 2 *number_of_patients --> t >= number_of_patients/6
confidence_rate = 0

# Titles of paragraphs
p0 = Div(text="""<div style="background-color:#00b33c;color:white;padding:20px;"><span><center><font size="+5", color="#ffffff", face="Times New Roman"><b>Allocation of proton radiotherapy over patients </b></font></center></span></div>""", width=1350, height=90)
p1 = Div(text="""<font size="+3", color="#154696"><b>Settings: </b></font>""", width=600, height=40)
p2 = Div(text="""<font size="+3", color="#154696"><b>Results: </b></font>""", width=600, height=50)
confidence = Paragraph()

# Titles of widgets
RadioButton_title = Div(text="""<font size="-0.5">Select the type of model: </font>""", width=600, height=15)

#Widgets
RadioButton = RadioButtonGroup(labels=["Linear", "Heuristic", "Automatic"], active=2)
radio_button_change = CustomJS(args = dict(radioButton = RadioButton), code="""
    var sel_model_options = {linear : 0 , heuristic : 1, auto : 2 };
    var sel_model = radioButton.attributes.active;
    if(sel_model == sel_model_options['heuristic']) {
        document.getElementsByClassName("bk-slider-parent")[0].style.display = "none";
    } else {
        document.getElementsByClassName("bk-slider-parent")[0].style.display = "block";
    }
""")
RadioButton.callback = radio_button_change

time = Slider(start = 1, end = 20, value = 10, step=1, title = "Set the time for calculation in hours")
add_params = ColumnDataSource(data=dict(min_lp_time = [MIN_LP_TIME], num_patients = [NUM_PATIENTS]))
capacity_change = CustomJS(args = {'add_params' : add_params, 'time' : time, 'radioButton': RadioButton}, code="""
    // JavaScript code goes here
    console.log(time.id);
    var sel_model_options = {linear : 0 , heuristic : 1, auto : 2 };
    var sel_model = radioButton.attributes.active;
    if(sel_model == sel_model_options['auto']){ // only for Automatic Model Choice
        capacity_val = parseInt(cb_obj.value);
        heur_time = capacity_val * 5 / 60;  // converting heur_time to hours
        num_patients = add_params.data.num_patients[0];
        min_lp_time = add_params.data.min_lp_time[0];
        use_heur = heur_time < min_lp_time;
        min_time = ((use_heur) ? heur_time : min_lp_time);
        if(use_heur) {
            document.getElementsByClassName("bk-slider-parent")[0].style.display = "none";
        } else {
            document.getElementsByClassName("bk-slider-parent")[0].style.display = "block";
            time.set('value', min_time);
            time.set('start', min_time);
        }
    }
""")

capacity = TextInput(value = "100", title="Set the capacity:", callback = capacity_change)

calculation_button = Button(label = 'Calculate', button_type="success")

# Initialization of table object
data = dict(patients=[],
            fractions=[])
source = ColumnDataSource(data)
columns = [TableColumn(field="patients", title="Patient ID"),
           TableColumn(field="fractions", title="Number of fractions")]
results_table = DataTable(source=source, columns=columns, width=700, height=450)

#============================FUNCTIONS==========================================
#Model selection
def model_selection(attr, old, new):
    if RadioButton.active == 1: #Heuristic
        time.title = "1"
        time.end = 1
        time.step = 2
        time.value = MIN_LP_TIME
    elif RadioButton.active == 0:
        time.title = 'Set the time for calculation'
        time.end = 10
        time.step = 1
        time.value = MIN_LP_TIME
        time.start = MIN_LP_TIME
    else:
        time.title = 'Set the time for calculation'
        time.end = 10
        time.step = 1
        heur_time = int(capacity.value) + NUM_PATIENTS
        time.start = time.value = heur_time if heur_time < MIN_LP_TIME else MIN_LP_TIME

def get_obj_val(solution, BED):
    total = 0
    for i, j in solution.items():
        total += BED[i, j]
    return total

# Table with results
def show_table():
    t = time.value
    c = int(capacity.value)
    if c >= NUM_PATIENTS * MAX_FRACTIONS: #if the capacity is bigger than num_patient * num_fractions, than no optimization is needed. Everyone will get MAX_FRACTIONS
        results = dict(patients=list(range(0, NUM_PATIENTS)), fractions=[MAX_FRACTIONS] * NUM_PATIENTS)
        confidence_rate = 0
    else:
        if RadioButton.active == 0:
            opt = SmartOptimizer().build(BED, capacity=c, max_time=t * 60, force_linear = True)
            confidence_rate = opt.get_confidence_rate()
        elif RadioButton.active == 1:
            opt = HeuristicOptimizer().build(BED, capacity = c)
            confidence_rate = 0
        else:
            opt = SmartOptimizer().build(BED, capacity = c, max_time = t*60)
            confidence_rate = opt.get_confidence_rate()
        output = opt.get_optimum()
        results = dict(patients=list(output.keys()),
                    fractions=list(output.values()))
    source.data = results
    confidence.text = "Confidence rate: " + str(confidence_rate)

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
table = widgetbox(p2, results_table, confidence, width=700, height=500)
l(p0, inputs, table)
os.system('bokeh serve --show bgui.py')