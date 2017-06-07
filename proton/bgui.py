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
from decimal import *
import os
import math


# Read data
df = pd.read_csv('data/PayoffMatrix.txt', delim_whitespace=True, header=None)
BED = df.values

#Constants
NUM_PATIENTS = BED.shape[0]
MAX_FRACTIONS = BED.shape[1] - 1
MIN_LP_TIME = math.ceil(NUM_PATIENTS / 6) # 12 * t[hrs] >= 2 *number_of_patients --> t >= number_of_patients/6
MAX_LP_TIME = math.ceil(NUM_PATIENTS * MAX_FRACTIONS * 5 / 60)
error_rate = 0

#CSS for nice visualization of all gui elements
css = """
<style>
.bk-root .bk-slider-parent input[type="text"] {
    width: 10px;
}
.bk-root {
    width: 1100px;
    margin:0 auto;
}
body {
    width:100%;
    margin: 0 auto;
    position:relative;
}
.bk-root .bk-layout-fixed.bk-grid-row>div
  {
    padding-right:20px;
  }
</style>
"""

# Titles of paragraphs
p0 = Div(text="""<div style="background-color:#386994;color:white;padding:9px; font-size: 29px;border-radius: 3px;"><span><center><b>Allocation of Proton Radiotherapy over Patients </b></center></span></div>""" + css, width=1024, height=39)
p1 = Div(text="""<font size="+2", color="#154696"><b>Settings</b></font>""", width=600, height=23)
p2 = Div(text="""<font size="+2", color="#154696"><b>Results </b></font>""", width=600, height=33)
error = Paragraph()
calculation_time = Paragraph()

total_BED = Paragraph()
model_type = Paragraph()
num_of_calls = Paragraph()

# Titles of widgets
RadioButton_title = Div(text="""<font size="-0.5">Select the type of model: </font>""", width=600, height=15)

#==============================Widgets==========================================
#RadioButton widget for model type selection
RadioButton = RadioButtonGroup(labels=["Linear", "Heuristic", "Automatic"], active=2)
#Add function to hide time slider if heuristic model was selected
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

#Time slider widget
time = Slider(start = MIN_LP_TIME, end = MAX_LP_TIME, value = 10, step=1, title = "Set the time for calculation [hrs]")

#Capacity widget
add_params = ColumnDataSource(data=dict(min_lp_time = [MIN_LP_TIME], num_patients = [NUM_PATIENTS]))
#Add function to change min time of slider baced on selected capasity value
capacity_change = CustomJS(args = {'add_params' : add_params, 'time' : time, 'radioButton': RadioButton}, code="""
    // JavaScript code goes here
    console.log(time.id);
    var sel_model_options = {linear : 0 , heuristic : 1, auto : 2 };
    var sel_model = radioButton.attributes.active;
    if(sel_model == sel_model_options['auto']){ // only for Automatic Model Choice
        capacity_val = parseInt(cb_obj.value);
        heur_time = (capacity_val + num_patients) * 5 / 60;  // converting heur_time to hours
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

#Calculation button widget
calculation_button = Button(label = 'Calculate', button_type="success")

#Table widget - Initialization of table object
data = dict(patients=[],
            fractions=[],
            BED_value=[])
source = ColumnDataSource(data)
columns = [TableColumn(field="patients", title="Patient ID"),
           TableColumn(field="fractions", title="Number of fractions"),
           TableColumn(field="BED_value", title="BED value")]
results_table = DataTable(source=source, columns=columns, width=700, height=450)

#============================FUNCTIONS==========================================
#Model selection: run when RadioButton pressed
def model_selection(attr, old, new):
    if RadioButton.active == 0:
        time.value = MIN_LP_TIME
        time.start = MIN_LP_TIME
        time.end = MAX_LP_TIME
    else:
        heur_time = int(capacity.value) + NUM_PATIENTS
        time.start = time.value = heur_time if heur_time < MIN_LP_TIME else MIN_LP_TIME
        time.end = MAX_LP_TIME

# Table with results: run when calculation button is pressed
def show_table():
    t = time.value
    c = int(capacity.value)
    # if the capacity is bigger than num_patient * num_fractions, than no optimization is needed. Everyone will get MAX_FRACTIONS
    if c >= NUM_PATIENTS * MAX_FRACTIONS:
        results = dict(patients=list(range(0, NUM_PATIENTS)), fractions=[MAX_FRACTIONS] * NUM_PATIENTS)
        error_rate = 0
        calc_time = 0
        number_calls_text = 0
    else:
        #Linear model case
        if RadioButton.active == 0:
            opt = SmartOptimizer().build(BED, capacity=c, max_time=t * 60, force_linear = True)
            error_rate = opt.get_error_rate()
            calc_time  = opt.get_calculation_time()
        #Heuristic model case
        elif RadioButton.active == 1:
            opt = HeuristicOptimizer().build(BED, capacity = c)
            error_rate = 0
            calc_time = opt.get_lookups()*5
        #Automatic model case
        else:
            opt = SmartOptimizer().build(BED, capacity = c, max_time = t*60)
            error_rate = opt.get_error_rate()
            heur_time = c * 5 / 60  # converting heur_time to hours
            if heur_time < MIN_LP_TIME:
                h = HeuristicOptimizer().build(BED, capacity = c)
                calc_time = h.get_lookups()*5
            else:
                calc_time  = opt.get_calculation_time()

        output = opt.get_optimum()
        number_calls_text = opt.get_lookups()
        output_value = [float(Decimal("%.2f" % e)) for e in list(opt.get_optimum_value().values())]

        results = dict(patients=list(output.keys()),
                       fractions=list(output.values()),
                       BED_value=output_value)
    # Returns
    source.data = results
    model_type.text = "Model type: " + opt.get_type()
    total_BED.text = "Total BED: " + str(format(opt.get_total_BED(), '.2f'))
    num_of_calls.text = "Number of calls: " + str(number_calls_text)
    error.text = "Error rate: " + str(format(error_rate, '.2f'))
    calculation_time.text = "Calculation time: " + str(format(calc_time/60, '.2f')) + " hours"
# Visualization function
def l(p0, inputs, table):
    curdoc().add_root(column(p0,
                            row(inputs, table, width = 1200)))

#===============================================================================

#Interactions
RadioButton.on_change("active", model_selection)
calculation_button.on_click(show_table)

# Webpage visualization
title = p0
inputs = widgetbox(p1, RadioButton_title, RadioButton, capacity, time, calculation_button, width = 250)
table = widgetbox(p2, results_table, model_type, total_BED, num_of_calls, error, calculation_time, height=600, width = 700)
l(p0, inputs, table)
os.system('bokeh serve --show bgui.py')
