# Import
from __future__ import division
from PIL import Image
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import TextInput, Button, Paragraph, Div, RadioButtonGroup, Slider, DataTable, TableColumn, Panel, Tabs
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import row, column, widgetbox
from bokeh.plotting import figure
import matplotlib.pyplot as plt
from optimizer import *
from estimator import BEDPredictorUpperBoundCorrect
import pandas as pd
from decimal import *
import os
import math
import base64
from io import StringIO
import bgui_js_handlers

def create_BED_figure(BED, BED_min, BED_max, looked_up_inds, patient_num, max_fractions, output_file):
    """Create plot with BED figure and save it into output_file."""
    Y_plots = 3  # plots per row
    X_plots = math.ceil((patient_num) / Y_plots)
    fig, axarr = plt.subplots(X_plots, Y_plots, sharex=True)  # sharex='col', sharey='row'
    for i in range(0, patient_num):
        x, y = math.floor(i / Y_plots), i % Y_plots
        BED_looked_up = BED[looked_up_inds]
        obj1, = axarr[x, y].plot(range(0, max_fractions + 1), BED.loc[i,], 'bo')
        obj2, = axarr[x, y].plot(looked_up_inds, BED_looked_up.loc[i,], 'ro')
        obj3, = axarr[x, y].plot(range(0, max_fractions + 1), BED_min.loc[i,], color='g')
        obj4, = axarr[x, y].plot(range(0, max_fractions + 1), BED_max.loc[i,], color='darkorange')
        axarr[x, y].set_title('Patient' + str(i))
    for i in range(0, Y_plots - (patient_num % Y_plots)):
        axarr[-1, -(i + 1)].axis('off')  # hidding plots that are not used
    plot_objs = [obj1, obj2, obj3, obj4]
    fig.subplots_adjust(hspace=0.3)
    fig.set_size_inches(14, 10.71/17 * patient_num)
    fig.legend(plot_objs, ('Ground-true points', 'Looked-up points', 'Lower bound', 'Upper Bound'), loc = 'upper center', ncol=2, fontsize = 16 )
    fig.text(0.5, 0.04, '#Proton fractions', ha='center', fontsize=16)
    fig.text(0.08, 0.5, 'BED value', va='center', rotation='vertical', fontsize=16)
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)

def init_constanst(BED):
    """Setting up constants. This is also called during app execution."""
    global NUM_PATIENTS, MAX_FRACTIONS, MIN_LP_TIME, MAX_LP_TIME
    NUM_PATIENTS = BED.shape[0]
    MAX_FRACTIONS = BED.shape[1] - 1
    MIN_LP_TIME = math.ceil(NUM_PATIENTS / 6) # 12 * t[hrs] >= 2 *number_of_patients --> t >= number_of_patients/6
    MAX_LP_TIME = math.ceil(NUM_PATIENTS * MAX_FRACTIONS * 5 / 60)

def get_img_obj(path):
    """Loads a img from file a return an object, that can bokeh process."""
    img_o = Image.open(path).convert('RGBA')
    xdim, ydim = img_o.size
    print("Dimensions: ({xdim}, {ydim})".format(**locals()))
    # Create an array representation for the image `img`, and an 8-bit "4
    # layer/RGBA" version of it `view`.
    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
    # Copy the RGBA image into view, flipping it so it comes right-side up
    # with a lower-left origin
    view[:, :, :] = np.flipud(np.asarray(img_o))
    return img

def update_BED_fig_img(gran):
    global BED_fig_ds
    if gran is not None:
        BED_min = pd.DataFrame(LinearBEDPredictor(BED).estimate(granularity=gran))
        BED_max_concave = pd.DataFrame(BEDPredictorUpperBoundCorrect(BED).estimate(granularity=gran))
        interp_step = MAX_FRACTIONS / (gran - 1)  # take out -1
        looked_up_inds = [int(i * interp_step) for i in range(gran)]
        create_BED_figure(df, BED_min, BED_max_concave, looked_up_inds, NUM_PATIENTS, MAX_FRACTIONS, BED_FIG_FILENAME)
        img = get_img_obj(BED_FIG_FILENAME)
    else:
        img = get_img_obj('static/under_construction.png')
    BED_fig_ds.data = dict(image=[img])

def upl_file_callback(attr, old, new):
    global BED, num_patients, NUM_PATIENTS
    print('filename:', upl_file_souce.data['file_name'])
    raw_contents = upl_file_souce.data['file_contents'][0]
    # remove the prefix that JS adds
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    file_io = StringIO(file_contents.decode("utf-8") )

    df = pd.read_csv(file_io, delim_whitespace=True, header=None)
    init_constanst(df.values) # change the constast
    selected_model_changed(None, None, None)  # update the time slider
    info_data_uploaded.text = upl_file_souce.data['file_name'][0]
    BED = df.values
    num_patients.data = dict(val = [BED.shape[0]])
    NUM_PATIENTS = BED.shape[0]

def hide_figure_non_essentials(fig):
    """Hide everything from a bokeh figure apart from the content part."""
    fig.toolbar.logo = None
    fig.toolbar_location = None
    fig.xaxis.visible = None
    fig.yaxis.visible = None
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    fig.outline_line_alpha = 0
    fig.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    fig.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    fig.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    fig.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    fig.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
    fig.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
    fig.axis.visible = False
    return fig

def selected_model_changed(attr, old, new):
    if in_model_rdn_btn.active == 0:
        in_time_slider.value = MIN_LP_TIME
        in_time_slider.start = MIN_LP_TIME
        in_time_slider.end = MAX_LP_TIME
    else:
        heur_time = int(in_capacity_txt.value) + NUM_PATIENTS
        in_time_slider.start = in_time_slider.value = heur_time if heur_time < MIN_LP_TIME else MIN_LP_TIME
        in_time_slider.end = MAX_LP_TIME

def calculation_btn_clicked():
    t = in_time_slider.value
    c = int(in_capacity_txt.value)
    gran = None
    # if the capacity is bigger than num_patient * num_fractions, than no optimization is needed. Everyone will get MAX_FRACTIONS
    if c >= NUM_PATIENTS * MAX_FRACTIONS:
        results = dict(patients=list(range(0, NUM_PATIENTS)), fractions=[MAX_FRACTIONS] * NUM_PATIENTS)
        error_rate = 0
        calc_time = 0
        number_calls_text = 0
    else:
        if in_model_rdn_btn.active == 1: # Heuristic model case
            opt = HeuristicOptimizer().build(BED, capacity = c)
            error_rate = 0
        else:# Automatic and linear case combined model case
            force_linear = in_model_rdn_btn.active == 0 #force smrat optimizer to execute LP
            opt = SmartOptimizer().build(BED, capacity = c, max_time = t*60, force_linear = force_linear)
            error_rate = opt.get_error_rate()
            gran = opt.get_granularity()
        calc_time = opt.get_lookups()*TIME_FOR_LOOK_UP
        output = opt.get_optimum()
        number_calls_text = opt.get_lookups()
        output_value = [float(Decimal("%.2f" % e)) for e in list(opt.get_optimum_value().values())]
        results = dict(patients=list(output.keys()),
                       fractions=list(output.values()),
                       BED_value=output_value)
    # Returns
    update_BED_fig_img(gran)
    BED_data_source.data = results
    model_type.text = "Model type: " + opt.get_type()
    total_BED.text = "Total BED: " + str(format(opt.get_total_BED(), '.2f'))
    num_of_calls.text = "Number of calls: " + str(number_calls_text)
    error.text = "Error rate: " + str(format(error_rate, '.2f'))
    calculation_time.text = "Calculation time: " + str(format(calc_time/60, '.2f')) + " hours"

def l(p0, inputs, table):
    # Layout visualization function of bokeh
    curdoc().add_root(column(p0, row(inputs, table, width = 1200)))

########## INITIALIZATION OF VALUES  #####
BED_FIG_FILENAME = 'static/graph.png'
default_file_BED = 'data/PayoffMatrix.txt'
df = pd.read_csv(default_file_BED, delim_whitespace=True, header=None)
BED = df.values
init_constanst(BED)
TIME_FOR_LOOK_UP = 5
error_rate = 0
css= "<style>" + open('static/css.css', 'r').read() + "</style>"


################## SETTING - bokeh models ###########
p0 = Div(text="""<div style="background-color:#386994;color:white;padding:9px; font-size: 29px;border-radius: 3px;"><span><center><b>Dispro v.0.1</b></center></span></div>""" + css, width=1024, height=39)
p1 = Div(text="""<font size="+2", color="#154696"><b>Settings</b></font>""", width=600, height=23)
p2 = Div(text="""<font size="+2", color="#154696"><b>Results </b></font>""", width=600, height=33)
error = Paragraph()
calculation_time = Paragraph()
total_BED = Paragraph()
model_type = Paragraph()
num_of_calls = Paragraph()
info_data_uploaded = Paragraph()
info_data_uploaded.text = default_file_BED + " loaded."
rdn_btn_title = Div(text="""<font size="-0.5">Select the type of model: </font>""", width=600, height=15)
in_model_rdn_btn = RadioButtonGroup(labels=["Linear", "Heuristic", "Automatic"], active=2)

in_model_rdn_btn.callback = CustomJS(args = dict(radioButton = in_model_rdn_btn), code= bgui_js_handlers.in_model_rdn_btn_callback_code)
in_model_rdn_btn.on_change("active", selected_model_changed)
in_min_time = ColumnDataSource(data=dict(val = [MIN_LP_TIME]))
in_time_slider = Slider(start = MIN_LP_TIME, end = MAX_LP_TIME, value = 10, step=1, title ="Set the time for calculation [hrs]")
in_capacity_txt = TextInput(value ="100", title="Set the capacity:")
in_capacity_txt.callback = CustomJS(args = {'min_time' : in_min_time, 'time' : in_time_slider, 'radioButton': in_model_rdn_btn},
                                    code= bgui_js_handlers.in_capacity_txt_callback_code)
calculation_button = Button(label = 'Calculate', button_type="success")
upl_file_souce = ColumnDataSource({'file_contents':[], 'file_name':[]})
upl_file_souce.on_change('data', upl_file_callback)
upload_btn = Button(label="Upload", button_type="default")
upload_btn.callback = CustomJS(args=dict(file_source=upl_file_souce), code = bgui_js_handlers.upload_btn_callback_code)


################### RESULTS - bokeh models #######################
BED_fig = figure(x_range=(0, 1151), y_range=(0, 2049), plot_width=750, plot_height=36 * NUM_PATIENTS) #toolbar_location="above"
BED_fig_ds = ColumnDataSource(data=dict(image=[]))
BED_fig.image_rgba(image ='image', source=BED_fig_ds, x=0, y=0, dw=1151, dh=2049)
BED_fig = hide_figure_non_essentials(BED_fig)
num_patients = ColumnDataSource(data=dict(val = [NUM_PATIENTS]))
calculation_button.callback = CustomJS(args=dict(fig=BED_fig, num_patients = num_patients),
                                       code = bgui_js_handlers.calculation_button_callback_code)
calculation_button.on_click(calculation_btn_clicked)

BED_data_source = ColumnDataSource(dict(patients=[], fractions=[], BED_value=[]))
BED_result_table_cols = [TableColumn(field="patients", title="Patient ID"),
                         TableColumn(field="fractions", title="Number of fractions"),
                         TableColumn(field="BED_value", title="BED value")]
BED_result_tbl = DataTable(source=BED_data_source, columns=BED_result_table_cols, width=700, height=450)
tab1 = Panel(child = BED_result_tbl, title="Numbers")
tab2 = Panel(child = BED_fig, title="Graphs")
results_tabs = Tabs(tabs=[ tab1, tab2 ])

# Webpage visualization
title = p0
inputs = widgetbox(p1, upload_btn, info_data_uploaded, rdn_btn_title, in_model_rdn_btn, in_capacity_txt, in_time_slider, calculation_button, width = 250)
table = widgetbox(p2, results_tabs, model_type, total_BED, num_of_calls, error, calculation_time, height=600, width = 700)
l(p0, inputs, table)
os.system('bokeh serve --show  bgui.py')
