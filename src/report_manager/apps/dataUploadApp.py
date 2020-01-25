import base64
import io
import os
import pandas as pd
import json
import py2neo
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import flask
import config.ckg_config as ckg_config
from apps import basicApp
from graphdb_connector import connector
import logging
import logging.config

driver = connector.getGraphDatabaseConnectionConfiguration()
DataTypes = ['experimental_design', 'clinical', 'proteomics', 'wes', 'longitudinal_proteomics', 'longitudinal_clinical']

class DataUploadApp(basicApp.BasicApp):
    """
    Defines what the dataUpload App is in the report_manager.
    Used to upload experimental and clinical data to correct project folder.

    .. warning:: There is a size limit of 55MB. Files bigger than this will have to be moved manually.
    """
    def __init__(self, title, subtitle, description, layout = [], logo = None, footer = None):
        self.pageType = "UploadDataPage"
        basicApp.BasicApp.__init__(self, title, subtitle, description, self.pageType, layout, logo, footer)
        self.buildPage()

    def buildPage(self):
        """
        Builds page with the basic layout from *basicApp.py* and adds relevant Dash components for project data upload.
        """
        self.add_basic_layout()
        layout = [html.Div([
                            html.Div([html.H4('Project identifier:', style={'marginTop':30, 'marginBottom':20}),
                                      dcc.Input(id='project_id', placeholder='e.g. P0000001', type='text', value='', debounce=True, maxLength=8, minLength=8, style={'width':'100%', 'height':'55px'}),
                                      dcc.Markdown(id='existing-project')], 
                                     style={'width':'20%'}),
                            html.Br(),
                            html.Div(id='upload-form',children=[
                                html.Div(children=[html.Label('Select upload data type:', style={'marginTop':10})],
                                               style={'width':'49%', 'marginLeft':'0%', 'verticalAlign':'top', 'fontSize':'18px'}),
                                html.Div(children=[dcc.RadioItems(id='upload-data-type-picker', options=[{'label':i, 'value':i} for i in DataTypes], value=None, labelStyle={'display': 'inline-block', 'margin-right': 20},
                                                              inputStyle={"margin-right": "5px"}, style={'display':'block', 'fontSize':'16px'})]),
                                html.Div(children=[html.H5('Proteomics tool:'), dcc.RadioItems(id='prot-tool', options=[{'label':i, 'value':i} for i in ['MaxQuant', 'Spectronaut']], value='', labelStyle={'display': 'inline-block', 'margin-right': 20},
                                                              inputStyle={"margin-right": "5px"}, style={'display':'block', 'fontSize':'16px'})], id='proteomics-tool', style={'padding-top':20}),
                                html.Div([html.H4('Upload Experiment file', style={'marginTop':30, 'marginBottom':20}),
                                      dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                                                 style={'width': '100%',
                                                        'height': '60px',
                                                        'lineHeight': '60px',
                                                        'borderWidth': '1px',
                                                        'borderStyle': 'dashed',
                                                        'borderRadius': '5px',
                                                        'textAlign': 'center',
                                                        'margin': '0px'},
                                                 multiple=True)]),
                                html.Br(),
                                html.Div(children=[dcc.Markdown('**Uploaded Files:**', id='markdown-title'), dcc.Markdown(id='uploaded-files')]),
                                html.Div([html.A("Upload Data to CKG",
                                             id='submit_button',
                                             title="Upload Data to CKG",
                                             href='',
                                             target='',
                                             n_clicks=0,
                                             className="button_link")],
                                      style={'width':'100%', 'padding-left':'87%', 'padding-right':'0%'})]),
                                
                                html.Div(children=html.A('Download Uploaded Files(.zip)',
                                            id='data_download_link',
                                            href='',
                                            n_clicks=0, 
                                            style={'display':'none'},
                                            className="button_link"),
                                         style={'width':'100%', 'padding-left':'87%', 'padding-right':'0%'}),
                                html.Div(id='data-upload-result', children=[dcc.Markdown(id='upload-result')], style={'fontSize':'20px', 'marginLeft':'70%'}),
                                html.Hr()])]

        self.extend_layout(layout)
