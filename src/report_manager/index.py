import io
import os
import shutil
import re
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime
from uuid import uuid4
import base64
import json
from natsort import natsorted
import flask
import urllib.parse
import user
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app import app, server as application
from apps import initialApp, projectCreationApp, dataUploadApp, dataUpload, projectApp, importsApp, homepageApp, loginApp, projectCreation
from graphdb_builder import builder_utils
from graphdb_builder.builder import loader, importer
from graphdb_builder.experiments import experiments_controller as eh
from report_manager import utils
import ckg_utils
import config.ckg_config as ckg_config

from worker import create_new_project
from graphdb_connector import connector

cwd = os.path.abspath(os.path.dirname(__file__))
importDir = os.path.join(cwd, '../../data/imports/experiments')
experimentDir = os.path.join(cwd, '../../data/experiments')
driver = connector.getGraphDatabaseConnectionConfiguration()
separator = '|'

app.layout = dcc.Loading(
    children=[html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', style={'padding-top':10}, className='container-fluid'),
])], style={'text-align':'center',
            'margin-top':'70px',
            'margin-bottom':'-60px','position':'absolute',
            'top':'50%','left':'50%', 'height':'200px'},
    type='circle', 
    color='#2b8cbe')

@app.callback([Output('page-content', 'children'), 
               Output('logout_form', 'style')],
              [Input('url','href')])
def display_page(pathname):
    session_cookie = flask.request.cookies.get('custom-auth-session')
    logged_in = session_cookie is not None
    if not logged_in:
        login_form = loginApp.LoginApp("Login", "", "", layout = [], logo = None, footer = None)
        return (login_form.layout, {'display': 'none'})
    elif pathname is not None:
        if '/apps/initial' in pathname:
            return (initialApp.layout, {'display': 'block',
                                        'position': 'absolute',
                                        'right': '50px'})
        elif '/apps/login' in pathname:
            if logged_in:
                stats_db = homepageApp.HomePageApp("CKG homepage", "Database Stats", "", layout = [], logo = None, footer = None)
                return (stats_db.layout, {'display': 'block',
                                          'position': 'absolute',
                                          'right': '50px'})
            else:
                login_form = loginApp.LoginApp("Login", "", "", layout = [], logo = None, footer = None)
                return (login_form.layout, {'display': 'none'})
        elif '/apps/projectCreationApp' in pathname:
            projectCreation_form = projectCreationApp.ProjectCreationApp("Project Creation", "", "", layout = [], logo = None, footer = None)
            return (projectCreation_form.layout, {'display': 'block',
                                             'position': 'absolute',
                                             'right': '50px'})
        elif '/apps/dataUploadApp' in pathname:
            projectId = pathname.split('/')[-1]
            session_id = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
            dataUpload_form = dataUploadApp.DataUploadApp(session_id, projectId, "Data Upload", "", "", layout = [], logo = None, footer = None)
            return (dataUpload_form.layout, {'display': 'block',
                                        'position': 'absolute',
                                        'right': '50px'})
        elif '/apps/project?' in pathname:
            project_id, force, session_id = get_project_params_from_url(pathname)
            if session_id is None:
                session_id = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
            if project_id is None:
                return (initialApp.layout, {'display': 'block',
                                            'position': 'absolute',
                                            'right': '50px'})
            else:
                project = projectApp.ProjectApp(session_id, project_id, project_id, "", "", layout = [], logo = None, footer = None, force=force)
                return (project.layout, {'display': 'block',
                                         'position': 'absolute',
                                         'right': '50px'})
        elif '/apps/imports' in pathname:
            imports = importsApp.ImportsApp("CKG imports monitoring", "Statistics", "", layout = [], logo = None, footer = None)
            return (imports.layout, {'display': 'block',
                                     'position': 'absolute',
                                     'right': '50px'})
        elif '/apps/homepage' in pathname or pathname.count('/') <= 3:
            stats_db = homepageApp.HomePageApp("CKG homepage", "Database Stats", "", layout = [], logo = None, footer = None)
            return (stats_db.layout, {'display': 'block',
                                      'position': 'absolute',
                                      'right': '50px'})
        else:
            return ('404',{'display': 'block',
                           'position': 'absolute',
                           'right': '50px'})
    return (None, None)



def get_project_params_from_url(pathname):
    force = False
    project_id = None
    session_id = None
    regex_id = r"project_id=(\w+)"
    regex_force = r"force=(\d)"
    regex_session = r"session=(.+)"
    match_id = re.search(regex_id, pathname)
    if match_id:
        project_id = match_id.group(1)
    match_force = re.search(regex_force,pathname)
    if match_force:
        force = bool(int(match_force.group(1)))
    match_session = re.search(regex_session,pathname)
    if match_session:
        session_id = match_session.group(1)
    
    return project_id, force, session_id


# Documentation files
@app.server.route("/docs/<value>")
def return_docs(value):
    docs_url = ckg_config.docs_url
    return flask.render_template(docs_url+"{}".format(value))
 
# Callback upload configuration files
@app.callback([Output('upload-config', 'style'), 
               Output('output-data-upload','children'),
               Output('upload-config', 'filename')],
              [Input('upload-config', 'contents'),
               Input('my-dropdown','value')],
              [State('upload-config', 'filename')])
def update_output(contents, value, fname):
    display = {'display': 'none'}
    uploaded = None
    if value is not None:
        page_id, dataset = value.split('/')      
        directory = os.path.join('../../data/tmp', page_id)
        if dataset != "defaults":  
            display = {'width': '50%',
                       'height': '60px',
                       'lineHeight': '60px',
                       'borderWidth': '2px',
                       'borderStyle': 'dashed',
                       'borderRadius': '15px',
                       'textAlign': 'center',
                       'margin-bottom': '20px',
                       'display': 'block'}
            if not os.path.exists('../../data/tmp'):
                os.makedirs('../../data/tmp')
            elif not os.path.exists(directory):
                os.makedirs(directory)
            if  fname is None:
                contents = None
            if contents is not None:
                with open(os.path.join(directory, dataset+'.yml'), 'wb') as out:
                    content_type, content_string = contents.split(',')
                    decoded = base64.b64decode(content_string)
                    out.write(decoded)
                uploaded = dcc.Markdown("**{} configuration uploaded: {}** &#x2705;".format(dataset.title(),fname))
                fname = None
                contents = None
            else:
                uploaded = None
        elif dataset == 'reset':
            display = {'display': 'none'}
            if os.path.exists(directory):
                shutil.rmtree(directory)                
    return display, uploaded, fname
                

##Callbacks for CKG homepage
@app.callback(Output('db-creation-date', 'children'),
             [Input('db_stats_df', 'data')])
def update_db_date(df):
    kernel = pd.read_json(df['kernel_monitor'], orient='records')
    db_date = kernel['storeCreationDate'][0]
    return html.H3('Store Creation date: {}'.format(db_date))

@app.callback([Output("db_indicator_1", "children"),
               Output("db_indicator_2", "children"),
               Output("db_indicator_3", "children"),
               Output("db_indicator_4", "children"),
               Output("db_indicator_5", "children"),
               Output("db_indicator_6", "children"),
               Output("db_indicator_7", "children"),
               Output("db_indicator_8", "children"),
               Output("db_indicator_9", "children"),
               Output("db_indicator_10", "children"),
               Output("db_indicator_11", "children"),
               Output("db_indicator_12", "children"),
               Output("db_indicator_13", "children"),
               Output("db_indicator_14", "children"),],
              [Input("db_stats_df", "data")])
def number_panel_update(df):
    projects = pd.read_json(df['projects'], orient='records')
    if not projects.empty and 'Projects' in projects:
        projects = projects['Projects'][0]

    meta_stats = pd.read_json(df['meta_stats'], orient='records')
    if not meta_stats.empty:
      if 'nodeCount' in meta_stats:
          ent = meta_stats['nodeCount'][0]
      else:
          ent = '0'
      if 'relCount' in meta_stats:
          rel = meta_stats['relCount'][0]
      else:
          rel = '0'
      if 'labelCount' in meta_stats:
          labels = meta_stats['labelCount'][0]
      else:
          labels = '0'
      if 'relTypeCount' in  meta_stats:
          types = meta_stats['relTypeCount'][0]
      else:
          types = '0'
      if 'propertyKeyCount' in meta_stats:
          prop = meta_stats['propertyKeyCount'][0]
      else:
          prop = '0'

    store_size = pd.read_json(df['store_size'], orient='records')
    if not store_size.empty and 'size' in store_size:
        ent_store = store_size['size'][2]
        rel_store = store_size['size'][4]
        prop_store = store_size['size'][3]
        string_store = store_size['size'][5]
        array_store = store_size['size'][0]
        log_store = store_size['size'][1]
    else:
        ent_store = '0 MB'
        rel_store = '0 MB'
        prop_store = '0 MB'
        string_store = '0 MB'
        array_store = '0 MB'
        log_store = '0 MB'

    transactions = pd.read_json(df['transactions'], orient='records')
    if not transactions.empty and 'name' in transactions:
        t_open = transactions.loc[transactions['name'] == 'NumberOfOpenedTransactions', 'value'].iloc[0]
        t_comm = transactions.loc[transactions['name'] == 'NumberOfCommittedTransactions', 'value'].iloc[0]
    else:
        t_open = '0'
        t_comm = '0'
    
    
    return [dcc.Markdown("**{}**".format(i)) for i in [ent,labels,rel,types,prop,ent_store,rel_store,prop_store,string_store,array_store,log_store,t_open,t_comm,projects]]

@app.callback(Output("project_url", "children"),
             [Input("project_option", "value")])
def update_project_url(value):
    if value is not None and len(value) > 1:
        return html.A(value[0].title(),
                        href='/apps/project?project_id={}&force=0'.format(value[1]),
                        target='', 
                        n_clicks=0,
                        className="button_link")
    else:
      return ''
  
# Create a login route
@app.server.route('/apps/login', methods=['POST'])
def route_login():
    data = flask.request.form
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        flask.abort(401)
    elif not user.User(username).verify_password(password):
        return dcc.Markdown('**Invalid login.** &#x274C;')
    else:
        rep = flask.redirect('/')
        rep.set_cookie('custom-auth-session', username)
        return rep

@app.server.route('/apps/logout', methods=['POST'])
def route_logout():
    # Redirect back to the index and remove the session cookie.
    rep = flask.redirect('/')
    rep.set_cookie('custom-auth-session', '', expires=0)
    
    return rep




###Callbacks for download project
@app.callback(Output('download-zip', 'href'),
             [Input('download-zip', 'n_clicks')],
             [State('url', 'href')])
def generate_report_url(n_clicks, pathname):
    project_id, force, session_id = get_project_params_from_url(pathname)
    return '/downloads/{}'.format(project_id)
    
@application.route('/downloads/<value>')
def route_report_url(value):
    uri = os.path.join(os.getcwd(),"../../data/downloads/"+value+'.zip')
    return flask.send_file(uri, attachment_filename = value+'.zip', as_attachment = True)

###Callback regenerate project
@app.callback(Output('regenerate', 'href'),
             [Input('regenerate', 'n_clicks'),
              Input('regenerate', 'title')],
             [State('url', 'href')])
def regenerate_report(n_clicks, title, pathname):
    basic_path = '/'.join(pathname.split('/')[0:3]) 
    project_id, force, session_id = get_project_params_from_url(pathname)
    return basic_path+'/apps/project?project_id={}&force=1&session={}'.format(project_id, title)






###Callbacks for project creation app
def image_formatter(im):
    data_im = base64.b64encode(im).decode('ascii')
    return f'<img src="data:image/jpeg;base64,{data_im}">'

def add_internal_identifiers_to_excel(driver, external_id, data):
    subject_ids = projectCreation.get_subjects_in_project(driver, external_id)
    subject_ids = natsorted([item for sublist in subject_ids for item in sublist], reverse=False)
    data.insert(loc=0, column='subject id', value=subject_ids)
    return data

@app.callback([Output('project-creation', 'children'),
               Output('update_project_id','children'),
               Output('update_project_id','style'),
               Output('download_button', 'style')],
              [Input('project_button', 'n_clicks')],
              [State('project name', 'value'),
               State('project acronym', 'value'),
               State('responsible-picker', 'value'),
               State('participant-picker', 'value'),
               State('data-types-picker', 'value'),
               State('number_timepoints', 'value'),
               State('disease-picker', 'value'),
               State('tissue-picker', 'value'),
               State('intervention-picker', 'value'),
               State('number_subjects', 'value'),
               State('project description', 'value'),
               State('date-picker-start', 'date'),
               State('date-picker-end', 'date')])
def create_project(n_clicks, name, acronym, responsible, participant, datatype, timepoints, disease, tissue, intervention, number_subjects, description, start_date, end_date):
    if n_clicks > 0:
        responsible = separator.join(responsible)
        participant = separator.join(participant)
        datatype = separator.join(datatype)
        disease = separator.join(disease)
        tissue = separator.join(tissue)
        intervention = separator.join(intervention)
        arguments = [name, number_subjects, datatype, disease, tissue, responsible]

        if any(not arguments[n] for n, i in enumerate(arguments)) == True:
            response = "Insufficient information to create project. Fill in all fields with '*'."
            return response, None, {'display': 'none'}, {'display': 'none'}
        
        if any(not arguments[n] for n, i in enumerate(arguments)) == False:
        # Get project data from filled-in fields
            projectData = pd.DataFrame([name, acronym, description, number_subjects, datatype, timepoints, disease, tissue, intervention, responsible, participant, start_date, end_date]).T
            projectData.columns = ['name', 'acronym', 'description', 'subjects', 'datatypes', 'timepoints', 'disease', 'tissue', 'intervention', 'responsible', 'participant', 'start_date', 'end_date']
            projectData['status'] = ''

            projectData.fillna(value=pd.np.nan, inplace=True)
            projectData.replace('', np.nan, inplace=True)

            # Generate project internal identifier bsed on timestamp
            # Excel file is saved in folder with internal id name
            epoch = time.time()
            internal_id = "%s%d" % ("CP", epoch)
            projectData.insert(loc=0, column='internal_id', value=internal_id)
           
            result = create_new_project.apply_async(args=[internal_id, projectData.to_json(), separator], task_id='project_creation_'+internal_id)
            result_output = result.get()
            external_id = list(result_output.keys())[0]

            if result is not None:
                if external_id != '':
                    response = "Project successfully submitted. Download Clinical Data template."
                else:
                    response = 'A project with the same name already exists in the database.'
            else:
                response = "There was a problem when creating the project."

            return response, '- '+external_id, {'display': 'inline-block'}, {'display': 'block'}
    else:
        return None, None, {'display': 'none'}, {'display': 'none'}


@app.callback(Output('project-creation', 'style'),
              [Input('project-creation', 'children')])
def change_style(style):
    if style is not None and 'successfully' in style:
        return {'fontSize':'20px', 'marginLeft':'70%', 'color': 'black'}
    else:
        return {'fontSize':'20px', 'marginLeft':'70%', 'color': 'red'}


@app.callback(Output('download_link', 'href'),
              [Input('update_project_id', 'children')])
def update_download_link(project):
    if project is not None and project != '':
        project_id = project.split()[-1]
        return '/apps/templates?value=ClinicalData_template_{}.xlsx'.format(project_id)
    else:
        return ''

@application.route('/apps/templates')
def serve_static():
    file = flask.request.args.get('value')
    filename = '_'.join(file.split('_')[:-1])+'.xlsx'
    project_id = file.split('_')[-1].split('.')[0]
    df = pd.read_excel('apps/templates/{}'.format(filename))
    df = add_internal_identifiers_to_excel(driver, project_id, df)
    str_io = io.StringIO()
    df.to_csv(str_io, sep='\t', index=False)
    mem = io.BytesIO()
    mem.write(str_io.getvalue().encode('utf-8'))
    mem.seek(0)
    str_io.close()
    return flask.send_file(mem,
                          mimetype='text/csv',
                          attachment_filename='ClinicalData_{}.tsv'.format(project_id),
                          as_attachment=True,
                          cache_timeout=0)


###Callbacks for data upload app
def parse_contents(file):
    ftype = file.split('.')[-1]
    if ftype == 'txt' or ftype == 'tsv':
        df = pd.read_csv(file, sep='\t', low_memory=False)
    elif ftype == 'csv':
        df = pd.read_csv(file, low_memory=False)
    elif ftype == 'xlsx' or ftype == 'xls':
        df = pd.read_excel(file)        
    return df

def export_contents(data, dataDir, filename):
    file = filename.split('.')[-1]
    
    if file == 'txt' or file == 'tsv':
        csv_string = data.to_csv(os.path.join(dataDir, filename), sep='\t', index=False, encoding='utf-8')
    elif file == 'csv':
        csv_string = data.to_csv(os.path.join(dataDir, filename), sep=',', index=False, encoding='utf-8')
    elif file == 'xlsx' or file == 'xls':
        csv_string = data.to_excel(os.path.join(dataDir, filename), index=False, encoding='utf-8')   
    return csv_string


@app.callback(Output('proteomics-tool', 'style'),
              [Input('upload-data-type-picker', 'value')])
def show_proteomics_options(datatype):
    if 'proteomics' in datatype or 'longitudinal_proteomics' in datatype:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(Output('uploaded-files', 'children'),
             [Input('upload-data-type-picker', 'value'),
              Input('prot-tool', 'value'),
              Input('upload-data', 'filename'),
              Input('upload-data', 'contents')])
def save_files_in_tmp(datatype, prot_tool, filenames, contents):
    if datatype != '':
        page_id, dataset = datatype.split('/')
        tmpDirectory = os.path.join('../../data/tmp', page_id)
        if not os.path.exists('../../data/tmp'):
            os.makedirs('../../data/tmp')
        elif not os.path.exists(tmpDirectory):
            os.makedirs(tmpDirectory)
        
        if 'proteomics' in datatype or 'longitudinal_proteomics' in datatype:
            prot_dir = os.path.join(tmpDirectory, dataset)
            directory = os.path.join(prot_dir, prot_tool.lower())
            if not os.path.exists(os.path.join(tmpDirectory, dataset)):
                os.makedirs(os.path.join(tmpDirectory, dataset))
            else:
                for f in os.listdir(prot_dir):
                    if os.path.isfile(os.path.join(prot_dir, f)):
                        os.remove(os.path.join(prot_dir, f))
        else:
            directory = os.path.join(tmpDirectory, dataset)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            for f in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, f)):
                    os.remove(os.path.join(directory, f))

        if contents is not None:
            for file in zip(filenames, contents):
                with open(os.path.join(directory, file[0]), 'wb') as out:
                    content_type, content_string = file[1].split(',')
                    decoded = base64.b64decode(content_string)
                    out.write(decoded)
            directory = None
            prot_tool = ''
            #Two or more spaces before '\n' will create a new line in Markdown
            return '   \n'.join(filenames)
        else:
            raise PreventUpdate
    else:
        return None


@app.callback([Output('data-upload', 'children'),
               Output('data_download_link', 'style')],
              [Input('submit_button', 'n_clicks')],
              [State('project_id', 'value'),
               State('upload-data-type-picker', 'value')])
def run_processing(n_clicks, project_id, dtype):
    if n_clicks > 0:
        page_id, dataset = dtype.split('/')
        destDir = os.path.join(experimentDir, project_id)
        tmpDirectory = os.path.join('../../data/tmp', page_id)
        datasets = builder_utils.listDirectoryFolders(tmpDirectory)
        
        for dataset in datasets:
            directory = os.path.join(tmpDirectory, dataset)
            dir_tree = os.listdir(directory)
            if 'clinical' in dataset:
                data = parse_contents(os.path.join(directory, dir_tree[0]))
                data.fillna(value=pd.np.nan, inplace=True)
                df = dataUpload.create_new_experiment_in_db(driver, project_id, data, separator=separator)
                export_contents(df, directory, dir_tree[0])
            #Copy files from 'tmp' to 'experiments'
            for branch in dir_tree:
                source = os.path.join(tmpDirectory, os.path.join(dataset, branch))
                destination = os.path.join(destDir, dataset)
                if os.path.isdir(source):
                    destination = os.path.join(destination, os.path.basename(source))
                    shutil.copytree(source, destination)
                else:
                    shutil.copy(source, destination)

            # # Uncomment when proteomics headers changing (to internal identifiers) has been implemented        
            # if 'proteomics' in dataset:
            #     datasetPath = os.path.join(os.path.join(importDir, project_id), 'proteomics')
            #     builder_utils.checkDirectory(datasetPath)
            #     eh.generate_dataset_imports(project_id, 'proteomics', datasetPath)
            #     print('FINISHED IMPORTER')

        loader.partialUpdate(imports=['project', 'experiment'])
        style = {'display':'block'}
        message = 'Files successfully uploaded.'
          
        return message, style
    else:
        return '', {'display':'none'}


@app.callback(Output('data_download_link', 'href'),
             [Input('data_download_link', 'n_clicks')],
             [State('project_id', 'value'),
              State('upload-data-type-picker', 'value')])
def generate_upload_zip(n_clicks, project_id, dtype):
    if dtype is not None and dtype != '':
        page_id, dataset = dtype.split('/')
        return '/tmp/{}_{}'.format(page_id, project_id)
    else:
        return None

@application.route('/tmp/<value>')
def route_upload_url(value):
    page_id, project_id = value.split('_')
    directory = os.path.join(cwd,'../../data/tmp/')
    filename = os.path.join(directory, 'Uploaded_files_'+project_id)
    utils.compress_directory(filename, os.path.join(directory, page_id), compression_format='zip')
    url = filename+'.zip'
    return flask.send_file(url, attachment_filename = filename.split('/')[-1]+'.zip', as_attachment = True)




# @app.callback(Output('data-upload', 'style'),
#               [Input('data-upload', 'children')])
# def change_style(message):
#     if message is None:
#         return {'fontSize':'20px', 'marginLeft':'70%', 'color': 'black'}
#     else:
#         if 'Error' in message:
#             return {'fontSize':'20px', 'marginLeft':'70%', 'color': 'red'}
#         else:
#             return {'fontSize':'20px', 'marginLeft':'70%', 'color': 'black'}




if __name__ == '__main__':
    print("IN MAIN")
    application.run(debug=True, host='0.0.0.0')