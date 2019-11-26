import os
import sys
import re
import pandas as pd
import numpy as np
import config.ckg_config as ckg_config
import ckg_utils
from graphdb_connector import connector
from graphdb_builder import builder_utils
from graphdb_builder.experiments.parsers import clinicalParser as cp
from report_manager.queries import query_utils
import logging
import logging.config
 
log_config = ckg_config.report_manager_log
logger = builder_utils.setup_logging(log_config, key="project_creation")
 
cwd = os.path.abspath(os.path.dirname(__file__))
experimentDir = os.path.join(cwd, '../../../data/experiments')
importDir = os.path.join(cwd, '../../../data/imports/experiments')
 
def get_project_creation_queries():
    """
    Reads the YAML file containing the queries relevant to user creation, parses the given stream and \
    returns a Python object (dict[dict]).
 
    :return: Nested dictionary.
    """
    try:
        cwd = os.path.abspath(os.path.dirname(__file__))
        queries_path = "../queries/project_creation_cypher.yml"
        project_creation_cypher = ckg_utils.get_queries(os.path.join(cwd, queries_path))
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading queries from file {}: {}, file: {},line: {}".format(queries_path, sys.exc_info(), fname, exc_tb.tb_lineno))
    return project_creation_cypher
 
def check_if_node_exists(driver, node_property, value):
    """
    Queries the graph database and checks if a node with a specific property and property value already exists.
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str node_property: property of the node.
    :param value: property value.
    :type value: str, int, float or bool
    :return: Pandas dataframe with user identifier if User with node_property and value already exists, \
            if User does not exist, returns and empty dataframe.
    """
    query_name = 'check_node'
    try:
        cypher = get_project_creation_queries()
        query = cypher[query_name]['query'].replace('PROPERTY', node_property)
        for q in query.split(';')[0:-1]:
            if '$' in q:
                result = connector.getCursorData(driver, q+';', parameters={'value':value})
            else:
                result = connector.getCursorData(driver, q+';')
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}, error: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno, err))
    return result
 
def get_new_project_identifier(driver, projectId):
    """
    Queries the database for the last project external identifier and returns a new sequential identifier.
 
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str projectId: internal project identifier (CPxxxxxxxxxxxx).
    :return: Project external identifier.
    :rtype: str
    """
    query_name = 'increment_project_id'
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        external_identifier = connector.getCursorData(driver, query).values[0][0]
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    return external_identifier
 
def get_new_subject_identifier(driver, projectId):
    """
    Queries the database for the last subject identifier and returns a new sequential identifier.
 
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str projectId: external project identifier (from the graph database).
    :return: Subject identifier.
    :rtype: str
    """
    query_name = 'increment_subject_id'
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        subject_identifier = connector.getCursorData(driver, query).values[0][0]
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    return subject_identifier
 
 
def get_subjects_in_project(driver, projectId):
    """
    Extracts the number of subjects included in a given project.
 
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str projectId: external project identifier (from the graph database).
    :return: Number of subjects.
    :rtype: Numpy ndarray
    """
    query_name = 'extract_project_subjects'
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        for q in query.split(';')[0:-1]:
            if '$' in q:
                result = connector.getCursorData(driver, q+';', parameters={'external_id': str(projectId)})
            else:
                result = connector.getCursorData(driver, q+';')
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    return result.values
 
 
def create_new_project(driver, projectId, data, separator='|'):
    """
    Creates a new project in the graph database, following the steps:
     
    1. Retrieves new project external identifier and creates project node and relationships in the graph database.
    2. Creates subjects, timepoints and intervention nodes.
    3. Saves all the entities and relationships to tab-delimited files.
    4. Returns the number of projects created and the project external identifier.
 
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str projectId: internal project identifier (CPxxxxxxxxxxxx).
    :param data: pandas Dataframe with project as row and other attributes as columns.
    :param str separator: character used to separate multiple entries in a project attribute.
    :return: Two strings: number of projects created and the project external identifier.
    """
    query_name = 'create_project'
    external_identifier='No Identifier Assigned'
    disease_ids = []
    tissue_ids = []
     
    try:
        db_project = check_if_node_exists(driver, 'name', data['name'][0])
        if db_project.empty:
            external_identifier = get_new_project_identifier(driver, projectId)        
            data['external_id'] = external_identifier
            project_creation_cypher = get_project_creation_queries()
            query = project_creation_cypher[query_name]['query']
            for q in query.split(';')[0:-1]:
                if '$' in q:
                    for parameters in data.to_dict(orient='records'):
                        result = connector.getCursorData(driver, q+';', parameters=parameters)
                else:
                    result = connector.getCursorData(driver, q+';')    

            subjects = create_new_subjects(driver, external_identifier, data['subjects'][0])

            if data['timepoints'][0] is None:
                pass
            else:
                timepoints = create_new_timepoint(driver, external_identifier, data, separator)
            if data['intervention'][0] == separator:
                pass
            else:
                interventions = create_intervention_relationship(driver, external_identifier, data, separator)
             
            for disease in data['disease'][0].split(separator):
                disease_ids.append(query_utils.map_node_name_to_id(driver, 'Disease', str(disease)))
            for tissue in data['tissue'][0].split(separator):
                tissue_ids.append(query_utils.map_node_name_to_id(driver, 'Tissue', str(tissue)))
 
            store_new_project(external_identifier, data, experimentDir, 'xlsx')
            store_as_file(external_identifier, data, external_identifier, importDir, 'tsv')
            store_new_relationships(external_identifier, data['responsible'][0].split(separator), [external_identifier], 'IS_RESPONSIBLE', 'responsibles', importDir, 'tsv')
            store_new_relationships(external_identifier, data['participant'][0].split(separator), [external_identifier], 'PARTICIPATES_IN', 'participants', importDir, 'tsv')
            store_new_relationships(external_identifier, [external_identifier], disease_ids, 'STUDIES_DISEASE', 'studies_disease', importDir, 'tsv')
            store_new_relationships(external_identifier, [external_identifier], tissue_ids, 'STUDIES_TISSUE', 'studies_tissue', importDir, 'tsv')
        else:
            result = pd.DataFrame([''])
            external_identifier = ''
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
    
    return result.values[0], external_identifier   
        
def create_new_subjects(driver, projectId, subjects):
    """
    Creates new graph database nodes for subjects participating in a project.
 
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str projectId: external project identifier (from the graph database).
    :param int subjects: number of subjects participating in the project.
    :return: Integer for the number of subjects created.
    """
    query_name = 'create_subjects'
    subject_identifier='No Identifier Assigned'
    subject_ids = []
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        for subject in list(np.arange(subjects)):
            done = 0
            subject_identifier = get_new_subject_identifier(driver, projectId)
            for q in query.split(';')[0:-1]:
                if '$' in q:
                    result = connector.getCursorData(driver, q+';', parameters={'subject_id': str(subject_identifier), 'external_id': str(projectId)})
                else:
                    result = connector.getCursorData(driver, q+';')
            subject_ids.append(subject_identifier)
            done += 1
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
 
    data = pd.DataFrame(subject_ids)
    data.insert(loc=0, column='', value=projectId)
    data.columns = ['START_ID', 'END_ID']
    data['TYPE'] = 'HAS_ENROLLED'
    store_as_file(projectId, data, projectId+'_project', importDir, 'tsv')
    return done
 
def create_new_timepoint(driver, projectId, data, separator='|'):
    """
    Creates new timepoints and relationships to project in the graph database, and saves the \
    data to a tab-delimited file in the project folder.
 
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str projectId: external project identifier (from the graph database).
    :param data: pandas Dataframe with project as row and other attributes as columns.
    :param str separator: character used to separate multiple entries in a project attribute.
    :return: Integer for the number of timepoints created.
    """
    query_name = 'create_timepoint'
    df = cp.extract_timepoints(data, separator=separator)
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        done = 0
        for index, row in df.iterrows():
            for q in query.split(';')[0:-1]:
                if '$' in q:
                    result = connector.getCursorData(driver, q+';', parameters={'timepoint': str(row['ID']), 'units': str(row['units'])})
                else:
                    result = connector.getCursorData(driver, q+';')
            done += 1
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
  
    store_as_file(projectId, df, projectId+'_timepoint', importDir, 'tsv')
    return done
 
def create_intervention_relationship(driver, projectId, data, separator='|'):
    """
    Creates new intervention relationships to project in the graph database, and saves the \
    data to a tab-delimited file in the project folder.
 
    :param driver: py2neo driver, which provides the connection to the neo4j graph database.
    :type driver: py2neo driver
    :param str projectId: external project identifier (from the graph database).
    :param data: pandas Dataframe with project as row and other attributes as columns.
    :param str separator: character used to separate multiple entries in a project attribute.
    :return: Integer for the number of interventions created.
    """
    query_name = 'create_intervention_relationship'
    data = cp.extract_project_intervention_rels(data, separator=separator)
    try:
        project_creation_cypher = get_project_creation_queries()
        query = project_creation_cypher[query_name]['query']
        done = 0
        for i in data['END_ID'].astype(str):
            for q in query.split(';')[0:-1]:
                if '$' in q:
                    result = connector.getCursorData(driver, q+';', parameters={'external_id': projectId, 'intervention_id': i})
                else:
                    result = connector.getCursorData(driver, q+';')
            done += 1
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Reading query {}: {}, file: {},line: {}".format(query_name, sys.exc_info(), fname, exc_tb.tb_lineno))
 
    store_as_file(projectId, data, projectId+'_studies_intervention', importDir, 'tsv')
    return done
 
def store_as_file(projectId, data, filename, folder, file_format):
    """
    Saves data provided as a Pandas DataFrame to an excel or tab-delimited file.
 
    :param str projectId: external project identifier (from the graph database).
    :param data: pandas Dataframe with nodes as rows and properties as columns.
    :param str filename: name of the file to be created.
    :param str folder: path to imports folder.
    :param str file_format: 'tsv' or 'xlsx'.
    """
    if data is not None:
        outputDir = os.path.join(folder, os.path.join(projectId,'clinical'))
        ckg_utils.checkDirectory(outputDir)
        outputfile = os.path.join(outputDir, filename+'.{}'.format(file_format))
        if file_format == 'tsv':
            with open(outputfile, 'w') as f:
                data.to_csv(path_or_buf = f, sep='\t',
                            header=True, index=False, quotechar='"',
                            line_terminator='\n', escapechar='\\')
        if file_format == 'xlsx':
            with pd.ExcelWriter(outputfile, mode='w') as e:
                data.to_excel(e, index=False)
 
def store_new_project(projectId, data, folder, file_format):
    """
    Saves new project data to an excel or tab-delimited file.
 
    :param str projectId: external project identifier (from the graph database).
    :param data: pandas Dataframe with nodes as rows and properties as columns.
    :param str folder: path to imports folder.
    :param str file_format: 'tsv' or 'xlsx'.
    """
    if data is not None:
        filename = 'ProjectData_{}'.format(projectId)
        store_as_file(projectId, data, filename, folder, file_format)
 
def store_new_relationships(projectId, start_node_list, end_node_list, relationship, filename, folder, file_format):
    """
    Creates a Pandas DataFrame of relationships between the provided lists of start and end nodes, and saves \
    it to an excel or tab-delimited file.
 
    :param str projectId: external project identifier (from the graph database).
    :param list start_node_list: list of source nodes.
    :param list end_node_list: list of target nodes.
    :param str relationship: name of the relationship between start node and end node.
    :param str filename: name of the file to be created.
    :param str folder: path to imports folder.
    :param str file_format: 'tsv' or 'xlsx'.
    """
    length = int(len(max([start_node_list, end_node_list], key=len)))
    data = pd.DataFrame(index=np.arange(length), columns=['START_ID', 'END_ID', 'TYPE'])
    if len(start_node_list) == len(data.index):
        data['START_ID'] = start_node_list
    else: data['START_ID'] = start_node_list[0]
    if len(end_node_list) == len(data.index):
        data['END_ID'] = end_node_list
    else: data['END_ID'] = end_node_list[0]
    data['TYPE'] = relationship
     
    filename = projectId+'_{}'.format(filename)
    store_as_file(projectId, data, filename, folder, file_format)