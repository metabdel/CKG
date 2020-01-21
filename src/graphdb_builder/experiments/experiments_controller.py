import sys
import re
import os.path
import pandas as pd
import numpy as np
from collections import defaultdict
from graphdb_connector import connector
from graphdb_builder import builder_utils
from graphdb_builder.experiments.parsers import clinicalParser, proteomicsParser, wesParser
import config.ckg_config as ckg_config
import ckg_utils
import logging
import logging.config

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="experiments_controller")

def generate_dataset_imports(projectId, dataType, dataset_import_dir):
    stats = set()
    builder_utils.checkDirectory(dataset_import_dir)
    try:
        if dataType == 'clinical':
            data = clinicalParser.parser(projectId)
            for dtype, ot in data:
                print(dtype, ot)
                generate_graph_files(data[(dtype, ot)],dtype, projectId, stats, ot, dataset_import_dir)
        elif dataType == "proteomics":
            data = proteomicsParser.parser(projectId)
            for dtype, ot in data:
                generate_graph_files(data[(dtype, ot)],dtype, projectId, stats, ot, dataset_import_dir)
        elif dataType == "wes":
            data = wesParser.parser(projectId)
            for dtype, ot in data:
                generate_graph_files(data[(dtype, ot)],dtype, projectId, stats, ot, dataset_import_dir)
        else:
            raise Exception("Error when importing experiment for project {}. Non-existing parser for data type {}".format(projectId, dataType))
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Experiment {}: {} file: {}, line: {}".format(projectId, sys.exc_info(), fname, exc_tb.tb_lineno))
        raise Exception("Error when importing experiment {}.\n {}".format(projectId, err))

def generate_graph_files(data, dataType, projectId, stats, ot = 'w', dataset_import_dir='experiments'):
    if dataType.lower() == '':
        outputfile = os.path.join(dataset_import_dir, projectId+".tsv")
    else:
        outputfile = os.path.join(dataset_import_dir, projectId+"_"+dataType.lower()+".tsv")
    print(data)
    print(outputfile)
    with open(outputfile, ot) as f:
        data.to_csv(path_or_buf = f, sep='\t',
            header=True, index=False, quotechar='"',
            line_terminator='\n', escapechar='\\')
    
    logger.info("Experiment {} - Number of {} relationships: {}".format(projectId, dataType, data.shape[0]))
    stats.add(builder_utils.buildStats(data.shape[0], "relationships", dataType, "Experiment", outputfile))
    
def map_experiment_files(project_id, datasetPath, mapping):
    files = builder_utils.listDirectoryFiles(datasetPath)
    
    for f in files:
        data = builder_utils.readDataset(f)
        data = map_experimental_data(data, mapping)
        data.to_csv(path_or_buf = f, sep='\t',
					header=True, index=False, quotechar='"',
					line_terminator='\n', escapechar='\\')

def map_experimental_data(data, mapping):
    mapping_cols = {}

    if not data.empty:
        for column in data.columns:
            for external_id in mapping:
                if external_id in column:
                    mapping_cols[column] = column.replace(external_id, mapping[external_id])
        data = data.rename(columns=mapping_cols)


    return data

def get_mapping_analytical_samples(project_id, driver):
    mapping = {}
    query = "MATCH (p:Project)-[:HAS_ENROLLED]-(:Subject)-[:BELONGS_TO_SUBJECT]-()-[:SPLITTED_INTO]-(a:Analytical_sample) WHERE p.id='{}' RETURN a.external_id, a.id".format(project_id)
    mapping = connector.getCursorData(driver,query)
    if not mapping.empty:
        mapping = mapping.set_index("a.external_id").to_dict(orient='dict')["a.id"]
    
    return mapping


if __name__ == "__main__":
    pass
