import os
import pandas as pd
import datetime
from celery import Celery
from report_manager.apps import projectCreation, dataUpload
from graphdb_connector import connector


celery_app = Celery('create_new_project')

celery_app.conf.update(broker_url = 'redis://localhost:6379',
					   result_backend = 'redis://localhost:6379/0')


@celery_app.task
def create_new_project(identifier, data, separator='|'):
    driver = connector.getGraphDatabaseConnectionConfiguration()
    project_result, projectId = projectCreation.create_new_project(driver, identifier, pd.read_json(data), separator=separator)
    return {str(projectId): str(project_result)}

@celery_app.task
def create_new_identifiers(project_id, data, directory, filename):
	driver = connector.getGraphDatabaseConnectionConfiguration()
	upload_result = dataUpload.create_experiment_internal_identifiers(driver, project_id, pd.read_json(data, dtype = {'subject external_id': object, 'biological_sample external_id': object, 'analytical_sample external_id': object}), directory, filename)
	res_n = dataUpload.check_samples_in_project(driver, project_id)
	
	return {str(project_id):str(upload_result), 'res_n':res_n.to_dict()}
