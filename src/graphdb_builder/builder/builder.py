"""
    **builder.py**
    Builds the database in two main steps:
    1) Imports all the data from ontologies, databases and experiments
    2) Loads these data into the database
    The module can perform full updates, executing both steps for all the ontologies,
    databases and experiments or a partial update. Partial updates can execute step 1 or
    step 2 for specific data.
"""
import os
import sys
import argparse
from graphdb_builder.builder import importer, loader
import config.ckg_config as ckg_config
import ckg_utils
from graphdb_builder import builder_utils
import logging
import logging.config

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="builder")

try:
    config = builder_utils.setup_config('builder')
    dbconfig = builder_utils.setup_config('databases')
    oconfig = builder_utils.setup_config('ontologies')
except Exception as err:
    logger.error("builder - Reading configuration > {}.".format(err))

def set_arguments():
    """
    This function sets the arguments to be used as input for **builder.py** in the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--build_type", help="define the type of build you want (import, load or both)", type=str, choices=['import', 'load', 'full'], default='full')
    parser.add_argument("-i", "--import_types", help="define which data types (ontologies, experiments, databases) you want to import (partial import)", nargs='+', default=None, choices=['experiments', 'databases', 'ontologies', 'users'])
    parser.add_argument("-l", "--load_entities",  help="define which entities you want to load into the database (partial load)",  nargs='+', default=config["graph"])
    parser.add_argument("-d", "--data", help="define which ontology/ies, experiment/s or database/s you want to import",  nargs='+', default=None)
    parser.add_argument("-n", "--n_jobs", help="define number of cores used when importing data", type=int, default=4)
    parser.add_argument("-w", "--download", help="define whether or not to download imported data", type=str, default='True')
    parser.add_argument("-u", "--user", help="define the user that will build the database", type=str, required=True)
    
    return parser

if __name__ == '__main__':
    parser =  set_arguments()
    args = parser.parse_args()
    
    if args.build_type == 'full':
        logger.info("The user {} chose to perform a full build".format(args.user))
        logger.info("Building database > step 1: Importing data from ontologies, databases and experiments")
        importer.fullImport()
        logger.info("Building database > step 2: Loading all data imported into the database")
        loader.fullUpdate()
    elif args.build_type == 'import':
        logger.info("The user chose to perform a partial build")
        if args.import_types is not None:
            if len(args.data) > 0:
                logger.info("The build will import data from {}".format("".join(args.import_types)))
                for import_type in args.import_types:
                    logger.info("Importing {}: {}".format(import_type, "".join(args.data)))
                    if import_type.lower() == 'experiments' or import_type.lower() == 'experiment':
                        importer.experimentsImport(projects=args.data, n_jobs=1)
                    elif import_type.lower() == 'users' or import_type.lower() == 'user':
                        importer.usersImport(importDirectory='../../../data/imports')
                    elif import_type.lower() == 'databases' or import_type.lower() == 'database':
                        databases = [d.lower() for d in dbconfig['databases']]
                        valid_entities = [x.lower() for x in args.data if x.lower() in databases]
                        if len(valid_entities) > 0:
                            logger.info("These entities will be imported: {}".format(", ".join(valid_entities)))
                            print("These entities will be imported: {}".format(", ".join(valid_entities)))
                            importer.databasesImport(importDirectory='../../../data/imports', databases=valid_entities, n_jobs=args.n_jobs, download=bool(args.download))
                        else:
                            logger.error("The indicated entities (--data) cannot be imported: {}".format(args.data))
                            print("The indicated entities (--data) cannot be imported: {}".format(args.data))
                    elif import_type.lower() == 'ontologies' or import_type.lower() == 'ontology':
                        ontologies = [d.lower   () for d in oconfig['ontologies']]
                        valid_entities = [x.capitalize() for x in args.data if x.lower() in ontologies]
                        if len(valid_entities) > 0:
                            logger.info("These entities will be imported: {}".format(", ".join(valid_entities)))
                            print("These entities will be loaded into the database: {}".format(", ".join(valid_entities)))
                            importer.ontologiesImport(importDirectory='../../../data/imports', ontologies=valid_entities)
                        else:
                            logger.error("The indicated entities (--data) cannot be imported: {}".format(args.data))
                            print("The indicated entities (--data) cannot be imported: {}".format(args.data))
            else:
                print("Indicate the data to be imported by passing the argument --data and the list to be imported. \
                                Example: python builder.py --build_type import --import_types databases --data UniProt")
    elif args.build_type == 'load':
        logger.info("The build will load data into the database: {}".format("".join(args.load_entities)))
        valid_entities = []
        if len(args.load_entities) > 0:
            valid_entities = [x.lower() for x in args.load_entities if x.lower() in config['graph']]
        else:
            valid_entities = config['graph']
            
        if len(valid_entities) > 0:
            logger.info("These entities will be loaded into the database: {}".format(", ".join(valid_entities)))
            print("These entities will be loaded into the database: {}".format(", ".join(valid_entities)))
            loader.partialUpdate(imports=valid_entities)
        else:
            logger.error("The indicated entities (--load_entities) cannot be loaded: {}".format(args.load_entities))
            print("The indicated entities (--load_entities) cannot be loaded into the database: {}".format(args.load_entities))
    else:
        print("Indicate the type of build you want to perform, either import (generate csv files to be loaded into the database), \
                                    load (load csv files into the database) or full (import and then load all the data into the database) \
                                    Example: Import > python builder.py --build_type import --import_types databases --data UniProt\n \
                                    Load > python builder.py --build_type load --load_types Mentions\n \
                                    Full > python builder.py --build_type full or simpy python builder.py")
