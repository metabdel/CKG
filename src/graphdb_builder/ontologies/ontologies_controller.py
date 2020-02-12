import urllib
from graphdb_builder import mapping as mp, builder_utils
import config.ckg_config as ckg_config
import ckg_utils
from graphdb_builder.ontologies.parsers import *
import os.path
from collections import defaultdict
import pandas as pd
import csv
import obonet
import re
from datetime import date
import logging
import logging.config
import sys

log_config = ckg_config.graphdb_builder_log
logger = builder_utils.setup_logging(log_config, key="ontologies_controller")

try:
    config = builder_utils.setup_config('ontologies')
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))

#########################
# General functionality #
#########################

def entries_to_remove(entries, the_dict):
    """
    This function removes pairs from a given dictionary, based on a list of provided keys.
    
    :param list entries: list of keys to be deleted from dictionary.
    :param dict the_dict: dictionary.
    :return: The original dictionary minus the key,value pairs from the provided entries list.
    """
    for key in entries:
        if key in the_dict:
            del the_dict[key]

############################
# Calling the right parser #
############################
def parse_ontology(ontology, download=True):
    """
    Parses and extracts data from a given ontology file(s), and returns a tuple with multiple dictionaries.

    :param str ontology: acronym of the ontology to be parsed (e.g. Disease Ontology:'DO').
    :param bool download: wether database is to be downloaded.
    :return: Tuple with three nested dictionaries: terms, relationships between terms, and definitions of the terms.\
            For more information on the returned dictionaries, see the documentation for any ontology parser.
    """
    directory = config["ontologies_directory"]
    ontology_directory = os.path.join(directory, ontology)
    builder_utils.checkDirectory(ontology_directory)
    ontology_files = []
    if ontology in config["ontology_types"]:
        otype = config["ontology_types"][ontology]
        if 'urls' in config:
            if otype in config['urls']:
                urls = config['urls'][otype]
                for url in urls:
                    f = url.split('/')[-1]
                    ontology_files.append(os.path.join(ontology_directory, f))
                    if download:
                        builder_utils.downloadDB(url, directory=ontology_directory)
            elif otype in config["files"]:
                ofiles = config["files"][otype]
                ###Check SNOMED-CT files exist
                for f in ofiles:
                    if os.path.isfile(os.path.join(directory, f)):
                        ontology_files.append(os.path.join(directory, f))
        filters = None
        if otype in config["parser_filters"]:
            filters = config["parser_filters"][otype]
    if len(ontology_files) > 0:
        if ontology == "SNOMED-CT":
            ontologyData = snomedParser.parser(ontology_files, filters)
        elif ontology == "ICD":
            ontologyData = icdParser.parser(ontology_files)
        else:
            ontologyData = oboParser.parser(ontology, ontology_files)
            mp.buildMappingFromOBO(ontology_files[0], ontology)
    else:
        if ontology == "SNOMED-CT":
            logger.info("WARNING: SNOMED-CT terminology needs to be downloaded manually since it requires UMLS License. More information available here: https://www.nlm.nih.gov/databases/umls.html")
        else:
            logger.info("WARNING: Ontology {} could not be downloaded. Check that the link in configuration works.".format(ontology))
    return ontologyData

#########################
#       Graph files     #
#########################
def generate_graphFiles(import_directory, ontologies=None, download=True):
    """
    This function parses and extracts data from a given list of ontologies. If no ontologies are provided, \
    all availables ontologies are used. Terms, relationships and definitions are saved as .tsv files to be loaded into \
    the graph database.

    :param str import_directory: relative path from current python module to 'imports' directory.
    :param ontologies: list of ontologies to be imported. If None, all available ontologies are imported.
    :type ontologies: list or None
    :param bool download: wether database is to be downloaded.
    :return: Dictionary of tuples. Each tuple corresponds to a unique label/relationship type, date, time, \
            database, and number of nodes and relationships.
    """
    entities = config["ontologies"]
    if ontologies is not None:
        entities = {}
        for ontology in ontologies:
            entities.update({ontology:config["ontologies"][ontology]})
    
    updated_on = None
    if download:
        updated_on = str(date.today())

    stats = set()
    for entity in entities:
        ontology = config["ontologies"][entity]
        if ontology in config["ontology_types"]:
            ontologyType = config["ontology_types"][ontology]
        try:
            terms, relationships, definitions = parse_ontology(ontology, download)
            for namespace in terms:
                if namespace in config["entities"]:
                    name = config["entities"][namespace]
                entity_outputfile = os.path.join(import_directory, name+".tsv")
                with open(entity_outputfile, 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t', escapechar='\\', quotechar='"', quoting=csv.QUOTE_ALL)
                    writer.writerow(['ID', ':LABEL', 'name', 'description', 'type', 'synonyms'])
                    for term in terms[namespace]:
                        writer.writerow([term, entity, list(terms[namespace][term])[0], definitions[term], ontologyType, ",".join(terms[namespace][term])])
                logger.info("Ontology {} - Number of {} entities: {}".format(ontology, name, len(terms[namespace])))
                stats.add(builder_utils.buildStats(len(terms[namespace]), "entity", name, ontology, entity_outputfile, updated_on))
                if namespace in relationships:
                    relationships_outputfile = os.path.join(import_directory, name+"_has_parent.tsv")
                    relationshipsDf = pd.DataFrame(list(relationships[namespace]))
                    relationshipsDf.columns = ['START_ID', 'END_ID', 'TYPE']
                    relationshipsDf.to_csv(path_or_buf=relationships_outputfile,
                                                sep='\t',
                                                header=True, index=False, quotechar='"',
                                                quoting=csv.QUOTE_ALL,
                                                line_terminator='\n', escapechar='\\')
                    logger.info("Ontology {} - Number of {} relationships: {}".format(ontology, name+"_has_parent", len(relationships[namespace])))
                    stats.add(builder_utils.buildStats(len(relationships[namespace]), "relationships", name+"_has_parent", ontology, relationships_outputfile, updated_on))
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("Ontology {}: {}, file: {},line: {}".format(ontology, sys.exc_info(), fname, exc_tb.tb_lineno))
            raise Exception("Error when importing ontology {}.\n {}".format(ontology, err))
    return stats

if __name__ == "__main__":
    generate_graphFiles(import_directory='../../../data/imports', download=True) 
