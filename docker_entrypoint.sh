#!/bin/bash

echo "$USER"
service neo4j version
service neo4j start
sleep 60
cat /var/log/neo4j/neo4j.log

echo "Setting up the config files"
python3 /CKG/setup_config_files.py

echo "Running jupyterHub"
jupyterhub -f /etc/jupyterhub/jupyterhub.py --no-ssl &

echo "Running redis-server"
./etc/redis/bin/redis-server &

echo "Initiating queue celery"
cd /CKG/src/report_manager
celery -A worker worker --loglevel=DEBUG --concurrency=3 -E &

echo "Building CKG graph database"
cd /CKG/src/graphdb_builder/builder
python3 builder.py -b full -u ckg

if [ -e /debug1 ]; then
  echo "Running app in debug mode!"
  python3 /CKG/src/report_manager/index.py
else
  echo "Running app in production mode!"
  nginx && uwsgi --ini /etc/uwsgi/apps-enabled/uwsgi.ini
fi
