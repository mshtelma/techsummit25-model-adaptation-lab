import json
import logging

import IPython

# from gql.transport.requests import log as requests_logger
# from gql.transport.websockets import log as websockets_logger

logger = logging.getLogger(__name__)

def batchify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def get_current_cluster_id():
    return \
        json.loads(get_dbutils().notebook.entry_point.getDbutils().notebook().getContext().safeToJson())['attributes'][
            'clusterId']


def get_user_email():
    return get_spark().sql('select current_user() as user').collect()[0]['user']


def get_user_name():
    username = get_user_email().split('@')[0].replace('.', '_')
    return username


def set_or_create_catalog_and_database(catalog: str, db_name: str):
    get_spark().sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    get_spark().sql(f"ALTER CATALOG {catalog} OWNER TO `account users`")
    get_spark().sql(f"USE CATALOG {catalog} ")
    get_spark().sql(f"CREATE DATABASE IF NOT EXISTS  `{catalog}`.`{db_name}` ")
    get_spark().sql(f"GRANT CREATE, USAGE on DATABASE `{catalog}`.`{db_name}` TO `account users`")
    get_spark().sql(f"ALTER SCHEMA `{catalog}`.`{db_name}` OWNER TO `account users`")
    get_spark().sql(f"USE `{catalog}`.`{db_name}`")
    get_spark().sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{db_name}`.data")


def get_dbutils():
    return IPython.get_ipython().user_ns["dbutils"]


def get_spark():
    return IPython.get_ipython().user_ns["spark"]


def display(*args, **kwargs):
    return IPython.get_ipython().user_ns["display"](*args, **kwargs)


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("sh.command").setLevel(logging.ERROR)

    # requests_logger.setLevel(logging.WARNING)
    # websockets_logger.setLevel(logging.WARNING)
