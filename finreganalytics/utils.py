import logging

import IPython

# from gql.transport.requests import log as requests_logger
# from gql.transport.websockets import log as websockets_logger

logger = logging.getLogger(__name__)


def get_current_cluster_id():
    import json
    return \
    json.loads(get_dbutils().notebook.entry_point.getDbutils().notebook().getContext().safeToJson())['attributes'][
        'clusterId']


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
