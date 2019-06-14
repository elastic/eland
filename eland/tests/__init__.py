import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define test files and indices
ELASTICSEARCH_HOST = 'localhost' # TODO externalise this

FLIGHTS_INDEX_NAME = 'flights'
FLIGHTS_FILE_NAME = ROOT_DIR + '/flights.json.gz'

ECOMMERCE_INDEX_NAME = 'ecommerce'
ECOMMERCE_FILE_NAME = ROOT_DIR + '/ecommerce.json.gz'
