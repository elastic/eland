import os
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define test files and indices
ELASTICSEARCH_HOST = 'localhost' # TODO externalise this

FLIGHTS_INDEX_NAME = 'flights'
FLIGHTS_FILE_NAME = ROOT_DIR + '/flights.json.gz'

ECOMMERCE_INDEX_NAME = 'ecommerce'
ECOMMERCE_FILE_NAME = ROOT_DIR + '/ecommerce.json.gz'

TEST_MAPPING1 = {
      'mappings': {
        'properties': {
          'city': {
            'type': 'text',
            'fields': {
              'raw': {
                'type': 'keyword'
              }
            }
          },
          'text': {
            'type': 'text',
            'fields': {
              'english': {
                'type': 'text',
                'analyzer': 'english'
              }
            }
          },
          'origin_location': {
            'properties': {
              'lat': {
                'type': 'text',
                'index_prefixes': {},
                'fields': {
                  'keyword': {
                    'type': 'keyword',
                    'ignore_above': 256
                  }
                }
              },
              'lon': {
                'type': 'text',
                'fields': {
                  'keyword': {
                    'type': 'keyword',
                    'ignore_above': 256
                  }
                }
              }
            }
          },
          'maps-telemetry': {
            'properties': {
              'attributesPerMap': {
                'properties': {
                  'dataSourcesCount': {
                    'properties': {
                      'avg': {
                        'type': 'long'
                      },
                      'max': {
                        'type': 'long'
                      },
                      'min': {
                        'type': 'long'
                      }
                    }
                  },
                  'emsVectorLayersCount': {
                    'dynamic': 'true',
                    'properties': {
                      'france_departments': {
                        'properties': {
                          'avg': {
                            'type': 'float'
                          },
                          'max': {
                            'type': 'long'
                          },
                          'min': {
                            'type': 'long'
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          },
          'type': {
            'type': 'keyword'
          },
          'name': {
            'type': 'text'
          },
          'user_name': {
            'type': 'keyword'
          },
          'email': {
            'type': 'keyword'
          },
          'content': {
            'type': 'text'
          },
          'tweeted_at': {
            'type': 'date'
          },
          'dest_location': {
            'type': 'geo_point'
          },
          'user': {
            'type': 'nested'
          },
          'my_join_field': {
            'type': 'join',
            'relations': {
              'question': ['answer', 'comment'],
              'answer': 'vote'
            }
          }
        }
      }
    }

TEST_MAPPING1_INDEX_NAME = 'mapping1'

TEST_MAPPING1_EXPECTED = {
    'city': 'text',
    'city.raw': 'keyword',
    'content': 'text',
    'dest_location': 'geo_point',
    'email': 'keyword',
    'maps-telemetry.attributesPerMap.dataSourcesCount.avg': 'long',
    'maps-telemetry.attributesPerMap.dataSourcesCount.max': 'long',
    'maps-telemetry.attributesPerMap.dataSourcesCount.min': 'long',
    'maps-telemetry.attributesPerMap.emsVectorLayersCount.france_departments.avg': 'float',
    'maps-telemetry.attributesPerMap.emsVectorLayersCount.france_departments.max': 'long',
    'maps-telemetry.attributesPerMap.emsVectorLayersCount.france_departments.min': 'long',
    'my_join_field': 'join',
    'name': 'text',
    'origin_location.lat': 'text',
    'origin_location.lat.keyword': 'keyword',
    'origin_location.lon': 'text',
    'origin_location.lon.keyword': 'keyword',
    'text': 'text',
    'text.english': 'text',
    'tweeted_at': 'date',
    'type': 'keyword',
    'user': 'nested',
    'user_name': 'keyword'
}

TEST_MAPPING1_EXPECTED_DF = pd.DataFrame.from_dict(data=TEST_MAPPING1_EXPECTED, orient='index', columns=['datatype'])
