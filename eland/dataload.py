#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

from typing import TYPE_CHECKING, Any, Dict

from elasticsearch import helpers

if TYPE_CHECKING:
    import pandas as pd  # type: ignore
    from elasticsearch import Elasticsearch

FLIGHTS_INDEX_NAME = "flights"
FLIGHTS_MAPPING = {
    "mappings": {
        "properties": {
            "AvgTicketPrice": {"type": "float"},
            "Cancelled": {"type": "boolean"},
            "Carrier": {"type": "keyword"},
            "Dest": {"type": "keyword"},
            "DestAirportID": {"type": "keyword"},
            "DestCityName": {"type": "keyword"},
            "DestCountry": {"type": "keyword"},
            "DestLocation": {"type": "geo_point"},
            "DestRegion": {"type": "keyword"},
            "DestWeather": {"type": "keyword"},
            "DistanceKilometers": {"type": "float"},
            "DistanceMiles": {"type": "float"},
            "FlightDelay": {"type": "boolean"},
            "FlightDelayMin": {"type": "integer"},
            "FlightDelayType": {"type": "keyword"},
            "FlightNum": {"type": "keyword"},
            "FlightTimeHour": {"type": "float"},
            "FlightTimeMin": {"type": "float"},
            "Origin": {"type": "keyword"},
            "OriginAirportID": {"type": "keyword"},
            "OriginCityName": {"type": "keyword"},
            "OriginCountry": {"type": "keyword"},
            "OriginLocation": {"type": "geo_point"},
            "OriginRegion": {"type": "keyword"},
            "OriginWeather": {"type": "keyword"},
            "dayOfWeek": {"type": "byte"},
            "timestamp": {"type": "date", "format": "strict_date_hour_minute_second"},
        }
    }
}


ECOMMERCE_INDEX_NAME = "ecommerce"
ECOMMERCE_MAPPING = {
    "mappings": {
        "properties": {
            "category": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "currency": {"type": "keyword"},
            "customer_birth_date": {"type": "date"},
            "customer_first_name": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "customer_full_name": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "customer_gender": {"type": "text"},
            "customer_id": {"type": "keyword"},
            "customer_last_name": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "customer_phone": {"type": "keyword"},
            "day_of_week": {"type": "keyword"},
            "day_of_week_i": {"type": "integer"},
            "email": {"type": "keyword"},
            "geoip": {
                "properties": {
                    "city_name": {"type": "keyword"},
                    "continent_name": {"type": "keyword"},
                    "country_iso_code": {"type": "keyword"},
                    "location": {"type": "geo_point"},
                    "region_name": {"type": "keyword"},
                }
            },
            "manufacturer": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "order_date": {"type": "date"},
            "order_id": {"type": "keyword"},
            "products": {
                "properties": {
                    "_id": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                    "base_price": {"type": "half_float"},
                    "base_unit_price": {"type": "half_float"},
                    "category": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "created_on": {"type": "date"},
                    "discount_amount": {"type": "half_float"},
                    "discount_percentage": {"type": "half_float"},
                    "manufacturer": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "min_price": {"type": "half_float"},
                    "price": {"type": "half_float"},
                    "product_id": {"type": "long"},
                    "product_name": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                        "analyzer": "english",
                    },
                    "quantity": {"type": "integer"},
                    "sku": {"type": "keyword"},
                    "tax_amount": {"type": "half_float"},
                    "taxful_price": {"type": "half_float"},
                    "taxless_price": {"type": "half_float"},
                    "unit_discount_amount": {"type": "half_float"},
                }
            },
            "sku": {"type": "keyword"},
            "taxful_total_price": {"type": "float"},
            "taxless_total_price": {"type": "float"},
            "total_quantity": {"type": "integer"},
            "total_unique_products": {"type": "integer"},
            "type": {"type": "keyword"},
            "user": {"type": "keyword"},
        }
    }
}


def pandas_to_es(
    pd_df: "pd.DataFrame",
    es_client: "Elasticsearch",
    es_index_name: str,
    es_mapping: Dict[str, Any],
) -> None:
    """Read df and index records into Elasticsearch

    Parameters
    ----------
    pd_f: Pandas DataFrame
        Data to index into Elasticsearch
    es_client: Elasticsearch
        elasticsearch-py Elasticsearch client
    es_index_name: str
        Name of Elasticsearch index to be appended to
    es_mapping: Dict
        Elasticsearch mapping definition
    """
    print("Deleting index:", es_index_name)
    es_client.options(ignore_status=[400, 404]).indices.delete(index=es_index_name)
    print("Creating index:", es_index_name)
    es_client.indices.create(index=es_index_name, **es_mapping)

    actions = []
    n = 0

    print("Adding", pd_df.shape[0], "items to index:", es_index_name)
    for index, row in pd_df.iterrows():
        values = row.to_dict()

        # Use integer as id field for repeatable results
        action = {"_index": es_index_name, "_source": values, "_id": str(n)}
        actions.append(action)

        n = n + 1
        if n % 10000 == 0:
            helpers.bulk(es_client, actions)
            actions = []

    helpers.bulk(es_client, actions)
    del actions

    print("Done", es_index_name)
