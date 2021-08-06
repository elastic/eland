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

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    TextIO,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd  # type: ignore
from pandas.core.dtypes.common import (  # type: ignore
    is_bool_dtype,
    is_datetime_or_timedelta_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.inference import is_list_like

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
    from numpy.typing import DTypeLike


ES_FLOAT_TYPES: Set[str] = {"double", "float", "half_float", "scaled_float"}
ES_INTEGER_TYPES: Set[str] = {"long", "integer", "short", "byte"}
ES_COMPATIBLE_TYPES: Dict[str, Set[str]] = {
    "double": ES_FLOAT_TYPES,
    "scaled_float": ES_FLOAT_TYPES,
    "float": ES_FLOAT_TYPES,
    "half_float": ES_FLOAT_TYPES,
    "long": ES_INTEGER_TYPES,
    "integer": ES_INTEGER_TYPES,
    "short": ES_INTEGER_TYPES,
    "byte": ES_INTEGER_TYPES,
    "date": {"date_nanos"},
    "date_nanos": {"date"},
    "keyword": {"text"},
}


class Field(NamedTuple):
    """Holds all information on a particular field in the mapping"""

    column: str
    es_field_name: str
    is_source: bool
    es_dtype: str
    es_date_format: Optional[str]
    pd_dtype: type
    is_searchable: bool
    is_aggregatable: bool
    is_scripted: bool
    aggregatable_es_field_name: str

    @property
    def is_numeric(self) -> bool:
        return is_integer_dtype(self.pd_dtype) or is_float_dtype(self.pd_dtype)

    @property
    def is_timestamp(self) -> bool:
        return is_datetime_or_timedelta_dtype(self.pd_dtype)

    @property
    def is_bool(self) -> bool:
        return is_bool_dtype(self.pd_dtype)

    @property
    def np_dtype(self):
        return np.dtype(self.pd_dtype)

    def is_es_agg_compatible(self, es_agg) -> bool:
        # Unpack the actual aggregation if this is 'extended_stats/percentiles'
        if isinstance(es_agg, tuple):
            if es_agg[0] == "extended_stats":
                es_agg = es_agg[1]
            elif es_agg[0] == "percentiles":
                es_agg = "percentiles"

        # Except "median_absolute_deviation" which doesn't support bool
        if es_agg == "median_absolute_deviation" and self.is_bool:
            return False
        # Cardinality, Count and mode work for all types
        # Numerics and bools work for all aggs
        if (
            es_agg in {"cardinality", "value_count", "mode"}
            or self.is_numeric
            or self.is_bool
        ):
            return True
        # Timestamps also work for 'min', 'max' and 'avg'
        if es_agg in {"min", "max", "avg", "percentiles"} and self.is_timestamp:
            return True
        return False

    @property
    def nan_value(self) -> Any:
        """Returns NaN for any field except datetimes which use NaT"""
        if self.is_timestamp:
            return pd.NaT
        return np.float64(np.NaN)


class FieldMappings:
    """
    General purpose to manage Elasticsearch to/from pandas mappings

    Attributes
    ----------

    _mappings_capabilities: pandas.DataFrame
        A data frame summarising the capabilities of the index mapping

        column (index)              - the eland display name

        es_field_name               - the Elasticsearch field name
        is_source                   - is top level field (i.e. not a multi-field sub-field)
        es_dtype                    - Elasticsearch field datatype
        es_date_format              - Elasticsearch date format (or None)
        pd_dtype                    - Pandas datatype
        is_searchable               - is the field searchable?
        is_aggregatable             - is the field aggregatable?
        is_scripted                 - is the field a scripted_field?
        aggregatable_es_field_name  - either es_field_name (if aggregatable),
                                      or es_field_name.keyword (if exists) or None
    """

    ES_DTYPE_TO_PD_DTYPE: Dict[str, str] = {
        "text": "object",
        "keyword": "object",
        "long": "int64",
        "integer": "int64",
        "short": "int64",
        "byte": "int64",
        "binary": "int64",
        "double": "float64",
        "float": "float64",
        "half_float": "float64",
        "scaled_float": "float64",
        "date": "datetime64[ns]",
        "date_nanos": "datetime64[ns]",
        "boolean": "bool",
    }

    # the labels for each column (display_name is index)
    column_labels: List[str] = [
        "es_field_name",
        "is_source",
        "es_dtype",
        "es_date_format",
        "pd_dtype",
        "is_searchable",
        "is_aggregatable",
        "is_scripted",
        "aggregatable_es_field_name",
    ]

    def __init__(
        self,
        client: "Elasticsearch",
        index_pattern: str,
        display_names: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        client: elasticsearch.Elasticsearch
            Elasticsearch client

        index_pattern: str
            Elasticsearch index pattern

        display_names: list of str
            Field names to display
        """
        if (client is None) or (index_pattern is None):
            raise ValueError(
                f"Can not initialise mapping without client "
                f"or index_pattern {client} {index_pattern}",
            )

        get_mapping = client.indices.get_mapping(index=index_pattern)
        if not get_mapping:  # dict is empty
            raise ValueError(
                f"Can not get mapping for {index_pattern} "
                f"check indexes exist and client has permission to get mapping."
            )

        # Get all fields (including all nested) and then all field_caps
        all_fields = FieldMappings._extract_fields_from_mapping(get_mapping)
        all_fields_caps = client.field_caps(index=index_pattern, fields="*")

        # Get top level (not sub-field multifield) mappings
        source_fields = FieldMappings._extract_fields_from_mapping(
            get_mapping, source_only=True
        )

        # Populate capability matrix of fields
        self._mappings_capabilities = FieldMappings._create_capability_matrix(
            all_fields, source_fields, all_fields_caps
        )

        if display_names is not None:
            self.display_names = display_names

    @staticmethod
    def _extract_fields_from_mapping(
        mappings: Dict[str, Any], source_only: bool = False
    ) -> Dict[str, str]:
        """
        Extract all field names and types from a mapping.
        ```
        {
          "my_index": {
            "mappings": {
              "properties": {
                "city": {
                  "type": "text",
                  "fields": {
                    "keyword": {
                      "type": "keyword"
                    }
                  }
                }
              }
            }
          }
        }

        or (6.x)

        {
          "my_index": {
            "mappings": {
              "doc": {
                "properties": {
                  "city": {
                    "type": "text",
                    "fields": {
                      "keyword": {
                        "type": "keyword"
                      }
                    }
                  }
                }
              }
            }
          }
        }
        ```
        if source_only == False:
            return {'city': ('text', None), 'city.keyword': ('keyword', None)}
        else:
            return {'city': ('text', None)}

        Note: first field name type wins. E.g.

        ```
        PUT my_index1 {"mappings":{"properties":{"city":{"type":"text"}}}}
        PUT my_index2 {"mappings":{"properties":{"city":{"type":"long"}}}}

        Returns {'city': 'text'}
        ```

        Parameters
        ----------
        mappings: dict
            Return from get_mapping

        Returns
        -------
        fields, dates_format: tuple(dict, dict)
            where:
                fields: dict of field names and types
                dates_format: Dict of date field names and format
        """
        fields = {}

        # Recurse until we get a 'type: xxx'
        def flatten(x, name=""):
            if isinstance(x, dict):
                for a in x:
                    if a == "type" and isinstance(
                        x[a], str
                    ):  # 'type' can be a name of a field
                        field_name = name[:-1]
                        field_type = x[a]
                        # if field_type is 'date' keep track of the format info when available
                        date_format = None
                        if field_type == "date" and "format" in x:
                            date_format = x["format"]
                        # If there is a conflicting type, warn - first values added wins
                        if field_name in fields and fields[field_name] != (
                            field_type,
                            date_format,
                        ):
                            warnings.warn(
                                f"Field {field_name} has conflicting types "
                                f"{fields[field_name]} != {field_type}",
                                UserWarning,
                            )
                        else:
                            fields[field_name] = (field_type, date_format)
                    elif a == "properties" or (not source_only and a == "fields"):
                        flatten(x[a], name)
                    elif not (
                        source_only and a == "fields"
                    ):  # ignore multi-field fields for source_only
                        flatten(x[a], name + a + ".")

        for index in mappings:
            if "properties" in mappings[index]["mappings"]:
                properties = mappings[index]["mappings"]["properties"]
            else:
                # Pre Elasticsearch 7.0 mappings had types. Support these
                # in case eland is connected to 6.x index - this is not
                # officially supported, but does help usability
                es_types = list(mappings[index]["mappings"].keys())
                if len(es_types) != 1:
                    raise NotImplementedError(
                        f"eland only supports 0 or 1 Elasticsearch types. es_types={es_types}"
                    )
                properties = mappings[index]["mappings"][es_types[0]]["properties"]

            flatten(properties)

        return fields

    @staticmethod
    def _create_capability_matrix(all_fields, source_fields, all_fields_caps):
        """
        {
          "fields": {
            "rating": {
              "long": {
                "searchable": true,
                "aggregatable": false,
                "indices": ["index1", "index2"],
                "non_aggregatable_indices": ["index1"]
              },
              "keyword": {
                "searchable": false,
                "aggregatable": true,
                "indices": ["index3", "index4"],
                "non_searchable_indices": ["index4"]
              }
            },
            "title": {
              "text": {
                "searchable": true,
                "aggregatable": false

              }
            }
          }
        }
        """
        all_fields_caps_fields = all_fields_caps["fields"]

        capability_matrix = {}

        for field, field_caps in all_fields_caps_fields.items():
            if field in all_fields:
                # v = {'long': {'type': 'long', 'searchable': True, 'aggregatable': True}}
                for kk, vv in field_caps.items():
                    _source = field in source_fields
                    es_field_name = field
                    es_dtype = vv["type"]
                    es_date_format = all_fields[field][1]
                    pd_dtype = FieldMappings._es_dtype_to_pd_dtype(vv["type"])
                    is_searchable = vv["searchable"]
                    is_aggregatable = vv["aggregatable"]
                    scripted = False
                    aggregatable_es_field_name = None  # this is populated later

                    caps = [
                        es_field_name,
                        _source,
                        es_dtype,
                        es_date_format,
                        pd_dtype,
                        is_searchable,
                        is_aggregatable,
                        scripted,
                        aggregatable_es_field_name,
                    ]

                    capability_matrix[field] = caps

                    if "non_aggregatable_indices" in vv:
                        warnings.warn(
                            f"Field {field} has conflicting aggregatable fields across indexes "
                            f"{str(vv['non_aggregatable_indices'])}",
                            UserWarning,
                        )
                    if "non_searchable_indices" in vv:
                        warnings.warn(
                            f"Field {field} has conflicting searchable fields across indexes "
                            f"{str(vv['non_searchable_indices'])}",
                            UserWarning,
                        )

        capability_matrix_df = pd.DataFrame.from_dict(
            capability_matrix, orient="index", columns=FieldMappings.column_labels
        )

        def find_aggregatable(row, df):
            # convert series to dict so we can add 'aggregatable_es_field_name'
            row_as_dict = row.to_dict()
            if not row_as_dict["is_aggregatable"]:
                # if not aggregatable, then try field.keyword
                es_field_name_keyword = row.es_field_name + ".keyword"
                try:
                    series = df.loc[df.es_field_name == es_field_name_keyword]
                    if not series.empty and series.is_aggregatable.squeeze():
                        row_as_dict[
                            "aggregatable_es_field_name"
                        ] = es_field_name_keyword
                    else:
                        row_as_dict["aggregatable_es_field_name"] = None
                except KeyError:
                    row_as_dict["aggregatable_es_field_name"] = None
            else:
                row_as_dict["aggregatable_es_field_name"] = row_as_dict["es_field_name"]

            return pd.Series(data=row_as_dict)

        # add aggregatable_es_field_name column by applying action to each row
        capability_matrix_df = capability_matrix_df.apply(
            find_aggregatable, args=(capability_matrix_df,), axis="columns"
        )

        # return just source fields (as these are the only ones we display)
        return capability_matrix_df[capability_matrix_df.is_source].sort_index()

    @classmethod
    def _es_dtype_to_pd_dtype(cls, es_dtype):
        """
        Mapping Elasticsearch types to pandas dtypes
        --------------------------------------------

        Elasticsearch field datatype              | Pandas dtype
        --
        text                                      | object
        keyword                                   | object
        long, integer, short, byte, binary        | int64
        double, float, half_float, scaled_float   | float64
        date, date_nanos                          | datetime64
        boolean                                   | bool
        TODO - add additional mapping types
        """
        return cls.ES_DTYPE_TO_PD_DTYPE.get(es_dtype, "object")

    @staticmethod
    def _pd_dtype_to_es_dtype(pd_dtype) -> Optional[str]:
        """
        Mapping pandas dtypes to Elasticsearch dtype
        --------------------------------------------

        ```
        Pandas dtype Python type NumPy type Usage
        object str string_, unicode_ Text
        int64 int int_, int8, int16, int32, int64, uint8, uint16, uint32, uint64 Integer numbers
        float64 float float_, float16, float32, float64 Floating point numbers
        bool bool bool_ True/False values
        datetime64 NA datetime64[ns] Date and time values
        timedelta[ns] NA NA Differences between two datetimes
        category NA NA Finite list of text values
        ```
        """
        es_dtype: Optional[str] = None

        # Map all to 64-bit - TODO map to specifics: int32 -> int etc.
        if is_float_dtype(pd_dtype):
            es_dtype = "double"
        elif is_integer_dtype(pd_dtype):
            es_dtype = "long"
        elif is_bool_dtype(pd_dtype):
            es_dtype = "boolean"
        elif is_string_dtype(pd_dtype):
            es_dtype = "keyword"
        elif is_datetime_or_timedelta_dtype(pd_dtype):
            es_dtype = "date"
        else:
            warnings.warn(
                f"No mapping for pd_dtype: [{pd_dtype}], using default mapping"
            )

        return es_dtype

    @staticmethod
    def _generate_es_mappings(
        dataframe: "pd.DataFrame", es_type_overrides: Optional[Mapping[str, str]] = None
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Given a pandas dataframe, generate the associated Elasticsearch mapping

        Parameters
        ----------
            dataframe : pandas.DataFrame
                pandas.DataFrame to create schema from
            es_type_overrides : dict
                Dictionary of Elasticsearch types to override defaults  for certain fields
                (e.g. { 'location': 'geo_point' })

        Returns
        -------
            mapping : str
        """
        es_dtype: Union[str, Dict[str, Any]]

        mapping_props: Dict[str, Any] = {}

        if es_type_overrides is not None:
            non_existing_columns: List[str] = [
                key for key in es_type_overrides.keys() if key not in dataframe.columns
            ]
            if non_existing_columns:
                raise KeyError(
                    f"{repr(non_existing_columns)[1:-1]} column(s) not in given dataframe"
                )

        for column, dtype in dataframe.dtypes.iteritems():
            if es_type_overrides is not None and column in es_type_overrides:
                es_dtype = es_type_overrides[column]
                if es_dtype == "text":
                    es_dtype = {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    }
            else:
                es_dtype = FieldMappings._pd_dtype_to_es_dtype(dtype)

            if isinstance(es_dtype, str):
                mapping_props[column] = {"type": es_dtype}
            else:
                mapping_props[column] = es_dtype

        return {"mappings": {"properties": mapping_props}}

    def aggregatable_field_name(self, display_name: str) -> Optional[str]:
        """
        Return a single aggregatable field_name from display_name

        Logic here is that field_name names are '_source' fields and keyword fields
        may be nested beneath the field. E.g.
        customer_full_name: text
        customer_full_name.keyword: keyword

        customer_full_name.keyword is the aggregatable field for customer_full_name

        Parameters
        ----------
        display_name: str

        Returns
        -------
        aggregatable_es_field_name: str or None
            The aggregatable field name associated with display_name. This could be the field_name, or the
            field_name.keyword.

        raise KeyError if the field_name doesn't exist in the mapping, or isn't aggregatable
        """
        mapping: Optional[pd.Series] = None

        try:
            mapping = self._mappings_capabilities.loc[display_name]
        except KeyError:
            raise KeyError(
                f"Can not get aggregatable field name for invalid display name {display_name}"
            ) from None

        if mapping is not None and mapping.aggregatable_es_field_name is None:
            warnings.warn(f"Aggregations not supported for '{display_name}'")

        return mapping.aggregatable_es_field_name

    def aggregatable_field_names(self) -> Dict[str, str]:
        """
        Return a list of aggregatable Elasticsearch field_names for all display names.
        If field is not aggregatable_field_names, return nothing.

        Logic here is that field_name names are '_source' fields and keyword fields
        may be nested beneath the field. E.g.
        customer_full_name: text
        customer_full_name.keyword: keyword

        customer_full_name.keyword is the aggregatable field for customer_full_name

        Returns
        -------
        Dict of aggregatable_field_names
            key = aggregatable_field_name, value = field_name
            e.g. {'customer_full_name.keyword': 'customer_full_name', ...}
        """
        non_aggregatables = self._mappings_capabilities[
            self._mappings_capabilities.aggregatable_es_field_name.isna()
        ]
        if not non_aggregatables.empty:
            warnings.warn(f"Aggregations not supported for '{non_aggregatables}'")

        aggregatables = self._mappings_capabilities[
            self._mappings_capabilities.aggregatable_es_field_name.notna()
        ]

        # extract relevant fields and convert to dict
        # <class 'dict'>: {'category.keyword': 'category', 'currency': 'currency', ...
        return dict(
            aggregatables[["aggregatable_es_field_name", "es_field_name"]].to_dict(
                orient="split"
            )["data"]
        )

    def date_field_format(self, es_field_name: str) -> str:
        """
        Parameters
        ----------
        es_field_name: str


        Returns
        -------
        str
            A string (for date fields) containing the date format for the field
        """
        return self._mappings_capabilities.loc[
            self._mappings_capabilities.es_field_name == es_field_name
        ].es_date_format.squeeze()

    def field_name_pd_dtype(self, es_field_name: str) -> str:
        """
        Parameters
        ----------
        es_field_name: str

        Returns
        -------
        pd_dtype: str
            The pandas data type we map to

        Raises
        ------
        KeyError
            If es_field_name does not exist in mapping
        """
        if es_field_name not in self._mappings_capabilities.es_field_name:
            raise KeyError(f"es_field_name {es_field_name} does not exist")

        pd_dtype = self._mappings_capabilities.loc[
            self._mappings_capabilities.es_field_name == es_field_name
        ].pd_dtype.squeeze()
        return pd_dtype

    def add_scripted_field(
        self, scripted_field_name: str, display_name: str, pd_dtype: str
    ) -> None:
        # if this display name is used somewhere else, drop it
        if display_name in self._mappings_capabilities.index:
            self._mappings_capabilities = self._mappings_capabilities.drop(
                index=[display_name]
            )

        # ['es_field_name', 'is_source', 'es_dtype', 'es_date_format', 'pd_dtype', 'is_searchable',
        # 'is_aggregatable', 'is_scripted', 'aggregatable_es_field_name']

        capabilities = {
            display_name: [
                scripted_field_name,
                False,
                self._pd_dtype_to_es_dtype(pd_dtype),
                None,
                pd_dtype,
                True,
                True,
                True,
                scripted_field_name,
            ]
        }

        capability_matrix_row = pd.DataFrame.from_dict(
            capabilities, orient="index", columns=FieldMappings.column_labels
        )

        self._mappings_capabilities = self._mappings_capabilities.append(
            capability_matrix_row
        )

    def numeric_source_fields(self) -> List[str]:
        _, es_field_names, _ = self.metric_source_fields()
        return es_field_names

    def all_source_fields(self) -> List[Field]:
        """
        This method is used to return all Field Mappings for fields

        Returns
        -------
        A list of Field Mappings

        """
        source_fields: List[Field] = []
        for column, row in self._mappings_capabilities.iterrows():
            row = row.to_dict()
            row["column"] = column
            source_fields.append(Field(**row))
        return source_fields

    def groupby_source_fields(self, by: List[str]) -> Tuple[List[Field], List[Field]]:
        """
        This method returns all Field Mappings for groupby and non-groupby fields

        Parameters
        ----------
        by:
            A list of groupby fields

        Returns
        -------
        A Tuple consisting of a list of field mappings for groupby and non-groupby fields

        """
        groupby_fields: Dict[str, Field] = {}
        aggregatable_fields: List[Field] = []
        for column, row in self._mappings_capabilities.iterrows():
            row = row.to_dict()
            row["column"] = column
            if column not in by:
                aggregatable_fields.append(Field(**row))
            else:
                groupby_fields[column] = Field(**row)

        # Maintain groupby order as given input
        return [groupby_fields[column] for column in by], aggregatable_fields

    def metric_source_fields(
        self, include_bool: bool = False, include_timestamp: bool = False
    ) -> Tuple[List["DTypeLike"], List[str], Optional[List[str]]]:
        """
        Returns
        -------
        pd_dtypes: list of np.dtype
            List of pd_dtypes for es_field_names
        es_field_names: list of str
            List of source fields where pd_dtype == (int64 or float64 or bool or timestamp)
        es_date_formats: list of str (can be None)
            List of es date formats for es_field

        TODO - not very efficient, but unless called per row, this should be ok
        """
        pd_dtypes = []
        es_field_names = []
        es_date_formats = []
        for index, row in self._mappings_capabilities.iterrows():
            pd_dtype = row["pd_dtype"]
            es_field_name = row["es_field_name"]
            es_date_format = row["es_date_format"]

            if is_integer_dtype(pd_dtype) or is_float_dtype(pd_dtype):
                pd_dtypes.append(np.dtype(pd_dtype))
                es_field_names.append(es_field_name)
                es_date_formats.append(es_date_format)
            elif include_bool and is_bool_dtype(pd_dtype):
                pd_dtypes.append(np.dtype(pd_dtype))
                es_field_names.append(es_field_name)
                es_date_formats.append(es_date_format)
            elif include_timestamp and is_datetime_or_timedelta_dtype(pd_dtype):
                pd_dtypes.append(np.dtype(pd_dtype))
                es_field_names.append(es_field_name)
                es_date_formats.append(es_date_format)

        # return in display_name order
        return pd_dtypes, es_field_names, es_date_formats

    def get_field_names(self, include_scripted_fields: bool = True) -> List[str]:
        if include_scripted_fields:
            return self._mappings_capabilities.es_field_name.to_list()

        return self._mappings_capabilities[  # noqa: E712
            self._mappings_capabilities.is_scripted == False
        ].es_field_name.to_list()

    def _get_display_names(self):
        return self._mappings_capabilities.index.to_list()

    def _set_display_names(self, display_names: List[str]):
        if not is_list_like(display_names):
            raise ValueError(f"'{display_names}' is not list like")

        if list(set(display_names) - set(self.display_names)):
            raise KeyError(f"{display_names} not in display names {self.display_names}")

        self._mappings_capabilities = self._mappings_capabilities.reindex(display_names)

    display_names = property(_get_display_names, _set_display_names)

    def dtypes(self):
        """
        Returns
        -------
        dtypes: pd.Series
            Index: Display name
            Values: pd_dtype as np.dtype
        """
        pd_dtypes = self._mappings_capabilities["pd_dtype"]

        # Set name of the returned series as None
        pd_dtypes.name = None

        # Convert return from 'str' to 'np.dtype'
        return pd_dtypes.apply(lambda x: np.dtype(x))

    def es_dtypes(self):
        """
        Returns
        -------
        dtypes: pd.Series
            Index: Display name
            Values: es_dtype as a string
        """
        es_dtypes = self._mappings_capabilities["es_dtype"]

        # Set name of the returned series as None
        es_dtypes.name = None
        return es_dtypes

    def es_info(self, buf: TextIO) -> None:
        buf.write("Mappings:\n")
        buf.write(f" capabilities:\n{self._mappings_capabilities.to_string()}\n")

    def rename(self, old_name_new_name_dict):
        """
        Renames display names in-place

        Parameters
        ----------
        old_name_new_name_dict

        Returns
        -------
        Nothing

        Notes
        -----
        For the names that do not exist this is a no op
        """
        self._mappings_capabilities = self._mappings_capabilities.rename(
            index=old_name_new_name_dict
        )

    def get_renames(self):
        # return dict of renames { old_name: new_name, ... } (inefficient)
        renames = {}

        for display_name in self.display_names:
            field_name = self._mappings_capabilities.loc[display_name].es_field_name
            if field_name != display_name:
                renames[field_name] = display_name

        return renames


def verify_mapping_compatibility(
    ed_mapping: Mapping[str, Mapping[str, Mapping[str, Mapping[str, str]]]],
    es_mapping: Mapping[str, Mapping[str, Mapping[str, Mapping[str, str]]]],
    es_type_overrides: Optional[Mapping[str, str]] = None,
) -> None:
    """Given a mapping generated by Eland and an existing ES index mapping
    attempt to see if the two are compatible. If not compatible raise ValueError
    with a list of problems between the two to be reported to the user.
    """
    problems = []
    es_type_overrides = es_type_overrides or {}

    ed_mapping = ed_mapping["mappings"]["properties"]
    es_mapping = es_mapping["mappings"]["properties"]

    for key in sorted(es_mapping.keys()):
        if key not in ed_mapping:
            problems.append(f"- {key!r} is missing from DataFrame columns")

    for key, key_def in sorted(ed_mapping.items()):
        if key not in es_mapping:
            problems.append(f"- {key!r} is missing from ES index mapping")
            continue

        key_type = es_type_overrides.get(key, key_def["type"])
        es_key_type = es_mapping[key]["type"]
        if key_type != es_key_type and es_key_type not in ES_COMPATIBLE_TYPES.get(
            key_type, ()
        ):
            problems.append(
                f"- {key!r} column type ({key_type!r}) not compatible with "
                f"ES index mapping type ({es_key_type!r})"
            )

    if problems:
        problems_message = "\n".join(problems)
        raise ValueError(
            f"DataFrame dtypes and Elasticsearch index mapping "
            f"aren't compatible:\n{problems_message}"
        )
