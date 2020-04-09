# Copyright 2020 Elasticsearch BV
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from io import StringIO

import numpy as np


class ArithmeticObject(ABC):
    @property
    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def dtype(self):
        pass

    @abstractmethod
    def resolve(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class ArithmeticString(ArithmeticObject):
    def __init__(self, value):
        self._value = value

    def resolve(self):
        return self.value

    @property
    def dtype(self):
        return np.dtype(object)

    @property
    def value(self):
        return f"'{self._value}'"

    def __repr__(self):
        return self.value


class ArithmeticNumber(ArithmeticObject):
    def __init__(self, value, dtype):
        self._value = value
        self._dtype = dtype

    def resolve(self):
        return self.value

    @property
    def value(self):
        return f"{self._value}"

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return self.value


class ArithmeticSeries(ArithmeticObject):
    def __init__(self, query_compiler, display_name, dtype):
        task = query_compiler.get_arithmetic_op_fields()
        if task is not None:
            self._value = task._arithmetic_series.value
            self._tasks = task._arithmetic_series._tasks.copy()
            self._dtype = dtype
        else:
            aggregatable_field_name = query_compiler.display_name_to_aggregatable_name(
                display_name
            )
            if aggregatable_field_name is None:
                raise ValueError(
                    f"Can not perform arithmetic operations on non aggregatable fields"
                    f"{display_name} is not aggregatable."
                )

            self._value = f"doc['{aggregatable_field_name}'].value"
            self._tasks = []
            self._dtype = dtype

    @property
    def value(self):
        return self._value

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        buf = StringIO()
        buf.write(f"Series: {self.value} ")
        buf.write("Tasks: ")
        for task in self._tasks:
            buf.write(f"{task!r} ")
        return buf.getvalue()

    def resolve(self):
        value = self._value

        for task in self._tasks:
            if task.op_name == "__add__":
                value = f"({value} + {task.object.resolve()})"
            elif task.op_name == "__truediv__":
                value = f"({value} / {task.object.resolve()})"
            elif task.op_name == "__floordiv__":
                value = f"Math.floor({value} / {task.object.resolve()})"
            elif task.op_name == "__mod__":
                value = f"({value} % {task.object.resolve()})"
            elif task.op_name == "__mul__":
                value = f"({value} * {task.object.resolve()})"
            elif task.op_name == "__pow__":
                value = f"Math.pow({value}, {task.object.resolve()})"
            elif task.op_name == "__sub__":
                value = f"({value} - {task.object.resolve()})"
            elif task.op_name == "__radd__":
                value = f"({task.object.resolve()} + {value})"
            elif task.op_name == "__rtruediv__":
                value = f"({task.object.resolve()} / {value})"
            elif task.op_name == "__rfloordiv__":
                value = f"Math.floor({task.object.resolve()} / {value})"
            elif task.op_name == "__rmod__":
                value = f"({task.object.resolve()} % {value})"
            elif task.op_name == "__rmul__":
                value = f"({task.object.resolve()} * {value})"
            elif task.op_name == "__rpow__":
                value = f"Math.pow({task.object.resolve()}, {value})"
            elif task.op_name == "__rsub__":
                value = f"({task.object.resolve()} - {value})"

        return value

    def arithmetic_operation(self, op_name, right):
        # check if operation is supported (raises on unsupported)
        self.check_is_supported(op_name, right)

        task = ArithmeticTask(op_name, right)
        self._tasks.append(task)
        return self

    def check_is_supported(self, op_name, right):
        # supported set is
        # series.number op_name number (all ops)
        # series.string op_name string (only add)
        # series.string op_name int (only mul)
        # series.string op_name float (none)
        # series.int op_name string (none)
        # series.float op_name string (none)

        # see end of https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html?highlight=dtype for
        # dtype heirarchy
        if np.issubdtype(self.dtype, np.number) and np.issubdtype(
            right.dtype, np.number
        ):
            # series.number op_name number (all ops)
            return True
        elif np.issubdtype(self.dtype, np.object_) and np.issubdtype(
            right.dtype, np.object_
        ):
            # series.string op_name string (only add)
            if op_name == "__add__" or op_name == "__radd__":
                return True
        elif np.issubdtype(self.dtype, np.object_) and np.issubdtype(
            right.dtype, np.integer
        ):
            # series.string op_name int (only mul)
            if op_name == "__mul__":
                return True

        raise TypeError(
            f"Arithmetic operation on incompatible types {self.dtype} {op_name} {right.dtype}"
        )


class ArithmeticTask:
    def __init__(self, op_name, object):
        self._op_name = op_name

        if not isinstance(object, ArithmeticObject):
            raise TypeError(f"Task requires ArithmeticObject not {type(object)}")
        self._object = object

    def __repr__(self):
        buf = StringIO()
        buf.write(f"op_name: {self.op_name} object: {self.object!r} ")
        return buf.getvalue()

    @property
    def op_name(self):
        return self._op_name

    @property
    def object(self):
        return self._object
