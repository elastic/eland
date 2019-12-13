#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
from abc import ABC, abstractmethod
from io import StringIO


class ArithmeticObject(ABC):
    @property
    @abstractmethod
    def value(self):
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
    def value(self):
        return "'{}'".format(self._value)

    def __repr__(self):
        return self.value

class ArithmeticNumber(ArithmeticObject):
    def __init__(self, value):
        self._value = value

    def resolve(self):
        return self.value

    @property
    def value(self):
        return "{}".format(self._value)

    def __repr__(self):
        return self.value

class ArithmeticSeries(ArithmeticObject):
    def __init__(self, query_compiler, display_name):
        task = query_compiler.get_arithmetic_op_fields()
        if task is not None:
            self._value = task._arithmetic_series.value
            self._tasks = task._arithmetic_series._tasks.copy()
        else:
            self._value = "doc['{}'].value".format(display_name)
            self._tasks = []

    @property
    def value(self):
        return self._value

    def __repr__(self):
        buf = StringIO()
        buf.write("Series: {} ".format(self.value))
        buf.write("Tasks: ")
        for task in self._tasks:
            buf.write("{} ".format(repr(task)))
        return buf.getvalue()

    def resolve(self):
        value = self._value

        for task in self._tasks:
            if task.op_name == '__add__':
                value = "({} + {})".format(value, task.object.resolve())
            elif task.op_name == '__truediv__':
                value = "({} / {})".format(value, task.object.resolve())
            elif task.op_name == '__floordiv__':
                value = "Math.floor({} / {})".format(value, task.object.resolve())
            elif task.op_name == '__mod__':
                value = "({} % {})".format(value, task.object.resolve())
            elif task.op_name == '__mul__':
                value = "({} * {})".format(value, task.object.resolve())
            elif task.op_name == '__pow__':
                value = "Math.pow({}, {})".format(value, task.object.resolve())
            elif task.op_name == '__sub__':
                value = "({} - {})".format(value, task.object.resolve())
            elif task.op_name == '__radd__':
                value = "({} + {})".format(task.object.resolve(), value)
            elif task.op_name == '__rtruediv__':
                value = "({} / {})".format(task.object.resolve(), value)
            elif task.op_name == '__rfloordiv__':
                value = "Math.floor({} / {})".format(task.object.resolve(), value)
            elif task.op_name == '__rmod__':
                value = "({} % {})".format(task.object.resolve(), value)
            elif task.op_name == '__rmul__':
                value = "({} * {})".format(task.object.resolve(), value)
            elif task.op_name == '__rpow__':
                value = "Math.pow({}, {})".format(task.object.resolve(), value)
            elif task.op_name == '__rsub__':
                value = "({} - {})".format(task.object.resolve(), value)

        return value

    def arithmetic_operation(self, op_name, right):
        task = ArithmeticTask(op_name, right)
        self._tasks.append(task)
        return self

class ArithmeticTask:
    def __init__(self, op_name, object):
        self._op_name = op_name

        if not isinstance(object, ArithmeticObject):
            raise TypeError("Task requires ArithmeticObject not {}".format(type(object)))
        self._object = object

    def __repr__(self):
        buf = StringIO()
        buf.write("op_name: {} ".format(self.op_name))
        buf.write("object: {} ".format(repr(self.object)))
        return buf.getvalue()

    @property
    def op_name(self):
        return self._op_name

    @property
    def object(self):
        return self._object

