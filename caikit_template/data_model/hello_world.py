# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from caikit.core import (
    dataobject,
    DataObjectBase,
)


@dataobject(package="caikit_template.data_model")
class HelloWorldInput(DataObjectBase):
    """An example `domain primitive` input type for this library.
    This is analogous to a `Raw Document` for the `Natural Language Processing` domain."""

    name: str

@dataobject(package="caikit_template.data_model")
class HelloWorldTrainingType(DataObjectBase):
    """An example `training data` type for the `example_task` task."""

    text: str
    label: str

@dataobject(package="caikit_template.data_model")   
class HelloWorldPrediction(DataObjectBase):
    """A simple return type for the `example_task` task"""

    greeting: str
