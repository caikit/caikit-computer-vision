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
"""Data structures for classification in images."""

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from caikit.core import DataObjectBase, dataobject
from caikit.interfaces.common.data_model import ProducerId

log = alog.use_channel("DATAM")

@dataobject(package="caikit_data_model.caikit_computer_vision")
class ImageClassification(DataObjectBase):
    label: Annotated[str, FieldNumber(1)]
    score: Annotated[float, FieldNumber(2)]

@dataobject(package="caikit_data_model.caikit_computer_vision")
class ImageClassificationResult(DataObjectBase):
    classifications: Annotated[list[ImageClassification], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]