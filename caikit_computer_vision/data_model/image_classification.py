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

# Standard
from typing import List

# Third Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber

# First Party
from caikit.core import DataObjectBase, dataobject
from caikit.interfaces.common.data_model import ProducerId
import alog

log = alog.use_channel("DATAM")


@dataobject(package="caikit_data_model.caikit_computer_vision")
class ImageClassification(DataObjectBase):
    label: Annotated[str, FieldNumber(1)]
    score: Annotated[float, FieldNumber(2)]


@dataobject(package="caikit_data_model.caikit_computer_vision")
class ImageClassificationResult(DataObjectBase):
    classifications: Annotated[List[ImageClassification], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]
