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
"""Data model for flattened images.

NOTE: In the *very* near future, this will be merged to Caikit,
and support will be added for this structure in the Image data model
backend / its corresponding types.
"""

# Standard
from typing import List

# Third Party
import numpy as np
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber

# First Party
import alog
from caikit.core import DataObjectBase, dataobject
from caikit.interfaces.common.data_model import ProducerId
from caikit.core.exceptions import error_handler

log = alog.use_channel("DATAM")
error = error_handler.get(log)

@dataobject(package="caikit_data_model.caikit_computer_vision")
class FlatChannel(DataObjectBase):
    """A flat array of values representing a single channel in an image."""
    values: Annotated[List[np.uint32], FieldNumber(1)]

    def __post_init__(self):
        error.type_check(
            "CCV81813713E",
            tuple,
            list,
            values=self.values
        )
        error.value_check(
            "<CCV81819292E>",
            not self.values or (max(self.values) < 256 and min(self.values) >= 0),
            "Flat channel values must be of type uint8"
        )


@dataobject(package="caikit_data_model.caikit_computer_vision")
class FlatImage(DataObjectBase):
    """A series of flattened arrays & an image shape."""
    flat_channels: Annotated[List[FlatChannel], FieldNumber(1)]
    image_shape: Annotated[List[np.uint32], FieldNumber(2)]

    def __post_init__(self):
        # Every channel should be the same length when flat
        channel_lengths = [len(ch.values) for ch in self.flat_channels]
        error.value_check(
            "<CCV80491785E>",
            len(set(channel_lengths)) == 1,
            "All channels must have the same length!"
        )

        # The image shape should be of length 3, and should contain
        # the same number of elements that we have in our flattned channels
        # Every channel should be the same length when flat
        num_channel_elems = channel_lengths[0] * len(channel_lengths)
        expected_channel_elems = np.prod(self.image_shape)
        error.value_check(
            "<CCV80910185E>",
            num_channel_elems == expected_channel_elems,
            "Number of channel elements must match number of elements in provided shape"
        )
