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
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import numpy as np

# First Party
from caikit.core import DataObjectBase, dataobject
from caikit.core.exceptions import error_handler
from caikit.interfaces.common.data_model import ProducerId
import alog

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.caikit_computer_vision")
class FlatChannel(DataObjectBase):
    """A flat array of values representing a single channel in an image."""

    values: Annotated[List[np.uint32], FieldNumber(1)]

    def __post_init__(self):
        error.type_check("<CCV81813713E>", tuple, list, values=self.values)
        error.value_check(
            "<CCV81819292E>",
            not self.values or (max(self.values) < 256 and min(self.values) >= 0),
            "Flat channel values must be of type uint8",
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
            "All channels must have the same length!",
        )

        # The image shape should be of length 3, and should contain
        # the same number of elements that we have in our flattned channels
        # Every channel should be the same length when flat
        num_channel_elems = channel_lengths[0] * len(channel_lengths)
        expected_channel_elems = np.prod(self.image_shape)
        error.value_check(
            "<CCV80910185E>",
            num_channel_elems == expected_channel_elems,
            "Number of channel elements must match number of elements in provided shape",
        )

    @classmethod
    def from_numpy(cls, np_arr: np.ndarray) -> "FlatImage":
        """Builds a flattened image out of a numpy array.

        Args:
            FlatImage
                An instance of this class that is serializable.
        """
        # Build the flattened channels
        flat_channels = [
            FlatChannel(values=np_arr[:, :, ch_idx].flatten().tolist())
            for ch_idx in range(np_arr.shape[-1])
        ]
        # And from the flattened channels, build the flat image
        return cls(flat_channels=flat_channels, image_shape=np_arr.shape)

    def to_numpy(self) -> np.ndarray:
        """Converts the flattened image to a numpy array. This is accomplished by
        reconstructing each flattened channel, then stacking along the new channel
        axis.

        Returns:
            np.ndarray
                BGR numpy array representing the flattened image.
        """
        # Individually reconstruct each channel, each of which
        # were flattened in row major order...
        channel_shape = self.image_shape[:2]
        rebuilt_channels = [
            np.array(ch.values, dtype=np.uint8).reshape(channel_shape)
            for ch in self.flat_channels
        ]
        # And stack the reconstructed channels along a new channel axis
        return np.stack(rebuilt_channels, axis=2)
