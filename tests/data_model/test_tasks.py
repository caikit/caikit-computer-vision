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
"""Tests for the task definitions;

NOTE: these tests are shamelessly lifted from caikit:
https://github.com/caikit/caikit/blob/main/tests/core/helpers.py
"""

# Standard
from typing import Type

# Third Party
from caikit.core import ModuleBase, TaskBase, module
from caikit.core.registries import (
    module_backend_registry,
    module_registry,
)
import pytest

# Local
from caikit_computer_vision.data_model import tasks

## Helpers #####################################################################
@pytest.fixture
def reset_module_registry():
    """Fixture that will reset caikit.core module registry if a test modifies it"""
    orig_module_registry = {key: val for key, val in module_registry().items()}
    yield
    module_registry().clear()
    module_registry().update(orig_module_registry)

@pytest.fixture
def reset_module_backend_registry():
    """Fixture that will reset the module distribution registry if a test modifies them"""
    orig_module_backend_registry = {
        key: val for key, val in module_backend_registry().items()
    }
    yield
    module_backend_registry().clear()
    module_backend_registry().update(orig_module_backend_registry)

## Tests #######################################################################

class InvalidType:
    pass


@pytest.mark.parametrize(
    "task", (
        tasks.ObjectDetectionTask,
        tasks.ImageClassificationTask,
    ),
)
def test_tasks(reset_module_registry, reset_module_backend_registry, task: Type[TaskBase]):
    """Common tests for all tasks"""
    # Only support single required param named "inputs"
    assert set(task.get_required_parameters().keys()) == {"inputs"}
    input_type = task.get_required_parameters()["inputs"]
    output_type = task.get_output_type()

    # Version with the right signature and nothing else
    @module(id="foo1", name="Foo", version="0.0.0", task=task)
    class Foo1(ModuleBase):
        def run(self, inputs: input_type) -> output_type:
            return output_type()

    # Version with the right signature plus extra args
    @module(id="foo2", name="Foo", version="0.0.0", task=task)
    class Foo2(ModuleBase):
        def run(
            self,
            inputs: input_type,
            workit: bool,
            makeit: bool,
            doit: bool,
        ) -> output_type:
            return output_type()

    # Version with missing required argument
    with pytest.raises(TypeError):

        @module(id="foo3", name="Foo", version="0.0.0", task=task)
        class Foo3(ModuleBase):
            def run(self, other_name: str) -> output_type:
                return output_type()

    # Version with bad required argument type
    with pytest.raises(TypeError):

        @module(id="foo4", name="Foo", version="0.0.0", task=task)
        class Foo4(ModuleBase):
            def run(self, inputs: InvalidType) -> output_type:
                return output_type()

    # Version with bad return type
    with pytest.raises(TypeError):

        @module(id="foo", name="Foo", version="0.0.0", task=task)
        class Foo(ModuleBase):
            def run(self, inputs: input_type) -> InvalidType:
                return "hi there"
