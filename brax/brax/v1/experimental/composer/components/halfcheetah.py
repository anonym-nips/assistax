# Copyright 2024 The Brax Authors.
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

"""Halfcheetah."""
# pylint:disable=protected-access
from brax.v1.envs import half_cheetah
from brax.v1.experimental.composer import component_editor

COLLIDES = ("torso", "bfoot", "ffoot")

ROOT = "torso"

DEFAULT_OBSERVERS = ("root_z_joints",)

TERM_FN = None


def get_specs():
    return dict(
        message_str=component_editor.filter_message_str(
            half_cheetah._SYSTEM_CONFIG_SPRING, "floor"
        ),
        collides=COLLIDES,
        root=ROOT,
        term_fn=TERM_FN,
        observers=DEFAULT_OBSERVERS,
    )
