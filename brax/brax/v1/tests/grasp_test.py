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

"""Tests for brax.envs.grasp."""

from absl.testing import absltest
from brax.v1 import envs
from brax.v1 import jumpy as jp
import jax


class GraspTest(absltest.TestCase):

    def testGrasp(self):
        env = envs.create("grasp")
        grasp_action = jp.array(
            [
                -0.3,
                0.0,
                0.0,
                -1,
                0.0,
                0.0,
                -1,
                -0,
                0.0,  # gripper arm 1
                0.3,
                0.0,
                0.0,
                1,
                0.0,
                0.0,
                1,
                0,
                0.0,  # gripper arm 2
                0.3,
                0.0,
                0.0,
                1,
                0.0,
                0.0,
                1,
                0,
                0.0,  # gripper arm 3
                0.3,
                0.0,
                0.0,
                1,
                0.0,
                0.0,
                1,
                0,
                0.0,  # gripper arm 4
                0,
                0,
                -1.0,  # position action
            ]
        )

        def raise_action(i):
            return jp.array(
                [
                    -0.3,
                    0.0,
                    0.0,
                    -1,
                    0.0,
                    0.0,
                    -1,
                    -0,
                    0.0,  # gripper arm 1
                    0.3,
                    0.0,
                    0.0,
                    1,
                    0.0,
                    0.0,
                    1,
                    0,
                    0.0,  # gripper arm 2
                    0.3,
                    0.0,
                    0.0,
                    1,
                    0.0,
                    0.0,
                    1,
                    0,
                    0.0,  # gripper arm 3
                    0.3,
                    0.0,
                    0.0,
                    1,
                    0.0,
                    0.0,
                    1,
                    0,
                    0.0,  # gripper arm 4
                    0,
                    0,
                    -1.0 * ((250 - i) / 250) + -0.5 * (i / 250),  # position action
                ]
            )

        state = env.reset(jp.random_prngkey(0))
        step = jax.jit(env.step)
        # grasp
        for _ in range(250):
            state = step(state, grasp_action)
        # slowly lift
        for i in range(250):
            state = step(state, raise_action(i))

        self.assertGreater(state.qp.pos[1, 2], 1.36)  # ball lifted off ground
        self.assertLess(state.qp.ang[1, 2], 0.01)  # ball not rolling


if __name__ == "__main__":
    absltest.main()
