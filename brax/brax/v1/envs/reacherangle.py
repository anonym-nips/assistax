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

"""Trains a reacher to reach a target.

Based on the OpenAI Gym MuJoCo Reacher environment.
"""

from typing import Tuple

import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1.envs import env


class ReacherAngle(env.Env):
    """Trains a reacher arm to touch a random target via angle actuators."""

    def __init__(self, legacy_spring=False, **kwargs):
        config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)
        self.target_idx = self.sys.body.index["target"]
        self.arm_idx = self.sys.body.index["body1"]

        # map the [-1, 1] action space into a valid angle for the actuators
        limits = []
        for j in self.sys.config.joints:
            for l in j.angle_limit:
                limits.append((l.min, l.max))
        self._min_act = jp.array([l[0] for l in limits])
        self._range_act = jp.array([l[1] - l[0] for l in limits])

    def reset(self, rng: jp.ndarray) -> env.State:
        rng, rng1, rng2 = jp.random_split(rng, 3)
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -0.1, 0.1
        )
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -0.005, 0.005)
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        rng, target = self._random_target(rng)
        pos = jp.index_update(qp.pos, self.target_idx, target)
        qp = qp.replace(pos=pos)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "rewardDist": zero,
            "rewardCtrl": zero,
        }
        return env.State(qp, obs, reward, done, metrics)

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        action = self._min_act + self._range_act * ((action + 1) / 2.0)

        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)

        # vector from tip to target is last 3 entries of obs vector
        reward_dist = -jp.norm(obs[-3:])

        reward = reward_dist

        metrics = {
            "rewardDist": reward_dist,
            "rewardCtrl": 0.0,
        }

        return state.replace(qp=qp, obs=obs, reward=reward, metrics=metrics)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Egocentric observation of target and arm body."""

        # some pre-processing to pull joint angles and velocities
        joint_angle, _ = self.sys.joints[0].angle_vel(qp)

        # qpos:
        # x,y coord of target
        qpos = [qp.pos[self.target_idx, :2]]

        # dist to target and speed of tip
        arm_qps = jp.take(qp, jp.array(self.arm_idx))
        tip_pos, tip_vel = arm_qps.to_world(jp.array([0.11, 0.0, 0.0]))
        tip_to_target = [tip_pos - qp.pos[self.target_idx]]
        cos_sin_angle = [jp.cos(joint_angle), jp.sin(joint_angle)]

        # qvel:
        # velocity of tip
        qvel = [tip_vel[:2]]

        return jp.concatenate(cos_sin_angle + qpos + qvel + tip_to_target)

    def _random_target(self, rng: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        """Returns a target location in a random circle slightly above xy plane."""
        rng, rng1, rng2 = jp.random_split(rng, 3)
        # we perform a sqrt here so that the final distribution of sampled targets
        # is uniformly random in 2D
        dist = 0.2 * jp.sqrt(jp.random_uniform(rng1))
        ang = jp.pi * 2.0 * jp.random_uniform(rng2)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        target_z = 0.01
        target = jp.array([target_x, target_y, target_z]).transpose()
        return rng, target


_SYSTEM_CONFIG = """
  bodies {
    name: "ground"
    colliders {
      plane {
      }
    }
    mass: 1.0
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    frozen {
      all: true
    }
  }
  bodies {
    name: "body0"
    colliders {
      position {
        x: 0.05
      }
      rotation {
        y: 90.0
      }
      capsule {
        radius: 0.01
        length: 0.12
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.035604715
  }
  bodies {
    name: "body1"
    colliders {
      position {
        x: 0.05
      }
      rotation {
        y: 90.0
      }
      capsule {
        radius: 0.01
        length: 0.12
      }
    }
    colliders {
      position { x: .11 }
      sphere {
        radius: 0.01
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.035604715
  }
  bodies {
    name: "target"
    colliders {
      position {
      }
      sphere {
        radius: 0.009
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
    frozen { all: true }
  }
  joints {
    name: "joint0"
    parent: "ground"
    child: "body0"
    parent_offset {
      z: 0.01
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angle_limit {
        min: -180
        max: 180
      }
    angular_damping: 10.0
  }
  joints {
    name: "joint1"
    parent: "body0"
    child: "body1"
    parent_offset {
      x: 0.1
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -180
      max: 180
    }
    angular_damping: 10.0
  }
  actuators {
    name: "joint0"
    joint: "joint0"
    strength: 20.0
    angle {
    }
  }
  actuators {
    name: "joint1"
    joint: "joint1"
    strength: 20.0
    angle {
    }
  }
  collide_include {
  }
  gravity {
    z: -9.81
  }
  dt: 0.02
  substeps: 4
  frozen {
    position {
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
    }
  }
  dynamics_mode: "pbd"
  """

_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "ground"
    colliders {
      plane {
      }
    }
    mass: 1.0
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    frozen {
      all: true
    }
  }
  bodies {
    name: "body0"
    colliders {
      position {
        x: 0.05
      }
      rotation {
        y: 90.0
      }
      capsule {
        radius: 0.01
        length: 0.12
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.035604715
  }
  bodies {
    name: "body1"
    colliders {
      position {
        x: 0.05
      }
      rotation {
        y: 90.0
      }
      capsule {
        radius: 0.01
        length: 0.12
      }
    }
    colliders {
      position { x: .11 }
      sphere {
        radius: 0.01
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.035604715
  }
  bodies {
    name: "target"
    colliders {
      position {
      }
      sphere {
        radius: 0.009
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
    frozen { all: true }
  }
  joints {
    name: "joint0"
    stiffness: 100.0
    parent: "ground"
    child: "body0"
    parent_offset {
      z: 0.01
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angle_limit {
        min: -180
        max: 180
      }
    limit_strength: 0.0
    spring_damping: 3.0
    angular_damping: 10.0
  }
  joints {
    name: "joint1"
    stiffness: 100.0
    parent: "body0"
    child: "body1"
    parent_offset {
      x: 0.1
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -180
      max: 180
    }
    limit_strength: 0.0
    spring_damping: 3.0
    angular_damping: 10.0
  }
  actuators {
    name: "joint0"
    joint: "joint0"
    strength: 20.0
    angle {
    }
  }
  actuators {
    name: "joint1"
    joint: "joint1"
    strength: 20.0
    angle {
    }
  }
  collide_include {
  }
  gravity {
    z: -9.81
  }
  baumgarte_erp: 0.1
  dt: 0.02
  substeps: 4
  frozen {
    position {
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
    }
  }
  dynamics_mode: "legacy_spring"
  """
