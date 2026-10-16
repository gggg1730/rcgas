"""제자리 점프 → 공중 360° 플립 → 살포시 착지 RL 환경
바닥에 정지한 박스가 노즐로 뛰어올라, 몸체 x축(롤) 또는 y축(피치)으로 한 바퀴 돌고 착지한다.
추진계·탱크는 landing_env와 같다 (기본: 바퀴 자리 바닥 노즐 4개 × 20 N).
"""
from collections import deque

import mujoco
import numpy as np
from gymnasium import spaces

from box_env import CTRL_EVERY, axis_quat, landing_eval
from cold_gas_box import BOX, DELAY_STEPS, DT, FLOW_PER_N
from landing_env import (ESCAPE_PENALTY, OK_TILT, OK_VH, OK_VZ, OK_W, TANK_G, TIME_COST, Z_MAX,
                         LanderEnv, potential)

T_MAX = 5.0                # 에피소드 최대 시간 [s]
FLIP = 2 * np.pi           # 한 바퀴 [rad]
MIN_AIR = 0.1              # 이보다 짧게 떴다 닿으면 그냥 들썩인 것으로 보고 착지로 치지 않음 [s]
THRUST_JITTER = 0.10       # 노즐별 추력 편차 (에피소드마다 무작위) [비율]


class JumpFlipEnv(LanderEnv):
    def __init__(self, layout="wheels", axis="any", min_z=0.0):
        super().__init__(layout)
        self.min_z = min_z     # 무게중심이 이 높이 이상일 때만 회전을 플립으로 셈 (낮게 대충 도는 꼼수 방지) [m]
        self.axis = {"any": [0, 1], "roll": [0], "pitch": [1]}[axis]   # 플립으로 인정할 회전축
        self.gear0 = self.m.actuator_gear.copy()
        self.observation_space = spaces.Box(-np.inf, np.inf, (21,), np.float32)

    def _obs(self):
        return np.concatenate([super()._obs(), self.theta / FLIP,
                               [float(self.airborne), float(self.flipped)]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super(LanderEnv, self).reset(seed=seed)
        rng = self.np_random
        mujoco.mj_resetData(self.m, self.d)
        scale = 1 + rng.uniform(-THRUST_JITTER, THRUST_JITTER, self.m.nu)
        self.m.actuator_gear[:] = self.gear0 * scale[:, None]
        self.d.qpos[0:3] = [0, 0, BOX[2]]
        self.d.qpos[3:7] = axis_quat([0, 0, 1], rng.uniform(-np.pi, np.pi))
        mujoco.mj_forward(self.m, self.d)
        self.queue = deque([np.zeros(len(self.names))] * DELAY_STEPS, maxlen=DELAY_STEPS + 1)
        self.gas = 0.0
        self.theta = np.zeros(2)            # 몸체 x·y축 누적 회전 [rad]
        self.airborne, self.flipped, self.t_off = False, False, None
        self.apex = self.d.qpos[2]                 # 최고 높이 [m]
        self.phi = None
        return self._obs(), {}

    def _progress(self):
        return min(np.max(np.abs(self.theta[self.axis])), FLIP)

    def step(self, action):
        a = np.clip(action, 0.0, 1.0)
        gas0, prog0 = self.gas, self._progress()
        landed = False
        for _ in range(CTRL_EVERY):
            self.queue.append(a)
            u = self.queue[0] * (self.gas < TANK_G)
            self.d.ctrl[:] = u
            self.gas += (u @ self.fmax) * FLOW_PER_N * DT
            xmat, qvel = self.d.xmat[self.b].copy(), self.d.qvel.copy()
            mujoco.mj_step(self.m, self.d)
            if self.d.qpos[2] >= self.min_z:
                self.theta += self.d.qvel[3:5] * DT    # 자유관절 각속도는 몸체 좌표
            if self.d.ncon == 0 and not self.airborne:
                self.airborne, self.t_off = True, self.d.time
            if self.airborne and self.d.ncon > 0:
                if self.d.time - self.t_off > MIN_AIR:
                    landed = True
                    break
                self.airborne = False                    # 살짝 들썩였다 닿음 → 아직 지상

        self.apex = max(self.apex, self.d.qpos[2])
        tilt, yaw, w, _ = landing_eval(xmat, qvel)
        dist = np.linalg.norm(self.d.qpos[0:2])
        reward = -0.01 * (self.gas - gas0) - TIME_COST
        if not self.flipped:
            reward += 3.0 * (self._progress() - prog0)  # 회전 진행 보상 (한 바퀴 ≈ +18.8)
            if self._progress() >= FLIP:
                self.flipped = True
                self.phi = potential(tilt, w, qvel, dist)
        else:
            phi = potential(tilt, w, qvel, dist)        # 플립 후에는 착륙 길잡이
            reward += phi - self.phi
            self.phi = phi

        trunc = bool(not landed and (self.d.time > T_MAX or self.d.qpos[2] > Z_MAX))
        info = {}
        if landed:
            vz, vh = -qvel[2], np.linalg.norm(qvel[0:2])
            ok = bool(self.flipped and tilt < OK_TILT and w < OK_W and vz < OK_VZ and vh < OK_VH)
            if self.flipped:
                reward += (10.0 * ok + 5.0 * np.exp(-(tilt / 8) ** 2) + 5.0 * np.exp(-vz ** 2)
                           + 3.0 * np.exp(-vh ** 2))
            else:
                reward -= 10.0
            info = dict(tilt=tilt, yaw=yaw, w=w, vz=vz, vh=vh, dist=dist, success=ok, gas=self.gas,
                        flipped=self.flipped, rot=np.degrees(self.theta), air=self.d.time - self.t_off,
                        apex=self.apex)
        elif trunc:
            reward -= ESCAPE_PENALTY
        return self._obs(), float(reward), landed, trunc, info
