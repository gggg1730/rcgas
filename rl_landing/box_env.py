"""냉가스 박스 착지 RL 환경 (Gymnasium)"""
import os
import sys
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# 박스 모델·노즐 정의는 상위 폴더의 cold_gas_box.py를 그대로 사용
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cold_gas_box import DELAY_STEPS, DT, FLOW_PER_N, GAS_CAP, attitude, bang, build_xml  # noqa: E402

CTRL_EVERY = 10            # 제어 주기 100 Hz (물리 1 kHz)
T_MAX = 3.0                # 에피소드 최대 시간 [s]
OK_TILT, OK_W = 5.0, 30.0  # 성공 기준 [deg], [deg/s]


def axis_quat(axis, a):
    return np.r_[np.cos(a / 2), np.sin(a / 2) * np.asarray(axis, dtype=float)]


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def random_throw(rng):
    r, p, y = np.radians(rng.uniform([-30, -35, -30], [30, 35, 30]))
    q = qmul(qmul(axis_quat([0, 0, 1], y), axis_quat([0, -1, 0], p)), axis_quat([1, 0, 0], r))
    return dict(z0=rng.uniform(0.25, 0.45), quat=q,
                v=np.array([rng.uniform(3, 7), rng.uniform(-0.5, 0.5), rng.uniform(1.5, 3.0)]),
                w=np.radians(rng.uniform(-120, 120, 3)))


def landing_eval(xmat, qvel):
    """접촉 직전 상태로 평가 (mj_step은 접촉을 감지한 스텝에서 바로 충격력을 가하므로)"""
    R = xmat.reshape(3, 3)
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    w = np.degrees(np.linalg.norm(qvel[3:6]))
    return tilt, yaw, w, bool(tilt < OK_TILT and w < OK_W)


def baseline_action(d, b):
    """기존 방뱅 제어 (비교용)"""
    ang, rate = attitude(d, b)
    bp, br, by = (bang(a, r) for a, r in zip(ang, rate))
    return np.array([bp > 0, bp < 0, br > 0, br < 0, by > 0, by < 0, by < 0, by > 0], dtype=np.float32)


class ColdGasBoxEnv(gym.Env):
    def __init__(self, thrust=1.0):
        self.m = mujoco.MjModel.from_xml_string(build_xml())
        self.m.actuator_gear[:, :3] *= thrust
        self.d = mujoco.MjData(self.m)
        self.thrust = thrust
        self.b = self.m.body("box").id
        self.action_space = spaces.Box(0.0, 1.0, (8,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), np.float32)

    def _obs(self):
        return np.concatenate([self.d.xmat[self.b], self.d.sensor("gyro").data,
                               [self.d.qpos[2], self.d.qvel[2]]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.m, self.d)
        th = random_throw(self.np_random)
        self.d.qpos[0:3] = [0, 0, th["z0"]]
        self.d.qpos[3:7] = th["quat"]
        self.d.qvel[0:3] = th["v"]
        self.d.qvel[3:6] = th["w"]
        mujoco.mj_forward(self.m, self.d)
        self.queue = deque([np.zeros(8)] * DELAY_STEPS, maxlen=DELAY_STEPS + 1)
        self.gas = 0.0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, 0.0, 1.0)
        gas0 = self.gas
        for _ in range(CTRL_EVERY):
            self.queue.append(a)
            u = self.queue[0] * (self.gas < GAS_CAP)
            self.d.ctrl[:] = u
            self.gas += u.sum() * self.thrust * FLOW_PER_N * DT
            xmat, qvel = self.d.xmat[self.b].copy(), self.d.qvel.copy()
            mujoco.mj_step(self.m, self.d)
            if self.d.ncon > 0:
                break

        tilt, yaw, w, ok = landing_eval(xmat, qvel)
        reward = -0.5 * (self.gas - gas0) - 0.002 * tilt
        landed = self.d.ncon > 0
        info = {}
        if landed:
            reward += 10.0 * ok + 10.0 * np.exp(-(tilt / 8) ** 2) + 2.0 * np.exp(-(yaw / 15) ** 2) - 0.02 * w
            info = dict(tilt=tilt, yaw=yaw, w=w, success=ok, gas=self.gas)
        return self._obs(), float(reward), landed, self.d.time > T_MAX, info
