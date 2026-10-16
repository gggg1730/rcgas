"""큰 탱크 추력 착륙 RL 환경
던져진 박스가 공중에서 자세를 잡고, 낙하와 전진을 늦춰 제자리에 가깝게 살포시 내려앉는 것이 목표.
배치(layout):
  main     : 자세 노즐 8개(2 N) + 하강 메인 노즐 + 전방 제동 노즐
  roadster : 노즐 8개뿐. 바닥 4개를 세게(드론처럼 차등 추력으로 감속·자세·기울여 제동), 옆 4개는 요
  wheels   : roadster와 같되 바닥 4개를 바퀴 자리(네 모서리, 쿼드콥터 X형)로 옮김
"""
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from box_env import CTRL_EVERY, baseline_action, landing_eval, random_throw
from cold_gas_box import BOX, DELAY_STEPS, DT, FLOW_PER_N, MASS, NOZZLES

# ===== 추진계 =====
TANK_G = 190.0             # 고압 공기 탱크 (0.8 L, 200 bar 가정) [g]
TANK_MASS = 0.8            # 탱크 무게 [kg] (가스 소모로 인한 질량 감소는 무시)
ATT_N = 2.0                # 자세 노즐 추력 [N]
MAIN_N = 60.0              # 하강 메인 노즐 최대 추력 [N] (총중량 2.8 kg → 27.5 N 이상이면 호버 가능)
BRAKE_N = 25.0             # 전방 제동 노즐 최대 추력 [N]
BOTTOM_N = 20.0            # roadster 배치: 바닥 노즐 1개당 최대 추력 [N] (4개 합 80 N)

T_MAX = 4.0                # 에피소드 최대 시간 [s]
Z_MAX = 2.0                # 이 높이를 넘으면 이탈로 보고 종료 [m]
ESCAPE_PENALTY = 30.0      # 시간 초과·고도 이탈 감점 (어떤 착지보다 나쁘게)
TIME_COST = 0.02           # 공중 체류 감점 [/제어 스텝]
OK_TILT, OK_W = 5.0, 30.0  # 성공 기준: 기울기 [deg], 회전 [deg/s]
OK_VZ, OK_VH = 0.5, 0.5    # 성공 기준: 수직·수평 착지 속도 [m/s]
WILD_W = 360.0             # 막 던지기: 각 축 회전 범위 [deg/s]

bx, by, bz = BOX
BOTTOM = ("front", "rear", "left", "right")
WHEEL_X, WHEEL_Y = bx - 0.03, by - 0.02   # 바퀴 자리 [m]


def make_thrusters(layout):
    """이름: (위치, 미는 방향, 최대 추력 [N])"""
    if layout == "roadster":
        return {n: (p, f, BOTTOM_N if n in BOTTOM else ATT_N) for n, (p, f) in NOZZLES.items()}
    if layout == "wheels":
        th = {n: ((sx * WHEEL_X, sy * WHEEL_Y, -bz), (0, 0, 1), BOTTOM_N)
              for n, sx, sy in [("fl", 1, 1), ("fr", 1, -1), ("rl", -1, 1), ("rr", -1, -1)]}
        th.update({n: (p, f, ATT_N) for n, (p, f) in NOZZLES.items() if n not in BOTTOM})
        return th
    th = {n: (p, f, ATT_N) for n, (p, f) in NOZZLES.items()}
    th["main"] = ((0, 0, -bz), (0, 0, 1), MAIN_N)       # 바닥 중앙, 위로 밀기
    th["brake"] = ((bx, 0, 0), (-1, 0, 0), BRAKE_N)     # 앞면, 뒤로 밀기 (가스는 앞으로)
    return th


CAR_WHEEL_X, CAR_WHEEL_R = 0.14, 0.03    # 차 모양 바퀴 위치·반지름 (보기용) [m]


def car_xml():
    """아이오닉 6 느낌의 유선형 세단 외형. 전부 class="car"(질량·충돌 없음)라 물리는 박스 그대로"""
    wz = -bz + CAR_WHEEL_R
    wheels = "".join(
        f'<geom type="cylinder" fromto="{sx * CAR_WHEEL_X} {sy * 0.082} {wz} {sx * CAR_WHEEL_X} {sy * 0.1} {wz}" '
        f'size="{CAR_WHEEL_R}" material="tire"/>'
        f'<geom type="cylinder" fromto="{sx * CAR_WHEEL_X} {sy * 0.099} {wz} {sx * CAR_WHEEL_X} {sy * 0.1015} {wz}" '
        f'size="{CAR_WHEEL_R * 0.72}" material="rim"/>'
        for sx in (1, -1) for sy in (1, -1))
    body = f"""
      <geom type="box" size="0.165 0.084 0.006" pos="0 0 {-bz + 0.008}" material="chassis"/>
      <geom type="box" size="0.15 0.093 0.02" pos="0 0 {-bz + 0.032}" material="paint"/>
      <geom type="ellipsoid" size="0.06 0.093 0.022" pos="0.14 0 {-bz + 0.034}" material="paint"/>
      <geom type="ellipsoid" size="0.065 0.093 0.026" pos="-0.135 0 {-bz + 0.036}" material="paint"/>
      <geom type="ellipsoid" size="0.2 0.092 0.03" pos="0 0 {-bz + 0.048}" material="paint"/>
      <geom type="ellipsoid" size="0.13 0.074 0.034" pos="-0.02 0 {-bz + 0.066}" material="glass"/>
      <geom type="box" size="0.006 0.009 0.004" pos="0.055 0.092 {-bz + 0.072}" material="glass"/>
      <geom type="box" size="0.006 0.009 0.004" pos="0.055 -0.092 {-bz + 0.072}" material="glass"/>
      <geom type="box" size="0.004 0.082 0.006" pos="0.195 0 {-bz + 0.046}" material="glass"/>
      <geom type="box" size="0.004 0.06 0.004" pos="0.195 0 {-bz + 0.022}" material="chassis"/>
      <geom type="box" size="0.006 0.088 0.006" pos="-0.193 0 {-bz + 0.054}" material="glass"/>"""
    return (body + wheels + lamps_xml()).replace("<geom ", '<geom class="car" ')


def lamps_xml():
    """아이오닉 6 느낌 픽셀 램프 + 빛 번짐 (스포트라이트는 MuJoCo 렌더러에서 바닥 전체를 어둡게 만들어 뺐음)"""
    zf, zr = -bz + 0.047, -bz + 0.054
    drl = (f'<geom type="box" size="0.002 0.074 0.0016" pos="0.1995 0 {zf}" material="headlamp"/>'
           f'<geom type="box" size="0.003 0.079 0.0042" pos="0.1955 0 {zf}" material="glow_w"/>')
    pix = "".join(f'<geom type="box" size="0.0015 0.0032 0.0022" pos="0.1995 {sy * (0.05 + 0.009 * i)} {zf - 0.0042}" '
                  f'material="headlamp"/>' for sy in (1, -1) for i in range(3))
    tail = "".join(f'<geom type="box" size="0.0015 0.0028 0.0028" pos="-0.1995 {y:.4f} {zr}" material="taillamp"/>'
                   for y in np.linspace(-0.08, 0.08, 19))
    tail += f'<geom type="box" size="0.003 0.086 0.0055" pos="-0.1955 0 {zr}" material="glow_r"/>'
    # 주광(햇빛): 차를 따라다니며(trackcom) 그림자 계산 범위가 항상 차 주변에 오도록
    sun = ('<light mode="trackcom" pos="-1 -1.5 4" dir="1 1.5 -4" directional="true" '
           'diffuse=".62 .6 .56" specular=".45 .45 .45"/>')
    return drl + pix + tail + sun


def build_xml(thrusters):
    sites = "".join(f'<site name="{n}" pos="{p[0]} {p[1]} {p[2]}"/>' for n, (p, _, _) in thrusters.items())
    # 노즐 모양 (보기용: 질량·충돌 없음 → 물리 결과에 영향 없음). 분사 방향(= -미는 방향)으로 짧은 금속 원통
    bells = "".join(
        f'<geom class="car" type="cylinder" fromto="{p[0] + f[0] * L} {p[1] + f[1] * L} {p[2] + f[2] * L} '
        f'{p[0] - f[0] * 0.004} {p[1] - f[1] * 0.004} {p[2] - f[2] * 0.004}" size="{0.007 + 0.0003 * F}" material="metal"/>'
        for p, f, F in thrusters.values() for L in [0.012 + 0.0004 * F])
    motors = "".join(f'<motor name="{n}" site="{n}" gear="{f[0] * F} {f[1] * F} {f[2] * F} 0 0 0" ctrlrange="0 1"/>'
                     for n, (_, f, F) in thrusters.items())
    return f"""
<mujoco model="cold_gas_lander">
  <option timestep="{DT}"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".86 .89 .93" rgb2=".38 .45 .53" width="512" height="3072"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".44 .45 .47" rgb2=".38 .39 .41" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="15 15" reflectance=".08"/>
    <material name="metal" rgba=".72 .74 .8 1" specular="1" shininess=".9" reflectance=".2"/>
    <material name="paint" rgba=".95 .96 .97 1" specular=".7" shininess=".8" reflectance=".08" emission=".08"/>
    <material name="glass" rgba=".05 .06 .08 1" specular="1" shininess="1" reflectance=".35"/>
    <material name="chassis" rgba=".12 .12 .13 1" specular=".3"/>
    <material name="tire" rgba=".05 .05 .05 1" specular=".1" shininess=".1"/>
    <material name="rim" rgba=".6 .62 .66 1" specular="1" shininess=".8" reflectance=".2"/>
    <material name="headlamp" rgba="1 1 1 1" emission="1"/>
    <material name="taillamp" rgba="1 .08 .06 1" emission="1"/>
    <material name="glow_w" rgba=".85 .93 1 .28" emission="1"/>
    <material name="glow_r" rgba="1 .1 .05 .3" emission="1"/>
  </asset>
  <default>
    <site type="sphere" size="0.005" rgba="0 0 0 0"/>
    <default class="car">
      <geom mass="0" contype="0" conaffinity="0"/>
    </default>
  </default>
  <visual>
    <headlight ambient=".42 .43 .45" diffuse=".38 .38 .38" specular=".15 .15 .15"/>
    <quality shadowsize="4096"/>
    <map fogstart="5" fogend="18" shadowclip="1.5"/>
    <rgba fog=".8 .84 .89 1"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <worldbody>
    <!-- 주광: 비스듬한 따뜻한 햇빛 + 부드러운 그림자 / 보조광: 반대편 차가운 하늘빛 -->
    <light pos="3 3 3" dir="-1 -1 -0.8" directional="true" diffuse=".32 .35 .4" specular=".1 .1 .1" castshadow="false"/>
    <geom name="floor" type="plane" size="30 6 .1" material="grid"/>
    <body name="box" pos="0 0 0.3">
      <freejoint/>
      <!-- 물리용 박스 (투명). 보이는 건 car_xml()의 차 외형 -->
      <geom type="box" size="{bx} {by} {bz}" mass="{MASS + TANK_MASS}" group="3"/>
      <geom type="box" size="0.03 0.03 0.003" pos="{bx - 0.05} 0 {bz}" mass="0.001"
            contype="0" conaffinity="0" group="3"/>{car_xml()}
      <site name="imu" size="0.005"/>
      {sites}
      {bells}
    </body>
  </worldbody>
  <actuator>{motors}</actuator>
  <sensor><gyro name="gyro" site="imu"/></sensor>
</mujoco>"""


def wild_throw(rng):
    """막 던지기: 완전 무작위 자세(뒤집힘 포함) + 빠른 회전. 속도·높이는 기본 던지기와 같음"""
    th = random_throw(rng)
    q = rng.normal(size=4)
    th["quat"] = q / np.linalg.norm(q)
    th["w"] = np.radians(rng.uniform(-WILD_W, WILD_W, 3))
    return th


def potential(tilt, w, qvel, dist):
    """LunarLander식 길잡이: 속도·기울기·회전·이동이 작을수록 높음 (스텝 보상 = 증가분)"""
    return -(3.0 * abs(qvel[2]) + 1.0 * np.linalg.norm(qvel[0:2]) + 0.05 * tilt + 0.01 * w + 0.5 * dist)


def baseline_landing_action(env):
    """비교용: 방뱅 자세 제어만 (메인·제동 노즐 없음)"""
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    a[:8] = baseline_action(env.d, env.b)
    return a


class LanderEnv(gym.Env):
    def __init__(self, layout="main", p_wild=0.0):
        self.p_wild = p_wild       # 막 던지기 비율 (0~1)
        self.thrusters = make_thrusters(layout)
        self.names = list(self.thrusters)
        self.fmax = np.array([t[2] for t in self.thrusters.values()])
        self.m = mujoco.MjModel.from_xml_string(build_xml(self.thrusters))
        self.d = mujoco.MjData(self.m)
        self.b = self.m.body("box").id
        self.action_space = spaces.Box(0.0, 1.0, (len(self.names),), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (17,), np.float32)

    def _obs(self):
        d = self.d
        return np.concatenate([d.xmat[self.b], d.sensor("gyro").data, [d.qpos[2]], d.qvel[0:3],
                               [1 - self.gas / TANK_G]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.m, self.d)
        rng = self.np_random
        th = wild_throw(rng) if self.p_wild > 0 and rng.random() < self.p_wild else random_throw(rng)
        self.d.qpos[0:3] = [0, 0, th["z0"]]
        self.d.qpos[3:7] = th["quat"]
        self.d.qvel[0:3] = th["v"]
        self.d.qvel[3:6] = th["w"]
        mujoco.mj_forward(self.m, self.d)
        self.queue = deque([np.zeros(len(self.names))] * DELAY_STEPS, maxlen=DELAY_STEPS + 1)
        self.gas = 0.0
        tilt, _, w, _ = landing_eval(self.d.xmat[self.b], self.d.qvel)
        self.phi = potential(tilt, w, self.d.qvel, 0.0)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, 0.0, 1.0)
        gas0 = self.gas
        for _ in range(CTRL_EVERY):
            self.queue.append(a)
            u = self.queue[0] * (self.gas < TANK_G)
            self.d.ctrl[:] = u
            self.gas += (u @ self.fmax) * FLOW_PER_N * DT
            xmat, qvel = self.d.xmat[self.b].copy(), self.d.qvel.copy()
            mujoco.mj_step(self.m, self.d)
            if self.d.ncon > 0:
                break

        tilt, yaw, w, _ = landing_eval(xmat, qvel)
        dist = np.linalg.norm(self.d.qpos[0:2])
        phi = potential(tilt, w, qvel, dist)
        reward = phi - self.phi - 0.01 * (self.gas - gas0) - TIME_COST
        self.phi = phi
        landed = self.d.ncon > 0
        trunc = bool(not landed and (self.d.time > T_MAX or self.d.qpos[2] > Z_MAX))
        info = {}
        if landed:
            vz, vh = -qvel[2], np.linalg.norm(qvel[0:2])
            ok = bool(tilt < OK_TILT and w < OK_W and vz < OK_VZ and vh < OK_VH)
            reward += (10.0 * ok + 5.0 * np.exp(-(tilt / 8) ** 2) + 5.0 * np.exp(-vz ** 2)
                       + 3.0 * np.exp(-vh ** 2))
            info = dict(tilt=tilt, yaw=yaw, w=w, vz=vz, vh=vh, dist=dist, success=ok, gas=self.gas)
        elif trunc:
            reward -= ESCAPE_PENALTY
        return self._obs(), float(reward), landed, trunc, info
