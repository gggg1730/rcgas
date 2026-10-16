"""RC카 냉가스 자세제어 MuJoCo 시뮬레이션 v2
- 바퀴별 서스펜션, 앞바퀴 조향, 4WD 구동
- 냉가스 노즐 4개(피치/롤), CO2 카트리지·레귤레이터·밸브 질량 반영
- 잔량에 따른 추력 감소
"""
import argparse
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

# ===== 차량 치수 [m] — ※가정값: 실차 또는 MuSHR URDF 값으로 교체 =====
WHEELBASE = 0.32               # 앞뒤 바퀴 중심 거리
TRACK = 0.25                   # 좌우 바퀴 중심 거리
R_W = 0.055                    # 바퀴 반지름
W_HALF = 0.02                  # 바퀴 폭의 절반
CHASSIS = (0.22, 0.09, 0.03)   # 섀시 반치수 (x, y, z)
H_SUS = 0.03                   # 섀시 중심과 바퀴 중심의 높이차

# ===== 질량 [kg] — ※가정값: 저울로 실측해서 교체 =====
M_CHASSIS = 1.40               # 섀시 + 배터리 + 전장
M_WHEEL = 0.08
M_LINK = 0.02                  # 서스펜션·조향 링크
M_CO2_SHELL = 0.030            # 빈 카트리지
M_REG = 0.08                   # 레귤레이터
M_VALVE = 0.04                 # 밸브 1개

# ===== 서스펜션 (바퀴당) =====
K_SUS = 560.0                  # 스프링 [N/m]
C_SUS = 15.0                   # 댐퍼 [N·s/m]
SUS_TRAVEL = 0.03              # 스트로크 [m]

# ===== 주행 / 램프 =====
V0 = 7.0                       # 주행 속도 [m/s]
X_START = -2.0                 # 출발 위치 [m]
RAMP_DEG = 20
RAMP_LEN = 1.0

# ===== 냉가스 =====
TAU = 0.12                     # 선행 시간 [s]
DB = np.radians(2.0)           # 데드밴드 [rad]
DELAY_STEPS = 15               # 밸브 지연 [step]
GAS_CAP = 12.0                 # 카트리지 용량 [g]
FLOW_PER_N = 1.3 / 0.5         # 1 N당 가스 소모 [g/s]
LIQUID_FRAC = 0.15             # 이 잔량 비율 아래에서 추력 감소 시작
KICK_ROLL = np.radians(60)     # 이륙 순간 롤 교란 [rad/s]

# ===== 시뮬 =====
DT = 0.001
T_END = 3.5
AIR_CONFIRM = 20               # 공중 판정 디바운스 [step]

NOZZLES = ["front", "rear", "left", "right"]
CORNERS = ["fl", "fr", "rl", "rr"]


def build_xml():
    cx, cy, cz = CHASSIS
    zt = cz + 0.015  # 부품 장착 높이
    noz_xy = {"front": (cx - 0.02, 0), "rear": (-cx + 0.02, 0), "left": (0, cy), "right": (0, -cy)}
    link = f'<inertial pos="0 0 0" mass="{M_LINK}" diaginertia="1e-5 1e-5 1e-5"/>'

    def corner(n, x, y, front):
        w = (f'<body name="w_{n}"><joint name="w_{n}" type="hinge" axis="0 1 0" damping="1e-4"/>'
             f'<geom class="wheel"/></body>')
        if front:
            w = (f'<body name="s_{n}">{link}'
                 f'<joint name="steer_{n}" type="hinge" axis="0 0 1" range="-0.4 0.4" damping="0.03"/>{w}</body>')
        return (f'<body name="c_{n}" pos="{x} {y} {-H_SUS}">{link}'
                f'<joint name="sus_{n}" type="slide" axis="0 0 1" range="{-SUS_TRAVEL} {SUS_TRAVEL}" '
                f'stiffness="{K_SUS}" damping="{C_SUS}"/>{w}</body>')

    a, b = WHEELBASE / 2, TRACK / 2
    corners = (corner("fl", a, b, True) + corner("fr", a, -b, True)
               + corner("rl", -a, b, False) + corner("rr", -a, -b, False))
    valves = "".join(f'<geom type="box" size="0.01 0.01 0.012" pos="{x} {y} {zt}" mass="{M_VALVE}" rgba=".3 .3 .6 1"/>'
                     for x, y in noz_xy.values())
    sites = "".join(f'<site name="thr_{n}" pos="{x} {y} {zt + 0.02}"/>' for n, (x, y) in noz_xy.items())
    motors = "".join(f'<motor name="n_{n}" site="thr_{n}" gear="0 0 1 0 0 0" ctrlrange="0 1"/>' for n in NOZZLES)
    drives = "".join(f'<velocity name="d_{c}" joint="w_{c}" kv="0.02" forcerange="-0.6 0.6"/>' for c in CORNERS)

    th = np.radians(RAMP_DEG)
    rx = 0.5 * RAMP_LEN * np.cos(th) + 0.02 * np.sin(th)
    rz = 0.5 * RAMP_LEN * np.sin(th) - 0.02 * np.cos(th)

    return f"""
<mujoco model="rc_cold_gas_v2">
  <compiler autolimits="true"/>
  <option timestep="{DT}"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .2 .3" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="15 15" reflectance=".1"/>
  </asset>
  <default>
    <site type="sphere" size="0.012" rgba=".5 .5 .5 1"/>
    <default class="car">
      <geom contype="1" conaffinity="0" friction="1.2 0.005 0.0001"/>
      <default class="wheel">
        <geom type="cylinder" size="{R_W} {W_HALF}" euler="90 0 0" mass="{M_WHEEL}" rgba=".1 .1 .1 1"/>
      </default>
    </default>
  </default>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="30 6 .1" material="grid"/>
    <geom name="ramp" type="box" size="{RAMP_LEN / 2} 0.4 0.02" pos="{rx} 0 {rz}" euler="0 {-RAMP_DEG} 0" rgba=".7 .5 .3 1"/>
    <body name="car" childclass="car" pos="{X_START} 0 {R_W + H_SUS + 0.005}">
      <freejoint name="root"/>
      <geom name="chassis" type="box" size="{cx} {cy} {cz}" mass="{M_CHASSIS}" rgba=".2 .2 .2 1"/>
      <site name="imu" size="0.005"/>
      <body name="co2" pos="-0.06 0 {zt}" euler="0 90 0">
        <geom name="co2" type="cylinder" size="0.011 0.042" mass="{M_CO2_SHELL + GAS_CAP / 1000}" rgba=".8 .8 .85 1"/>
      </body>
      <geom name="regulator" type="box" size="0.02 0.015 0.015" pos="0.02 0 {zt}" mass="{M_REG}" rgba=".6 .6 .2 1"/>
      {valves}{sites}{corners}
    </body>
  </worldbody>
  <actuator>
    {motors}
    <position name="steer_fl" joint="steer_fl" kp="2" ctrlrange="-0.35 0.35"/>
    <position name="steer_fr" joint="steer_fr" kp="2" ctrlrange="-0.35 0.35"/>
    {drives}
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="acc" site="imu"/>
  </sensor>
</mujoco>"""


def build(thrust, v0):
    m = mujoco.MjModel.from_xml_string(build_xml())
    d = mujoco.MjData(m)
    for n in NOZZLES:
        m.actuator_gear[m.actuator("n_" + n).id, 2] = thrust
    d.qvel[0] = v0
    for c in CORNERS:
        d.qvel[m.joint("w_" + c).dofadr[0]] = v0 / R_W
    mujoco.mj_forward(m, d)
    return m, d


def touching(m, d, car):
    n = d.ncon
    geoms = np.r_[d.contact.geom1[:n], d.contact.geom2[:n]]
    return bool(np.any(m.body_rootid[m.geom_bodyid[geoms]] == car))


def attitude(d, car):
    """피치(기수 들림 +), 롤(왼쪽 들림 +)과 각속도"""
    R = d.xmat[car].reshape(3, 3)
    pitch = np.arcsin(np.clip(R[2, 0], -1, 1))
    roll = np.arcsin(np.clip(R[2, 1], -1, 1))
    w = d.sensor("gyro").data
    return pitch, roll, -w[1], w[0]


def bang(angle, rate):
    s = angle + TAU * rate
    return -1 if s > DB else (1 if s < -DB else 0)


def thrust_scale(gas_used):
    left = max(GAS_CAP - gas_used, 0) / GAS_CAP
    return 1.0 if left > LIQUID_FRAC else left / LIQUID_FRAC


def run(m, d, ctrl_on, thrust, v0, viewer=None, slow=1.0):
    car = m.body("car").id
    noz = [m.actuator("n_" + n).id for n in NOZZLES]
    drv = [m.actuator("d_" + c).id for c in CORNERS]
    sid = [m.site("thr_" + n).id for n in NOZZLES]
    sus = [m.joint("sus_" + c).qposadr[0] for c in CORNERS]

    queue = deque([np.zeros(4)] * DELAY_STEPS, maxlen=DELAY_STEPS + 1)
    gas, pulses, prev = 0.0, 0, np.zeros(4)
    air_cnt, flying, takeoff, landed, max_comp = 0, False, None, None, 0.0
    log, step, t0 = [], 0, time.perf_counter()
    d.ctrl[drv] = v0 / R_W

    while d.time < T_END:
        if viewer is not None and not viewer.is_running():
            break

        contact = touching(m, d, car)
        air_cnt = 0 if contact else air_cnt + 1
        if not flying and air_cnt >= AIR_CONFIRM and d.time > 0.2 and takeoff is None:
            flying, takeoff = True, d.time
            d.qvel[3] += KICK_ROLL            # 이륙 순간 롤 교란
        if flying and contact:
            flying = False
            p, r, _, _ = attitude(d, car)
            landed = (d.time, p, r)
        if takeoff is not None:
            max_comp = max(max_comp, d.qpos[sus].max())

        pitch, roll, pr, rr = attitude(d, car)
        cmd = np.zeros(4)
        if ctrl_on and flying:
            bp, br = bang(pitch, pr), bang(roll, rr)
            cmd[:] = [bp > 0, bp < 0, br > 0, br < 0]
        queue.append(cmd)
        valve = queue[0]

        sc = thrust_scale(gas)
        d.ctrl[noz] = valve * sc
        gas += valve.sum() * thrust * sc * FLOW_PER_N * DT
        pulses += int(np.sum((valve > 0) & (prev == 0)))
        prev = valve.copy()
        log.append([d.time, pitch, roll, flying, *valve])

        mujoco.mj_step(m, d)
        step += 1

        if viewer is not None and step % 10 == 0:
            for i in range(4):
                m.site_rgba[sid[i]] = [1, .35, .1, 1] if valve[i] else [.5, .5, .5, 1]
            viewer.sync()
            lag = t0 + d.time / slow - time.perf_counter()
            if lag > 0:
                time.sleep(lag)

    return dict(log=np.array(log), gas=gas, pulses=pulses, takeoff=takeoff,
                landed=landed, max_comp=max_comp)


def report(name, res):
    print(f"[{name}]")
    if res["takeoff"] is None:
        print("  이륙 감지 안 됨 (속도·램프 설정 확인)")
        return
    if res["landed"] is None:
        print("  착지 감지 안 됨 (T_END를 늘릴 것)")
    else:
        t, p, r = res["landed"]
        print(f"  체공 {(t - res['takeoff']) * 1000:.0f} ms | 착지 피치 {np.degrees(p):+.1f}° | 착지 롤 {np.degrees(r):+.1f}°")
    comp = res["max_comp"]
    warn = "  ← 서스펜션 바닥 침" if comp >= 0.95 * SUS_TRAVEL else ""
    print(f"  서스펜션 최대 압축 {comp * 1000:.1f} / {SUS_TRAVEL * 1000:.0f} mm{warn}")
    print(f"  가스 {res['gas']:.2f} / {GAS_CAP} g | 펄스 {res['pulses']}회")


def plot(on, off):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    for res, ls, lab in [(off, "--", "no control"), (on, "-", "control")]:
        L = res["log"]
        ax[0].plot(L[:, 0], np.degrees(L[:, 1]), ls, label=lab)
        ax[1].plot(L[:, 0], np.degrees(L[:, 2]), ls, label=lab)
    L = on["log"]
    for a in ax[:2]:
        a.fill_between(L[:, 0], -90, 90, where=L[:, 3] > 0, alpha=.1, color="gray")
        a.set_ylim(-60, 60)
        a.legend(loc="upper right")
    for i, n in enumerate(NOZZLES):
        ax[2].fill_between(L[:, 0], i, i + 0.8 * L[:, 4 + i], step="post")
    ax[0].set_ylabel("pitch [deg]")
    ax[1].set_ylabel("roll [deg]")
    ax[2].set_yticks(np.arange(4) + .4, NOZZLES)
    ax[2].set_xlabel("time [s]  (gray = airborne)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="뷰어 없이 제어/무제어 비교 후 그래프")
    ap.add_argument("--no-ctrl", action="store_true")
    ap.add_argument("--thrust", type=float, default=1.0, help="노즐당 추력 [N]")
    ap.add_argument("--speed", type=float, default=V0, help="주행 속도 [m/s]")
    ap.add_argument("--slow", type=float, default=0.25, help="재생 배속")
    a = ap.parse_args()

    if a.headless:
        on = run(*build(a.thrust, a.speed), True, a.thrust, a.speed)
        off = run(*build(a.thrust, a.speed), False, a.thrust, a.speed)
        report("제어", on)
        report("무제어", off)
        plot(on, off)
    else:
        m, d = build(a.thrust, a.speed)
        with mujoco.viewer.launch_passive(m, d) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = m.body("car").id
            v.cam.distance = 2.5
            res = run(m, d, not a.no_ctrl, a.thrust, a.speed, viewer=v, slow=a.slow)
        report("제어 OFF" if a.no_ctrl else "제어 ON", res)