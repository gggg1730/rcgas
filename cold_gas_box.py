"""냉가스 박스 자세제어 MuJoCo 시뮬레이션 (피치/롤/요 3축) + 분사 시각화
박스를 비스듬히 던져 올리고, 착지 전까지 냉가스 노즐로 수평·정방향 자세를 맞춘다.
"""
import argparse
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

# ===== 박스 =====
BOX = (0.2, 0.1, 0.05)         # 반치수 [m]
MASS = 2.0                     # [kg]

# ===== 던지기 초기조건 =====
Z0 = 0.25                      # 시작 높이 [m]
V_FWD, V_UP = 6.0, 2.3         # 초기 속도 [m/s]
PITCH0 = np.radians(20)        # 기수 들림
KICK_ROLL = np.radians(60)     # 롤 교란 [rad/s]
KICK_YAW = np.radians(40)      # 요 교란 [rad/s]

# ===== 냉가스 =====
TAU = 0.12                     # 선행 시간 [s]
DB = np.radians(2.0)           # 데드밴드 [rad]
DELAY_STEPS = 15               # 밸브 지연 [step]
GAS_CAP = 12.0                 # 카트리지 용량 [g]
FLOW_PER_N = 1.3 / 0.5         # 1 N당 가스 소모 [g/s]

DT = 0.001
T_AFTER = 0.5                  # 착지 후 더 보여줄 시간 [s]
PLUME_LEN = 0.15               # 분사 기둥 길이 [m] (시각화용)

bx, by, bz = BOX
# 이름: (노즐 위치, 박스를 미는 방향) — 박스 좌표계
# 가스는 미는 방향의 반대로 나가므로, 노즐은 그쪽 면에 둔다
NOZZLES = {
    "front": ((bx - 0.02, 0, -bz), (0, 0, 1)),
    "rear":  ((-bx + 0.02, 0, -bz), (0, 0, 1)),
    "left":  ((0, by - 0.01, -bz), (0, 0, 1)),
    "right": ((0, -by + 0.01, -bz), (0, 0, 1)),
    "yfl":   ((bx - 0.02, -by, 0), (0, 1, 0)),    # 앞쪽을 왼쪽으로 밀기
    "yfr":   ((bx - 0.02, by, 0), (0, -1, 0)),    # 앞쪽을 오른쪽으로 밀기
    "yrl":   ((-bx + 0.02, -by, 0), (0, 1, 0)),   # 뒤쪽을 왼쪽으로 밀기
    "yrr":   ((-bx + 0.02, by, 0), (0, -1, 0)),   # 뒤쪽을 오른쪽으로 밀기
}
NAMES = list(NOZZLES)


def build_xml():
    sites = "".join(f'<site name="{n}" pos="{p[0]} {p[1]} {p[2]}"/>' for n, (p, _) in NOZZLES.items())
    motors = "".join(f'<motor name="{n}" site="{n}" gear="{f[0]} {f[1]} {f[2]} 0 0 0" ctrlrange="0 1"/>'
                     for n, (_, f) in NOZZLES.items())
    return f"""
<mujoco model="cold_gas_box">
  <option timestep="{DT}"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .2 .3" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="15 15" reflectance=".1"/>
  </asset>
  <default>
    <site type="sphere" size="0.012" rgba=".5 .5 .5 1"/>
  </default>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="30 6 .1" material="grid"/>
    <body name="box" pos="0 0 {Z0}">
      <freejoint/>
      <geom type="box" size="{bx} {by} {bz}" mass="{MASS}" rgba=".2 .2 .2 1"/>
      <geom type="box" size="0.03 0.03 0.003" pos="{bx - 0.05} 0 {bz}" mass="0.001"
            contype="0" conaffinity="0" rgba=".9 .3 .2 1"/>
      <site name="imu" size="0.005"/>
      {sites}
    </body>
  </worldbody>
  <actuator>{motors}</actuator>
  <sensor><gyro name="gyro" site="imu"/></sensor>
</mujoco>"""


def build(thrust):
    m = mujoco.MjModel.from_xml_string(build_xml())
    d = mujoco.MjData(m)
    m.actuator_gear[:, :3] *= thrust
    h = PITCH0 / 2
    d.qpos[3:7] = [np.cos(h), 0, -np.sin(h), 0]   # 기수 들림
    d.qvel[0:3] = [V_FWD, 0, V_UP]                # 선속도 (월드 좌표)
    d.qvel[3:6] = [KICK_ROLL, 0, KICK_YAW]        # 각속도 (박스 좌표)
    mujoco.mj_forward(m, d)
    return m, d


def attitude(d, b):
    """[피치(기수↑+), 롤(왼쪽↑+), 요(왼쪽 회전+)]와 각속도"""
    R = d.xmat[b].reshape(3, 3)
    ang = np.array([np.arcsin(np.clip(R[2, 0], -1, 1)),
                    np.arcsin(np.clip(R[2, 1], -1, 1)),
                    np.arctan2(R[1, 0], R[0, 0])])
    w = d.sensor("gyro").data
    return ang, np.array([-w[1], w[0], w[2]])


def bang(angle, rate):
    s = angle + TAU * rate
    return -1 if s > DB else (1 if s < -DB else 0)


def draw_plumes(viewer, d, sid, on):
    """켜진 노즐에서 추력 반대 방향으로 가스 기둥을 그린다 (on: 0~1 듀티, 길이 비례)"""
    scn = viewer.user_scn
    scn.ngeom = 0
    for i, n in enumerate(NAMES):
        if on[i] < 0.05 or scn.ngeom >= scn.maxgeom:
            continue
        p = d.site_xpos[sid[i]]
        f = d.site_xmat[sid[i]].reshape(3, 3) @ np.array(NOZZLES[n][1], dtype=float)
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.array([.75, .9, 1, .45], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, p, p - f * PLUME_LEN * on[i])
        scn.ngeom += 1


def run(m, d, ctrl_on, thrust, viewer=None, slow=1.0):
    b = m.body("box").id
    sid = [m.site(n).id for n in NAMES]
    queue = deque([np.zeros(8)] * DELAY_STEPS, maxlen=DELAY_STEPS + 1)
    gas, pulses, prev = 0.0, 0, np.zeros(8)
    landed, log, step, t0 = None, [], 0, time.perf_counter()
    vis = np.zeros(8, dtype=bool)

    while (landed is None or d.time < landed[0] + T_AFTER) and d.time < 5.0:
        if viewer is not None and not viewer.is_running():
            break
        ang, rate = attitude(d, b)
        if landed is None and d.ncon > 0:
            landed = (d.time, *ang)

        cmd = np.zeros(8)
        if ctrl_on and landed is None and gas < GAS_CAP:
            bp, br, byaw = (bang(a, r) for a, r in zip(ang, rate))
            cmd[:] = [bp > 0, bp < 0, br > 0, br < 0, byaw > 0, byaw < 0, byaw < 0, byaw > 0]
        queue.append(cmd)
        valve = queue[0]

        d.ctrl[:] = valve
        gas += valve.sum() * thrust * FLOW_PER_N * DT
        pulses += int(np.sum((valve > 0) & (prev == 0)))
        prev = valve.copy()
        vis |= valve > 0
        log.append([d.time, *ang, *valve])

        mujoco.mj_step(m, d)
        step += 1

        if viewer is not None and step % 10 == 0:
            with viewer.lock():
                for i in range(8):
                    m.site_rgba[sid[i]] = [1, .35, .1, 1] if vis[i] else [.5, .5, .5, 1]
                draw_plumes(viewer, d, sid, vis)
            viewer.sync()
            vis[:] = False
            lag = t0 + d.time / slow - time.perf_counter()
            if lag > 0:
                time.sleep(lag)

    return dict(log=np.array(log), gas=gas, pulses=pulses, landed=landed)


def report(name, res):
    print(f"[{name}]")
    if res["landed"] is None:
        print("  착지 전에 종료됨")
    else:
        t, p, r, y = res["landed"]
        print(f"  체공 {t * 1000:.0f} ms | 착지 피치 {np.degrees(p):+.1f}° | 롤 {np.degrees(r):+.1f}° | 요 {np.degrees(y):+.1f}°")
    print(f"  가스 {res['gas']:.2f} / {GAS_CAP} g | 펄스 {res['pulses']}회")


def plot(on, off):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(9, 9))
    labels = ["pitch [deg]", "roll [deg]", "yaw [deg]"]
    for res, ls, lab in [(off, "--", "no control"), (on, "-", "control")]:
        L = res["log"]
        for k in range(3):
            ax[k].plot(L[:, 0], np.degrees(L[:, 1 + k]), ls, label=lab)
    t_land = on["landed"][0] if on["landed"] else None
    for k in range(3):
        if t_land:
            ax[k].axvline(t_land, color="gray", lw=1)
        ax[k].set_ylabel(labels[k])
        ax[k].legend(loc="upper right")
    L = on["log"]
    for i, n in enumerate(NAMES):
        ax[3].fill_between(L[:, 0], i, i + 0.8 * L[:, 4 + i], step="post")
    ax[3].set_yticks(np.arange(8) + .4, NAMES)
    ax[3].set_xlabel("time [s]  (gray line = landing)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="뷰어 없이 제어/무제어 비교 후 그래프")
    ap.add_argument("--no-ctrl", action="store_true")
    ap.add_argument("--thrust", type=float, default=1.0, help="노즐당 추력 [N]")
    ap.add_argument("--slow", type=float, default=0.25, help="재생 배속")
    a = ap.parse_args()

    if a.headless:
        on = run(*build(a.thrust), True, a.thrust)
        off = run(*build(a.thrust), False, a.thrust)
        report("제어", on)
        report("무제어", off)
        plot(on, off)
    else:
        m, d = build(a.thrust)
        with mujoco.viewer.launch_passive(m, d) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = m.body("box").id
            v.cam.distance = 2.0
            res = run(m, d, not a.no_ctrl, a.thrust, viewer=v, slow=a.slow)
        report("제어 OFF" if a.no_ctrl else "제어 ON", res)