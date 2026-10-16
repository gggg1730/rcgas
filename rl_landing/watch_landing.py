"""추력 착륙 RL 정책 보기 (mjpython으로 실행). 가스 기둥 길이 = 노즐 출력에 비례"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from box_env import CTRL_EVERY
from cold_gas_box import DT
from landing_env import LanderEnv

PLUME_PER_N = 0.035        # 1 N당 기둥 길이 [m] (시각화용)
PLUME_SEGS = 9             # 증기층 겹 수
CORE = np.array([1.0, 1.0, 1.0])       # 노즐 앞 코어: 흰색
VAPOR = np.array([0.62, 0.82, 1.0])    # 퍼지는 증기: 연한 하늘색
_rng = np.random.default_rng()


def _add(scn, gtype, size, pos, mat, rgba, emission=0.0):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), mat.flatten(),
                        np.asarray(rgba, np.float32))
    g.emission = emission
    g.category = mujoco.mjtCatBit.mjCAT_DECOR    # 장식 취급 → 가스가 바닥에 그림자를 드리우지 않음
    scn.ngeom += 1


def add_plumes(scn, env, sid, act):
    """냉가스 분사: 밝은 코어 + 점점 넓어지고 옅어지는 증기층 + 프레임마다 일렁임. 길이·굵기·진하기는 추력에 비례"""
    d, fmax = env.d, env.fmax
    for i, n in enumerate(env.names):
        t = float(act[i])
        if t < 0.05:
            continue
        F = fmax[i] * t
        p = d.site_xpos[sid[i]]
        f = d.site_xmat[sid[i]].reshape(3, 3) @ np.array(env.thrusters[n][1], dtype=float)
        out = -f                                         # 가스가 나가는 방향
        quat, mat = np.zeros(4), np.zeros(9)
        mujoco.mju_quatZ2Vec(quat, out)
        mujoco.mju_quat2Mat(mat, quat)
        mat = mat.reshape(3, 3)
        L = min(0.7, 0.05 + PLUME_PER_N * F) * (1 + 0.12 * _rng.uniform(-1, 1))
        r0 = 0.006 + 0.0005 * F
        dens = 0.35 + 0.65 * np.sqrt(t)
        # 코어: 노즐 바로 앞의 밝고 가는 흰 줄기
        _add(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [r0 * 0.55, r0 * 0.55, L * 0.08],
             p + out * L * 0.08, mat, [*CORE, 0.6 * dens], emission=0.5)
        # 증기층: 뒤로 갈수록 넓고 옅게
        seg = L / PLUME_SEGS
        for k in range(PLUME_SEGS):
            s = (k + 0.5) / PLUME_SEGS
            r = r0 * (1.0 + 5.0 * s ** 0.8) * (1 + 0.1 * _rng.uniform(-1, 1))
            color = (1 - s) * CORE + s * VAPOR
            alpha = dens * (0.26 * (1 - s) ** 1.6 + 0.025)
            _add(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [r, r, seg * 1.1],
                 p + out * (0.01 + L * s), mat, [*color, alpha], emission=0.12 * (1 - s))


def draw_plumes(viewer, env, sid, act):
    scn = viewer.user_scn
    scn.ngeom = 0
    add_plumes(scn, env, sid, act)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo_lander.zip")
    ap.add_argument("--slow", type=float, default=0.25)
    ap.add_argument("--wild", type=float, default=0.0, help="막 던지기 비율 (1이면 전부 막 던지기)")
    ap.add_argument("--layout", choices=["main", "roadster", "wheels"], default="main")
    a = ap.parse_args()

    env = LanderEnv(a.layout, a.wild)
    model = PPO.load(a.model, device="cpu")
    sid = [env.m.site(n).id for n in env.names]

    with mujoco.viewer.launch_passive(env.m, env.d) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = env.b
        v.cam.distance = 2.5
        ep = 0
        while v.is_running():
            obs, _ = env.reset(seed=10_000 + ep)
            done, info = False, {}
            while not done and v.is_running():
                act = model.predict(obs, deterministic=True)[0]
                obs, _, term, trunc, info = env.step(act)
                done = term or trunc
                with v.lock():
                    draw_plumes(v, env, sid, np.clip(act, 0, 1))
                v.sync()
                time.sleep(CTRL_EVERY * DT / a.slow)
            if info:
                print(f"#{ep} 기울기 {info['tilt']:.1f}° | 회전 {info['w']:.0f}°/s | 수직 {info['vz']:.2f} m/s | "
                      f"수평 {info['vh']:.2f} m/s | 이동 {info['dist']:.2f} m | "
                      f"{'성공' if info['success'] else '실패'} | 가스 {info['gas']:.0f} g")
            elif done:
                print(f"#{ep} 시간 초과")
            time.sleep(1.0)
            ep += 1
