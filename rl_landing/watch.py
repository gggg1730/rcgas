"""학습된 RL 정책이 무작위 던지기를 착지시키는 모습 보기 (mjpython으로 실행)"""
import argparse
import time

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from box_env import CTRL_EVERY, ColdGasBoxEnv
from cold_gas_box import DT, NAMES, draw_plumes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo_box.zip")
    ap.add_argument("--slow", type=float, default=0.25)
    ap.add_argument("--thrust", type=float, default=1.0, help="노즐당 추력 [N]")
    a = ap.parse_args()

    env = ColdGasBoxEnv(a.thrust)
    model = PPO.load(a.model, device="cpu")
    sid = [env.m.site(n).id for n in NAMES]

    with mujoco.viewer.launch_passive(env.m, env.d) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = env.b
        v.cam.distance = 2.0
        ep = 0
        while v.is_running():
            obs, _ = env.reset(seed=10_000 + ep)
            done, info = False, {}
            while not done and v.is_running():
                act = model.predict(obs, deterministic=True)[0]
                obs, _, term, trunc, info = env.step(act)
                done = term or trunc
                with v.lock():
                    draw_plumes(v, env.d, sid, np.clip(act, 0, 1))
                v.sync()
                time.sleep(CTRL_EVERY * DT / a.slow)
            if info:
                print(f"#{ep} 기울기 {info['tilt']:.1f}° | 회전 {info['w']:.0f}°/s | "
                      f"{'성공' if info['success'] else '실패'} | 가스 {info['gas']:.2f} g")
            time.sleep(1.0)
            ep += 1
