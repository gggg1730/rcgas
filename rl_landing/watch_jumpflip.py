"""제자리 점프 플립 정책 보기 (mjpython으로 실행)"""
import argparse
import glob
import os
import time

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from box_env import CTRL_EVERY
from cold_gas_box import DT
from jumpflip_env import JumpFlipEnv
from watch_landing import draw_plumes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo_jumpflip.zip")
    ap.add_argument("--slow", type=float, default=0.25)
    ap.add_argument("--follow", help="체크포인트 폴더 (예: checkpoints/ppo_jumpflip). 매 에피소드 최신 모델로 교체")
    ap.add_argument("--layout", choices=["main", "roadster", "wheels"], default="wheels")
    a = ap.parse_args()

    env = JumpFlipEnv(a.layout)
    def latest():
        files = glob.glob(os.path.join(a.follow, "*.zip")) if a.follow else []
        return max(files, key=os.path.getmtime) if files else a.model

    path = latest()
    model = PPO.load(path, device="cpu")
    sid = [env.m.site(n).id for n in env.names]

    with mujoco.viewer.launch_passive(env.m, env.d) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = env.b
        v.cam.distance = 3.0
        ep = 0
        while v.is_running():
            if a.follow and latest() != path:
                path = latest()
                model = PPO.load(path, device="cpu")
                print(f"--- 모델 교체: {os.path.basename(path)}")
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
                print(f"#{ep} 최고 {info['apex']:.2f} m | 회전 x {info['rot'][0]:+.0f}° y {info['rot'][1]:+.0f}° | "
                      f"착지 기울기 {info['tilt']:.1f}° | 수직 {info['vz']:.2f} m/s | "
                      f"{'성공' if info['success'] else '실패'} | 가스 {info['gas']:.0f} g")
            elif done:
                print(f"#{ep} 시간 초과")
            time.sleep(1.0)
            ep += 1
