"""제자리 점프 플립: N회 평가 (매회 방향·노즐 추력 편차가 다름)"""
import argparse

import numpy as np
from stable_baselines3 import PPO

from evaluate_landing import rollout
from jumpflip_env import JumpFlipEnv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="ppo_jumpflip.zip")
    ap.add_argument("--flip-axis", choices=["any", "roll", "pitch"], default="any")
    ap.add_argument("--flip-min-z", type=float, default=0.0)
    ap.add_argument("--layout", choices=["main", "roadster", "wheels"], default="wheels")
    a = ap.parse_args()

    env = JumpFlipEnv(a.layout, a.flip_axis, a.flip_min_z)
    model = PPO.load(a.model, device="cpu")
    results = [rollout(env, s, lambda o, e: model.predict(o, deterministic=True)[0]) for s in range(a.n)]
    res = [r for r in results if r]
    med = lambda k: np.median([r[k] for r in res])
    axis = [("롤(x)" if abs(r["rot"][0]) > abs(r["rot"][1]) else "피치(y)") for r in res if r["flipped"]]
    print(f"{a.n}회 | 성공 {np.mean([r['success'] for r in res]) * 100:5.1f}% | "
          f"플립 완료 {np.mean([r['flipped'] for r in res]) * 100:5.1f}% | 시간초과 {a.n - len(res)}회")
    print(f"중앙값 | 최고 높이 {med('apex'):.2f} m | 체공 {med('air'):.2f} s | 기울기 {med('tilt'):.1f}° | "
          f"회전 {med('w'):.0f}°/s | 수직 {med('vz'):.2f} m/s | 수평 {med('vh'):.2f} m/s | "
          f"이동 {med('dist'):.2f} m | 가스 {np.mean([r['gas'] for r in res]):.0f} g")
    if axis:
        print("플립 축:", {k: axis.count(k) for k in set(axis)})
