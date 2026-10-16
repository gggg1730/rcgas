"""추력 착륙: 같은 무작위 던지기 N회로 무제어 / 방뱅(자세만) / RL 비교"""
import argparse
import os

import numpy as np

from landing_env import LanderEnv, baseline_landing_action


def rollout(env, seed, policy):
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, term, trunc, info = env.step(policy(obs, env))
        if term:
            return info
        if trunc:
            return None


def summarize(name, results):
    res = [r for r in results if r]
    med = lambda k: np.median([r[k] for r in res])
    ok = np.mean([r["success"] for r in res]) * 100
    print(f"{name:6s} | 성공 {ok:5.1f}% | 기울기 {med('tilt'):5.1f}° | 회전 {med('w'):5.1f}°/s | "
          f"수직 {med('vz'):4.2f} m/s | 수평 {med('vh'):4.2f} m/s | 이동 {med('dist'):4.2f} m | "
          f"가스 {np.mean([r['gas'] for r in res]):5.1f} g | 시간초과 {len(results) - len(res)}회")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="ppo_lander.zip")
    ap.add_argument("--wild", type=float, default=0.0, help="막 던지기 비율 (1이면 전부 막 던지기)")
    ap.add_argument("--layout", choices=["main", "roadster", "wheels"], default="main")
    a = ap.parse_args()

    env = LanderEnv(a.layout, a.wild)
    policies = {
        "무제어": lambda o, e: np.zeros(e.action_space.shape, dtype=np.float32),
        "방뱅": lambda o, e: baseline_landing_action(e),
    }
    if os.path.exists(a.model):
        from stable_baselines3 import PPO
        model = PPO.load(a.model, device="cpu")
        policies["RL"] = lambda o, e: model.predict(o, deterministic=True)[0]
    else:
        print(f"{a.model} 없음 → RL 평가 생략")

    print(f"무작위 던지기 {a.n}회 (중앙값, 모든 방식이 같은 던지기)")
    for name, pol in policies.items():
        summarize(name, [rollout(env, s, pol) for s in range(a.n)])
