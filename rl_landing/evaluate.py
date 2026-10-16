"""같은 무작위 던지기 N회로 무제어 / 방뱅 / RL 비교"""
import argparse
import os

import numpy as np

from box_env import ColdGasBoxEnv, baseline_action


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
    ok = np.mean([r["success"] for r in res]) * 100
    tilt = np.median([r["tilt"] for r in res])
    w = np.median([r["w"] for r in res])
    gas = np.mean([r["gas"] for r in res])
    print(f"{name:8s} | 성공률 {ok:5.1f}% | 기울기 중앙값 {tilt:5.1f}° | 회전 중앙값 {w:6.1f}°/s | 평균 가스 {gas:5.2f} g")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="ppo_box.zip")
    ap.add_argument("--thrust", type=float, default=1.0, help="노즐당 추력 [N]")
    a = ap.parse_args()

    env = ColdGasBoxEnv(a.thrust)
    policies = {
        "무제어": lambda o, e: np.zeros(8, dtype=np.float32),
        "방뱅": lambda o, e: baseline_action(e.d, e.b),
    }
    if os.path.exists(a.model):
        from stable_baselines3 import PPO
        model = PPO.load(a.model, device="cpu")
        policies["RL"] = lambda o, e: model.predict(o, deterministic=True)[0]
    else:
        print(f"{a.model} 없음 → RL 평가 생략")

    print(f"무작위 던지기 {a.n}회 (모든 방식이 같은 던지기)")
    for name, pol in policies.items():
        summarize(name, [rollout(env, s, pol) for s in range(a.n)])
