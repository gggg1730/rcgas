"""PPO 학습"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from box_env import ColdGasBoxEnv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--out", default="ppo_box")
    ap.add_argument("--thrust", type=float, default=1.0, help="노즐당 추력 [N] (자세 제어 과제)")
    ap.add_argument("--task", choices=["attitude", "landing", "jumpflip"], default="attitude",
                    help="attitude: 자세만 / landing: 큰 탱크 추력 착륙 / jumpflip: 제자리 점프 플립 착지")
    ap.add_argument("--wild", type=float, default=0.0, help="막 던지기(뒤집힘·빠른 회전) 비율 0~1")
    ap.add_argument("--flip-axis", choices=["any", "roll", "pitch"], default="any",
                    help="jumpflip: 플립 축 (pitch = 앞구르기)")
    ap.add_argument("--flip-min-z", type=float, default=0.0, help="jumpflip: 이 높이 이상에서만 회전 인정 [m]")
    ap.add_argument("--layout", choices=["main", "roadster", "wheels"], default="main", help="착륙 노즐 배치")
    a = ap.parse_args()

    if a.task == "jumpflip":
        from jumpflip_env import JumpFlipEnv
        env = make_vec_env(JumpFlipEnv, n_envs=8, env_kwargs=dict(layout=a.layout, axis=a.flip_axis, min_z=a.flip_min_z), vec_env_cls=SubprocVecEnv)
    elif a.task == "landing":
        from landing_env import LanderEnv
        env = make_vec_env(LanderEnv, n_envs=8, env_kwargs=dict(layout=a.layout, p_wild=a.wild), vec_env_cls=SubprocVecEnv)
    else:
        env = make_vec_env(ColdGasBoxEnv, n_envs=8, env_kwargs=dict(thrust=a.thrust), vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, n_steps=512, batch_size=1024, learning_rate=3e-4,
                gamma=0.99, policy_kwargs=dict(net_arch=[128, 128]), device="cpu", verbose=1)
    # 25만 스텝마다 중간 모델 저장 → watch_*.py --follow 로 학습 중 동작 보기
    ckpt = CheckpointCallback(save_freq=250_000 // env.num_envs, save_path=f"checkpoints/{a.out}", name_prefix="step")
    model.learn(total_timesteps=a.steps, callback=ckpt)
    model.save(a.out)
    print(f"저장: {a.out}.zip")
