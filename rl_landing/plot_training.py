"""학습 곡선 그리기. --live면 로그를 계속 다시 읽어 실시간 갱신
예) python plot_training.py train_jumpflip.log --live
    python plot_training.py train_roadster_wild.log train_wheels_wild.log   (여러 개 비교)
"""
import argparse
import os
import re

import matplotlib.pyplot as plt

KEYS = {"ep_rew_mean": "평균 보상", "ep_len_mean": "에피소드 길이 [스텝]"}
plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def parse(path):
    """SB3 로그 표에서 (스텝, 값) 열을 뽑는다"""
    data = {k: [] for k in KEYS}
    cur = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.match(r"\|\s+(\w+)\s+\|\s+([-\d.e+]+)\s+\|", line)
            if m:
                cur[m.group(1)] = float(m.group(2))
            elif line.startswith("---") and "total_timesteps" in cur:
                for k in KEYS:
                    if k in cur:
                        data[k].append((cur["total_timesteps"], cur[k]))
                cur = {}
    return data


def draw(axes, paths):
    for ax in axes:
        ax.cla()
    for p in paths:
        data = parse(p)
        for ax, k in zip(axes, KEYS):
            if data[k]:
                x, y = zip(*data[k])
                ax.plot([v / 1e6 for v in x], y, label=os.path.basename(p))
    for ax, k in zip(axes, KEYS):
        ax.set_ylabel(KEYS[k])
        ax.grid(alpha=.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("학습 스텝 [백만]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--live", action="store_true", help="2초마다 다시 읽기 (창 닫으면 종료)")
    ap.add_argument("--save", help="PNG로 저장할 경로")
    a = ap.parse_args()

    fig, axes = plt.subplots(len(KEYS), 1, sharex=True, figsize=(8, 6))
    fig.suptitle(" / ".join(os.path.basename(p) for p in a.logs))
    draw(axes, a.logs)
    if a.save:
        fig.savefig(a.save, dpi=120, bbox_inches="tight")
    elif a.live:
        while plt.fignum_exists(fig.number):
            draw(axes, a.logs)
            plt.pause(2.0)
    else:
        plt.show()
