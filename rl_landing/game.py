"""냉가스 로드스터 게임 모드 — 직접 조종 + RL 자동 착지 + RL 점프 플립 + 경사로·단상·벽 코스
실행: python game.py   (mjpython 아님. 전용 창을 직접 띄움)
      python game.py --speed 0.5   (0.5배속으로 시작)

조작
  땅 위    W/S 가속·후진   A/D 조향   Q/E 제자리 회전(요 노즐 분사)   → 경사로를 빠르게 타면 날아오름
  공중     W/S/A/D 기울여 이동 (놓으면 자동 수평)   Q/E 회전
  Space    떠오르기 / 상승     C  하강     (아무것도 안 누르면 그 높이에서 정지 비행)
  L        RL 자동 착지 (공중에서)          F  RL 옆구르기 플립 (땅: 점프 플립 / 공중 0.8 m+: 공중 플립 후 정지 비행)
  G        RL 앞구르기 플립 (F와 같은 방식)  공중 옆구르기 중 F 한 번 더 → 더블 플립
  X        랜덤 던지기: 뒤집고 돌리며 허공으로 내던짐 → RL이 받아서 착지
  T        가스 무한 ↔ 실제 탱크 190 g      R  처음부터     P  일시정지
  - / =    배속 내리기 / 올리기 (0.1 · 0.25 · 0.5 · 1 · 2배)
  1/2/3/4  카메라 (추격 / 옆 / 멀리 / 운전석 1인칭)   마우스 휠  줌      Esc  종료
  5 / 마우스 드래그  자유 시점: 왼쪽 드래그 = 차 주위로 돌리기, 오른쪽 드래그 = 시점 옮기기, 5 다시 = 차 중심으로
"""
import os
import time
from collections import deque

import glfw
import mujoco
import numpy as np
from stable_baselines3 import PPO

from box_env import CTRL_EVERY
from cold_gas_box import BOX, DELAY_STEPS, DT, FLOW_PER_N, MASS
from jumpflip_env import FLIP, MIN_AIR
from landing_env import TANK_G, TANK_MASS, WHEEL_X, WHEEL_Y, LanderEnv, build_xml, make_thrusters
from watch_landing import add_plumes

bx, by, bz = BOX
M = MASS + TANK_MASS
G = 9.81
NOZZLE_N = 20.0            # 바닥 노즐 1개 최대 추력 [N]
YAW_ARM = 0.18             # 요 노즐 팔 길이 [m]

# ===== 조종 보조 (자세 PD + 수직 속도 유지) =====
TILT_CMD = np.radians(25)  # W/S/A/D 최대 기울기
YAW_RATE_CMD = 2.0         # Q/E 회전 속도 [rad/s]
CLIMB, SINK = 1.5, 1.2     # Space/C 상승·하강 속도 [m/s]
KP = np.array([0.8, 2.6])  # 롤·피치 자세 이득
KD = np.array([0.15, 0.5, 0.4])  # 롤·피치·요 각속도 이득
KV = 4.0                   # 수직 속도 이득
BRAKE_GAIN = 0.15          # 키를 놓았을 때 수평 속도 1 m/s당 반대로 기울일 각 [rad]
TOUCH_VZ = 0.12            # 착지: 땅 바로 위 하강 속도 [m/s]
GLIDE_GAIN = 0.8           # 착지: 높이 1 m당 더해지는 하강 속도 [m/s] (높을수록 빨리 내려옴)
TOUCH_OFFSET = 0.06        # 이 높이 아래에서는 TOUCH_VZ로 (속도가 늦게 따라오는 만큼 미리 줄임) [m]
SLOW_ZONE = 0.3            # 이 높이 아래에서는 수평·정지가 덜 맞았으면 더 천천히 내려옴 [m]
READY_VH, READY_TILT = 0.6, np.radians(15)   # 이만큼 움직이거나 기울어 있으면 "덜 맞음"
MIN_SINK_FRAC = 0.3        # 덜 맞았을 때 최저 하강 비율 (0.12 × 0.3 ≈ 초당 4 cm, 멈추지는 않음)
HANDOFF_TILT, HANDOFF_W = np.radians(25), np.radians(120)  # RL·플립·점프가 땅 근처에서 이만큼 진정되면 자동 착지로 넘김
KV_TOUCHDOWN = 8.0         # 착지(C·자동 착지) 중 수직 속도 이득 — 목표 하강 속도를 빨리 따라가게
LOW_TILT_H = 0.4           # 이 높이 아래에서는 최대 기울기를 줄임 (모서리부터 닿지 않게) [m]
SIDE_N = 2.0               # 옆 노즐 1개 추력 [N] (앞뒤 같은 방향으로 쏘면 회전 없이 옆으로 밂)
SIDE_BRAKE = 3.0           # 키를 놓았을 때 옆 미끄럼 1 m/s당 옆 노즐 힘 [N]

# ===== 주행 (땅 위, 아케이드식) =====
DRIVE_F, DRIVE_DRAG = 25.0, 4.0   # 가속력 [N], 속도 저항 → 최고 약 6 m/s
PARK_BRAKE = 25.0          # W/S를 안 누르면 주차 브레이크 [N per m/s] (낮춘 마찰 때문에 미끄러지지 않게)
GRIP = 10.0                # 옆미끄럼 억제
STEER_RATE = 2.2           # 최대 조향 회전 [rad/s]

# ===== 공중 플립 (점프 플립 정책 재사용) =====
AIRFLIP_MIN_H = {"roll": 0.8, "pitch": 1.0}   # 공중 플립 최소 높이: 도는 동안 옆 약 0.6 m, 앞 약 0.9 m 떨어짐 [m]
AIRFLIP_FAKE_H = {"roll": 0.25, "pitch": 0.25}  # 정책은 낮은 높이에서만 돌아 봤으므로 높이를 이 값으로 알려 줌 (시험: 0.25가 최선) [m]
FRONT_MIN_Z = 0.35         # 앞구르기 정책은 이 높이 이상에서만 회전을 세며 학습함 (학습과 똑같이 셈) [m]
NEXT_FLIP_MIN_H = 0.8      # 연속 플립: 한 바퀴 끝났을 때 이 높이 이상이어야 다음 바퀴 [m]
SETTLE_W, SETTLE_TILT = np.radians(150), np.radians(20)   # 플립 뒤 이만큼 진정되면 비행 보조에게 넘김
SETTLE_MAX_T = 0.8         # 진정 단계 최대 시간 [s]
MAX_EXTRA_FLIPS = 1        # 연타로 예약할 수 있는 추가 바퀴 (기존 정책 재사용으로는 더블까지 안정. 트리플은 전용 학습 필요)
FLIP_NAMES = {2: "DOUBLE", 3: "TRIPLE", 4: "QUAD"}
COCKPIT = '<camera name="cockpit" pos="0.07 0 0.078" xyaxes="0 -1 0 0 0 1" fovy="80"/>'   # 운전석 시점

# ===== 코스 (x 방향으로 진행) =====
COURSE = """
    <geom name="ramp" type="box" size="0.8 0.6 0.02" pos="3.8 0 0.17" euler="0 -13 0" material="ramp"/>
    <geom name="deck1" type="box" size="0.6 0.6 0.15" pos="10 0 0.15" material="deck"/>
    <geom name="deck2" type="box" size="0.5 0.5 0.3" pos="12.5 0 0.3" material="deck"/>
    <geom name="wall" type="box" size="0.05 1.5 0.25" pos="15 0 0.25" material="wall"/>
    <geom name="pad" type="cylinder" size="0.6 0.003" pos="17.5 0 0.003" material="pad"/>
    <geom name="pad_in" type="cylinder" size="0.35 0.004" pos="17.5 0 0.004" material="pad_in"
          contype="0" conaffinity="0"/>
"""
COURSE_MAT = """
    <material name="ramp" rgba=".95 .6 .1 1" specular=".3"/>
    <material name="deck" rgba=".55 .58 .62 1" specular=".2"/>
    <material name="wall" rgba=".85 .2 .2 1" specular=".2"/>
    <material name="pad" rgba=".95 .95 .95 1" emission=".2"/>
    <material name="pad_in" rgba=".1 .5 1 1" emission=".3"/>
"""


def game_xml(thrusters):
    xml = build_xml(thrusters)
    body = f'size="{bx} {by} {bz}" mass="{M}"'
    assert body in xml
    # 바퀴 굴림 흉내: 차체 마찰을 낮추고(우선순위 1), 구동·접지는 아래 drive()가 힘으로 처리
    xml = xml.replace(body, body + ' priority="1" friction="0.03 0.005 0.0001"')
    xml = xml.replace('size="30 6 .1" material="grid"', 'size="60 20 .1" material="grid"')
    xml = xml.replace("  </asset>", COURSE_MAT + "  </asset>")
    xml = xml.replace("<freejoint/>", "<freejoint/>" + COCKPIT, 1)
    return xml.replace("  </worldbody>", COURSE + "  </worldbody>")


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


class Game(LanderEnv):
    """LanderEnv와 같은 기체·노즐(wheels)·지연·관측을 쓰되, 코스가 있고 끝나지 않는 세계"""

    def __init__(self):
        self.thrusters = make_thrusters("wheels")
        self.names = list(self.thrusters)
        self.fmax = np.array([t[2] for t in self.thrusters.values()])
        self.m = mujoco.MjModel.from_xml_string(game_xml(self.thrusters))
        self.d = mujoco.MjData(self.m)
        self.b = self.m.body("box").id
        self.sid = [self.m.site(n).id for n in self.names]
        self.lander = PPO.load("ppo_wheels_wild.zip", device="cpu")
        self.flippers = {"roll": PPO.load("ppo_jumpflip.zip", device="cpu")}   # 앞구르기는 G 누를 때 로드
        self.m.vis.map.znear = 0.0005                     # 운전석 카메라가 보닛을 잘라먹지 않게
        self.infinite_gas = True
        self.reset_game()

    # ---------- 상태 ----------
    def reset_game(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = [0, 0, bz]
        mujoco.mj_forward(self.m, self.d)
        self.queue = deque([np.zeros(8)] * DELAY_STEPS, maxlen=DELAY_STEPS + 1)
        self.gas = 0.0
        self.mode = "drive"          # drive / fly / ballistic / rl_land / rl_flip
        self.act = np.zeros(8)
        self.sub = 0                 # 제어 주기 안의 물리 스텝 번호 (0~9)
        self.touch_msg, self.touch_v = "", 0.0
        self.msg, self.msg_t = "", 0.0

    def say(self, text):
        self.msg, self.msg_t = text, time.time()

    def height(self):
        """바로 아래 지면까지 COM 높이 (단상 위에서도 RL이 평지처럼 느끼게)"""
        geomid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(self.m, self.d, self.d.xpos[self.b], np.array([0, 0, -1.0]),
                             None, 1, self.b, geomid)
        return dist if dist >= 0 else self.d.qpos[2]

    def lander_obs(self):
        d = self.d
        return np.concatenate([d.xmat[self.b], d.sensor("gyro").data, [self.height()], d.qvel[0:3],
                               [1 - self.gas / TANK_G]]).astype(np.float32)

    def flip_obs(self):
        return np.concatenate([self.lander_obs(), self.theta / FLIP,
                               [float(self.airborne), float(self.flipped)]]).astype(np.float32)

    @property
    def grounded(self):
        return self.d.ncon > 0

    def gas_left(self):
        return np.inf if self.infinite_gas else max(0.0, TANK_G - self.gas)

    # ---------- 조종 보조 ----------
    def assist(self, keys):
        """키 입력 → 목표 자세·수직 속도 → PD → 노즐 듀티 (X형 4개 + 요 4개)"""
        d = self.d
        R = d.xmat[self.b].reshape(3, 3)
        w = d.sensor("gyro").data
        psi = np.arctan2(R[1, 0], R[0, 0])
        h = self.height() - bz
        tilt_max = TILT_CMD * np.clip(h / LOW_TILT_H, 0.3, 1.0)   # 땅 근처에서는 덜 기울임
        pitch_d = tilt_max * (keys["W"] - keys["S"])          # +y 회전 = 기수 내림 = 전진
        roll_d = tilt_max * (keys["D"] - keys["A"])           # +x 회전 = 오른쪽 내림 = 우측 이동
        side = SIDE_N * 2 * (keys["A"] - keys["D"])           # 옆 노즐로도 직접 밂 (+ = 왼쪽)
        if self.mode in ("fly", "auto_land") and not (keys["W"] or keys["S"] or keys["A"] or keys["D"]):
            # 키를 놓으면 기울여서 수평 속도를 죽임 (DJI식 위치 유지) + 옆 미끄럼은 옆 노즐도 거듦
            v = d.qvel[0:3]
            vf = v @ np.array([np.cos(psi), np.sin(psi), 0])
            vl = v @ np.array([-np.sin(psi), np.cos(psi), 0])
            pitch_d = np.clip(-BRAKE_GAIN * vf, -tilt_max, tilt_max)
            roll_d = np.clip(BRAKE_GAIN * vl, -tilt_max, tilt_max)
            side = np.clip(-SIDE_BRAKE * vl, -2 * SIDE_N, 2 * SIDE_N)
        R_d = rot("z", psi) @ rot("y", pitch_d) @ rot("x", roll_d)
        e = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        r_d = YAW_RATE_CMD * (keys["Q"] - keys["E"])
        tau = np.array([-KP[0] * e[0] - KD[0] * w[0],
                        -KP[1] * e[1] - KD[1] * w[1],
                        -KD[2] * (w[2] - r_d)])

        collective = 0.0
        if self.mode in ("fly", "auto_land") and R[2, 2] > 0.2:
            vz_d = CLIMB * keys["SPACE"]
            if keys["C"] and not keys["SPACE"]:
                # 착지: 멈추지 않고 계속 내려오면서 수평·감속을 함께 맞춤 (절대 다시 올라가지 않음)
                #   높을수록 빨리, 땅에 가까울수록 TOUCH_VZ로. 땅 근처에서 덜 맞았으면 더 천천히 내려옴
                vh = np.linalg.norm(d.qvel[0:2])
                tilt = np.arccos(np.clip(R[2, 2], -1, 1))
                ready = np.clip(1 - vh / READY_VH, 0, 1) * np.clip(1 - tilt / READY_TILT, 0, 1)
                glide = np.clip(TOUCH_VZ + GLIDE_GAIN * max(0.0, h - TOUCH_OFFSET), TOUCH_VZ, SINK)
                vz_d = -glide * (1.0 if h > SLOW_ZONE else MIN_SINK_FRAC + (1 - MIN_SINK_FRAC) * ready)
            kv = KV_TOUCHDOWN if (self.mode == "auto_land" or keys["C"]) else KV
            fz = M * (G + kv * (vz_d - d.qvel[2]))
            collective = np.clip(fz / max(R[2, 2], 0.5), 0, 4 * NOZZLE_N)

        sx, sy = np.array([1, 1, -1, -1]), np.array([1, -1, 1, -1])   # fl fr rl rr
        f = collective / 4 + tau[0] * sy / (4 * WHEEL_Y) - tau[1] * sx / (4 * WHEEL_X)
        f += max(0.0, -f.min())                               # 음수 없이 토크 내기
        act = np.zeros(8)
        act[:4] = np.clip(f / NOZZLE_N, 0, 1)
        # 옆 노즐: 앞쪽 합력 Ff, 뒤쪽 합력 Fr (+ = 왼쪽으로 밂)
        #   옆으로 미는 힘 = Ff + Fr,  회전 토크 = 팔길이 × (Ff − Fr)
        if self.mode == "ballistic":
            side = 0.0                                        # 경사로 비행 중엔 자세만 (가스 아낌)
        ff = np.clip((side + tau[2] / YAW_ARM) / 2, -SIDE_N, SIDE_N)
        fr = np.clip((side - tau[2] / YAW_ARM) / 2, -SIDE_N, SIDE_N)
        act[4], act[5] = max(ff, 0) / SIDE_N, max(-ff, 0) / SIDE_N    # yfl(+y) / yfr(−y)
        act[6], act[7] = max(fr, 0) / SIDE_N, max(-fr, 0) / SIDE_N    # yrl(+y) / yrr(−y)
        return act

    def drive(self, keys):
        """땅 위 아케이드 주행: 전진력 + 옆미끄럼 억제 + 조향 토크 (월드 좌표 외력)"""
        R = self.d.xmat[self.b].reshape(3, 3)
        self.d.xfrc_applied[self.b] = 0
        if self.mode != "drive" or not self.grounded or R[2, 2] < 0.8:
            return
        psi = np.arctan2(R[1, 0], R[0, 0])
        h = np.array([np.cos(psi), np.sin(psi), 0])
        n = np.array([-np.sin(psi), np.cos(psi), 0])
        v = self.d.qvel[0:3]
        vf, vl = v @ h, v @ n
        drag = DRIVE_DRAG if (keys["W"] or keys["S"]) else PARK_BRAKE
        F = h * (DRIVE_F * (keys["W"] - keys["S"]) - drag * vf) - n * (M * GRIP * vl)
        r_d = (STEER_RATE * (keys["A"] - keys["D"]) * np.clip(vf / 1.5, -1, 1)
               + YAW_RATE_CMD * (keys["Q"] - keys["E"]))
        self.d.xfrc_applied[self.b, 0:3] = F
        self.d.xfrc_applied[self.b, 5] = 0.6 * (r_d - self.d.qvel[5])

    # ---------- 제어: 10 ms마다 판단, 물리는 1 ms씩 (배속을 낮춰도 부드럽게) ----------
    def control(self, keys):
        """제어 한 주기(10 ms)를 통째로 진행 (시험·스크립트용)"""
        while not self.tick(keys):
            pass

    def tick(self, keys):
        """물리 1 ms 진행. 주기 첫 스텝에서 판단하고, 주기가 끝나면 True"""
        if self.sub == 0:
            self.decide(keys)
        end = self.physics_step(keys)
        self.sub += 1
        if end or self.sub >= CTRL_EVERY:
            self.sub = 0
            self.update_mode(keys)
            return True
        return False

    def decide(self, keys):
        if self.mode == "rl_land":
            act = self.lander.predict(self.lander_obs(), deterministic=True)[0]
        elif self.mode == "rl_flip":
            act = self.flip_policy.predict(self.flip_obs(), deterministic=True)[0]
        elif self.mode == "rl_airflip":
            obs = self.flip_obs()
            obs[12] = AIRFLIP_FAKE_H[self.flip_axis]
            act = self.flip_policy.predict(obs, deterministic=True)[0]
        elif self.mode == "drive":
            act = np.zeros(8)                                 # 땅 위 Q/E: 요 노즐로 제자리 회전
            act[[4, 7]] = keys["Q"]                           # yfl·yrr → 왼쪽
            act[[5, 6]] = keys["E"]                           # yfr·yrl → 오른쪽
        else:
            if self.mode == "auto_land":                      # 자동 착지 = C를 누른 것과 같은 동작
                keys = {**dict.fromkeys(keys, 0.0), "C": 1.0}
            act = self.assist(keys)
        self.act = np.clip(act, 0, 1)

    def physics_step(self, keys):
        """물리 1 ms. 모드가 바뀌어 이번 주기를 끊어야 하면 True"""
        d = self.d
        self.drive(keys)
        self.queue.append(self.act)
        u = self.queue[0] * (self.gas_left() > 0)
        d.ctrl[:] = u
        if not self.infinite_gas:
            self.gas += (u @ self.fmax) * FLOW_PER_N * DT
        vz_before = d.qvel[2]
        mujoco.mj_step(self.m, d)
        if not self.grounded:
            self.touch_v = -vz_before
        if self.mode == "rl_airflip" and self.flipped:   # 진정 단계: 정책이 회전을 멈추고 수평을 잡게 둠
            R22 = d.xmat[self.b][8]
            calm = np.linalg.norm(d.qvel[3:6]) < SETTLE_W and R22 > np.cos(SETTLE_TILT)
            if calm or d.time - self.settle_t0 > SETTLE_MAX_T:
                n = self.flip_count
                self.mode = "fly"
                self.say(f"{FLIP_NAMES.get(n, f'x{n}')} AIR FLIP!" if n > 1 else "AIR FLIP!")
                return True
        elif self.mode == "rl_airflip":
            self.theta += d.qvel[3:5] * DT
            if self.flip_progress() >= FLIP:              # 한 바퀴 완료
                self.flip_count += 1
                if self.flips_left > 0 and self.height() - bz >= NEXT_FLIP_MIN_H:
                    self.flips_left -= 1                  # 연타 예약분 → 정책에게 "아직 안 돌았다"고 다시 시작
                    self.theta[:] = 0
                    self.flip_t0 = d.time
                elif self.flip_axis == "pitch":           # 앞구르기: 끝 무렵 이미 회전이 느려 비행 보조가 바로 받음
                    n = self.flip_count
                    self.mode = "fly"
                    self.say(f"{FLIP_NAMES.get(n, f'x{n}')} AIR FRONT FLIP!" if n > 1 else "AIR FRONT FLIP!")
                    return True
                else:                                     # 옆구르기: 아직 빨리 돌므로 정책이 진정시킨 뒤 넘김
                    self.flipped, self.settle_t0 = True, d.time
        if self.mode == "rl_flip":
            if self.flip_axis != "pitch" or self.height() >= FRONT_MIN_Z:
                self.theta += d.qvel[3:5] * DT
            if self.flip_progress() >= FLIP:
                self.flipped = True
            if not self.grounded and not self.airborne:
                self.airborne, self.t_off = True, d.time
            if self.airborne and self.grounded:
                if d.time - self.t_off > MIN_AIR:
                    self.mode = "drive"
                    laps = int(np.max(np.abs(self.theta)) / FLIP + 0.1)   # 정책이 스스로 두 바퀴 도는 경우가 많음
                    kind = "FRONT FLIP" if self.flip_axis == "pitch" else "FLIP"
                    self.say(f"{FLIP_NAMES.get(laps, '')} {kind} LANDED".strip() if self.flipped
                             else "landed (no flip)")
                    return True
                self.airborne = False
        return False

    def handoff_ready(self):
        """땅 근처 + 어느 정도 수평 + 회전이 잦아듦 → 마지막 착지는 자동 착지(12 cm에서 맞추고 천천히)에 맡김"""
        d = self.d
        vz = d.qvel[2]
        near = self.height() - bz < 0.35 + max(0.0, -vz) ** 2 / 20   # 빨리 떨어질수록 높은 곳에서 넘겨 멈출 여유를 줌
        calm = d.xmat[self.b][8] > np.cos(HANDOFF_TILT) and np.linalg.norm(d.qvel[3:6]) < HANDOFF_W
        return near and calm and not self.grounded

    def update_mode(self, keys):
        R = self.d.xmat[self.b].reshape(3, 3)
        if self.mode == "rl_land" and self.handoff_ready():
            self.mode, self.touch_msg = "auto_land", "RL LANDED"
        elif self.mode == "ballistic" and self.d.qvel[2] < -0.3 and self.handoff_ready():
            self.mode, self.touch_msg = "auto_land", "LANDED"      # 경사로 점프 후 떨어지는 중
        elif self.mode == "rl_flip" and self.flipped and self.airborne and self.handoff_ready():
            laps = int(np.max(np.abs(self.theta)) / FLIP + 0.1)
            kind = "FRONT FLIP" if self.flip_axis == "pitch" else "FLIP"
            self.mode, self.touch_msg = "auto_land", f"{FLIP_NAMES.get(laps, '')} {kind} LANDED".strip()
        if self.mode == "auto_land" and self.grounded:
            self.mode = "drive"
            self.say(f"{self.touch_msg}  ({self.touch_v:.2f} m/s)")
        elif self.mode == "rl_land" and self.grounded:         # 넘기기 전에 닿아 버린 경우
            self.mode = "drive"
            self.say(f"RL LANDED  tilt {np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))):.1f} deg")
        elif self.mode in ("fly", "ballistic") and self.grounded and not keys["SPACE"]:
            self.mode = "drive"
        elif self.mode == "drive" and not self.grounded:
            self.mode = "ballistic"                           # 경사로에서 날아오름 → 자세 보조만
        if self.mode == "rl_flip" and self.d.time - self.flip_t0 > 5.0:
            self.mode = "ballistic"
        if self.mode == "rl_airflip" and (self.grounded or self.d.time - self.flip_t0 > 2.0 + SETTLE_MAX_T):
            self.mode = "drive" if self.grounded else "fly"
            self.say("air flip failed")
        if self.d.qpos[2] < -1 or abs(self.d.qpos[1]) > 19 or abs(self.d.qpos[0]) > 59:
            self.reset_game()
            self.say("out of bounds - reset")

    # ---------- 플립 ----------
    def flip_progress(self):
        """한 바퀴 진행량: 옆구르기 정책은 어느 축이든(학습 때와 같게), 앞구르기는 피치축만"""
        return np.max(np.abs(self.theta[[1] if self.flip_axis == "pitch" else [0, 1]]))

    def start_flip(self, axis):
        name = "front flip" if axis == "pitch" else "flip"
        if axis not in self.flippers:                         # 최신 앞구르기 모델, 없으면 v1(공중에서는 충분)
            path = next((p for p in ("ppo_frontflip.zip", "ppo_frontflip_v1.zip") if os.path.exists(p)), None)
            if path is None:
                self.say("front flip model not found")
                return
            self.flippers[axis] = PPO.load(path, device="cpu")
        if self.mode == "rl_airflip":                         # 공중 플립 중 연타 → 추가 바퀴 예약
            if self.flip_axis == "pitch":
                self.say("front flip: single only")           # 앞구르기 더블은 시험에서 실패
            elif self.flips_left < MAX_EXTRA_FLIPS and not self.flipped:
                self.flips_left += 1
                self.say(f"{FLIP_NAMES.get(self.flip_count + self.flips_left + 1, '')} {name.upper()}!")
            return
        R = self.d.xmat[self.b].reshape(3, 3)
        ground_ok = self.grounded and R[2, 2] > 0.95 and np.linalg.norm(self.d.qvel[:3]) < 0.5
        air_ok = self.mode in ("fly", "ballistic") and self.height() - bz >= AIRFLIP_MIN_H[axis]
        if not (ground_ok or air_ok):
            self.say(f"air {name} needs {AIRFLIP_MIN_H[axis]:.1f} m+" if not self.grounded
                     else f"{name}: stop on level ground first")
            return
        self.flip_policy, self.flip_axis = self.flippers[axis], axis
        self.mode = "rl_flip" if ground_ok else "rl_airflip"
        self.flip_t0, self.theta = self.d.time, np.zeros(2)
        self.airborne, self.flipped, self.t_off = not ground_ok, False, None
        self.flip_count, self.flips_left = 0, 0
        self.say(f"RL {'jump' if ground_ok else 'air'} {name}!")

    def random_throw(self):
        """X: 뒤집고 돌리며 허공으로 내던짐 → RL 착지가 받음.
        속도는 막 던지기 학습 범위(월드 +x로 3~7 m/s, 위로 1.5~3 m/s)에 맞춤 — 벗어나면 정책이 처음 보는 상황"""
        rng = np.random.default_rng()
        d = self.d
        surface = d.qpos[2] - self.height()
        d.qpos[2] = surface + rng.uniform(0.25, 0.45)
        q = rng.normal(size=4)
        d.qpos[3:7] = q / np.linalg.norm(q)
        d.qvel[0:3] = [rng.uniform(3, 7), rng.uniform(-0.5, 0.5), rng.uniform(1.5, 3.0)]
        d.qvel[3:6] = np.radians(rng.uniform(-360, 360, 3))
        mujoco.mj_forward(self.m, d)
        self.queue = deque([np.zeros(8)] * DELAY_STEPS, maxlen=DELAY_STEPS + 1)
        self.mode = "rl_land"
        self.say("THROWN!  RL catching...")

    # ---------- 키 이벤트 ----------
    def on_key(self, key):
        if key == glfw.KEY_R:
            self.reset_game()
            self.say("reset")
        elif key == glfw.KEY_SPACE and self.mode in ("drive", "ballistic"):
            self.mode = "fly"
        elif key == glfw.KEY_L:
            if self.grounded:
                self.say("L works in the air")
            else:
                self.mode = "rl_land"
                self.say("RL auto-landing")
        elif key in (glfw.KEY_F, glfw.KEY_G):
            self.start_flip("roll" if key == glfw.KEY_F else "pitch")
        elif key == glfw.KEY_X:
            self.random_throw()
        elif key == glfw.KEY_T:
            self.infinite_gas = not self.infinite_gas
            self.gas = 0.0
            self.say("gas: infinite" if self.infinite_gas else f"gas: real tank {TANK_G:.0f} g")


# ======================= 창·렌더링 =======================
HELP = ("W/S  drive / tilt fwd-back\nA/D  steer / tilt sideways\nQ/E  yaw (ground & air)\nSpace  take off / climb\n"
        "C  descend\nL  RL auto-land (air)\nF  RL side flip (ground / air 0.8m+)\nG  RL front flip (air 1.0m+)\n"
        "F again in air  double flip\nX  random throw -> RL catch\nT  gas mode\nR  reset   P  pause\n-/=  slower / faster\n"
        "1/2/3/4  camera (4 = cockpit)\n5 / mouse drag  free camera\n  (L-drag orbit, R-drag pan)\nwheel  zoom   Esc  quit")
MODE_NAME = {"drive": "DRIVE", "fly": "FLY (assist)", "ballistic": "AIRBORNE (auto-level)",
             "rl_land": "RL AUTO-LAND", "rl_flip": "RL JUMP FLIP", "rl_airflip": "RL AIR FLIP",
             "auto_land": "AUTO TOUCHDOWN (level & slow)"}


FLIP_GAS = 100.0           # 플립 1회에 드는 가스 (평가 평균 약 99 g)


def draw_gas_bar(game, vp, con):
    """화면 아래 가운데 가스 막대: 초록 → 노랑 → 빨강, 플립에 필요한 양 눈금, 무한이면 파랑"""
    bw, bh = int(vp.width * 0.32), max(14, vp.height // 45)
    x0, y0 = (vp.width - bw) // 2, int(vp.height * 0.05)
    mujoco.mjr_rectangle(mujoco.MjrRect(x0 - 3, y0 - 3, bw + 6, bh + 6), 0.05, 0.05, 0.07, 0.75)
    if game.infinite_gas:
        frac, rgb, label = 1.0, (0.25, 0.55, 1.0), "GAS  INF   (T: real tank)"
    else:
        left = game.gas_left()
        frac = left / TANK_G
        rgb = (0.2, 0.85, 0.35) if frac > 0.6 else (0.95, 0.8, 0.15) if frac > 0.3 else (0.95, 0.25, 0.2)
        label = f"GAS  {left:.0f} / {TANK_G:.0f} g" + ("   flip needs ~100 g" if left < FLIP_GAS else "")
    if frac > 0:
        mujoco.mjr_rectangle(mujoco.MjrRect(x0, y0, int(bw * frac), bh), *rgb, 0.95)
    if not game.infinite_gas:                                 # 플립 필요량 눈금
        xm = x0 + int(bw * FLIP_GAS / TANK_G)
        mujoco.mjr_rectangle(mujoco.MjrRect(xm - 1, y0 - 3, 3, bh + 6), 1, 1, 1, 0.9)
    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                       mujoco.MjrRect(x0 - 3, y0 + bh + 4, bw + 6, 4 * bh), label, None, con)


SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0]   # - / = 키로 고르는 배속


def main(speed=1.0):
    game = Game()
    m, d = game.m, game.d
    if not glfw.init():
        raise SystemExit("glfw 초기화 실패")
    win = glfw.create_window(1280, 800, "RC Cold-Gas Roadster", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)

    cam, opt, scn = mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvScene(m, maxgeom=4000)
    pert = mujoco.MjvPerturb()
    con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    view = {"mode": 1, "dist": 2.2, "paused": False, "az": 0.0, "speed": speed, "carry": 0.0,
            "free_az": 0.0, "free_el": -15.0, "pan": np.zeros(3), "drag": None, "xy": (0.0, 0.0)}

    def enter_free():
        """지금 보던 각도 그대로 자유 시점으로 전환"""
        if view["mode"] != 5:
            view["free_az"], view["free_el"] = cam.azimuth, cam.elevation
            view["mode"] = 5

    def key_cb(window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key in (glfw.KEY_MINUS, glfw.KEY_EQUAL):        # 배속: 목록에서 한 칸씩
            i = int(np.argmin([abs(v - view["speed"]) for v in SPEEDS]))
            i = max(0, i - 1) if key == glfw.KEY_MINUS else min(len(SPEEDS) - 1, i + 1)
            view["speed"] = SPEEDS[i]
            game.say(f"speed x{view['speed']:g}")
        elif key == glfw.KEY_P:
            view["paused"] = not view["paused"]
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            view["mode"] = key - glfw.KEY_0
            view["dist"] = {1: 2.2, 2: 3.0, 3: 7.0, 4: 2.2}[view["mode"]]
        elif key == glfw.KEY_5:
            if view["mode"] == 5:
                view["pan"][:] = 0                            # 다시 누르면 차 중심으로
            enter_free()
        else:
            game.on_key(key)

    def scroll_cb(window, dx, dy):
        view["dist"] = float(np.clip(view["dist"] * 0.9 ** dy, 0.6, 30))

    def button_cb(window, button, action, mods):
        if action == glfw.PRESS and button in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            view["drag"], view["xy"] = button, glfw.get_cursor_pos(window)
            enter_free()
        elif action == glfw.RELEASE:
            view["drag"] = None

    def cursor_cb(window, x, y):
        if view["drag"] is None:
            return
        dx, dy = x - view["xy"][0], y - view["xy"][1]
        view["xy"] = (x, y)
        if view["drag"] == glfw.MOUSE_BUTTON_LEFT:            # 차 주위로 돌리기
            view["free_az"] -= 0.3 * dx
            view["free_el"] = float(np.clip(view["free_el"] - 0.3 * dy, -89, 89))
        else:                                                 # 화면 평면으로 시점 옮기기
            a = np.radians(view["free_az"])
            right = np.array([np.sin(a), -np.cos(a), 0.0])
            k = 0.0015 * view["dist"]
            view["pan"] += k * (-dx * right + dy * np.array([0, 0, 1.0]))

    glfw.set_key_callback(win, key_cb)
    glfw.set_scroll_callback(win, scroll_cb)
    glfw.set_mouse_button_callback(win, button_cb)
    glfw.set_cursor_pos_callback(win, cursor_cb)

    names = {"W": glfw.KEY_W, "S": glfw.KEY_S, "A": glfw.KEY_A, "D": glfw.KEY_D, "Q": glfw.KEY_Q,
             "E": glfw.KEY_E, "C": glfw.KEY_C, "SPACE": glfw.KEY_SPACE}
    last = time.perf_counter()
    while not glfw.window_should_close(win):
        now = time.perf_counter()
        dt, last = min(now - last, 0.05), now
        keys = {k: float(glfw.get_key(win, v) == glfw.PRESS) for k, v in names.items()}
        if not view["paused"]:
            view["carry"] += dt * view["speed"]             # 배속: 실제 1초에 시뮬레이션을 speed초 진행
            n = int(view["carry"] / DT)
            view["carry"] -= n * DT
            for _ in range(n):
                game.tick(keys)

        # 카메라: 1 추격(차 뒤) / 2 옆 / 3 멀리 높게 / 4 운전석 / 5 자유 시점
        R = d.xmat[game.b].reshape(3, 3)
        heading = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if view["mode"] == 4:
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = m.camera("cockpit").id
        elif view["mode"] == 1:
            diff = (heading - view["az"] + 180) % 360 - 180
            view["az"] += 0.08 * diff
            cam.azimuth, cam.elevation = view["az"], -14
        elif view["mode"] == 2:
            cam.azimuth, cam.elevation = 270, -8
        elif view["mode"] == 3:
            cam.azimuth, cam.elevation = 235, -28
        else:
            cam.azimuth, cam.elevation = view["free_az"], view["free_el"]
        cam.lookat[:] = d.xpos[game.b] + np.array([0, 0, 0.05]) + (view["pan"] if view["mode"] == 5 else 0)
        cam.distance = view["dist"]

        w, h = glfw.get_framebuffer_size(win)
        vp = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(m, d, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        add_plumes(scn, game, game.sid, game.act)
        mujoco.mjr_render(vp, scn, con)

        speed = np.linalg.norm(d.qvel[0:3])
        gas = "inf" if game.infinite_gas else f"{game.gas_left():.0f} / {TANK_G:.0f} g"
        tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        status = (f"{MODE_NAME[game.mode]}{'  [PAUSED]' if view['paused'] else ''}\n"
                  f"x{view['speed']:g}\n{speed:.1f} m/s\n{game.height() - bz:.2f} m\n{tilt:.0f} deg\n{gas}\n{d.qpos[0]:.1f} m")
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp,
                           "MODE\nTIME\nSPEED\nALT\nTILT\nGAS\nX", status, con)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, vp,
                           "CONTROLS", HELP, con)
        draw_gas_bar(game, vp, con)
        if game.msg and time.time() - game.msg_t < 2.5:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp,
                               game.msg, None, con)
        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=1.0, help="시작 배속 (예: 0.5 = 절반 속도)")
    main(ap.parse_args().speed)
