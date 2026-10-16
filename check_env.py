"""MuJoCo 환경 점검"""
import os
import platform
import sys


def ok(msg): print(f"[OK]   {msg}")
def ng(msg): print(f"[FAIL] {msg}")
def info(msg): print(f"[INFO] {msg}")


# 1. 파이썬 / OS
info(f"Python {sys.version.split()[0]} | {platform.system()} {platform.machine()}")
(ok if sys.version_info >= (3, 10) else ng)("Python 3.10 이상")

# 2. 패키지
try:
    import mujoco
    import numpy as np
    ok(f"mujoco {mujoco.__version__} | numpy {np.__version__}")
except Exception as e:
    ng(f"패키지 import 실패: {e}")
    sys.exit(1)
try:
    import matplotlib
    ok(f"matplotlib {matplotlib.__version__}")
except Exception as e:
    ng(f"matplotlib 없음 (--headless 그래프 불가): {e}")

# 3. 물리 계산: 자유낙하 0.3초 후 높이 ≈ 1 - 0.5*9.81*0.3^2 = 0.559 m
XML = """<mujoco><option timestep="0.002"/><worldbody>
<geom type="plane" size="5 5 .1"/>
<body pos="0 0 1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
</worldbody></mujoco>"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
while d.time < 0.3 - 1e-9:
    mujoco.mj_step(m, d)
z = d.qpos[2]
(ok if abs(z - 0.559) < 0.01 else ng)(f"물리 계산 (z={z:.3f} m, 기대값 0.559)")

# 4. 오프스크린 렌더링
try:
    with mujoco.Renderer(m, 240, 320) as r:
        r.update_scene(d)
        px = r.render()
    ok(f"오프스크린 렌더링 {px.shape}")
except Exception as e:
    ng(f"렌더링 실패: {e}")
    info("Linux 서버라면 MUJOCO_GL=egl 또는 MUJOCO_GL=osmesa 환경변수를 설정")
info(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '(미설정)')}")

# 5. 뷰어
try:
    import mujoco.viewer  # noqa: F401
    ok("mujoco.viewer import")
    if platform.system() == "Darwin":
        info("macOS: 단독 뷰어는 python, launch_passive 시뮬은 mjpython으로 실행")
except Exception as e:
    ng(f"뷰어 import 실패: {e}")

# 6. v2 모델 컴파일 (같은 폴더에 rc_cold_gas_v2.py가 있을 때)
try:
    from rc_cold_gas_v2 import build_xml
    mv = mujoco.MjModel.from_xml_string(build_xml())
    car = mv.body("car").id
    mass = mv.body_subtreemass[car]
    ok(f"v2 모델 컴파일 | body {mv.nbody}, joint {mv.njnt}, actuator {mv.nu}, 총질량 {mass:.3f} kg")
    (ok if abs(mass - 2.122) < 0.01 else ng)("총질량 기대값 2.122 kg")
    (ok if mv.nu == 10 else ng)("액추에이터 10개 (노즐 4 + 조향 2 + 구동 4)")
except ModuleNotFoundError:
    info("rc_cold_gas_v2.py 없음 → 모델 검사 생략")
except Exception as e:
    ng(f"v2 모델 컴파일 실패: {e}")