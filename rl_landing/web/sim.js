// 냉가스 로드스터 시뮬레이션 코어 — 물리(강체·노즐·밸브 지연·가스·충돌) + 게임 로직 + RL 정책 실행
// 화면과 분리되어 있어 브라우저와 Node(검증용) 양쪽에서 쓴다. game.py를 그대로 옮긴 것.
const SIM = (() => {
  // ===== 기체 (MuJoCo 모델과 같은 값) =====
  const DT = 0.001, CTRL_EVERY = 10, DELAY_STEPS = 15;
  const BX = 0.2, BY = 0.1, BZ = 0.05;
  const M = 2.801, GRAV = 9.81;
  const INERTIA = [0.01166947, 0.03969196, 0.04668976];   // 몸체 x·y·z축
  const FLOW_PER_N = 1.3 / 0.5, TANK_G = 190.0;
  const WHEEL_X = BX - 0.03, WHEEL_Y = BY - 0.02;
  const NOZZLE_N = 20.0, SIDE_N = 2.0, YAW_ARM = 0.18;
  // 노즐: [위치(몸체), 미는 방향(몸체), 최대 추력 N]  순서 = fl fr rl rr yfl yfr yrl yrr
  const THRUSTERS = [
    [[WHEEL_X, WHEEL_Y, -BZ], [0, 0, 1], NOZZLE_N],
    [[WHEEL_X, -WHEEL_Y, -BZ], [0, 0, 1], NOZZLE_N],
    [[-WHEEL_X, WHEEL_Y, -BZ], [0, 0, 1], NOZZLE_N],
    [[-WHEEL_X, -WHEEL_Y, -BZ], [0, 0, 1], NOZZLE_N],
    [[BX - 0.02, -BY, 0], [0, 1, 0], SIDE_N],
    [[BX - 0.02, BY, 0], [0, -1, 0], SIDE_N],
    [[-BX + 0.02, -BY, 0], [0, 1, 0], SIDE_N],
    [[-BX + 0.02, BY, 0], [0, -1, 0], SIDE_N],
  ];

  // ===== 조종 보조·주행·플립 설정 (game.py와 같은 값) =====
  const rad = (d) => d * Math.PI / 180;
  const TILT_CMD = rad(25), YAW_RATE_CMD = 2.0, CLIMB = 1.5, SINK = 1.2;
  const KP = [0.8, 2.6], KD = [0.15, 0.5, 0.4], KV = 4.0, BRAKE_GAIN = 0.15;
  const TOUCH_VZ = 0.12, GLIDE_GAIN = 0.8, TOUCH_OFFSET = 0.06, SLOW_ZONE = 0.3;
  const READY_VH = 0.6, READY_TILT = rad(15), MIN_SINK_FRAC = 0.3;
  const HANDOFF_TILT = rad(25), HANDOFF_W = rad(120), KV_TOUCHDOWN = 8.0;
  const LOW_TILT_H = 0.4, SIDE_BRAKE = 3.0;
  const DRIVE_F = 25.0, DRIVE_DRAG = 4.0, PARK_BRAKE = 25.0, GRIP = 10.0, STEER_RATE = 2.2;
  const FLIP = 2 * Math.PI, MIN_AIR = 0.1;
  const AIRFLIP_MIN_H = { roll: 0.8, pitch: 1.0 }, AIRFLIP_FAKE_H = { roll: 0.25, pitch: 0.25 };
  const FRONT_MIN_Z = 0.35, NEXT_FLIP_MIN_H = 0.8;
  const SETTLE_W = rad(150), SETTLE_TILT = rad(20), SETTLE_MAX_T = 0.8, MAX_EXTRA_FLIPS = 1;
  const FLIP_NAMES = { 2: 'DOUBLE', 3: 'TRIPLE', 4: 'QUAD' };

  // ===== 충돌 (게임용 근사: 차체 표면 26점 vs 바닥·코스 상자, 스프링-댐퍼) =====
  const CONTACT_K = 30000, CONTACT_C = 300, FRICTION = 0.03;   // 차체 마찰은 MuJoCo 게임과 같이 낮게
  const PROBES = [];
  for (const sx of [-1, 0, 1]) for (const sy of [-1, 0, 1]) for (const sz of [-1, 0, 1])
    if (sx || sy || sz) PROBES.push([sx * BX, sy * BY, sz * BZ]);
  const rotY = (a) => [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]];
  const I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  // 코스: [이름, 중심, 반치수, 회전, 판(얇은 판이면 항상 윗면 방향으로 밀어냄)]  (game.py COURSE와 같음. 패드는 보기용만)
  const COURSE = [
    ['ramp', [3.8, 0, 0.17], [0.8, 0.6, 0.02], rotY(rad(-13)), true],
    ['deck1', [10, 0, 0.15], [0.6, 0.6, 0.15], I3, false],
    ['deck2', [12.5, 0, 0.3], [0.5, 0.5, 0.3], I3, false],
    ['wall', [15, 0, 0.25], [0.05, 1.5, 0.25], I3, false],
  ];
  const PAD = [17.5, 0];

  // ===== 작은 선형대수 =====
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);
  const mv = (R, v) => [dot(R[0], v), dot(R[1], v), dot(R[2], v)];
  const mtv = (R, v) => [R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
    R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2], R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2]];
  const mm = (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
  const tr = (A) => [0, 1, 2].map((i) => [A[0][i], A[1][i], A[2][i]]);
  const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const quatMat = ([w, x, y, z]) => [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
  const quatMul = (a, b) => [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
  const rotAxis = (axis, a) => {
    const c = Math.cos(a), s = Math.sin(a);
    return axis === 'x' ? [[1, 0, 0], [0, c, -s], [0, s, c]]
      : axis === 'y' ? [[c, 0, s], [0, 1, 0], [-s, 0, c]] : [[c, -s, 0], [s, c, 0], [0, 0, 1]];
  };

  // ===== RL 정책 (SB3 PPO MlpPolicy: tanh 128-128, 결정적 출력을 0~1로 자름) =====
  function makePolicy(p) {
    const layer = (W, b, x, act) => W.map((row, i) => {
      let s = b[i];
      for (let j = 0; j < row.length; j++) s += row[j] * x[j];
      return act ? Math.tanh(s) : s;
    });
    return (obs) => layer(p.W2, p.b2, layer(p.W1, p.b1, layer(p.W0, p.b0, obs, true), true), false)
      .map((a) => clip(a, 0, 1));
  }

  class Game {
    constructor(policies, rng = Math.random) {
      this.pol = { lander: makePolicy(policies.lander), roll: makePolicy(policies.flip),
        pitch: makePolicy(policies.frontflip) };
      this.rng = rng;
      this.infiniteGas = true;
      this.msg = ''; this.msgId = 0;
      this.reset();
    }

    // ---------- 상태 ----------
    reset() {
      this.pos = [0, 0, BZ]; this.quat = [1, 0, 0, 0]; this.vel = [0, 0, 0]; this.omega = [0, 0, 0];
      this.time = 0;
      this.queue = Array.from({ length: DELAY_STEPS }, () => new Array(8).fill(0));
      this.gas = 0; this.mode = 'drive'; this.act = new Array(8).fill(0); this.u = new Array(8).fill(0);
      this.sub = 0; this.touchMsg = ''; this.touchV = 0; this.ext = { f: [0, 0, 0], t: [0, 0, 0] };
      this.theta = [0, 0]; this.airborne = false; this.flipped = false; this.flipAxis = 'roll';
      this.flipCount = 0; this.flipsLeft = 0; this.flipT0 = 0; this.settleT0 = 0; this.tOff = 0;
      this.R = quatMat(this.quat); this.grounded = false; this.contacts();
    }

    say(text) { this.msg = text; this.msgId++; }
    gasLeft() { return this.infiniteGas ? Infinity : Math.max(0, TANK_G - this.gas); }
    tilt() { return Math.acos(clip(this.R[2][2], -1, 1)); }

    // 바로 아래 지면까지 COM 높이 (단상 위에서도 RL이 평지처럼 느끼게) — MuJoCo mj_ray와 같은 역할
    height() {
      const o = this.pos;
      let best = o[2] >= 0 ? o[2] : Infinity;
      for (const [, c, h, Rm] of COURSE) {
        const ol = mtv(Rm, sub(o, c)), dl = mtv(Rm, [0, 0, -1]);
        let t0 = -Infinity, t1 = Infinity;
        let miss = false;
        for (let k = 0; k < 3; k++) {
          if (Math.abs(dl[k]) < 1e-12) { if (Math.abs(ol[k]) > h[k]) miss = true; continue; }
          let a = (-h[k] - ol[k]) / dl[k], b = (h[k] - ol[k]) / dl[k];
          if (a > b) [a, b] = [b, a];
          t0 = Math.max(t0, a); t1 = Math.min(t1, b);
        }
        if (!miss && t1 >= Math.max(t0, 0) && t0 >= 0) best = Math.min(best, t0);
      }
      return best === Infinity ? o[2] : best;
    }

    landerObs() {
      const R = this.R;
      return [...R[0], ...R[1], ...R[2], ...this.omega, this.height(), ...this.vel, 1 - this.gas / TANK_G];
    }
    flipObs() {
      return [...this.landerObs(), this.theta[0] / FLIP, this.theta[1] / FLIP, +this.airborne, +this.flipped];
    }

    // ---------- 물리 ----------
    contacts() {
      // 바닥·코스 상자와의 침투를 스프링-댐퍼 힘으로 (월드 힘, 월드 토크 반환)
      const R = this.R, wW = mv(R, this.omega);
      let F = [0, 0, 0], T = [0, 0, 0], touching = false;
      for (const p of PROBES) {
        const r = mv(R, p), x = add(this.pos, r), v = add(this.vel, cross(wW, r));
        const hits = [];
        if (x[2] < 1e-4) hits.push([-x[2], [0, 0, 1]]);
        for (const [, c, h, Rm, plate] of COURSE) {
          const l = mtv(Rm, sub(x, c));
          const pen = [h[0] - Math.abs(l[0]), h[1] - Math.abs(l[1]), h[2] - Math.abs(l[2])];
          if (pen[0] > -1e-4 && pen[1] > -1e-4 && pen[2] > -1e-4) {
            const k = plate ? 2 : pen.indexOf(Math.min(...pen));   // 경사로 모서리에 걸리지 않고 위로 타고 오르게
            const nl = [0, 0, 0]; nl[k] = Math.sign(l[k]) || 1;
            hits.push([pen[k], mv(Rm, nl)]);
          }
        }
        for (const [pen, n] of hits) {
          touching = true;
          if (pen <= 0) continue;
          const vn = dot(v, n);
          const fn = Math.max(0, CONTACT_K * pen - CONTACT_C * vn);
          const vt = sub(v, scl(n, vn)), s = norm(vt);
          const ft = s > 1e-9 ? scl(vt, -Math.min(FRICTION * fn, 50 * s) / s) : [0, 0, 0];
          const f = add(scl(n, fn), ft);
          F = add(F, f); T = add(T, cross(r, f));
        }
      }
      this.grounded = touching;
      return [F, T];
    }

    step() {
      // MuJoCo Euler 적분과 같은 순서: 현재 상태로 힘 → 속도 갱신 → 새 속도로 위치·자세 갱신
      const R = this.R;
      const [Fc, Tc] = this.contacts();
      let Fb = [0, 0, 0], Tb = [0, 0, 0];
      for (let i = 0; i < 8; i++) {
        if (!this.u[i]) continue;
        const [p, d, Fmax] = THRUSTERS[i];
        const f = scl(d, Fmax * this.u[i]);
        Fb = add(Fb, f); Tb = add(Tb, cross(p, f));
      }
      const Fw = add(add(add(mv(R, Fb), Fc), this.ext.f), [0, 0, -M * GRAV]);
      const tau = add(Tb, mtv(R, add(Tc, this.ext.t)));
      const w = this.omega, Iw = [INERTIA[0] * w[0], INERTIA[1] * w[1], INERTIA[2] * w[2]];
      const rhs = sub(tau, cross(w, Iw));
      this.vel = add(this.vel, scl(Fw, DT / M));
      this.omega = [w[0] + DT * rhs[0] / INERTIA[0], w[1] + DT * rhs[1] / INERTIA[1], w[2] + DT * rhs[2] / INERTIA[2]];
      this.pos = add(this.pos, scl(this.vel, DT));
      const wn = norm(this.omega), ang = wn * DT;
      if (ang > 0) {
        const s = Math.sin(ang / 2) / wn;
        this.quat = quatMul(this.quat, [Math.cos(ang / 2), this.omega[0] * s, this.omega[1] * s, this.omega[2] * s]);
      }
      const qn = Math.hypot(...this.quat); this.quat = this.quat.map((c) => c / qn);
      this.R = quatMat(this.quat);
      this.time += DT;
    }

    // ---------- 조종 보조 ----------
    assist(keys) {
      const R = this.R, w = this.omega, v = this.vel;
      const psi = Math.atan2(R[1][0], R[0][0]);
      const h = this.height() - BZ;
      const tiltMax = TILT_CMD * clip(h / LOW_TILT_H, 0.3, 1.0);
      let pitchD = tiltMax * (keys.W - keys.S), rollD = tiltMax * (keys.D - keys.A);
      let side = SIDE_N * 2 * (keys.A - keys.D);
      const hold = this.mode === 'fly' || this.mode === 'auto_land';
      if (hold && !(keys.W || keys.S || keys.A || keys.D)) {
        const vf = v[0] * Math.cos(psi) + v[1] * Math.sin(psi);
        const vl = -v[0] * Math.sin(psi) + v[1] * Math.cos(psi);
        pitchD = clip(-BRAKE_GAIN * vf, -tiltMax, tiltMax);
        rollD = clip(BRAKE_GAIN * vl, -tiltMax, tiltMax);
        side = clip(-SIDE_BRAKE * vl, -2 * SIDE_N, 2 * SIDE_N);
      }
      const Rd = mm(mm(rotAxis('z', psi), rotAxis('y', pitchD)), rotAxis('x', rollD));
      const A = mm(tr(Rd), R), B = mm(tr(R), Rd);
      const e = [0.5 * (A[2][1] - B[2][1]), 0.5 * (A[0][2] - B[0][2])];
      const rD = YAW_RATE_CMD * (keys.Q - keys.E);
      const tau = [-KP[0] * e[0] - KD[0] * w[0], -KP[1] * e[1] - KD[1] * w[1], -KD[2] * (w[2] - rD)];

      let collective = 0;
      if (hold && R[2][2] > 0.2) {
        let vzD = CLIMB * keys.SPACE;
        if (keys.C && !keys.SPACE) {
          // 착지: 멈추지 않고 계속 내려오면서 수평·감속을 함께 맞춤 (절대 다시 올라가지 않음)
          const vh = Math.hypot(v[0], v[1]);
          const ready = clip(1 - vh / READY_VH, 0, 1) * clip(1 - this.tilt() / READY_TILT, 0, 1);
          const glide = clip(TOUCH_VZ + GLIDE_GAIN * Math.max(0, h - TOUCH_OFFSET), TOUCH_VZ, SINK);
          vzD = -glide * (h > SLOW_ZONE ? 1 : MIN_SINK_FRAC + (1 - MIN_SINK_FRAC) * ready);
        }
        const kv = (this.mode === 'auto_land' || keys.C) ? KV_TOUCHDOWN : KV;
        const fz = M * (GRAV + kv * (vzD - v[2]));
        collective = clip(fz / Math.max(R[2][2], 0.5), 0, 4 * NOZZLE_N);
      }
      const sx = [1, 1, -1, -1], sy = [1, -1, 1, -1];
      const f = sx.map((_, i) => collective / 4 + tau[0] * sy[i] / (4 * WHEEL_Y) - tau[1] * sx[i] / (4 * WHEEL_X));
      const lift = Math.max(0, -Math.min(...f));
      const act = f.map((x) => clip((x + lift) / NOZZLE_N, 0, 1));
      if (this.mode === 'ballistic') side = 0;
      const ff = clip((side + tau[2] / YAW_ARM) / 2, -SIDE_N, SIDE_N);
      const fr = clip((side - tau[2] / YAW_ARM) / 2, -SIDE_N, SIDE_N);
      act.push(Math.max(ff, 0) / SIDE_N, Math.max(-ff, 0) / SIDE_N, Math.max(fr, 0) / SIDE_N, Math.max(-fr, 0) / SIDE_N);
      return act;
    }

    drive(keys) {
      // 땅 위 아케이드 주행: 전진력 + 옆미끄럼 억제 + 조향 토크 (월드 좌표 외력)
      this.ext = { f: [0, 0, 0], t: [0, 0, 0] };
      const R = this.R;
      if (this.mode !== 'drive' || !this.grounded || R[2][2] < 0.8) return;
      const psi = Math.atan2(R[1][0], R[0][0]);
      const h = [Math.cos(psi), Math.sin(psi), 0], n = [-Math.sin(psi), Math.cos(psi), 0];
      const vf = dot(this.vel, h), vl = dot(this.vel, n);
      const drag = (keys.W || keys.S) ? DRIVE_DRAG : PARK_BRAKE;
      const F = sub(scl(h, DRIVE_F * (keys.W - keys.S) - drag * vf), scl(n, M * GRIP * vl));
      const rD = STEER_RATE * (keys.A - keys.D) * clip(vf / 1.5, -1, 1) + YAW_RATE_CMD * (keys.Q - keys.E);
      this.ext = { f: F, t: [0, 0, 0.6 * (rD - this.omega[2])] };   // game.py와 같이 몸체 z 회전속도 사용
    }

    // ---------- 제어: 10 ms마다 판단, 물리는 1 ms씩 ----------
    tick(keys) {
      if (this.sub === 0) this.decide(keys);
      const end = this.physicsStep(keys);
      this.sub++;
      if (end || this.sub >= CTRL_EVERY) { this.sub = 0; this.updateMode(keys); return true; }
      return false;
    }
    control(keys) { while (!this.tick(keys)); }

    decide(keys) {
      let act;
      if (this.mode === 'rl_land') act = this.pol.lander(this.landerObs());
      else if (this.mode === 'rl_flip') act = this.pol[this.flipAxis](this.flipObs());
      else if (this.mode === 'rl_airflip') {
        const obs = this.flipObs(); obs[12] = AIRFLIP_FAKE_H[this.flipAxis];
        act = this.pol[this.flipAxis](obs);
      } else if (this.mode === 'drive') {
        act = [0, 0, 0, 0, keys.Q, keys.E, keys.E, keys.Q];   // 땅 위 Q/E: 요 노즐로 제자리 회전
      } else {
        if (this.mode === 'auto_land') keys = { W: 0, S: 0, A: 0, D: 0, Q: 0, E: 0, SPACE: 0, C: 1 };
        act = this.assist(keys);
      }
      this.act = act.map((a) => clip(a, 0, 1));
    }

    physicsStep(keys) {
      this.drive(keys);
      this.queue.push(this.act);
      const head = this.queue.shift();
      this.u = this.gasLeft() > 0 ? head.slice() : new Array(8).fill(0);
      if (!this.infiniteGas) this.gas += this.u.reduce((s, x, i) => s + x * THRUSTERS[i][2], 0) * FLOW_PER_N * DT;
      const vzBefore = this.vel[2];
      this.step();
      if (!this.grounded) this.touchV = -vzBefore;
      const R22 = this.R[2][2];
      if (this.mode === 'rl_airflip' && this.flipped) {
        const calm = norm(this.omega) < SETTLE_W && R22 > Math.cos(SETTLE_TILT);
        if (calm || this.time - this.settleT0 > SETTLE_MAX_T) {
          const n = this.flipCount;
          this.mode = 'fly';
          this.say(n > 1 ? `${FLIP_NAMES[n] || 'x' + n} AIR FLIP!` : 'AIR FLIP!');
          return true;
        }
      } else if (this.mode === 'rl_airflip') {
        this.theta[0] += this.omega[0] * DT; this.theta[1] += this.omega[1] * DT;
        if (this.flipProgress() >= FLIP) {
          this.flipCount++;
          if (this.flipsLeft > 0 && this.height() - BZ >= NEXT_FLIP_MIN_H) {
            this.flipsLeft--; this.theta = [0, 0]; this.flipT0 = this.time;
          } else if (this.flipAxis === 'pitch') {
            const n = this.flipCount;
            this.mode = 'fly';
            this.say(n > 1 ? `${FLIP_NAMES[n] || 'x' + n} AIR FRONT FLIP!` : 'AIR FRONT FLIP!');
            return true;
          } else { this.flipped = true; this.settleT0 = this.time; }
        }
      }
      if (this.mode === 'rl_flip') {
        if (this.flipAxis !== 'pitch' || this.height() >= FRONT_MIN_Z) {
          this.theta[0] += this.omega[0] * DT; this.theta[1] += this.omega[1] * DT;
        }
        if (this.flipProgress() >= FLIP) this.flipped = true;
        if (!this.grounded && !this.airborne) { this.airborne = true; this.tOff = this.time; }
        if (this.airborne && this.grounded) {
          if (this.time - this.tOff > MIN_AIR) {
            this.mode = 'drive';
            this.say(this.flipped ? this.flipLabel() : 'landed (no flip)');
            return true;
          }
          this.airborne = false;
        }
      }
      return false;
    }

    flipLabel() {
      const laps = Math.floor(Math.max(Math.abs(this.theta[0]), Math.abs(this.theta[1])) / FLIP + 0.1);
      const kind = this.flipAxis === 'pitch' ? 'FRONT FLIP' : 'FLIP';
      return `${FLIP_NAMES[laps] ? FLIP_NAMES[laps] + ' ' : ''}${kind} LANDED`;
    }

    handoffReady() {
      // 땅 근처 + 어느 정도 수평 + 회전이 잦아듦 → 마지막 착지는 자동 착지에 맡김
      const vz = this.vel[2];
      const near = this.height() - BZ < 0.35 + Math.max(0, -vz) ** 2 / 20;
      const calm = this.R[2][2] > Math.cos(HANDOFF_TILT) && norm(this.omega) < HANDOFF_W;
      return near && calm && !this.grounded;
    }

    updateMode(keys) {
      if (this.mode === 'rl_land' && this.handoffReady()) {
        this.mode = 'auto_land'; this.touchMsg = 'RL LANDED';
      } else if (this.mode === 'ballistic' && this.vel[2] < -0.3 && this.handoffReady()) {
        this.mode = 'auto_land'; this.touchMsg = 'LANDED';
      } else if (this.mode === 'rl_flip' && this.flipped && this.airborne && this.handoffReady()) {
        this.mode = 'auto_land'; this.touchMsg = this.flipLabel();
      }
      if (this.mode === 'auto_land' && this.grounded) {
        this.mode = 'drive'; this.say(`${this.touchMsg}  (${this.touchV.toFixed(2)} m/s)`);
      } else if (this.mode === 'rl_land' && this.grounded) {
        this.mode = 'drive'; this.say(`RL LANDED  tilt ${(this.tilt() * 180 / Math.PI).toFixed(1)} deg`);
      } else if ((this.mode === 'fly' || this.mode === 'ballistic') && this.grounded && !keys.SPACE) {
        this.mode = 'drive';
      } else if (this.mode === 'drive' && !this.grounded) {
        this.mode = 'ballistic';
      }
      if (this.mode === 'rl_flip' && this.time - this.flipT0 > 5.0) this.mode = 'ballistic';
      if (this.mode === 'rl_airflip' && (this.grounded || this.time - this.flipT0 > 2.0 + SETTLE_MAX_T)) {
        this.mode = this.grounded ? 'drive' : 'fly'; this.say('air flip failed');
      }
      if (this.pos[2] < -1 || Math.abs(this.pos[1]) > 19 || Math.abs(this.pos[0]) > 59) {
        this.reset(); this.say('out of bounds - reset');
      }
    }

    // ---------- 플립·던지기 ----------
    flipProgress() {
      return this.flipAxis === 'pitch' ? Math.abs(this.theta[1])
        : Math.max(Math.abs(this.theta[0]), Math.abs(this.theta[1]));
    }

    startFlip(axis) {
      const name = axis === 'pitch' ? 'front flip' : 'flip';
      if (this.mode === 'rl_airflip') {
        if (this.flipAxis === 'pitch') this.say('front flip: single only');
        else if (this.flipsLeft < MAX_EXTRA_FLIPS && !this.flipped) {
          this.flipsLeft++;
          this.say(`${FLIP_NAMES[this.flipCount + this.flipsLeft + 1] || ''} ${name.toUpperCase()}!`);
        }
        return;
      }
      const groundOk = this.grounded && this.R[2][2] > 0.95 && norm(this.vel) < 0.5;
      const airOk = (this.mode === 'fly' || this.mode === 'ballistic') && this.height() - BZ >= AIRFLIP_MIN_H[axis];
      if (!(groundOk || airOk)) {
        this.say(!this.grounded ? `air ${name} needs ${AIRFLIP_MIN_H[axis].toFixed(1)} m+` : `${name}: stop on level ground first`);
        return;
      }
      this.flipAxis = axis;
      this.mode = groundOk ? 'rl_flip' : 'rl_airflip';
      this.flipT0 = this.time; this.theta = [0, 0];
      this.airborne = !groundOk; this.flipped = false; this.flipCount = 0; this.flipsLeft = 0;
      this.say(`RL ${groundOk ? 'jump' : 'air'} ${name}!`);
    }

    randomThrow() {
      // 막 던지기 학습 범위(월드 +x로 3~7 m/s, 위로 1.5~3 m/s, 무작위 자세, 각 축 ±360°/s)
      const u = (a, b) => a + (b - a) * this.rng();
      const gauss = () => Math.sqrt(-2 * Math.log(1 - this.rng())) * Math.cos(2 * Math.PI * this.rng());
      const surface = this.pos[2] - this.height();
      this.pos[2] = surface + u(0.25, 0.45);
      const q = [gauss(), gauss(), gauss(), gauss()], qn = Math.hypot(...q);
      this.quat = q.map((c) => c / qn); this.R = quatMat(this.quat);
      this.vel = [u(3, 7), u(-0.5, 0.5), u(1.5, 3.0)];
      this.omega = [rad(u(-360, 360)), rad(u(-360, 360)), rad(u(-360, 360))];
      this.queue = Array.from({ length: DELAY_STEPS }, () => new Array(8).fill(0));
      this.mode = 'rl_land';
      this.say('THROWN!  RL catching...');
    }

    onKey(k) {
      if (k === 'R') { this.reset(); this.say('reset'); }
      else if (k === 'SPACE' && (this.mode === 'drive' || this.mode === 'ballistic')) this.mode = 'fly';
      else if (k === 'L') {
        if (this.grounded) this.say('L works in the air');
        else { this.mode = 'rl_land'; this.say('RL auto-landing'); }
      } else if (k === 'F' || k === 'G') this.startFlip(k === 'F' ? 'roll' : 'pitch');
      else if (k === 'X') this.randomThrow();
      else if (k === 'T') {
        this.infiniteGas = !this.infiniteGas; this.gas = 0;
        this.say(this.infiniteGas ? 'gas: infinite' : `gas: real tank ${TANK_G} g`);
      }
    }
  }

  return { Game, THRUSTERS, COURSE, PAD, BX, BY, BZ, TANK_G, DT, quatMat, makePolicy, FLIP };
})();
if (typeof module !== 'undefined') module.exports = SIM;
