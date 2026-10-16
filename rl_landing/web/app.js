// 화면·입력: three.js 렌더링 (z축이 위), HUD, 키보드·마우스·터치. 시뮬레이션은 sim.js의 SIM.Game
(() => {
  const { Game, THRUSTERS, COURSE, PAD, BX, BY, BZ, TANK_G } = SIM;
  THREE.Object3D.DefaultUp.set(0, 0, 1);
  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ---------- 렌더러·장면 ----------
  const canvas = $('view');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;
  const scene = new THREE.Scene();
  const SKY_TOP = new THREE.Color(css('--sky-top')), SKY_HORIZON = new THREE.Color(css('--sky-horizon'));
  scene.fog = new THREE.Fog(SKY_HORIZON, 14, 60);
  {
    // 하늘: 위는 짙은 파랑, 지평선은 옅은 회청색 그라데이션
    const c = document.createElement('canvas'); c.width = 4; c.height = 256;
    const x = c.getContext('2d'), grad = x.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0, '#' + SKY_TOP.getHexString()); grad.addColorStop(1, '#' + SKY_HORIZON.getHexString());
    x.fillStyle = grad; x.fillRect(0, 0, 4, 256);
    const tex = new THREE.CanvasTexture(c); tex.encoding = THREE.sRGBEncoding;
    const sky = new THREE.Mesh(new THREE.SphereGeometry(200, 32, 16),
      new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide, fog: false, depthWrite: false }));
    sky.rotation.x = Math.PI / 2;
    scene.add(sky);
  }
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 400);
  camera.up.set(0, 0, 1);

  scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x5a5650, 0.75));
  const sun = new THREE.DirectionalLight(0xfff1dc, 1.55);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.1, far: 20 });
  sun.shadow.bias = -0.0004;
  scene.add(sun, sun.target);
  const fill = new THREE.DirectionalLight(0xbcd2ff, 0.35);
  fill.position.set(4, 4, 3);
  scene.add(fill);

  // ---------- 바닥·코스 ----------
  {
    const c = document.createElement('canvas'); c.width = c.height = 512;
    const x = c.getContext('2d');
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      x.fillStyle = (i + j) % 2 ? '#6d7177' : '#62666c'; x.fillRect(i * 256, j * 256, 256, 256);
    }
    x.strokeStyle = 'rgba(255,255,255,0.06)'; x.lineWidth = 2;
    for (let k = 0; k <= 512; k += 64) { x.beginPath(); x.moveTo(k, 0); x.lineTo(k, 512); x.moveTo(0, k); x.lineTo(512, k); x.stroke(); }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(60, 20);
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy(); tex.encoding = THREE.sRGBEncoding;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(120, 40),
      new THREE.MeshStandardMaterial({ map: tex, roughness: 0.92, metalness: 0 }));
    floor.receiveShadow = true;
    scene.add(floor);
  }
  const courseMat = {
    ramp: new THREE.MeshStandardMaterial({ color: 0xe0761a, roughness: 0.6 }),
    deck1: new THREE.MeshStandardMaterial({ color: 0x8c949e, roughness: 0.8 }),
    deck2: new THREE.MeshStandardMaterial({ color: 0x8c949e, roughness: 0.8 }),
    wall: new THREE.MeshStandardMaterial({ color: 0xd8382d, roughness: 0.7 }),
  };
  for (const [name, c, h, Rm] of COURSE) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(2 * h[0], 2 * h[1], 2 * h[2]), courseMat[name]);
    mesh.position.set(...c);
    mesh.setRotationFromMatrix(new THREE.Matrix4().set(...Rm[0], 0, ...Rm[1], 0, ...Rm[2], 0, 0, 0, 0, 1));
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
  }
  {
    // 목표 패드: 흰 원판 + 파란 안쪽 원 + 십자
    const pad = new THREE.Group(); pad.position.set(PAD[0], PAD[1], 0.003);
    const disc = (r, color, z) => {
      const m = new THREE.Mesh(new THREE.CircleGeometry(r, 48), new THREE.MeshStandardMaterial({ color, roughness: 0.5 }));
      m.position.z = z; m.receiveShadow = true; pad.add(m);
    };
    disc(0.6, 0xf3f4f6, 0); disc(0.35, 0x1d72d8, 0.001);
    for (const [w, hgt] of [[0.5, 0.04], [0.04, 0.5]]) {
      const bar = new THREE.Mesh(new THREE.PlaneGeometry(w, hgt), new THREE.MeshBasicMaterial({ color: 0xffffff }));
      bar.position.z = 0.002; pad.add(bar);
    }
    scene.add(pad);
  }
  {
    // 거리 표시: 출발점에서 2 m마다 코스 옆 바닥에 숫자
    const label = (text) => {
      const c = document.createElement('canvas'); c.width = 128; c.height = 64;
      const x = c.getContext('2d'); x.fillStyle = 'rgba(255,255,255,0.75)';
      x.font = '600 40px "IBM Plex Mono", monospace'; x.textAlign = 'center'; x.textBaseline = 'middle';
      x.fillText(text, 64, 32);
      return new THREE.CanvasTexture(c);
    };
    for (let d = 2; d <= 20; d += 2) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 0.25),
        new THREE.MeshBasicMaterial({ map: label(`${d} m`), transparent: true, depthWrite: false }));
      m.position.set(d, -0.9, 0.002);
      scene.add(m);
      const tick = new THREE.Mesh(new THREE.PlaneGeometry(0.03, 0.3), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 }));
      tick.position.set(d, -0.62, 0.002);
      scene.add(tick);
    }
  }

  // ---------- 차 (landing_env.car_xml과 같은 치수, 질량·충돌 없는 보기용) ----------
  const car = new THREE.Group();
  scene.add(car);
  const MAT = {
    paint: new THREE.MeshPhysicalMaterial({ color: 0xf2f3f5, roughness: 0.28, metalness: 0.05, clearcoat: 1, clearcoatRoughness: 0.12 }),
    glass: new THREE.MeshPhysicalMaterial({ color: 0x0c0f14, roughness: 0.08, metalness: 0.4, clearcoat: 1 }),
    chassis: new THREE.MeshStandardMaterial({ color: 0x1d1f23, roughness: 0.7 }),
    tire: new THREE.MeshStandardMaterial({ color: 0x111214, roughness: 0.9 }),
    rim: new THREE.MeshStandardMaterial({ color: 0xa8adb6, roughness: 0.25, metalness: 0.9 }),
    metal: new THREE.MeshStandardMaterial({ color: 0xbfc4cc, roughness: 0.3, metalness: 0.85 }),
    head: new THREE.MeshBasicMaterial({ color: 0xffffff }),
    tail: new THREE.MeshBasicMaterial({ color: 0xff1a12 }),
    glowW: new THREE.MeshBasicMaterial({ color: 0xd8ecff, transparent: true, opacity: 0.35, depthWrite: false }),
    glowR: new THREE.MeshBasicMaterial({ color: 0xff2a1a, transparent: true, opacity: 0.4, depthWrite: false }),
  };
  const z0 = -BZ;
  const part = (geo, mat, pos, scale) => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(...pos); if (scale) m.scale.set(...scale);
    m.castShadow = mat !== MAT.glowW && mat !== MAT.glowR && mat !== MAT.head && mat !== MAT.tail;
    m.receiveShadow = true;
    car.add(m); return m;
  };
  const box = (half, mat, pos) => part(new THREE.BoxGeometry(2 * half[0], 2 * half[1], 2 * half[2]), mat, pos);
  const UNIT_SPHERE = new THREE.SphereGeometry(1, 40, 24);
  const ell = (r, mat, pos) => part(UNIT_SPHERE, mat, pos, r);
  box([0.165, 0.084, 0.006], MAT.chassis, [0, 0, z0 + 0.008]);
  box([0.15, 0.093, 0.02], MAT.paint, [0, 0, z0 + 0.032]);
  ell([0.06, 0.093, 0.022], MAT.paint, [0.14, 0, z0 + 0.034]);
  ell([0.065, 0.093, 0.026], MAT.paint, [-0.135, 0, z0 + 0.036]);
  ell([0.2, 0.092, 0.03], MAT.paint, [0, 0, z0 + 0.048]);
  ell([0.13, 0.074, 0.034], MAT.glass, [-0.02, 0, z0 + 0.066]);
  box([0.006, 0.009, 0.004], MAT.glass, [0.055, 0.092, z0 + 0.072]);
  box([0.006, 0.009, 0.004], MAT.glass, [0.055, -0.092, z0 + 0.072]);
  box([0.004, 0.082, 0.006], MAT.glass, [0.195, 0, z0 + 0.046]);
  box([0.004, 0.06, 0.004], MAT.chassis, [0.195, 0, z0 + 0.022]);
  box([0.006, 0.088, 0.006], MAT.glass, [-0.193, 0, z0 + 0.054]);
  const zf = z0 + 0.047, zr = z0 + 0.054;
  box([0.002, 0.074, 0.0016], MAT.head, [0.1995, 0, zf]);
  box([0.003, 0.079, 0.0042], MAT.glowW, [0.1955, 0, zf]);
  for (const sy of [1, -1]) for (let i = 0; i < 3; i++) box([0.0015, 0.0032, 0.0022], MAT.head, [0.1995, sy * (0.05 + 0.009 * i), zf - 0.0042]);
  for (let i = 0; i < 19; i++) box([0.0015, 0.0028, 0.0028], MAT.tail, [-0.1995, -0.08 + 0.16 * i / 18, zr]);
  box([0.003, 0.086, 0.0055], MAT.glowR, [-0.1955, 0, zr]);
  const WHEEL_R = 0.03, wz = z0 + WHEEL_R;
  for (const sx of [1, -1]) for (const sy of [1, -1]) {
    const tire = part(new THREE.CylinderGeometry(WHEEL_R, WHEEL_R, 0.018, 32), MAT.tire, [sx * 0.14, sy * 0.091, wz]);
    const rim = part(new THREE.CylinderGeometry(WHEEL_R * 0.72, WHEEL_R * 0.72, 0.0025, 32), MAT.rim, [sx * 0.14, sy * 0.10, wz]);
    tire.rotation.z = rim.rotation.z = 0;   // CylinderGeometry 축은 y → 그대로 차축(y)
  }
  // 노즐 모양: 분사 방향으로 짧은 금속 원통
  const axisQuat = (v) => new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...v));
  for (const [p, f, F] of THRUSTERS) {
    const L = 0.012 + 0.0004 * F, r = 0.007 + 0.0003 * F;
    const m = part(new THREE.CylinderGeometry(r, r * 0.85, L + 0.004, 20), MAT.metal,
      [p[0] + f[0] * (L - 0.004) / 2, p[1] + f[1] * (L - 0.004) / 2, p[2] + f[2] * (L - 0.004) / 2]);
    m.quaternion.copy(axisQuat(f));
  }
  const cockpitAnchor = new THREE.Object3D(); cockpitAnchor.position.set(0.07, 0, 0.078); car.add(cockpitAnchor);

  // ---------- 냉가스 분사 (watch_landing.add_plumes와 같은 모양: 흰 코어 + 넓어지며 옅어지는 증기 9겹) ----------
  const SEGS = 9, CORE = new THREE.Color(1, 1, 1), VAPOR = new THREE.Color(0.62, 0.82, 1);
  const plumes = THRUSTERS.map(([p, f]) => {
    const g = new THREE.Group();
    g.position.set(...p);
    g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(-f[0], -f[1], -f[2]));
    const mk = () => {
      const m = new THREE.Mesh(UNIT_SPHERE, new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, fog: true }));
      m.renderOrder = 2; g.add(m); return m;
    };
    const core = mk(), vapor = Array.from({ length: SEGS }, mk);
    car.add(g);
    return { g, core, vapor };
  });
  function drawPlumes(u) {
    plumes.forEach((pl, i) => {
      const t = u[i], F = THRUSTERS[i][2] * t;
      pl.g.visible = t >= 0.05;
      if (!pl.g.visible) return;
      const L = Math.min(0.7, 0.05 + 0.035 * F) * (1 + 0.12 * (Math.random() * 2 - 1));
      const r0 = 0.006 + 0.0005 * F, dens = 0.35 + 0.65 * Math.sqrt(t);
      pl.core.scale.set(r0 * 0.55, r0 * 0.55, L * 0.08); pl.core.position.set(0, 0, L * 0.08);
      pl.core.material.color.copy(CORE); pl.core.material.opacity = 0.6 * dens;
      const seg = L / SEGS;
      pl.vapor.forEach((m, k) => {
        const s = (k + 0.5) / SEGS, r = r0 * (1 + 5 * s ** 0.8) * (1 + 0.1 * (Math.random() * 2 - 1));
        m.scale.set(r, r, seg * 1.1); m.position.set(0, 0, 0.01 + L * s);
        m.material.color.copy(CORE).lerp(VAPOR, s);
        m.material.opacity = dens * (0.26 * (1 - s) ** 1.6 + 0.025);
      });
    });
  }

  // r128은 재질·조명 색을 선형 값으로 받으므로, 위에서 쓴 sRGB 색 코드를 선형으로 바꿔 둔다 (안 하면 색이 바램)
  scene.traverse((o) => {
    if (o.isLight) o.color.convertSRGBToLinear();
    if (o.isLight && o.groundColor) o.groundColor.convertSRGBToLinear();
    [].concat(o.material || []).forEach((m) => {
      if (m.userData.linear) return;
      if (m.color && !m.map) m.color.convertSRGBToLinear();
      m.userData.linear = true;
    });
  });
  scene.fog.color.convertSRGBToLinear();
  VAPOR.convertSRGBToLinear();

  // ---------- 상태·입력 ----------
  let game = null;
  const keys = { W: 0, S: 0, A: 0, D: 0, Q: 0, E: 0, C: 0, SPACE: 0 };
  const view = { mode: 1, dist: 1.6, az: 0, freeAz: 0, freeEl: -15, pan: new THREE.Vector3(), speed: 1, paused: false, carry: 0 };
  const SPEEDS = [0.1, 0.25, 0.5, 1, 2];
  const DIST = { 1: 1.6, 2: 2.6, 3: 6.5, 4: 1.6, 5: 1.6 };
  const keyName = (e) => (e.code === 'Space' ? 'SPACE' : e.code.startsWith('Key') ? e.code.slice(3)
    : e.code.startsWith('Digit') ? e.code.slice(5) : e.code === 'Minus' ? '-' : e.code === 'Equal' ? '=' : e.key);
  const ACTIONS = new Set(['R', 'L', 'F', 'G', 'X', 'T', 'SPACE']);

  function setSpeed(dir) {
    let i = SPEEDS.reduce((b, v, k) => (Math.abs(v - view.speed) < Math.abs(SPEEDS[b] - view.speed) ? k : b), 0);
    i = Math.max(0, Math.min(SPEEDS.length - 1, i + dir));
    view.speed = SPEEDS[i]; game.say(`speed x${view.speed}`);
  }
  function enterFree() {
    if (view.mode !== 5) { view.freeAz = view.camAz; view.freeEl = view.camEl; view.mode = 5; }
  }
  function press(k) {
    if (!game) return;
    if (k in keys) keys[k] = 1;
    if (ACTIONS.has(k)) game.onKey(k);
    else if (k === 'P') { view.paused = !view.paused; game.say(view.paused ? 'paused' : 'resumed'); }
    else if (k === '-') setSpeed(-1);
    else if (k === '=') setSpeed(1);
    else if (['1', '2', '3', '4'].includes(k)) { view.mode = +k; view.dist = DIST[view.mode]; }
    else if (k === '5') { if (view.mode === 5) view.pan.set(0, 0, 0); enterFree(); }
    else if (k === 'H') $('help').hidden = !$('help').hidden;
  }
  const release = (k) => { if (k in keys) keys[k] = 0; };
  window.addEventListener('keydown', (e) => {
    const k = keyName(e);
    if (['SPACE', 'ArrowUp', 'ArrowDown'].includes(k)) e.preventDefault();
    if (!e.repeat) press(k);
  });
  window.addEventListener('keyup', (e) => release(keyName(e)));
  window.addEventListener('blur', () => Object.keys(keys).forEach((k) => { keys[k] = 0; }));

  // 마우스·터치: 왼쪽(한 손가락) 드래그 = 차 주위로 돌리기, 오른쪽 드래그 = 시점 옮기기, 휠·두 손가락 = 줌
  const drag = { id: null, button: 0, x: 0, y: 0 }, pinch = new Map();
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pinch.set(e.pointerId, [e.clientX, e.clientY]);
    if (pinch.size === 1) { Object.assign(drag, { id: e.pointerId, button: e.button, x: e.clientX, y: e.clientY }); enterFree(); }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (pinch.has(e.pointerId) && pinch.size === 2) {
      const [a, b] = [...pinch.values()], before = Math.hypot(a[0] - b[0], a[1] - b[1]);
      pinch.set(e.pointerId, [e.clientX, e.clientY]);
      const [c, d] = [...pinch.values()], after = Math.hypot(c[0] - d[0], c[1] - d[1]);
      if (before > 0) view.dist = Math.min(30, Math.max(0.4, view.dist * before / after));
      return;
    }
    if (e.pointerId !== drag.id) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.button === 2) {
      const a = view.freeAz * Math.PI / 180, k = 0.0015 * view.dist;
      view.pan.add(new THREE.Vector3(-dx * k * Math.sin(a), dx * k * Math.cos(a), dy * k));   // 화면 오른쪽 = (sin a, −cos a, 0)
    } else {
      view.freeAz -= 0.3 * dx;
      view.freeEl = Math.max(-89, Math.min(89, view.freeEl - 0.3 * dy));
    }
  });
  const endPointer = (e) => { pinch.delete(e.pointerId); if (e.pointerId === drag.id) drag.id = null; };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); view.dist = Math.min(30, Math.max(0.4, view.dist * Math.pow(1.0015, e.deltaY))); }, { passive: false });

  // 화면 버튼(터치용): data-hold = 누르는 동안, data-tap = 한 번
  document.querySelectorAll('[data-hold],[data-tap]').forEach((btn) => {
    const k = btn.dataset.hold || btn.dataset.tap;
    btn.addEventListener('pointerdown', (e) => { e.preventDefault(); btn.setPointerCapture(e.pointerId); btn.classList.add('on'); press(k); });
    const up = () => { btn.classList.remove('on'); release(k); };
    btn.addEventListener('pointerup', up); btn.addEventListener('pointercancel', up);
  });
  $('help-toggle').addEventListener('click', () => { $('help').hidden = !$('help').hidden; });
  // 폰이나 낮은 창에서는 조작표가 화면을 가리므로 접어 두고 버튼으로 열게 함
  if (window.matchMedia('(pointer: coarse)').matches || window.innerHeight < 760) $('help').hidden = true;

  // ---------- HUD ----------
  const MODE_NAME = { drive: 'Drive', fly: 'Fly · assist', ballistic: 'Airborne · auto-level', rl_land: 'RL auto-land',
    rl_flip: 'RL jump flip', rl_airflip: 'RL air flip', auto_land: 'Touchdown · level & slow' };
  const MODE_KIND = { drive: 'ground', fly: 'assist', ballistic: 'assist', auto_land: 'assist', rl_land: 'rl', rl_flip: 'rl', rl_airflip: 'rl' };
  let lastMsgId = -1, msgUntil = 0, hudT = 0;
  function hud(now) {
    if (game.msgId !== lastMsgId) {
      lastMsgId = game.msgId; msgUntil = now + 2600;
      $('toast').textContent = game.msg; $('toast').hidden = !game.msg;
    }
    if (now > msgUntil) $('toast').hidden = true;
    if (now - hudT < 80) return;
    hudT = now;
    const v = game.vel, deg = game.tilt() * 180 / Math.PI;
    $('mode').textContent = MODE_NAME[game.mode] || game.mode;
    $('mode').dataset.kind = MODE_KIND[game.mode] || 'ground';
    $('v-speed').textContent = Math.hypot(...v).toFixed(1);
    $('v-alt').textContent = Math.max(0, game.height() - BZ).toFixed(2);
    $('v-tilt').textContent = deg.toFixed(0);
    $('v-x').textContent = game.pos[0].toFixed(1);
    $('v-time').textContent = view.paused ? 'paused' : `×${view.speed}`;
    const inf = game.infiniteGas, left = game.gasLeft();
    const frac = inf ? 1 : left / TANK_G;
    $('gas-fill').style.width = `${(frac * 100).toFixed(1)}%`;
    $('gas').dataset.level = inf ? 'inf' : frac > 0.6 ? 'ok' : frac > 0.3 ? 'mid' : 'low';
    $('gas-text').textContent = inf ? 'Gas unlimited · T for real 190 g tank' : `${left.toFixed(0)} / ${TANK_G} g${left < 100 ? ' · flip needs ~100 g' : ''}`;
    $('gas-mark').hidden = inf;
  }

  // ---------- 카메라 ----------
  const tmpV = new THREE.Vector3();
  function placeCamera() {
    const R = game.R, heading = Math.atan2(R[1][0], R[0][0]) * 180 / Math.PI;
    const target = new THREE.Vector3(game.pos[0], game.pos[1], game.pos[2] + 0.05);
    let az, el;
    if (view.mode === 4) {
      cockpitAnchor.getWorldPosition(tmpV); camera.position.copy(tmpV);
      const fwd = new THREE.Vector3(R[0][0], R[1][0], R[2][0]), up = new THREE.Vector3(R[0][2], R[1][2], R[2][2]);
      camera.up.copy(up); camera.lookAt(tmpV.clone().add(fwd)); camera.fov = 80; camera.updateProjectionMatrix();
      view.camAz = heading; view.camEl = -10;
      return;
    }
    camera.up.set(0, 0, 1);
    if (camera.fov !== 50) { camera.fov = 50; camera.updateProjectionMatrix(); }
    if (view.mode === 1) { view.az += 0.08 * (((heading - view.az + 540) % 360) - 180); az = view.az; el = -14; }
    else if (view.mode === 2) { az = 270; el = -8; }
    else if (view.mode === 3) { az = 235; el = -28; }
    else { az = view.freeAz; el = view.freeEl; target.add(view.pan); }
    view.camAz = az; view.camEl = el;
    const a = az * Math.PI / 180, e = el * Math.PI / 180;
    camera.position.set(target.x - view.dist * Math.cos(e) * Math.cos(a), target.y - view.dist * Math.cos(e) * Math.sin(a),
      target.z - view.dist * Math.sin(e));
    camera.lookAt(target);
  }

  // ---------- 루프 ----------
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    }
  }
  let last = performance.now();
  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.05); last = now;
    if (!view.paused) {
      view.carry += dt * view.speed;
      const n = Math.floor(view.carry / SIM.DT); view.carry -= n * SIM.DT;
      for (let i = 0; i < n; i++) game.tick(keys);
    }
    car.position.set(...game.pos);
    car.quaternion.set(game.quat[1], game.quat[2], game.quat[3], game.quat[0]);
    drawPlumes(game.u);
    sun.position.set(game.pos[0] - 2, game.pos[1] - 3, game.pos[2] + 6);
    sun.target.position.set(...game.pos);
    resize(); placeCamera(); hud(now);
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  fetch('policies.json').then((r) => {
    if (!r.ok) throw new Error(`policies.json ${r.status}`);
    return r.json();
  }).then((pol) => {
    game = new Game(pol);
    window.roadster = { get game() { return game; }, view };   // 콘솔에서 상태 확인용
    game.say('Ready · hold W to drive, Space to take off');
    $('loading').hidden = true;
    requestAnimationFrame((t) => { last = t; frame(t); });
  }).catch((err) => {
    $('loading').textContent = `Could not load the RL policies (${err.message}). Reload the page to try again.`;
  });
})();
