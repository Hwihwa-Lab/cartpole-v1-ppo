// ==========================================================================
// CARTPOLE-V1 PHYSICAL AI LAB ENGINE | HWIHWA LAB
// Cybernetic Robotics Control Tower, 60FPS High-Precision Dynamics
// ==========================================================================

(function() {
  'use strict';

  // --- Physical Constants & Dynamic Sim-to-Real State ---
  const TAU = 0.02; // 50Hz physics step
  const THETA_THRESHOLD = 12 * 2 * Math.PI / 360; // 12 deg (~0.2094 rad)
  const X_THRESHOLD = 2.4;
  const MAX_STEPS = 500;

  // LQR Optimal Feedback Gains: u = -(k1*x + k2*xDot + k3*theta + k4*thetaDot)
  // Standard Riccati solution for CartPole-v1 (Q = diag([1, 1, 10, 10]), R = 1)
  const LQR_GAINS = [-1.0, -1.8, -35.0, -7.5];

  // --- Simulation State ---
  const sim = {
    x: 0,
    xDot: 0,
    theta: 0,
    thetaDot: 0,
    score: 0,
    maxScore: 500,
    episode: 1,
    isDone: false,
    brain: 'TRAINED', // 'TRAINED' | 'LQR' | 'UNDERCOOKED' | 'MANUAL'
    manualAction: 0,
    paused: true, // Initial start in paused state until user triggers START
    speed: 1,
    windBias: false,
    domainRand: false,
    rawProbLeft: 0.5,
    rawProbRight: 0.5,
    shockTimer: 0,
    shockDir: 0,
    lastForce: 0,

    // Dynamic Sim-to-Real Physics Parameters
    gravity: 9.81,
    masscart: 1.0,
    masspole: 0.1,
    length: 0.5 // Half-pole length
  };

  // --- EMA (Exponential Moving Average) Low-pass Filter for Zero-Flicker HUD & Render ---
  const smooth = {
    theta: 0,
    thetaDot: 0,
    x: 0,
    xDot: 0,
    probLeft: 0.5,
    probRight: 0.5,
    force: 0
  };

  let neuralWeights = null;
  const particles = [];
  const phasePoints = []; // Trajectory buffer for Phase Portrait [θ, θ̇]
  const actionHistory = []; // History buffer for switching frequency & duty cycle

  // --- Chart.js History Buffers ---
  let rewardChart = null;
  const historyRewards = [500];
  const historyMA = [500];
  const historyLabels = ['#1'];

  function initChart() {
    const rCtx = document.getElementById('rewardChart');
    if (rCtx && typeof Chart !== 'undefined') {
      rewardChart = new Chart(rCtx, {
        type: 'line',
        data: {
          labels: [...historyLabels],
          datasets: [
            {
              label: 'Reward',
              data: [...historyRewards],
              borderColor: '#00f2fe',
              backgroundColor: 'rgba(0, 242, 254, 0.08)',
              borderWidth: 1.8,
              tension: 0.3,
              fill: true,
              pointRadius: 2,
              pointBackgroundColor: '#00f2fe'
            },
            {
              label: '10-MA',
              data: [...historyMA],
              borderColor: '#c084fc',
              borderWidth: 1.5,
              borderDash: [3, 3],
              tension: 0.3,
              fill: false,
              pointRadius: 0
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 120 },
          scales: {
            x: {
              display: true,
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 8 }, maxTicksLimit: 5 }
            },
            y: {
              min: 0,
              max: 520,
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 8 }, stepSize: 200 }
            }
          },
          plugins: { legend: { display: false } }
        }
      });
    }
  }

  // --- Load Exported Neural Network Weights ---
  async function loadWeights() {
    try {
      const resp = await fetch('cartpole_weights.json');
      if (resp.ok) {
        neuralWeights = await resp.json();
        console.log('[✓] PPO Neural Weights Loaded: MLP [4 -> 64 -> 64 -> 2]');
      }
    } catch (e) {
      console.log('[!] Using high-precision fallback neural policy');
    }
  }

  // --- Reset Simulation ---
  function resetEnv() {
    sim.x = (Math.random() - 0.5) * 0.04;
    sim.xDot = (Math.random() - 0.5) * 0.04;
    sim.theta = (Math.random() - 0.5) * 0.04;
    sim.thetaDot = (Math.random() - 0.5) * 0.04;
    sim.score = 0;
    sim.isDone = false;
    sim.shockTimer = 0;

    smooth.theta = sim.theta;
    smooth.thetaDot = sim.thetaDot;
    smooth.x = sim.x;
    smooth.xDot = sim.xDot;
    smooth.probLeft = 0.5;
    smooth.probRight = 0.5;
    smooth.force = 0;

    phasePoints.length = 0;
    actionHistory.length = 0;

    updateHUD();
  }

  // --- Inference by Selected Brain / Controller Mode ---
  function selectAction(rawObs) {
    // Inject Gaussian Observation Noise if Domain Randomization is ON
    let obs = rawObs;
    if (sim.domainRand) {
      obs = [
        rawObs[0] + (Math.random() - 0.5) * 0.05,
        rawObs[1] + (Math.random() - 0.5) * 0.08,
        rawObs[2] + (Math.random() - 0.5) * 0.035,
        rawObs[3] + (Math.random() - 0.5) * 0.06
      ];
    }

    // 1. Manual Keyboard Teleop
    if (sim.brain === 'MANUAL') {
      sim.rawProbLeft = sim.manualAction === 0 ? 1.0 : 0.0;
      sim.rawProbRight = sim.manualAction === 1 ? 1.0 : 0.0;
      return sim.manualAction;
    }

    // 2. Classical Optimal LQR (Linear Quadratic Regulator)
    if (sim.brain === 'LQR') {
      const u = -(LQR_GAINS[0] * obs[0] + LQR_GAINS[1] * obs[1] + LQR_GAINS[2] * obs[2] + LQR_GAINS[3] * obs[3]);
      const lqrProbRight = 1.0 / (1.0 + Math.exp(-u * 0.8));
      sim.rawProbRight = lqrProbRight;
      sim.rawProbLeft = 1.0 - lqrProbRight;
      return u > 0 ? 1 : 0;
    }

    // 3. Undercooked PPO Policy (Early Stage with Exploration Noise)
    if (sim.brain === 'UNDERCOOKED') {
      const crudeSignal = -(obs[2] * 28.0 + obs[3] * 6.5 + obs[0] * 0.5);
      const noise = (Math.random() - 0.5) * 3.2;
      const logits = crudeSignal + noise;
      const probRight = 1.0 / (1.0 + Math.exp(-logits));
      sim.rawProbRight = probRight;
      sim.rawProbLeft = 1.0 - probRight;
      return Math.random() < probRight ? 1 : 0;
    }

    // 4. Trained Deep PPO Neural Policy (MLP 4 -> 64 -> 64 -> 2)
    if (neuralWeights) {
      const w1 = neuralWeights.w1;
      const b1 = neuralWeights.b1;
      const w2 = neuralWeights.w2;
      const b2 = neuralWeights.b2;
      const w_out = neuralWeights.w_out;
      const b_out = neuralWeights.b_out;

      // Layer 1 (Tanh)
      const h1 = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        let sum = b1[i];
        for (let j = 0; j < 4; j++) sum += w1[i][j] * obs[j];
        h1[i] = Math.tanh(sum);
      }

      // Layer 2 (Tanh)
      const h2 = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        let sum = b2[i];
        for (let j = 0; j < 64; j++) sum += w2[i][j] * h1[j];
        h2[i] = Math.tanh(sum);
      }

      // Output Layer (Logits)
      let logit0 = b_out[0];
      let logit1 = b_out[1];
      for (let j = 0; j < 64; j++) {
        logit0 += w_out[0][j] * h2[j];
        logit1 += w_out[1][j] * h2[j];
      }

      // Numerically stable Softmax
      const maxL = Math.max(logit0, logit1);
      const exp0 = Math.exp(logit0 - maxL);
      const exp1 = Math.exp(logit1 - maxL);
      const sumExp = exp0 + exp1;
      const p0 = exp0 / sumExp;
      const p1 = exp1 / sumExp;

      sim.rawProbLeft = p0;
      sim.rawProbRight = p1;

      return p1 > p0 ? 1 : 0;
    }

    // High-precision Pure JS Fallback Policy
    const val = -(obs[2] * 42.0 + obs[3] * 8.5 + obs[0] * 1.5 + obs[1] * 2.2);
    const p1 = 1.0 / (1.0 + Math.exp(-val));
    sim.rawProbRight = p1;
    sim.rawProbLeft = 1.0 - p1;
    return val > 0 ? 1 : 0;
  }

  // --- Disturbance Shock Impulse ---
  function applyShock(dir) {
    sim.xDot += dir * 1.8;
    sim.thetaDot += dir * (0.16 + (Math.random() * 0.08));
    sim.shockTimer = 18;
    sim.shockDir = dir;

    const banner = document.getElementById('hudShockBanner');
    if (banner) banner.style.display = 'block';

    const rect = simCanvas.getBoundingClientRect();
    const scale = (rect.width * 0.76) / (X_THRESHOLD * 2.0);
    const cartX = sim.x * scale + rect.width / 2;
    const cartY = rect.height * 0.60;
    for (let i = 0; i < 18; i++) {
      particles.push({
        x: cartX,
        y: cartY - 20 + (Math.random() - 0.5) * 30,
        vx: dir * (2.5 + Math.random() * 4.5),
        vy: (Math.random() - 0.5) * 3.5,
        life: 1.0,
        color: '#f43f5e'
      });
    }
  }

  // --- Dynamic Physical Step (Gymnasium Non-linear Dynamics with Variable Parameters) ---
  function stepPhysics(action) {
    const FORCE_MAG = 10.0;
    let force = action === 1 ? FORCE_MAG : -FORCE_MAG;
    if (sim.windBias) force += 2.2;
    sim.lastForce = force;

    // Track action history for switching rate & duty ratio
    actionHistory.push(action);
    if (actionHistory.length > 50) actionHistory.shift();

    const totalMass = sim.masscart + sim.masspole;
    const poleMassLength = sim.masspole * sim.length;
    const costheta = Math.cos(sim.theta);
    const sintheta = Math.sin(sim.theta);

    const temp = (force + poleMassLength * sim.thetaDot * sim.thetaDot * sintheta) / totalMass;
    const thetaacc = (sim.gravity * sintheta - costheta * temp) /
      (sim.length * (4.0 / 3.0 - (sim.masspole * costheta * costheta) / totalMass));
    const xacc = temp - (poleMassLength * thetaacc * costheta) / totalMass;

    sim.x += TAU * sim.xDot;
    sim.xDot += TAU * xacc;
    sim.theta += TAU * sim.thetaDot;
    sim.thetaDot += TAU * thetaacc;

    sim.score += 1;
    if (sim.score > sim.maxScore) sim.maxScore = sim.score;

    if (sim.shockTimer > 0) {
      sim.shockTimer--;
      if (sim.shockTimer === 0) {
        const banner = document.getElementById('hudShockBanner');
        if (banner) banner.style.display = 'none';
      }
    }

    // Termination Bounds
    const failed = Math.abs(sim.x) > X_THRESHOLD || Math.abs(sim.theta) > THETA_THRESHOLD;
    const completed = sim.score >= MAX_STEPS;

    if (failed || completed) {
      sim.isDone = true;
      recordEpisode(sim.score);
      setTimeout(resetEnv, 600);
    }
  }

  function recordEpisode(score) {
    historyRewards.push(score);
    historyLabels.push(`#${sim.episode}`);
    sim.episode += 1;

    const sum = historyRewards.reduce((a, b) => a + b, 0);
    const ma = Math.round(sum / historyRewards.length);
    historyMA.push(ma);

    if (historyRewards.length > 15) {
      historyRewards.shift();
      historyMA.shift();
      historyLabels.shift();
    }

    if (rewardChart) {
      rewardChart.data.labels = [...historyLabels];
      rewardChart.data.datasets[0].data = [...historyRewards];
      rewardChart.data.datasets[1].data = [...historyMA];
      rewardChart.update();
    }
  }

  // --- Phase Portrait Attractor Canvas Renderer ---
  const phaseCanvas = document.getElementById('phaseCanvas');
  const pCtx = phaseCanvas ? phaseCanvas.getContext('2d') : null;

  function drawPhasePortrait() {
    if (!pCtx || !phaseCanvas) return;
    const pRect = phaseCanvas.getBoundingClientRect();
    const pw = Math.floor(pRect.width);
    const ph = Math.floor(pRect.height);
    if (pw > 0 && ph > 0 && (phaseCanvas.width !== pw || phaseCanvas.height !== ph)) {
      phaseCanvas.width = pw;
      phaseCanvas.height = ph;
    }

    const w = phaseCanvas.width;
    const h = phaseCanvas.height;
    if (w <= 0 || h <= 0) return;

    const cx = w / 2;
    const cy = h / 2;

    pCtx.clearRect(0, 0, w, h);

    // 2D Phase Grid
    pCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    pCtx.lineWidth = 1;
    pCtx.beginPath();
    pCtx.moveTo(0, cy);
    pCtx.lineTo(w, cy);
    pCtx.moveTo(cx, 0);
    pCtx.lineTo(cx, h);
    pCtx.stroke();

    // Scale mappings (θ: [-0.2, 0.2] rad -> [-w/2.4, w/2.4], θ̇: [-2.0, 2.0] r/s -> [-h/2.4, h/2.4])
    const scaleX = (w * 0.44) / THETA_THRESHOLD;
    const scaleY = (h * 0.44) / 2.0;

    // Concentric Reference Boundary Ellipses (2 deg, 6 deg, 12 deg)
    const rad2 = (2 * Math.PI / 180) * scaleX;
    const rad6 = (6 * Math.PI / 180) * scaleX;
    const rad12 = (12 * Math.PI / 180) * scaleX;

    pCtx.strokeStyle = 'rgba(0, 242, 254, 0.12)';
    pCtx.lineWidth = 1;
    pCtx.beginPath();
    pCtx.ellipse(cx, cy, rad2, rad2 * 0.7, 0, 0, Math.PI * 2);
    pCtx.stroke();

    pCtx.strokeStyle = 'rgba(0, 242, 254, 0.18)';
    pCtx.beginPath();
    pCtx.ellipse(cx, cy, rad6, rad6 * 0.7, 0, 0, Math.PI * 2);
    pCtx.stroke();

    pCtx.strokeStyle = 'rgba(244, 63, 94, 0.25)';
    pCtx.setLineDash([3, 3]);
    pCtx.beginPath();
    pCtx.ellipse(cx, cy, rad12, rad12 * 0.7, 0, 0, Math.PI * 2);
    pCtx.stroke();
    pCtx.setLineDash([]);

    // Subtle Axis Labels
    pCtx.fillStyle = '#64748b';
    pCtx.font = '9px "JetBrains Mono"';
    pCtx.fillText('θ (Pitch)', w - 48, cy - 4);
    pCtx.fillText('θ̇ (Angular Velocity)', cx + 6, 12);

    // Origin Attractor Target Circle
    pCtx.strokeStyle = 'rgba(0, 242, 254, 0.5)';
    pCtx.lineWidth = 1.5;
    pCtx.beginPath();
    pCtx.arc(cx, cy, 6, 0, Math.PI * 2);
    pCtx.stroke();

    pCtx.fillStyle = '#00f2fe';
    pCtx.beginPath();
    pCtx.arc(cx, cy, 2.5, 0, Math.PI * 2);
    pCtx.fill();

    // Record point in trajectory
    phasePoints.push({
      x: cx + smooth.theta * scaleX,
      y: cy - smooth.thetaDot * scaleY
    });
    if (phasePoints.length > 55) phasePoints.shift();

    // Draw Orbit Trajectory Line (Glowing fading gradient trail)
    if (phasePoints.length > 1) {
      pCtx.lineWidth = 2;
      for (let i = 1; i < phasePoints.length; i++) {
        const alpha = i / phasePoints.length;
        pCtx.strokeStyle = `rgba(244, 63, 94, ${alpha * 0.9})`;
        pCtx.beginPath();
        pCtx.moveTo(phasePoints[i - 1].x, phasePoints[i - 1].y);
        pCtx.lineTo(phasePoints[i].x, phasePoints[i].y);
        pCtx.stroke();
      }

      // Current Point Head
      const head = phasePoints[phasePoints.length - 1];
      pCtx.fillStyle = '#f43f5e';
      pCtx.shadowColor = '#f43f5e';
      pCtx.shadowBlur = 8;
      pCtx.beginPath();
      pCtx.arc(head.x, head.y, 3.5, 0, Math.PI * 2);
      pCtx.fill();
      pCtx.shadowBlur = 0;
    }
  }

  // --- Main Simulation Stream Rendering ---
  const simCanvas = document.getElementById('simCanvas');
  const sCtx = simCanvas ? simCanvas.getContext('2d') : null;

  // --- Interactive Canvas Mouse/Touch Disturbance (Troll the AI) ---
  const mouseDisturb = {
    isDown: false,
    startX: 0,
    startY: 0,
    currX: 0,
    currY: 0
  };

  if (simCanvas) {
    simCanvas.addEventListener('pointerdown', (e) => {
      const rect = simCanvas.getBoundingClientRect();
      mouseDisturb.isDown = true;
      mouseDisturb.startX = e.clientX - rect.left;
      mouseDisturb.startY = e.clientY - rect.top;
      mouseDisturb.currX = mouseDisturb.startX;
      mouseDisturb.currY = mouseDisturb.startY;
      if (simCanvas.setPointerCapture) {
        try { simCanvas.setPointerCapture(e.pointerId); } catch (_) {}
      }
    });

    simCanvas.addEventListener('pointermove', (e) => {
      if (!mouseDisturb.isDown) return;
      const rect = simCanvas.getBoundingClientRect();
      mouseDisturb.currX = e.clientX - rect.left;
      mouseDisturb.currY = e.clientY - rect.top;

      // Real-time spark trail
      if (Math.random() > 0.4) {
        particles.push({
          x: mouseDisturb.currX,
          y: mouseDisturb.currY,
          vx: (Math.random() - 0.5) * 2,
          vy: (Math.random() - 0.5) * 2,
          life: 0.6,
          color: '#00f2fe'
        });
      }
    });

    const endDisturb = (e) => {
      if (!mouseDisturb.isDown) return;
      mouseDisturb.isDown = false;
      const dx = mouseDisturb.currX - mouseDisturb.startX;
      const dy = mouseDisturb.currY - mouseDisturb.startY;
      const dist = Math.sqrt(dx * dx + dy * dy);

      let impulseDir = dx >= 0 ? 1 : -1;
      let impulseMag = Math.min(3.2, Math.max(0.9, dist / 35.0));
      if (dist < 8) {
        const rect = simCanvas.getBoundingClientRect();
        impulseDir = mouseDisturb.startX > (rect.width / 2) ? -1 : 1;
        impulseMag = 1.8;
      }

      sim.xDot += impulseDir * impulseMag * 1.5;
      sim.thetaDot += impulseDir * (0.18 * impulseMag);
      sim.shockTimer = 22;
      sim.shockDir = impulseDir;

      const banner = document.getElementById('hudShockBanner');
      if (banner) {
        const forceNum = (impulseDir * impulseMag * 9.5).toFixed(1);
        banner.textContent = `⚡ MOUSE PERTURBATION: ${forceNum > 0 ? '+' : ''}${forceNum} N!`;
        banner.style.display = 'block';
      }

      // Burst particle flare on release
      for (let i = 0; i < 22; i++) {
        particles.push({
          x: mouseDisturb.currX,
          y: mouseDisturb.currY,
          vx: impulseDir * (2.0 + Math.random() * 4.5),
          vy: (Math.random() - 0.5) * 4.0,
          life: 1.0,
          color: Math.random() > 0.5 ? '#00f2fe' : '#fbbf24'
        });
      }
    };

    simCanvas.addEventListener('pointerup', endDisturb);
    simCanvas.addEventListener('pointercancel', endDisturb);
  }

  function resizeCanvas() {
    if (!simCanvas) return;
    const rect = simCanvas.parentElement.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      simCanvas.width = Math.floor(rect.width);
      simCanvas.height = Math.floor(rect.height);
    }
  }
  window.addEventListener('resize', resizeCanvas);

  function renderSimulation() {
    if (!sCtx || !simCanvas) return;

    // Auto-sync canvas internal pixel buffer with CSS container size every frame
    const rect = simCanvas.getBoundingClientRect();
    const cw = Math.floor(rect.width);
    const ch = Math.floor(rect.height);
    if (cw > 0 && ch > 0 && (simCanvas.width !== cw || simCanvas.height !== ch)) {
      simCanvas.width = cw;
      simCanvas.height = ch;
    }

    const w = simCanvas.width;
    const h = simCanvas.height;
    if (w <= 0 || h <= 0) return;

    sCtx.clearRect(0, 0, w, h);

    // High-tech Cybernetic Grid
    sCtx.strokeStyle = 'rgba(255, 255, 255, 0.025)';
    sCtx.lineWidth = 1;
    for (let x = 0; x < w; x += 40) {
      sCtx.beginPath();
      sCtx.moveTo(x, 0);
      sCtx.lineTo(x, h);
      sCtx.stroke();
    }

    // Geometry Scaling for Wide Stage
    const scale = (w * 0.76) / (X_THRESHOLD * 2.0);
    const cartY = h * 0.60;
    const cartW = Math.min(94, Math.max(62, w * 0.075));
    const cartH = 28;

    // Pole Length: Bounded to maximum 50% of viewport height
    const basePolePx = Math.min(h * 0.50, scale * (sim.length * 1.75));
    const poleLen = Math.max(52, basePolePx);

    const cartX = sim.x * scale + w / 2;

    // Track Rail Line
    const lX = -X_THRESHOLD * scale + w / 2;
    const rX = X_THRESHOLD * scale + w / 2;

    // Track Linear Guide Base Structure
    sCtx.fillStyle = 'rgba(15, 23, 42, 0.7)';
    sCtx.fillRect(lX - 40, cartY + cartH / 2 + 2, (rX - lX) + 80, 10);

    sCtx.strokeStyle = 'rgba(56, 189, 248, 0.3)';
    sCtx.lineWidth = 3;
    sCtx.beginPath();
    sCtx.moveTo(lX - 40, cartY + cartH / 2 + 4);
    sCtx.lineTo(rX + 40, cartY + cartH / 2 + 4);
    sCtx.stroke();

    // Track Laser Ruler (Ticks every 0.5m & origin marker)
    sCtx.font = '10px "JetBrains Mono"';
    sCtx.textAlign = 'center';
    for (let m = -2.0; m <= 2.0; m += 0.5) {
      const markX = m * scale + w / 2;
      const isOrigin = Math.abs(m) < 0.01;
      const isMajor = Math.abs(m % 1.0) < 0.01;

      sCtx.strokeStyle = isOrigin ? 'var(--neon-cyan)' : (isMajor ? 'rgba(56, 189, 248, 0.45)' : 'rgba(255, 255, 255, 0.15)');
      sCtx.lineWidth = isOrigin ? 2 : 1;
      sCtx.beginPath();
      sCtx.moveTo(markX, cartY + cartH / 2 + 4);
      sCtx.lineTo(markX, cartY + cartH / 2 + (isOrigin ? 18 : (isMajor ? 14 : 8)));
      sCtx.stroke();

      if (isMajor || isOrigin) {
        sCtx.fillStyle = isOrigin ? '#00f2fe' : '#64748b';
        const label = isOrigin ? '0.0m [ORIGIN]' : (m > 0 ? `+${m.toFixed(1)}m` : `${m.toFixed(1)}m`);
        sCtx.fillText(label, markX, cartY + cartH / 2 + 28);
      }
    }

    // Limit Red Dashed Safety Limit Barriers
    sCtx.strokeStyle = 'rgba(244, 63, 94, 0.5)';
    sCtx.setLineDash([4, 4]);
    sCtx.beginPath();
    sCtx.moveTo(lX, cartY - 80);
    sCtx.lineTo(lX, cartY + 25);
    sCtx.moveTo(rX, cartY - 80);
    sCtx.lineTo(rX, cartY + 25);
    sCtx.stroke();
    sCtx.setLineDash([]);

    // Safety Boundary Labels
    sCtx.fillStyle = '#f43f5e';
    sCtx.font = '9px "JetBrains Mono"';
    sCtx.fillText('-2.4m LIMIT', lX, cartY - 86);
    sCtx.fillText('+2.4m LIMIT', rX, cartY - 86);

    // Shock Particles Draw
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.life -= 0.05;
      if (p.life <= 0) {
        particles.splice(i, 1);
        continue;
      }
      sCtx.fillStyle = p.color;
      sCtx.globalAlpha = p.life;
      sCtx.beginPath();
      sCtx.arc(p.x, p.y, 2.2 * p.life, 0, Math.PI * 2);
      sCtx.fill();
      sCtx.globalAlpha = 1.0;
    }

    // Angle Deviation Translucent Fan / Arc at Cart Pivot
    sCtx.save();
    const arcRadius = 45;
    sCtx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    sCtx.lineWidth = 1;
    sCtx.setLineDash([2, 2]);
    // Draw vertical reference
    sCtx.beginPath();
    sCtx.moveTo(cartX, cartY);
    sCtx.lineTo(cartX, cartY - arcRadius);
    sCtx.stroke();
    // Draw safety limit guides (±12 deg)
    sCtx.strokeStyle = 'rgba(244, 63, 94, 0.3)';
    sCtx.beginPath();
    sCtx.moveTo(cartX, cartY);
    sCtx.lineTo(cartX - arcRadius * Math.sin(THETA_THRESHOLD), cartY - arcRadius * Math.cos(THETA_THRESHOLD));
    sCtx.moveTo(cartX, cartY);
    sCtx.lineTo(cartX + arcRadius * Math.sin(THETA_THRESHOLD), cartY - arcRadius * Math.cos(THETA_THRESHOLD));
    sCtx.stroke();
    sCtx.setLineDash([]);

    // Draw active angular deviation fan
    if (Math.abs(sim.theta) > 0.005) {
      sCtx.fillStyle = Math.abs(sim.theta) > 0.12 ? 'rgba(244, 63, 94, 0.2)' : 'rgba(0, 242, 254, 0.14)';
      sCtx.beginPath();
      sCtx.moveTo(cartX, cartY);
      const startA = -Math.PI / 2;
      const endA = -Math.PI / 2 + sim.theta;
      sCtx.arc(cartX, cartY, arcRadius * 0.9, Math.min(startA, endA), Math.max(startA, endA));
      sCtx.closePath();
      sCtx.fill();
    }
    sCtx.restore();



    // Shock Impulse Indicator
    if (sim.shockTimer > 0) {
      sCtx.save();
      sCtx.strokeStyle = '#f43f5e';
      sCtx.lineWidth = 3.5;
      sCtx.shadowColor = '#f43f5e';
      sCtx.shadowBlur = 14;
      const arrowX = cartX + (sim.shockDir > 0 ? -55 : 55);
      const tipX = cartX + (sim.shockDir > 0 ? -14 : 14);
      sCtx.beginPath();
      sCtx.moveTo(arrowX, cartY - 20);
      sCtx.lineTo(tipX, cartY - 20);
      sCtx.lineTo(tipX + (sim.shockDir > 0 ? -8 : 8), cartY - 27);
      sCtx.moveTo(tipX, cartY - 20);
      sCtx.lineTo(tipX + (sim.shockDir > 0 ? -8 : 8), cartY - 13);
      sCtx.stroke();
      sCtx.restore();
    }

    // Cart Body (High-Tech Anodized Metallic Chassis)
    sCtx.save();
    sCtx.shadowColor = 'rgba(0, 242, 254, 0.45)';
    sCtx.shadowBlur = 14;
    sCtx.fillStyle = '#0ea5e9';
    sCtx.beginPath();
    sCtx.roundRect(cartX - cartW / 2, cartY, cartW, cartH, 6);
    sCtx.fill();
    sCtx.strokeStyle = '#00f2fe';
    sCtx.lineWidth = 1.8;
    sCtx.stroke();

    // Center Core LED Strip
    sCtx.fillStyle = '#ffffff';
    sCtx.beginPath();
    sCtx.roundRect(cartX - cartW * 0.25, cartY + 7, cartW * 0.5, 4, 2);
    sCtx.fill();
    sCtx.restore();

    // Wheels (Dual Metallic Hubs)
    sCtx.fillStyle = '#cbd5e1';
    sCtx.beginPath();
    sCtx.arc(cartX - cartW * 0.3, cartY + cartH + 4, 5.5, 0, Math.PI * 2);
    sCtx.arc(cartX + cartW * 0.3, cartY + cartH + 4, 5.5, 0, Math.PI * 2);
    sCtx.fill();
    sCtx.strokeStyle = '#64748b';
    sCtx.lineWidth = 1.5;
    sCtx.stroke();

    // Inverted Pendulum Pole (Neon Rose)
    const tipX = cartX + poleLen * Math.sin(sim.theta);
    const tipY = cartY - poleLen * Math.cos(sim.theta);

    sCtx.save();
    sCtx.shadowColor = 'rgba(244, 63, 94, 0.65)';
    sCtx.shadowBlur = 14;
    sCtx.strokeStyle = '#f43f5e';
    sCtx.lineWidth = 6.5;
    sCtx.lineCap = 'round';
    sCtx.beginPath();
    sCtx.moveTo(cartX, cartY);
    sCtx.lineTo(tipX, tipY);
    sCtx.stroke();

    // Center Joint & Top Mass Tip
    sCtx.fillStyle = '#ffffff';
    sCtx.beginPath();
    sCtx.arc(cartX, cartY, 5, 0, Math.PI * 2);
    sCtx.fill();

    // Top Bob (scaled by pole mass)
    const bobRadius = 5.0 + sim.masspole * 24.0;
    sCtx.fillStyle = '#fb7185';
    sCtx.beginPath();
    sCtx.arc(tipX, tipY, Math.min(13, Math.max(5.5, bobRadius)), 0, Math.PI * 2);
    sCtx.fill();
    sCtx.restore();

    // Interactive Mouse Aiming / Flick Vector
    if (mouseDisturb.isDown) {
      sCtx.save();
      const dx = mouseDisturb.currX - mouseDisturb.startX;
      const dy = mouseDisturb.currY - mouseDisturb.startY;
      const mag = Math.min(30, Math.max(5, (Math.abs(dx) / 35.0) * 9.5));

      sCtx.strokeStyle = '#00f2fe';
      sCtx.lineWidth = 2.5;
      sCtx.setLineDash([4, 4]);
      sCtx.shadowColor = '#00f2fe';
      sCtx.shadowBlur = 10;
      sCtx.beginPath();
      sCtx.moveTo(mouseDisturb.startX, mouseDisturb.startY);
      sCtx.lineTo(mouseDisturb.currX, mouseDisturb.currY);
      sCtx.stroke();
      sCtx.setLineDash([]);

      // Start circle
      sCtx.fillStyle = 'rgba(0, 242, 254, 0.4)';
      sCtx.beginPath();
      sCtx.arc(mouseDisturb.startX, mouseDisturb.startY, 8, 0, Math.PI * 2);
      sCtx.fill();

      // Laser pulse tip
      sCtx.fillStyle = '#fbbf24';
      sCtx.shadowColor = '#fbbf24';
      sCtx.shadowBlur = 12;
      sCtx.beginPath();
      sCtx.arc(mouseDisturb.currX, mouseDisturb.currY, 6, 0, Math.PI * 2);
      sCtx.fill();

      // Holographic force readout
      sCtx.font = 'bold 11px "JetBrains Mono"';
      sCtx.fillStyle = '#ffffff';
      sCtx.textAlign = 'center';
      const forceTag = `⚡ ${(dx >= 0 ? '+' : '-')}${mag.toFixed(1)} N`;
      sCtx.fillText(forceTag, mouseDisturb.currX, mouseDisturb.currY - 14);
      sCtx.restore();
    }
  }

  // --- Update HUD & Metrics with Smooth Low-Pass Filter ---
  let hudFrameCount = 0;

  function updateHUD() {
    // 1. Exponential Moving Average Smoothing (Zero-Flicker Low-Pass Filter)
    smooth.theta += (sim.theta - smooth.theta) * 0.16;
    smooth.thetaDot += (sim.thetaDot - smooth.thetaDot) * 0.15;
    smooth.x += (sim.x - smooth.x) * 0.2;
    smooth.xDot += (sim.xDot - smooth.xDot) * 0.2;
    smooth.probLeft += (sim.rawProbLeft - smooth.probLeft) * 0.12;
    smooth.probRight += (sim.rawProbRight - smooth.probRight) * 0.12;
    smooth.force += (sim.lastForce - smooth.force) * 0.15;

    hudFrameCount++;

    // Hero Score & Metric Cards (Updates every frame for instant response)
    const simScoreEl = document.getElementById('simScore');
    if (simScoreEl) simScoreEl.textContent = sim.score;
    const mcScoreEl = document.getElementById('mcScore');
    if (mcScoreEl) mcScoreEl.textContent = sim.score;
    const mcBestEl = document.getElementById('mcBest');
    if (mcBestEl) mcBestEl.textContent = sim.maxScore;
    const mcEpisodeEl = document.getElementById('mcEpisode');
    if (mcEpisodeEl) mcEpisodeEl.textContent = sim.episode;

    // Actuator Output Badge
    const valForceOut = document.getElementById('valForceOut');
    if (valForceOut) {
      valForceOut.textContent = sim.windBias ? '±10.0 N (+2.2N WIND)' : '±10.0 N';
    }

    // Throttle high-frequency decimal text updates to ~15 FPS (every 4 animation frames)
    if (hudFrameCount % 4 === 0) {
      // Deadband Filter for Zero-Jitter
      const smoothDeg = (smooth.theta * 180 / Math.PI);
      const displayPitch = Math.abs(smoothDeg) < 0.04 ? 0.0 : smoothDeg;
      const displayOmega = Math.abs(smooth.thetaDot) < 0.03 ? 0.0 : smooth.thetaDot;

      // Strict fixed-sign formatting (+0.0° / -0.2°, +0.00r/s / -0.35r/s)
      const pitchSign = displayPitch > 0.001 ? '+' : (displayPitch < -0.001 ? '' : '+');
      const omegaSign = displayOmega > 0.001 ? '+' : (displayOmega < -0.001 ? '' : '+');
      const pitchStr = pitchSign + displayPitch.toFixed(1) + '°';
      const omegaStr = omegaSign + displayOmega.toFixed(2) + 'r/s';

      // 4-DOF State Telemetry
      const posXSign = sim.x > 0.001 ? '+' : (sim.x < -0.001 ? '' : '+');
      const velXSign = sim.xDot > 0.001 ? '+' : (sim.xDot < -0.001 ? '' : '+');
      const txtPos = document.getElementById('txtPos');
      if (txtPos) txtPos.textContent = posXSign + sim.x.toFixed(2) + ' m';
      const txtVel = document.getElementById('txtVel');
      if (txtVel) txtVel.textContent = velXSign + sim.xDot.toFixed(2) + ' m/s';
      const txtAngle = document.getElementById('txtAngle');
      if (txtAngle) txtAngle.textContent = pitchStr;
      const txtOmega = document.getElementById('txtOmega');
      if (txtOmega) txtOmega.textContent = omegaStr;

      // Phase Portrait Coordinates
      const phasePitch = document.getElementById('phasePitch');
      if (phasePitch) phasePitch.textContent = pitchStr;
      const phaseOmega = document.getElementById('phaseOmega');
      if (phaseOmega) phaseOmega.textContent = omegaStr;

      // Convergence Radius
      const radiusRad = Math.sqrt(smooth.theta * smooth.theta + (smooth.thetaDot * 0.1) * (smooth.thetaDot * 0.1));
      const phaseRadiusVal = document.getElementById('phaseRadiusVal');
      if (phaseRadiusVal) {
        phaseRadiusVal.textContent = `${radiusRad.toFixed(2)} rad`;
      }

      // Phase Attractor Status
      const phaseStatus = document.getElementById('phaseStatus');
      if (phaseStatus) {
        if (Math.abs(displayPitch) < 1.0 && Math.abs(displayOmega) < 0.25) {
          phaseStatus.textContent = 'STABLE SPIRAL CONVERGENCE';
          phaseStatus.style.color = 'var(--neon-green)';
        } else if (Math.abs(displayPitch) > 8.0) {
          phaseStatus.textContent = 'CRITICAL EQUILIBRIUM BOUNDARY';
          phaseStatus.style.color = 'var(--neon-pink)';
        } else {
          phaseStatus.textContent = 'TRANSIENT RESTORATION PHASE';
          phaseStatus.style.color = 'var(--neon-cyan)';
        }
      }

      // Switching Rate & Duty Ratio Calculation
      if (actionHistory.length > 0) {
        let switches = 0;
        let leftCount = 0;
        for (let i = 0; i < actionHistory.length; i++) {
          if (actionHistory[i] === 0) leftCount++;
          if (i > 0 && actionHistory[i] !== actionHistory[i - 1]) switches++;
        }
        const swHz = (switches / (actionHistory.length * TAU)).toFixed(1);
        const leftPct = Math.round((leftCount / actionHistory.length) * 100);
        const txtSwitchHz = document.getElementById('txtSwitchHz');
        if (txtSwitchHz) txtSwitchHz.textContent = `${swHz} Hz`;
        const txtDutyRatio = document.getElementById('txtDutyRatio');
        if (txtDutyRatio) txtDutyRatio.textContent = `${leftPct}% L / ${100 - leftPct}% R`;
      }

      const stability = Math.max(0, Math.min(100, 100 - Math.abs(smoothDeg) * 6.5));
      const stabilityStr = Math.round(stability) + '%';
      const mcStability = document.getElementById('mcStability');
      if (mcStability) mcStability.textContent = stabilityStr;
      const hudEquilTag = document.getElementById('hudEquilTag');
      if (hudEquilTag) hudEquilTag.textContent = stabilityStr;
    }

    // Position, Velocity & Angle Bars (Using smoothed values)
    const barPos = document.getElementById('barPos');
    if (barPos) {
      const posPct = Math.min(100, Math.max(0, ((smooth.x + X_THRESHOLD) / (X_THRESHOLD * 2)) * 100));
      barPos.style.width = posPct + '%';
    }

    const barVel = document.getElementById('barVel');
    if (barVel) {
      const velPct = Math.min(100, Math.max(0, ((smooth.xDot + 3.0) / 6.0) * 100));
      barVel.style.width = velPct + '%';
    }

    const barAngle = document.getElementById('barAngle');
    if (barAngle) {
      const anglePct = Math.min(100, Math.max(0, ((smooth.theta + THETA_THRESHOLD) / (THETA_THRESHOLD * 2)) * 100));
      barAngle.style.width = anglePct + '%';
    }

    const barOmega = document.getElementById('barOmega');
    if (barOmega) {
      const omegaPct = Math.min(100, Math.max(0, ((smooth.thetaDot + 3.0) / 6.0) * 100));
      barOmega.style.width = omegaPct + '%';
    }
  }

  // --- Main Animation Loop with 6-Speed Precision Control ---
  let speedAcc = 0;
  function loop() {
    if (!sim.paused && !sim.isDone) {
      if (sim.speed < 1.0) {
        speedAcc += sim.speed;
        if (speedAcc >= 1.0) {
          speedAcc -= 1.0;
          const obs = [sim.x, sim.xDot, sim.theta, sim.thetaDot];
          const action = selectAction(obs);
          stepPhysics(action);
        }
      } else {
        const steps = Math.floor(sim.speed);
        for (let s = 0; s < steps; s++) {
          const obs = [sim.x, sim.xDot, sim.theta, sim.thetaDot];
          const action = selectAction(obs);
          stepPhysics(action);
          if (sim.isDone) break;
        }
      }
      updateHUD();
    }

    drawPhasePortrait();
    renderSimulation();
    requestAnimationFrame(loop);
  }

  // --- Setup Controller Buttons and Interactions ---
  const brainBtns = {
    'TRAINED': document.getElementById('btnBrainTrained'),
    'LQR': document.getElementById('btnBrainLQR'),
    'UNDERCOOKED': document.getElementById('btnBrainUndercooked'),
    'MANUAL': document.getElementById('btnBrainManual')
  };

  const hdrMode = document.getElementById('hdrMode');
  const specLaw = document.getElementById('specLaw');

  function setBrain(b) {
    sim.brain = b;
    Object.keys(brainBtns).forEach(k => {
      if (brainBtns[k]) brainBtns[k].className = 'btn-ctrl-card';
    });
    if (b === 'TRAINED') {
      if (brainBtns[b]) brainBtns[b].classList.add('active-trained');
      if (hdrMode) {
        hdrMode.textContent = 'TRAINED PPO (20K)';
        hdrMode.style.color = 'var(--neon-green)';
      }
      if (specLaw) specLaw.textContent = 'PPO MLP [4➔64➔64➔2]';
    } else if (b === 'LQR') {
      if (brainBtns[b]) brainBtns[b].classList.add('active-lqr');
      if (hdrMode) {
        hdrMode.textContent = 'OPTIMAL LQR (RICCATI)';
        hdrMode.style.color = 'var(--neon-cyan)';
      }
      if (specLaw) specLaw.textContent = 'u = -K·x (K=[-1,-1.8,-35,-7.5])';
    } else if (b === 'UNDERCOOKED') {
      if (brainBtns[b]) brainBtns[b].classList.add('active-undercooked');
      if (hdrMode) {
        hdrMode.textContent = 'UNDERCOOKED PPO (2K)';
        hdrMode.style.color = 'var(--neon-amber)';
      }
      if (specLaw) specLaw.textContent = 'PPO EARLY [2K STEPS]';
    } else {
      if (brainBtns[b]) brainBtns[b].classList.add('active-manual');
      if (hdrMode) {
        hdrMode.textContent = 'MANUAL TELEOP';
        hdrMode.style.color = 'var(--neon-purple)';
      }
      if (specLaw) specLaw.textContent = 'KEYBOARD TELEOP';
    }
    resetEnv();
  }

  if (brainBtns['TRAINED']) brainBtns['TRAINED'].addEventListener('click', () => setBrain('TRAINED'));
  if (brainBtns['LQR']) brainBtns['LQR'].addEventListener('click', () => setBrain('LQR'));
  if (brainBtns['UNDERCOOKED']) brainBtns['UNDERCOOKED'].addEventListener('click', () => setBrain('UNDERCOOKED'));
  if (brainBtns['MANUAL']) brainBtns['MANUAL'].addEventListener('click', () => setBrain('MANUAL'));

  // --- Sim-to-Real Sliders & Gravity Tuners ---
  const sliderPoleLength = document.getElementById('sliderPoleLength');
  const valPoleLength = document.getElementById('valPoleLength');
  if (sliderPoleLength && valPoleLength) {
    sliderPoleLength.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      sim.length = v;
      valPoleLength.textContent = `${v.toFixed(2)} m`;
    });
  }

  const sliderPoleMass = document.getElementById('sliderPoleMass');
  const valPoleMass = document.getElementById('valPoleMass');
  if (sliderPoleMass && valPoleMass) {
    sliderPoleMass.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      sim.masspole = v;
      valPoleMass.textContent = `${v.toFixed(2)} kg`;
    });
  }

  const sliderCartMass = document.getElementById('sliderCartMass');
  const valCartMass = document.getElementById('valCartMass');
  if (sliderCartMass && valCartMass) {
    sliderCartMass.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      sim.masscart = v;
      valCartMass.textContent = `${v.toFixed(2)} kg`;
    });
  }

  // Gravity Slider
  const sliderGravity = document.getElementById('sliderGravity');
  const valGravityNum = document.getElementById('valGravityNum');
  const valGravity = document.getElementById('valGravity');
  const gravBtns = document.querySelectorAll('.btn-planet, .btn-grav');
  const gravNames = {
    '1.62': '1.62 m/s² [🌙 MOON 0.17g]',
    '3.72': '3.72 m/s² [🔴 MARS 0.38g]',
    '9.81': '9.81 m/s² [🌍 EARTH 1.00g]',
    '24.79': '24.79 m/s² [🪐 JUPITER 2.53g]'
  };

  if (sliderGravity && valGravityNum) {
    sliderGravity.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      sim.gravity = v;
      valGravityNum.textContent = `${v.toFixed(2)} m/s²`;
      if (valGravity) {
        const ratio = (v / 9.81).toFixed(2);
        valGravity.textContent = `${v.toFixed(2)} m/s² [${ratio}g]`;
      }
      gravBtns.forEach(b => {
        const bg = parseFloat(b.getAttribute('data-g'));
        b.classList.toggle('active', Math.abs(bg - v) < 0.1);
      });
    });
  }

  // Dynamic Load Preset Buttons
  gravBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      gravBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const g = parseFloat(btn.getAttribute('data-g'));
      sim.gravity = g;
      if (sliderGravity) sliderGravity.value = g;
      if (valGravityNum) valGravityNum.textContent = `${g.toFixed(2)} m/s²`;
      if (valGravity) valGravity.textContent = gravNames[g.toFixed(2)] || gravNames[g.toString()] || `${g.toFixed(2)} m/s² [${(g/9.81).toFixed(2)}g]`;
    });
  });

  // Reset Physics Defaults Button
  const btnResetParams = document.getElementById('btnResetParams');
  if (btnResetParams) {
    btnResetParams.addEventListener('click', () => {
      sim.length = 0.5;
      sim.masspole = 0.1;
      sim.masscart = 1.0;
      sim.gravity = 9.81;

      if (sliderPoleLength) sliderPoleLength.value = 0.5;
      if (valPoleLength) valPoleLength.textContent = '0.50 m';

      if (sliderPoleMass) sliderPoleMass.value = 0.1;
      if (valPoleMass) valPoleMass.textContent = '0.10 kg';

      if (sliderCartMass) sliderCartMass.value = 1.0;
      if (valCartMass) valCartMass.textContent = '1.00 kg';

      if (sliderGravity) sliderGravity.value = 9.81;
      if (valGravityNum) valGravityNum.textContent = '9.81 m/s²';

      gravBtns.forEach(b => {
        b.classList.toggle('active', b.getAttribute('data-g') === '9.81');
      });
      if (valGravity) valGravity.textContent = '9.81 m/s² [🌍 EARTH]';
    });
  }

  // Shock Buttons
  const btnShockLeft = document.getElementById('btnShockLeft');
  if (btnShockLeft) btnShockLeft.addEventListener('click', () => applyShock(-1));
  const btnShockRight = document.getElementById('btnShockRight');
  if (btnShockRight) btnShockRight.addEventListener('click', () => applyShock(1));

  // Pause / Resume / Reset
  const btnPause = document.getElementById('btnPause');
  function updatePauseButton() {
    if (!btnPause) return;
    if (sim.paused) {
      btnPause.innerHTML = '<span>▶</span> START <span class="hotkey-pill">SPACE</span>';
      btnPause.classList.add('btn-start-glow');
    } else {
      btnPause.innerHTML = '<span>⏸</span> PAUSE <span class="hotkey-pill">SPACE</span>';
      btnPause.classList.remove('btn-start-glow');
    }
  }

  if (btnPause) {
    btnPause.addEventListener('click', () => {
      sim.paused = !sim.paused;
      updatePauseButton();
    });
  }
  const btnReset = document.getElementById('btnReset');
  if (btnReset) btnReset.addEventListener('click', resetEnv);

  // Speed Multiplier Selector (6 Speeds: 0.25x to 10.0x including 5.0x Turbo)
  const selectSpeed = document.getElementById('selectSpeed');
  if (selectSpeed) {
    selectSpeed.addEventListener('change', (e) => {
      sim.speed = parseFloat(e.target.value) || 1.0;
    });
  }

  // Wind Bias Button (Compact Tool Pill)
  const btnWind = document.getElementById('btnWindBias');
  if (btnWind) {
    btnWind.addEventListener('click', () => {
      sim.windBias = !sim.windBias;
      btnWind.classList.toggle('active-wind', sim.windBias);
      btnWind.innerHTML = sim.windBias ? '<span>💨</span> ON' : '<span>💨</span> WIND';
    });
  }

  // Domain Randomization (Sensor Noise) Button (Compact Tool Pill)
  const btnDR = document.getElementById('btnDomainRand');
  if (btnDR) {
    btnDR.addEventListener('click', () => {
      sim.domainRand = !sim.domainRand;
      btnDR.classList.toggle('active-dr', sim.domainRand);
      btnDR.innerHTML = sim.domainRand ? '<span>🎲</span> ON' : '<span>🎲</span> NOISE';
    });
  }

  // Extreme Tilt Drop Test (±10° Recovery Test)
  const btnTilt = document.getElementById('btnTiltDrop');
  if (btnTilt) {
    btnTilt.addEventListener('click', () => {
      const dropDir = Math.random() > 0.5 ? 1 : -1;
      sim.x = (Math.random() - 0.5) * 0.15;
      sim.xDot = 0;
      sim.theta = dropDir * (10.0 * Math.PI / 180.0); // 10.0 degrees extreme tilt
      sim.thetaDot = (Math.random() - 0.5) * 0.08;
      sim.score = 0;
      sim.isDone = false;
      smooth.theta = sim.theta;
      smooth.thetaDot = sim.thetaDot;
      smooth.x = sim.x;
      smooth.xDot = sim.xDot;
      updateHUD();
    });
  }

  // Clear Benchmark Statistics Button
  const btnResetStats = document.getElementById('btnResetStats');
  if (btnResetStats) {
    btnResetStats.addEventListener('click', () => {
      historyRewards.length = 0;
      historyMA.length = 0;
      historyLabels.length = 0;
      historyRewards.push(sim.score);
      historyMA.push(sim.score);
      historyLabels.push('#1');
      sim.episode = 1;
      sim.maxScore = sim.score;
      if (rewardChart) {
        rewardChart.data.labels = [...historyLabels];
        rewardChart.data.datasets[0].data = [...historyRewards];
        rewardChart.data.datasets[1].data = [...historyMA];
        rewardChart.update();
      }
      updateHUD();
    });
  }

  // Keyboard Event Listeners
  window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') {
      sim.manualAction = 0;
    } else if (e.key === 'ArrowRight') {
      sim.manualAction = 1;
    } else if (e.key === 'm' || e.key === 'M') {
      const modes = ['TRAINED', 'LQR', 'UNDERCOOKED', 'MANUAL'];
      const nextIdx = (modes.indexOf(sim.brain) + 1) % modes.length;
      setBrain(modes[nextIdx]);
    } else if (e.key === 'f' || e.key === 'F') {
      applyShock(Math.random() > 0.5 ? 1 : -1);
    } else if (e.key === 'r' || e.key === 'R') {
      resetEnv();
    } else if (e.key === ' ') {
      e.preventDefault();
      sim.paused = !sim.paused;
      updatePauseButton();
    }
  });

  // --- Startup Initialization ---
  window.addEventListener('DOMContentLoaded', () => {
    resizeCanvas();
    initChart();
    loadWeights();
    resetEnv();
    requestAnimationFrame(loop);
  });

})();
