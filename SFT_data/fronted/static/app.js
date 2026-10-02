/* ManiSkill SFT 轨迹查看器 — 原生 JS 单页应用 */
'use strict';

const $ = (sel) => document.querySelector(sel);
const state = {
  filters: { teacher: '', task: '', status: '', q: '', sort: 'recent', page: 1,
             seed: '', tool: '', min_turns: '', max_turns: '', reason: '' },
  list: { total: 0, items: [] },
  listKey: null,
  episodeId: null,
  episode: null,
  turnsData: null,
  turn: 1,
  cam: 'base_camera',
  side: 'after',
  playing: false,
  speed: 1,
  zoom: { s: 1, x: 0, y: 0 },
  blinkTimer: null,
  playTimer: null,
  summary: null,
};

const CAM_LABELS = { base_camera: '1 主视角', hand_camera: '2 手部', render: '3 渲染' };
const STATUS_LABELS = { accepted: '已验收', rejected: '导出拒绝', other: '未验收' };

/* ---------------- utils ---------------- */
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}
const mediaUrl = (id, w) => id ? `/api/media/${id}${w ? `?thumb=1&w=${w}` : ''}` : null;
const fmtS = (s) => s == null ? '—' : s >= 60 ? `${(s / 60).toFixed(1)} min` : `${Number(s).toFixed(1)} s`;
const fmtTime = (s) => s ? new Date(s * 1000).toLocaleString() : '—';

/* ---------------- hash routing ---------------- */
function buildHash() {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(state.filters))
    if (v !== '' && v !== null && !(k === 'page' && v === 1) && !(k === 'sort' && v === 'recent'))
      p.set(k, v);
  if (state.episodeId) {
    p.set('turn', state.turn);
    if (state.cam !== 'base_camera') p.set('cam', state.cam);
    if (state.side !== 'after') p.set('side', state.side);
    return `#/e/${state.episodeId}?${p}`;
  }
  const q = p.toString();
  return `#/${q ? '?' + q : ''}`;
}
function parseHash() {
  const h = location.hash.slice(1) || '/';
  const [path, qs] = h.split('?');
  const p = new URLSearchParams(qs || '');
  for (const k of Object.keys(state.filters))
    state.filters[k] = p.has(k) ? (k === 'page' ? +p.get(k) : p.get(k)) : (k === 'page' ? 1 : (k === 'sort' ? 'recent' : ''));
  const m = path.match(/^\/e\/([^/]+)/);
  state.episodeId = m ? m[1] : null;
  if (m) {
    state.turn = p.has('turn') ? Math.max(1, +p.get('turn')) : 1;
    state.cam = p.get('cam') || 'base_camera';
    state.side = p.get('side') || 'after';
  }
}
function syncHash() { history.replaceState(null, '', buildHash()); }

/* ---------------- summary / overview ---------------- */
async function loadSummary() {
  state.summary = await api('/api/summary');
  const tp = $('#teacher-progress');
  tp.textContent = '';
  for (const [name, t] of Object.entries(state.summary.teachers)) {
    const b = el('span', 'badge');
    b.append(el('span', null, `${name.toUpperCase()} `), el('b', null, `${t.accepted || 0}`),
             el('span', null, `/${t.quota_total || '?'} 验收 · ${t.rejected || 0} 拒绝 · ${t.other || 0} 未验收`));
    tp.append(b);
  }
  const taskSel = $('#f-task');
  const cur = taskSel.value;
  taskSel.textContent = '';
  taskSel.append(el('option', null, '全部任务')); taskSel.lastChild.value = '';
  for (const t of state.summary.task_catalog || [])
    taskSel.append(el('option', null, t.task_slug));
  taskSel.value = cur;
}

function renderOverview() {
  const ov = $('#overview');
  ov.textContent = '';
  const grid = el('div', 'ov-grid');
  for (const [name, t] of Object.entries(state.summary.teachers)) {
    const card = el('div', 'ov-card');
    card.append(el('h2', null, `教师 ${name.toUpperCase()}`));
    const stats = el('div', 'ov-stats');
    const mk = (label, val) => { const s = el('span'); s.append(`${label} `, el('b', null, String(val))); return s; };
    stats.append(
      mk('验收', `${t.accepted || 0}/${t.quota_total || '?'}`),
      mk('导出拒绝', t.rejected || 0),
      mk('未验收', t.other || 0),
      mk('官方成功但未验收', t.official_success_unaccepted || 0),
      mk('最近采集', fmtTime(t.latest_s)),
    );
    card.append(stats);
    card.append(el('h3', null, '任务覆盖 (点击进入该任务已验收列表)'));
    const tasks = state.summary.task_catalog || [];
    for (const ct of tasks) {
      const info = (t.tasks || {})[ct.task_slug] || { accepted: 0, quota: ct.quota };
      const row = el('div', 'task-bar');
      row.append(el('span', 'task-name', ct.task_slug));
      const track = el('div', 'task-track');
      const fill = el('div', 'task-fill');
      fill.style.width = `${Math.min(100, 100 * info.accepted / (info.quota || 1))}%`;
      track.append(fill);
      row.append(track, el('span', 'task-count', `${info.accepted}/${info.quota ?? '?'}`));
      row.onclick = () => {
        Object.assign(state.filters, { teacher: name, task: ct.task_slug, status: 'accepted', page: 1 });
        state.episodeId = null; location.hash = buildHash();
      };
      card.append(row);
    }
    const fr = t.failure_reasons || {};
    if (Object.keys(fr).length) {
      card.append(el('h3', null, '失败原因分布'));
      for (const [reason, n] of Object.entries(fr).sort((a, b) => b[1] - a[1])) {
        const row = el('div', 'fail-row');
        row.append(el('span', null, reason), el('span', 'n', String(n)));
        row.style.cursor = 'pointer';
        row.onclick = () => {
          Object.assign(state.filters, { teacher: name, status: 'rejected', reason, page: 1 });
          state.episodeId = null; location.hash = buildHash();
        };
        card.append(row);
      }
    }
    grid.append(card);
  }
  ov.append(grid);
  if (state.summary.index_built_at)
    ov.append(el('p', 'hint', `索引构建于 ${fmtTime(state.summary.index_built_at)} · 点击右上角“刷新索引”增量更新`));
}

/* ---------------- episode list ---------------- */
async function loadList(force) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(state.filters)) if (v !== '' && v !== null) p.set(k, v);
  const key = p.toString();
  if (!force && key === state.listKey) return;
  state.listKey = key;
  state.list = await api(`/api/episodes?${p}`);
  const listEl = $('#episode-list');
  listEl.textContent = '';
  $('#list-meta').textContent = `${state.list.total} 条轨迹 · 按${{ recent: '最近时间', turns_desc: 'turn 数↓', turns_asc: 'turn 数↑', duration_desc: '耗时↓', seed_asc: 'seed↑', tokens_desc: 'token↓' }[state.filters.sort] || '最近时间'}排序`;
  for (const it of state.list.items) {
    const row = el('div', 'ep-row' + (it.session_id === state.episodeId ? ' sel' : ''));
    row.dataset.sid = it.session_id;
    const thumbs = el('div', 'ep-thumbs');
    for (const fid of [it.first_frame, it.last_frame]) {
      const img = el('img');
      img.loading = 'lazy';
      if (fid) img.src = mediaUrl(fid, 104);
      img.title = fid === it.first_frame ? '首帧' : '末帧';
      thumbs.append(img);
    }
    const info = el('div', 'ep-info');
    info.append(el('div', 'ep-title', it.episode_id || it.session_id));
    const sub = el('div', 'ep-sub');
    sub.append(el('span', `tag ${it.status}`, STATUS_LABELS[it.status] || it.status));
    if (it.official_success && it.status !== 'accepted')
      sub.append(el('span', 'tag reason', '官方成功未验收'));
    sub.append(el('span', null, `${it.task_slug} · seed ${it.seed}`));
    sub.append(el('span', null, `${it.n_turns} turn · ${fmtS(it.duration_s ?? it.elapsed_s)}`));
    info.append(sub);
    const reasons = (it.reject_reasons || []).concat(it.failure_reason ? [it.failure_reason] : []);
    if (reasons.length) {
      const r = el('div', 'ep-sub');
      for (const x of reasons.slice(0, 3)) r.append(el('span', 'tag reason', x));
      info.append(r);
    }
    row.append(thumbs, info);
    row.onclick = () => { state.turn = 1; location.hash = `#/e/${it.session_id}?${new URLSearchParams(cleanFilterParams())}`; };
    listEl.append(row);
  }
  $('#pg-info').textContent = `${state.filters.page} / ${Math.max(1, Math.ceil(state.list.total / state.list.items.length || 1))}`;
  $('#pg-prev').disabled = state.filters.page <= 1;
  $('#pg-next').disabled = state.filters.page * (state.list.items.length || 30) >= state.list.total;
}
function cleanFilterParams() {
  const o = {};
  for (const [k, v] of Object.entries(state.filters))
    if (v !== '' && v !== null && !(k === 'page' && v === 1) && !(k === 'sort' && v === 'recent')) o[k] = v;
  return o;
}

/* ---------------- viewer ---------------- */
async function loadEpisode() {
  const sid = state.episodeId;
  const [ep, turnsData] = await Promise.all([
    api(`/api/episodes/${sid}`), api(`/api/episodes/${sid}/turns`)]);
  state.episode = ep;
  state.turnsData = turnsData;
  $('#overview').hidden = true;
  $('#viewer').hidden = false;
  $('#detail-pane').hidden = false;
  $('#timeline-bar').hidden = false;
  $('#snapshot-note').textContent = turnsData.snapshot_note || '';

  const h = $('#ep-header');
  h.textContent = '';
  const title = el('div', 'title');
  title.append(el('span', null, ep.episode_id || sid), ' ',
               el('span', `tag ${ep.status}`, STATUS_LABELS[ep.status] || ep.status));
  if (ep.official_success) title.append(' ', el('span', 'tag accepted', '官方成功'));
  const sub = el('div', 'sub',
    `${ep.teacher} · ${ep.teacher_model || ''} · ${ep.task_slug} · seed ${ep.seed} · ` +
    `${ep.n_turns} turn · 耗时 ${fmtS(ep.duration_s ?? ep.elapsed_s)} · token ${ep.total_tokens ?? '—'} · ` +
    `采集于 ${fmtTime(ep.updated_at_s)} · batch ${ep.batch_id || '—'}`);
  h.append(title, sub);
  if (ep.task_text) {
    const d = el('details');
    d.append(el('summary', null, '任务指令全文'), el('div', 'reasoning', ep.task_text));
    h.append(d);
  }

  const w = $('#ep-warnings');
  w.textContent = '';
  for (const msg of (turnsData.warnings || [])) w.append(el('div', 'warn-line', msg));

  renderCamTabs();
  drawTimeline();
  if (!state.turnsData.turns.length) {
    $('#turn-indicator').textContent = '无 turn 数据';
    $('#stage-empty').hidden = false;
    for (const s of ['#d-decision .body', '#d-tools .body', '#d-state .body', '#d-sft .body', '#d-audit .body'])
      $(s).textContent = '';
    $('#d-decision .body').append(el('div', null, '该 rollout 没有记录任何 turn (可能启动即失败), 只能查看清单信息。'));
    return;
  }
  const maxTurn = state.turnsData.turns.length;
  selectTurn(Math.min(state.turn, maxTurn));
}

function availableCams() {
  const cams = new Set();
  for (const t of state.turnsData.turns) {
    Object.keys(t.frames_before || {}).forEach((c) => cams.add(c));
    Object.keys(t.frames_after || {}).forEach((c) => cams.add(c));
  }
  return [...cams];
}
function renderCamTabs() {
  const tabs = $('#cam-tabs');
  tabs.textContent = '';
  for (const c of availableCams()) {
    const b = el('button', c === state.cam ? 'on' : '', CAM_LABELS[c] || c);
    b.onclick = () => { state.cam = c; syncHash(); renderCamTabs(); renderStage(); };
    tabs.append(b);
  }
}

function currentTurnRow() {
  return state.turnsData.turns.find((t) => t.turn_index === state.turn) || state.turnsData.turns[0];
}

async function selectTurn(n, push = true) {
  stopBlink();
  state.turn = n;
  if (push) syncHash();
  $('#turn-indicator').textContent = `turn ${state.turn} / ${state.turnsData.turns.length}`;
  document.querySelectorAll('#episode-list .ep-row').forEach((r) =>
    r.classList.toggle('sel', r.dataset.sid === state.episodeId));
  drawTimeline();
  renderStage();
  try {
    const d = await api(`/api/episodes/${state.episodeId}/turns/${state.turn}`);
    renderDetail(d);
  } catch (e) {
    $('#d-decision .body').textContent = `加载 turn 详情失败: ${e.message}`;
  }
}

/* ---- stage (image area) ---- */
function frameFor(side) {
  const t = currentTurnRow();
  if (!t) return null;
  const frames = side === 'before' ? t.frames_before : t.frames_after;
  return (frames || {})[state.cam] || null;
}
function renderStage() {
  const stage = $('#stage');
  stage.classList.toggle('split', state.side === 'split');
  let img2 = $('#stage-img2');
  if (state.side === 'split') {
    if (!img2) {
      img2 = el('img'); img2.id = 'stage-img2'; img2.draggable = false;
      stage.append(img2);
    }
  } else if (img2) img2.remove();
  applyZoom();

  const setSrc = (img, fid) => {
    const url = mediaUrl(fid);
    if (url) { img.style.display = ''; img.src = url; }
    else img.style.display = 'none';
  };
  const img = $('#stage-img');
  const empty = $('#stage-empty');
  if (state.side === 'split') {
    setSrc(img, frameFor('before'));
    setSrc(img2, frameFor('after'));
    empty.hidden = !!(frameFor('before') || frameFor('after'));
  } else {
    const fid = frameFor(state.side === 'before' ? 'before' : 'after');
    setSrc(img, fid);
    empty.hidden = !!fid;
  }
  // prefetch 前后各 2 步
  const turns = state.turnsData.turns;
  const i = turns.findIndex((t) => t.turn_index === state.turn);
  for (const j of [i - 2, i - 1, i + 1, i + 2]) {
    const t = turns[j];
    if (!t) continue;
    for (const f of [t.frames_before, t.frames_after]) {
      const id = (f || {})[state.cam];
      if (id) { const p = new Image(); p.src = mediaUrl(id); }
    }
  }
  if (state.side === 'blink') startBlink();
}
function applyZoom() {
  const t = `translate(${state.zoom.x}px, ${state.zoom.y}px) scale(${state.zoom.s})`;
  for (const img of document.querySelectorAll('#stage img')) img.style.transform = t;
}
function startBlink() {
  let showBefore = true;
  state.blinkTimer = setInterval(() => {
    showBefore = !showBefore;
    const fid = frameFor(showBefore ? 'before' : 'after');
    const img = $('#stage-img');
    if (fid) img.src = mediaUrl(fid);
  }, 500);
}
function stopBlink() {
  if (state.blinkTimer) { clearInterval(state.blinkTimer); state.blinkTimer = null; }
}

/* ---- detail panel ---- */
function jsonDetails(summary, obj) {
  const d = el('details');
  d.append(el('summary', null, summary));
  const pre = el('pre', 'json');
  pre.textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 1);
  d.append(pre);
  return d;
}
function renderDetail(d) {
  for (const msg of (d.join_warnings || []))
    if (!$('#ep-warnings').textContent.includes(msg))
      $('#ep-warnings').append(el('div', 'warn-line', msg));

  // 决策
  const dec = $('#d-decision .body');
  dec.textContent = '';
  if (d.decision) {
    const head = el('div');
    head.append(el('span', 'pill ok', d.decision.name || d.decision.kind),
                el('span', 'pill', `call #${d.decision.seq}`),
                el('span', 'pill', `${fmtS(d.decision.duration_s)}`));
    dec.append(head);
    if (d.decision.reasoning) dec.append(el('div', 'reasoning', d.decision.reasoning));
    dec.append(jsonDetails('工具参数', d.decision.parameters));
  } else {
    dec.append(el('div', null, '该 turn 未能对齐到已验收的模型调用 (见顶部警告)。'));
  }
  if (d.validation && d.validation.errors && d.validation.errors.length)
    dec.append(el('div', 'warn-line', `校验错误: ${d.validation.errors.join('; ')}`));

  // 工具事件
  const tools = $('#d-tools .body');
  tools.textContent = '';
  if (!d.tool_events.length) tools.append(el('div', null, '该 turn 时间窗口内无工具事件记录。'));
  for (const e of d.tool_events) {
    const box = el('div', 'tool-ev');
    const head = el('div');
    head.append(el('span', 'pill', `${e.name} · ${e.phase}`));
    if (e.operational_success === 1) head.append(el('span', 'pill ok', '执行成功'));
    if (e.operational_success === 0) head.append(el('span', 'pill bad', '执行失败'));
    box.append(head);
    if (e.content) box.append(el('div', 'reasoning', e.content));
    if (e.parameters && Object.keys(e.parameters).length) box.append(jsonDetails('参数', e.parameters));
    if (e.stdout) {
      const dd = el('details');
      dd.append(el('summary', null, 'stdout'));
      const pre = el('pre', 'stdout'); pre.textContent = e.stdout;
      dd.append(pre); box.append(dd);
    }
    tools.append(box);
  }

  // reward / 状态
  const st = $('#d-state .body');
  st.textContent = '';
  const pills = el('div');
  const r = d.receipt;
  pills.append(el('span', 'pill', `reward ${r.reward ?? '—'}`));
  if (r.task_success) pills.append(el('span', 'pill ok', '官方成功'));
  if (r.official_reward) pills.append(el('span', 'pill ok', 'official reward'));
  if (r.terminated) pills.append(el('span', 'pill', 'terminated'));
  if (r.truncated) pills.append(el('span', 'pill bad', 'truncated'));
  st.append(pills);
  const ob = d.turn.objects_before || [], oa = d.turn.objects_after || [];
  if (ob.length || oa.length) {
    const mapB = Object.fromEntries(ob.map((o) => [o.name, o.position]));
    for (const o of oa) {
      const b = mapB[o.name];
      if (!b || !o.position) continue;
      const delta = Math.hypot(o.position[0] - b[0], o.position[1] - b[1], o.position[2] - b[2]);
      const kv = el('div', 'kv');
      kv.append(el('span', 'k', o.name),
                el('span', `v ${delta > 0.005 ? 'diff-pos' : ''}`,
                   `${b.map((x) => x.toFixed(3))} → ${o.position.map((x) => x.toFixed(3))} (Δ${delta.toFixed(3)}m)`));
      st.append(kv);
    }
  }
  const rb = d.turn.robot_before, ra = d.turn.robot_after;
  if (rb && ra && rb.end_effector_pose && ra.end_effector_pose) {
    const pb = rb.end_effector_pose.position || rb.end_effector_pose;
    const pa = ra.end_effector_pose.position || ra.end_effector_pose;
    if (Array.isArray(pb) && Array.isArray(pa)) {
      const kv = el('div', 'kv');
      kv.append(el('span', 'k', '末端位姿'),
                el('span', 'v', `${pb.map((x) => (+x).toFixed(3))} → ${pa.map((x) => (+x).toFixed(3))}`));
      st.append(kv);
    }
  }

  // SFT 样本
  const sft = $('#d-sft .body');
  sft.textContent = '';
  if (d.sft_sample) {
    const btn = el('button', null, `加载样本 ${d.sft_sample.sample_id}`);
    btn.onclick = async () => {
      btn.disabled = true;
      const s = await api(`/api/sft/${encodeURIComponent(d.sft_sample.sample_id)}`);
      sft.textContent = '';
      sft.append(el('div', 'kv'), );
      sft.lastChild.append(el('span', 'k', 'sample_id'), el('span', 'v', s.sample_id));
      for (const m of s.messages || []) {
        const box = el('div', 'msg');
        box.append(el('div', 'role', m.role));
        const pre = el('pre', 'stdout'); pre.textContent = m.content;
        box.append(pre);
        sft.append(box);
      }
    };
    sft.append(btn, el('div', 'hint', '仅展示最终用于训练的样本, 不用它判定轨迹成功'));
  } else {
    sft.append(el('div', null, '无对应 SFT 样本 (未验收或该调用未入选)。'));
  }

  // 审计
  const au = $('#d-audit .body');
  au.textContent = '';
  const btnCtx = el('button', null, '原始模型输出');
  const btnRaw = el('button', null, '原始 transition JSON');
  const out = el('pre', 'json'); out.hidden = true;
  btnCtx.onclick = async () => {
    const c = await api(`/api/episodes/${state.episodeId}/turns/${state.turn}/context`);
    out.hidden = false;
    out.textContent = (c.payload || '(无)') + (c.payload_truncated ? '\n…(已截断)' : '');
  };
  btnRaw.onclick = async () => {
    const raw = await api(`/api/episodes/${state.episodeId}/turns/${state.turn}/raw`);
    out.hidden = false;
    out.textContent = JSON.stringify(raw, null, 1);
  };
  au.append(btnCtx, ' ', btnRaw, out);
}

/* ---- timeline ---- */
const EVENT_COLORS = {
  ik_preview_check: '#4f9cf9', move_to: '#3fb96f', gripper_control: '#d9a13b',
  observe: '#8a94a5', python_exec: '#9b7ed9', official_success: '#ffffff',
  terminated: '#e05d5d', truncated: '#e05d5d',
};
function drawTimeline() {
  const cv = $('#timeline');
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const turns = state.turnsData ? state.turnsData.turns : [];
  if (!turns.length) return;
  const n = turns.length;
  const bw = w / n;
  const rewards = turns.map((t) => t.reward || 0);
  const rmax = Math.max(1e-9, ...rewards);
  // reward 曲线
  ctx.strokeStyle = '#3fb96f'; ctx.beginPath();
  turns.forEach((t, i) => {
    const x = i * bw + bw / 2, y = h - 6 - (t.reward || 0) / rmax * (h - 26);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  // turn 块 + 事件点
  turns.forEach((t, i) => {
    const x = i * bw;
    ctx.fillStyle = t.task_success ? '#2c5c40' : (t.terminated || t.truncated) ? '#5c2c2c' : '#232936';
    ctx.fillRect(x + 0.5, h - 14, Math.max(1, bw - 1), 12);
    const evs = t.key_events || [];
    evs.slice(0, 4).forEach((ev, j) => {
      ctx.fillStyle = EVENT_COLORS[ev] || '#666';
      ctx.beginPath();
      ctx.arc(x + bw / 2, 10 + j * 9, 2.5, 0, 7);
      ctx.fill();
    });
  });
  // 当前位置
  const i = turns.findIndex((t) => t.turn_index === state.turn);
  if (i >= 0) {
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
    ctx.strokeRect(i * bw, 2, bw, h - 4);
    ctx.lineWidth = 1;
  }
}
$('#timeline').addEventListener('click', (e) => {
  if (!state.turnsData) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const i = Math.floor((e.clientX - rect.left) / rect.width * state.turnsData.turns.length);
  const t = state.turnsData.turns[Math.max(0, Math.min(state.turnsData.turns.length - 1, i))];
  if (t) selectTurn(t.turn_index);
});

/* ---- playback & keyboard ---- */
function stepTurn(delta) {
  const turns = state.turnsData.turns;
  if (!turns.length) return;
  const i = turns.findIndex((t) => t.turn_index === state.turn);
  const j = i + delta;
  if (j >= 0 && j < turns.length) selectTurn(turns[j].turn_index);
}
function jumpKeyEvent(dir) {
  const turns = state.turnsData.turns;
  if (!turns.length) return;
  const i = turns.findIndex((t) => t.turn_index === state.turn);
  for (let j = i + dir; j >= 0 && j < turns.length; j += dir) {
    const evs = turns[j].key_events || [];
    if (evs.some((e) => ['ik_preview_check', 'move_to', 'gripper_control', 'official_success', 'terminated', 'truncated'].includes(e))) {
      selectTurn(turns[j].turn_index);
      return;
    }
  }
}
function setPlaying(on) {
  state.playing = on;
  $('#btn-play').textContent = on ? '⏸' : '▶';
  if (state.playTimer) { clearInterval(state.playTimer); state.playTimer = null; }
  if (on) {
    state.playTimer = setInterval(() => {
      const turns = state.turnsData.turns;
      const i = turns.findIndex((t) => t.turn_index === state.turn);
      if (i >= turns.length - 1) { setPlaying(false); return; }
      stepTurn(1);
    }, 1200 / state.speed);
  }
}
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea') || !state.episodeId || !state.turnsData) return;
  const cams = availableCams();
  switch (e.key) {
    case ' ': e.preventDefault(); setPlaying(!state.playing); break;
    case 'ArrowLeft': e.preventDefault(); e.shiftKey ? jumpKeyEvent(-1) : stepTurn(-1); break;
    case 'ArrowRight': e.preventDefault(); e.shiftKey ? jumpKeyEvent(1) : stepTurn(1); break;
    case '1': case '2': case '3': {
      const c = cams[+e.key - 1];
      if (c) { state.cam = c; syncHash(); renderCamTabs(); renderStage(); }
      break;
    }
    case 'b': case 'B':
      state.side = state.side === 'before' ? 'after' : 'before';
      syncHash(); updateSideSeg(); renderStage(); break;
  }
});

/* ---- stage zoom/pan ---- */
(function () {
  const stage = $('#stage');
  stage.addEventListener('wheel', (e) => {
    e.preventDefault();
    state.zoom.s = Math.max(0.2, Math.min(12, state.zoom.s * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    applyZoom();
  }, { passive: false });
  let drag = null;
  stage.addEventListener('mousedown', (e) => { drag = { x: e.clientX - state.zoom.x, y: e.clientY - state.zoom.y }; });
  window.addEventListener('mousemove', (e) => {
    if (!drag) return;
    state.zoom.x = e.clientX - drag.x; state.zoom.y = e.clientY - drag.y;
    applyZoom();
  });
  window.addEventListener('mouseup', () => { drag = null; });
  stage.addEventListener('dblclick', () => { state.zoom = { s: 1, x: 0, y: 0 }; applyZoom(); });
})();

/* ---- controls ---- */
function updateSideSeg() {
  document.querySelectorAll('#side-toggle button').forEach((b) =>
    b.classList.toggle('on', b.dataset.side === state.side));
}
document.querySelectorAll('#side-toggle button').forEach((b) => {
  b.onclick = () => { stopBlink(); state.side = b.dataset.side; syncHash(); updateSideSeg(); renderStage(); };
});
document.querySelectorAll('#speed-seg button').forEach((b) => {
  b.onclick = () => {
    state.speed = +b.dataset.speed;
    document.querySelectorAll('#speed-seg button').forEach((x) => x.classList.toggle('on', x === b));
    if (state.playing) setPlaying(true);
  };
});
$('#btn-play').onclick = () => setPlaying(!state.playing);
$('#btn-prev').onclick = () => stepTurn(-1);
$('#btn-next').onclick = () => stepTurn(1);
$('#btn-prev-key').onclick = () => jumpKeyEvent(-1);
$('#btn-next-key').onclick = () => jumpKeyEvent(1);
$('#pg-prev').onclick = () => { state.filters.page--; state.listKey = null; location.hash = buildHash(); };
$('#pg-next').onclick = () => { state.filters.page++; state.listKey = null; location.hash = buildHash(); };

const PRESETS = {
  'long-success': { status: 'accepted', min_turns: 30 },
  'ik-fail': { status: 'rejected', tool: 'ik_preview_check' },
  'token-limit': { reason: 'token' },
  'no-frames': { max_turns: 0 },
};
function bindFilter(id, key) {
  $(id).addEventListener('change', (e) => {
    state.filters[key] = e.target.value;
    state.filters.page = 1;
    state.listKey = null;
    location.hash = buildHash();
  });
}
bindFilter('#f-teacher', 'teacher');
bindFilter('#f-task', 'task');
bindFilter('#f-status', 'status');
$('#f-q').addEventListener('change', (e) => {
  state.filters.q = e.target.value; state.filters.page = 1; state.listKey = null;
  location.hash = buildHash();
});
$('#f-preset').addEventListener('change', (e) => {
  const p = PRESETS[e.target.value];
  if (p) {
    Object.keys(state.filters).forEach((k) => { state.filters[k] = k === 'page' ? 1 : (k === 'sort' ? 'recent' : ''); });
    Object.assign(state.filters, p);
    state.listKey = null;
    location.hash = buildHash();
  }
  e.target.value = '';
});
function syncFilterInputs() {
  $('#f-teacher').value = state.filters.teacher;
  $('#f-task').value = state.filters.task;
  $('#f-status').value = state.filters.status;
  $('#f-q').value = state.filters.q;
}
$('#btn-reindex').onclick = async () => {
  const btn = $('#btn-reindex');
  btn.disabled = true; btn.textContent = '索引中…';
  await fetch('/api/reindex', { method: 'POST' });
  const poll = setInterval(async () => {
    const s = await api('/api/reindex/status');
    if (!s.running) {
      clearInterval(poll);
      btn.disabled = false; btn.textContent = '刷新索引';
      state.listKey = null;
      await loadSummary();
      route(true);
    }
  }, 1000);
};

/* ---- router ---- */
async function route(force) {
  parseHash();
  syncFilterInputs();
  updateSideSeg();
  if (state.episodeId) {
    await loadList();
    if (force || !state.turnsData || state.turnsData.session_id !== state.episodeId) {
      state.episode = null; state.turnsData = null;
      await loadEpisode();
    } else {
      selectTurn(state.turn, false);
    }
  } else {
    $('#overview').hidden = false;
    $('#viewer').hidden = true;
    $('#detail-pane').hidden = true;
    $('#timeline-bar').hidden = true;
    setPlaying(false);
    renderOverview();
    await loadList();
  }
}
window.addEventListener('hashchange', () => route());
window.addEventListener('resize', () => { if (state.turnsData) drawTimeline(); });

(async function init() {
  await loadSummary();
  await route();
})();
