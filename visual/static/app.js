const $ = id => document.getElementById(id);
// Resolve against the loaded script so index.html, query strings and proxy prefixes work.
const appBase = new URL('.', document.currentScript.src);
const api = path => new URL(path, appBase);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let source = 'baseline', rows = [], episode = null, selected = null, turn = 0, requestedTurn = 0;
let playing = false, timer = null, playbackVersion = 0, renderVersion = 0, selectionVersion = 0, listVersion = 0;
const cameras = new Map(), frameCache = new Map();
async function fetchJson(path) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(api(path), {signal: controller.signal, cache: 'no-store'});
    if (!response.ok) throw new Error(`请求失败：${response.status} · ${response.url}`);
    if (!response.headers.get('content-type')?.includes('application/json'))
      throw new Error('接口返回了网页，请检查代理地址或登录状态：' + response.url);
    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('请求超过 20 秒，请刷新重试并检查代理连接。');
    throw error;
  } finally { clearTimeout(timeout); }
}
const resultLabel = value => value === true ? '成功' : value === false ? '失败' : '未知';
function filtered() {
  const query = $('search').value.toLowerCase();
  return rows.filter(r => (!$('bench').value || r.bench === $('bench').value)
    && (!$('group').value || r.group === $('group').value)
    && (!$('model').value || r.model === $('model').value)
    && (!$('result').value || (r.success == null ? 'unknown' : String(r.success)) === $('result').value)
    && (!source.startsWith('expert') || !$('annotation').value || ($('annotation').value === 'some' ? r.annotated_turns > 0 : !r.annotated_turns))
    && (!query || [r.label,r.task,r.model,r.group,r.status].join(' ').toLowerCase().includes(query)));
}
function drawList() {
  const view = filtered(), success = view.filter(r => r.success === true).length;
  const known = view.filter(r => typeof r.success === 'boolean').length;
  $('summary').textContent = `${success} 成功 / ${known} 已判定 · ${known ? (100 * success / known).toFixed(1) + '%' : '无成功率'}（按轨迹/attempt）`;
  $('count').textContent = `${view.length} / ${rows.length} 条轨迹`;
  $('list').innerHTML = view.map((r, i) => `<button class="item ${selected === r.id ? 'active' : ''}" data-index="${i}"><b>${escapeHtml(r.bench)} · ${escapeHtml(r.model)}</b><small>${escapeHtml(r.label)}</small><small class="muted">${escapeHtml(r.group)} · ${r.turns} 步${r.source === 'expert' ? ` · 补写 ${r.annotated_turns}/${r.turns}` : ''}</small><span class="${r.success === true ? 'success' : r.success === false ? 'failure' : 'muted'}">${resultLabel(r.success)}</span> · ${escapeHtml(r.status)}${r.reward == null ? '' : ' · R ' + Number(r.reward).toFixed(3)}</button>`).join('');
  $('list').querySelectorAll('button').forEach(b => b.onclick = () => openEpisode(view[Number(b.dataset.index)]));
}
function options(id, field, label) {
  const value = $(id).value;
  $(id).innerHTML = `<option value="">${label}</option>` + [...new Set(rows.map(r => r[field]))].sort().map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join('');
  if ([...$(id).options].some(o => o.value === value)) $(id).value = value;
}
async function refresh() {
  const version = ++listVersion, currentSource = source;
  $('count').textContent = '正在读取轨迹，首次索引可能需要几秒…';
  try {
    const endpoint = 'api/episodes?background=1&source=' + currentSource;
    let listing = await fetchJson(endpoint);
    while (listing.loading) {
      if (version !== listVersion) return;
      $('source-root').textContent = listing.root;
      if (listing.items?.length) {
        rows = listing.items;
        options('bench', 'bench', '全部 benchmark'); options('group', 'group', '全部 run / round'); options('model', 'model', '全部模型 / 角色');
        drawList();
      }
      $('count').textContent = listing.total
        ? `正在更新索引：${listing.processed ?? 0} / ${listing.total} 个文件${rows.length ? ` · 已显示 ${rows.length} 条轨迹` : ''}`
        : `正在准备扫描${rows.length ? ` · 已显示 ${rows.length} 条轨迹` : ''}…`;
      await new Promise(resolve => setTimeout(resolve, 1000));
      if (version !== listVersion) return;
      listing = await fetchJson(endpoint + '&poll=1');
    }
    if (version !== listVersion) return;
    if (listing.error) throw new Error('索引失败：' + listing.error);
    rows = listing.items;
    $('source-root').textContent = listing.root;
    $('error').textContent = !listing.exists ? '数据目录不存在，请检查启动参数。' : listing.errors.length ? `${listing.errors.length} 个文件暂时无法读取（可能正在写入），刷新后重试。` : '';
    options('bench', 'bench', '全部 benchmark'); options('group', 'group', '全部 run / round'); options('model', 'model', '全部模型 / 角色');
    drawList();
    // Keep an open live trajectory in sync without changing its position.
    const row = rows.find(r => r.id === selected);
    if (row && episode && row.modified !== episode.modified) await openEpisode(row, true);
  } catch (error) { if (version === listVersion) { $('error').textContent = error.message; $('count').textContent = '读取失败，请点击刷新轨迹重试。'; } }
}
function framesAt(record, index) {
  if ($('phase').value === 'input') return index ? (record.steps[index - 1].input_frames || []) : [];
  return index ? record.steps[index - 1][$('phase').value === 'before' ? 'before_frames' : 'frames'] : record.initial_frames;
}
function loadFrame(url) {
  if (frameCache.has(url)) return frameCache.get(url);
  const image = new Image(); image.src = api(url);
  const pending = image.decode().then(() => image).catch(error => { frameCache.delete(url); throw error; });
  frameCache.set(url, pending);
  while (frameCache.size > 32) frameCache.delete(frameCache.keys().next().value);
  return pending;
}
function setupCameras(record) {
  const names = new Set(record.initial_frames.map(f => f.camera));
  record.steps.forEach(s => [...s.frames, ...s.before_frames, ...(s.input_frames || [])].forEach(f => names.add(f.camera)));
  cameras.clear(); $('images').replaceChildren();
  for (const name of names) {
    const figure = document.createElement('figure'), canvas = document.createElement('canvas'), caption = document.createElement('figcaption');
    canvas.width = canvas.height = 512; canvas.setAttribute('aria-label', name);
    figure.append(canvas, caption); $('images').append(figure); cameras.set(name, {canvas, caption});
  }
}
function drawChart() {
  const canvas = $('chart'), ctx = canvas.getContext('2d'), width = canvas.width, height = canvas.height;
  ctx.clearRect(0,0,width,height);
  const values = episode ? episode.steps.map(s => typeof s.reward === 'number' && Number.isFinite(s.reward) ? s.reward : null) : [];
  const known = values.filter(v => v != null);
  ctx.fillStyle = '#8a94a5'; ctx.font = '13px sans-serif';
  if (!known.length) { ctx.fillText('本轨迹未记录逐步环境奖励',12,23); return; }
  const low = Math.min(...known), high = Math.max(...known), span = high - low || 1;
  ctx.fillText(`环境 reward  ${low.toFixed(3)} — ${high.toFixed(3)} · 点击跳转`,12,17);
  const x = i => 15 + (width - 30) * i / Math.max(values.length - 1, 1);
  ctx.strokeStyle = '#4f9cf9'; ctx.lineWidth = 2; ctx.beginPath(); let connected = false;
  values.forEach((v,i) => { if (v == null) { connected = false; return; } const y = height - 12 - (v - low) / span * (height - 40); if (connected) ctx.lineTo(x(i),y); else ctx.moveTo(x(i),y); connected = true; }); ctx.stroke();
  ctx.strokeStyle = '#6bd8a5'; ctx.beginPath(); ctx.moveTo(x(Math.max(0,turn-1)),24); ctx.lineTo(x(Math.max(0,turn-1)),height); ctx.stroke();
}
async function openEpisode(row, keepPosition = false) {
  stop(); const version = ++selectionVersion, prior = keepPosition ? turn : 0;
  $('status').textContent = '加载轨迹…';
  try {
    const record = await fetchJson('api/episode/' + row.id);
    if (version !== selectionVersion) return;
    selected = row.id; episode = record; turn = requestedTurn = Math.min(prior, record.steps.length);
    frameCache.clear(); setupCameras(record);
    $('title').textContent = `${record.bench} / ${record.label}`;
    $('meta').textContent = `${record.task}${record.source === 'expert' ? ` ｜ 动作：专家回放 ｜ reasoning：模型事后补写 ${record.annotated_turns}/${record.turns} 步` : ''} ｜ ${record.protocol ?? 'unknown'}${record.images_retained === false ? ' ｜ 审计记录：图片已删除' : ''} ｜ ${record.model} ｜ ${resultLabel(record.success)} · ${record.status}${record.reward == null ? '' : ' ｜ rollout reward: ' + record.reward}`;
    $('audit').textContent = JSON.stringify({source:record.path, ...record.metadata}, null, 2);
    $('slider').max = record.steps.length; drawList(); await render(turn);
  } catch (error) { if (version === selectionVersion) $('status').textContent = error.message; }
}
async function render(index) {
  if (!episode) return false;
  const record = episode, version = ++renderVersion, target = Math.max(0,Math.min(index, record.steps.length));
  requestedTurn = target; const frames = framesAt(record,target); $('status').textContent = '加载帧…';
  const decoded = await Promise.all(frames.map(async f => {
    try { return {...f, image: await loadFrame(f.url)}; } catch (_) { return {...f, image:null}; }
  }));
  if (version !== renderVersion || record !== episode) return false;
  await new Promise(resolve => requestAnimationFrame(resolve));
  if (version !== renderVersion || record !== episode) return false;
  const available = new Map(decoded.map(f => [f.camera,f]));
  for (const [name, camera] of cameras) {
    const ctx = camera.canvas.getContext('2d'), frame = available.get(name);
    camera.canvas.parentElement.hidden = $('phase').value === 'input' ? !frame : name.startsWith('image-');
    ctx.fillStyle = '#000'; ctx.fillRect(0,0,512,512);
    if (frame?.image) {
      const scale = Math.min(512/frame.image.naturalWidth,512/frame.image.naturalHeight), w = frame.image.naturalWidth*scale, h = frame.image.naturalHeight*scale;
      ctx.drawImage(frame.image,(512-w)/2,(512-h)/2,w,h);
    }
    camera.caption.textContent = `${name} · ${$('phase').value === 'input' ? '模型实际输入' : target ? ($('phase').value === 'before' ? '动作前' : '动作后') : '初始状态'}${!frame?.image ? ' · 无可用图像' : ''}`;
  }
  turn = target; $('slider').value = target; $('position').textContent = `${target} / ${record.steps.length}`;
  const step = target ? record.steps[target-1] : null;
  $('detail').textContent = step ? `第 ${target} 步 · reward: ${step.reward ?? '未记录'} · 步骤成功: ${resultLabel(step.success)}\n\n${record.output_label || '模型原始输出'}\n${step.annotated_xml || step.text || '本轮无文本输出'}${record.source === 'expert' ? `\n\n模型事后补写 reasoning（${step.reasoning_annotation?.reasoning_model || '未补写'}）：\n${step.reasoning_annotation?.reasoning || '本步尚无补写内容'}${step.annotated_xml ? '\n\n补写前 XML（原始记录）：\n' + step.text : ''}` : ''}\n\n动作与回执\n${JSON.stringify(step.raw,null,2)}` : `初始状态\n\n${record.task}\n\n最终判定：${resultLabel(record.success)} · ${record.status}\n${record.source === 'training' ? '逐步曲线是环境 reward；rollout reward 见顶部和元数据。' : ''}`;
  const missing = decoded.filter(f => !f.image).length;
  $('status').textContent = missing ? `${missing} 张图像缺失或加载失败，仍可查看动作与文本。` : frames.length ? '逐步快照 · 空格播放/暂停 · ← → 切换步骤' : '本步未保存相机帧';
  if ($('phase').value === 'input') $('detail').textContent = step?.model_input ? `本轮模型实际输入（不含宿主评分）\n\n${JSON.stringify(step.model_input,null,2)}` : '本轮未记录统一输入包，不能把宿主图片当作模型输入。';
  $('chart').hidden = $('phase').value === 'input';
  drawChart();
  for (let next=target+1;next<=Math.min(target+2,record.steps.length);next++) framesAt(record,next).forEach(f => loadFrame(f.url).catch(()=>{}));
  return true;
}
function stop() { playing=false; ++playbackVersion; ++renderVersion; requestedTurn=turn; clearTimeout(timer); timer=null; $('play').textContent='播放'; }
async function playStep(version) {
  if (!playing || version !== playbackVersion) return;
  const ok = await render(turn >= episode.steps.length ? 0 : turn+1);
  if (!playing || version !== playbackVersion) return;
  if (!ok || turn >= episode.steps.length) { stop(); return; }
  timer=setTimeout(()=>playStep(version),Number($('speed').value));
}
function seek(index) { stop(); render(index); }
$('play').onclick=()=>{ if(playing){stop();return;} if(!episode)return; playing=true; $('play').textContent='暂停'; playStep(++playbackVersion); };
$('prev').onclick=()=>seek(requestedTurn-1); $('next').onclick=()=>seek(requestedTurn+1);
$('slider').oninput=e=>seek(Number(e.target.value)); $('phase').onchange=()=>seek(turn);
$('chart').onclick=e=>{ if(episode?.steps.length)seek(1+Math.round(Math.max(0,Math.min(1,(e.offsetX/e.currentTarget.clientWidth*1000-15)/970))*(episode.steps.length-1))); };
$('refresh').onclick=refresh;
for (const id of ['bench','group','model','result','annotation']) $(id).onchange=drawList;
$('search').oninput=drawList;
document.querySelectorAll('[data-source]').forEach(button=>button.onclick=()=>{
  if(source===button.dataset.source)return;
  stop(); ++selectionVersion; ++listVersion; source=button.dataset.source; rows=[]; episode=null; selected=null;
  turn=requestedTurn=0; $('slider').value=0; $('slider').max=0; $('position').textContent='';
  $('images').replaceChildren(); cameras.clear(); frameCache.clear(); $('detail').textContent='选择左侧记录查看。'; $('audit').textContent=''; $('title').textContent='选择一条轨迹'; $('meta').textContent=''; $('status').textContent=''; $('error').textContent='';
  $('legacy').style.display=source==='sft'?'block':'none';
  $('annotation').style.display=source.startsWith('expert')?'block':'none';
  for(const id of ['bench','group','model','result','search','annotation'])$(id).value='';
  if(source==='expert')$('annotation').value='some';
  document.querySelectorAll('[data-source]').forEach(b=>b.classList.toggle('active',b===button));
  drawChart(); drawList(); refresh();
});
document.onkeydown=e=>{ if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName))return; if(e.key==='ArrowRight'){e.preventDefault();seek(requestedTurn+1);} if(e.key==='ArrowLeft'){e.preventDefault();seek(requestedTurn-1);} if(e.key===' '){e.preventDefault();$('play').click();} };
setInterval(()=>{ if($('auto').checked && !document.hidden)refresh(); },30000);
fetchJson('api/context').then(c=>{$('context').textContent=`当前分支：${c.branch} · baseline：${c.baseline_branch}`;}).catch(e=>{$('error').textContent=e.message;});
drawChart();
const initialSource = new URLSearchParams(location.search).get('source');
const initialButton = [...document.querySelectorAll('[data-source]')].find(button => button.dataset.source === initialSource);
if(initialButton && initialSource !== source)initialButton.click(); else refresh();
