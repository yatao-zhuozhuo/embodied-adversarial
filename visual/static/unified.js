const $ = id => document.getElementById(id);
const base = new URL('.', document.currentScript.src);
const escapeHtml = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let tasks = [], selected = null, detail = null, labels = {}, version = 0;
async function get(path) {
  const response = await fetch(new URL(path, base), {cache:'no-store'});
  if (!response.ok) throw new Error(`读取失败（${response.status}），请刷新重试`);
  const data = await response.json();
  if (data.error) throw new Error(data.error);
  return data;
}
function drawList() {
  const query = $('task-search').value.toLowerCase();
  const view = tasks.filter(t => (!$('task-bench').value || t.bench === $('task-bench').value)
    && (!$('task-role').value || t.role === $('task-role').value)
    && [t.task_key,t.task,t.family].join(' ').toLowerCase().includes(query));
  $('task-count').textContent = `${view.length} / ${tasks.length} 道题`;
  $('task-list').innerHTML = view.map(t => `<button class="task-item ${t.id === selected ? 'active' : ''}" data-task="${escapeHtml(t.id)}"><b>${escapeHtml(t.task_id)}</b><small>${escapeHtml(t.bench)} · ${t.role === 'shared' ? '共同题' : '泛化预留'} · ${escapeHtml(t.family)}</small><small>${Object.entries(t.case_counts).map(([k,n]) => `${escapeHtml(labels[k])} ${n}`).join(' / ')}</small></button>`).join('');
  $('task-list').querySelectorAll('button').forEach(b => b.onclick = () => openTask(b.dataset.task));
}
function dl(id, entries) {
  $(id).innerHTML = entries.map(([k,v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v ?? '未记录')}</dd>`).join('');
}
function drawDetail(data) {
  const t = data.task, c = data.selected_case, state = c.initial_state, p = data.preview;
  const observed = data.runtime_check || {};
  detail = data;
  $('task-title').textContent = t.task_id;
  $('task-badge').textContent = `${t.bench.toUpperCase()} · ${t.role === 'shared' ? '共同训练 / 验证 / 测试题' : '预留泛化题'}${t.generalization_scope?.includes('robot') ? ' · 同时涉及机器人 / 任务配置泛化' : ''}`;
  $('task-instruction').textContent = observed.task_instruction || c.task;
  $('verification').textContent = observed.capture_success
    ? observed.initial_success ? '初始已成功 · 需检查'
      : observed.action_dim_matches === false ? '动作契约需检查'
      : observed.restoration_passed === false ? '缓存状态恢复需检查'
      : observed.fresh_instance_checked && !observed.fresh_instance_reproducible ? '跨环境复现需检查'
      : observed.same_instance_reproducible && observed.fresh_instance_reproducible ? '跨环境复现已核验'
      : observed.same_instance_reproducible ? '同环境复现已核验' : '已采集 · 复现需检查'
    : observed.error_category === 'missing_assets' ? '缺资产 · 未能检查'
      : observed.error ? '运行检查受阻' : c.runtime_verified ? '已核验' : '待运行核验';
  const counts = tasks.find(x => x.id === selected).case_counts;
  $('split-counts').innerHTML = Object.entries(counts).map(([k,n]) => `<div class="split-chip ${escapeHtml(k)}"><strong>${n}</strong><span>${escapeHtml(labels[k])}初态</span></div>`).join('');
  $('case-phase').innerHTML = Object.keys(counts).map(k => `<option value="${escapeHtml(k)}">${escapeHtml(labels[k])}</option>`).join('');
  $('case-phase').value = c.phase;
  const phaseCases = data.cases.filter(x => x.phase === c.phase);
  $('case-picker').innerHTML = phaseCases.map((x,i) => `<option value="${escapeHtml(x.case_id)}">初态 ${i+1} / ${phaseCases.length} · ${x.initial_state.kind === 'reference_demo' ? `示范 #${x.initial_state.reference_demo_index} · 恢复前 seed ${x.seed}` : `seed ${x.seed}`} · ${x.runtime_capture_available ? '真实截图已采集' : '尚未采集真实截图'}</option>`).join('');
  $('case-picker').value = c.case_id;
  $('preview-label').textContent = p.label;
  $('preview-status').textContent = p.frames.length ? '正在读取保存的图片…' : '未保存此初态的相机图像';
  $('task-previews').replaceChildren();
  if (!p.frames.length) {
    const box = document.createElement('div'); box.className = 'empty-preview';
    const title = document.createElement('b'); title.textContent = '所选初态尚未生成真实截图';
    const note = document.createElement('span'); note.textContent = '目前不能从图片比较这些 seed 的场景差异；需要实际 reset 或恢复缓存状态后采集截图。';
    box.append(title,note); $('task-previews').append(box);
  }
  let loaded = 0, failed = 0;
  p.frames.forEach(frame => {
    const figure = document.createElement('figure'), image = document.createElement('img'), caption = document.createElement('figcaption');
    image.alt = `${t.task_id} · ${frame.camera}`;
    caption.textContent = `${frame.camera} · ${p.kind === 'actual_reset' ? '所选初态的真实截图' : p.kind === 'cached_demo' ? '缓存示范首帧' : '历史同题示例'}`;
    const settled = () => { if (detail !== data) return; $('preview-status').textContent = `${loaded} / ${p.frames.length} 张图片已加载${failed ? ` · ${failed} 张不可用` : ''}`; };
    image.onload = () => { loaded++; settled(); };
    image.onerror = () => { failed++; caption.textContent += ' · 图片不可用'; image.hidden = true; settled(); };
    image.src = new URL(frame.url, base); figure.append(image,caption); $('task-previews').append(figure);
  });
  const example = data.example_preview;
  $('history-example').hidden = !example;
  $('history-example').open = false;
  $('history-previews').replaceChildren();
  $('history-source').textContent = example ? `${example.label} · ${example.source}${example.example_seed != null ? ` · 历史 seed ${example.example_seed}` : ''}` : '';
  $('history-example').ontoggle = () => {
    if (!$('history-example').open || !example || $('history-previews').children.length) return;
    example.frames.forEach(frame => {
      const figure = document.createElement('figure'), image = document.createElement('img'), caption = document.createElement('figcaption');
      image.alt = `${t.task_id} · 历史参考 · 与所选 seed 无关`;
      caption.textContent = `${frame.camera} · 同题历史参考，不代表所选初态`;
      image.onerror = () => { image.hidden = true; caption.textContent += ' · 图片不可用'; };
      image.src = new URL(frame.url, base); figure.append(image,caption); $('history-previews').append(figure);
    });
  };
  dl('case-info', [['用途',labels[c.phase]],['初态筛选',c.diversity_selection_method === 'task_geometry_maximin' ? '按实测 / 缓存物体与目标差异选择' : '尚未采集几何信息，差异待确认'],['Seed',c.seed],['初态来源',state.kind === 'reference_demo' ? `缓存示范 #${state.reference_demo_index}` : '原生 seed'],['Seed 含义',c.seed_role === 'pre_restore_seed' ? '恢复缓存状态前使用；状态以示范编号为准' : '创建 / reset seed'],['实际状态指纹',observed.state_fingerprint || '待实际 reset 捕获'],['重复重置差异',observed.repeat_max_difference ?? '未检查'],['跨环境复现',observed.fresh_instance_checked ? (observed.fresh_instance_reproducible ? '通过' : '需检查') : '此 case 未单独检查'],['缓存状态指纹',state.state_fingerprint || '不适用'],['预览来源',p.source || '无保存图像'],...(observed.error ? [['检查错误',observed.error]] : [])]);
  dl('task-info', [['机器人',t.robot],['动作维度',t.expected_action_dim],['控制器',t.environment_config.control_mode],['控制步上限',t.horizon],['泛化范围',t.generalization_scope?.includes('robot') ? '任务 / 配置 + 机器人迁移（单列成绩）' : t.role === 'generalization' ? '未训练任务' : '共同任务的独立初态'],['选题理由',t.selection_reason]]);
  $('task-raw').textContent = JSON.stringify({task:t,selected_case:c,runtime_check:observed,initial_state_check:data.initial_state_check,preview:p,example_preview:example},null,2);
}
async function openTask(id, caseId) {
  const request = ++version;
  selected = id; drawList(); $('task-error').textContent = ''; $('preview-status').textContent = '读取题目…';
  $('task-previews').replaceChildren();
  $('preview-label').textContent = '正在读取所选初态…';
  $('history-example').open = false;
  $('history-example').hidden = true;
  $('history-previews').replaceChildren();
  try {
    const data = await get(`api/unified/tasks/${encodeURIComponent(id)}${caseId ? '?case=' + encodeURIComponent(caseId) : ''}`);
    if (request !== version) return;
    drawDetail(data);
    const url = new URL(location.href); url.searchParams.set('task',id); history.replaceState(null,'',url);
  } catch (error) { if (request === version) $('task-error').textContent = error.message; }
}
async function refresh() {
  $('task-error').textContent = '';
  try {
    const data = await get('api/unified/tasks'); tasks = data.items; labels = data.phase_labels;
    $('revision').textContent = `划分 v${data.split_revision} · ${tasks.filter(t=>t.role === 'shared').length} 个共同题 + ${tasks.filter(t=>t.role === 'generalization').length} 个泛化题 · SFT ${data.case_counts.sft_train} / RL ${data.case_counts.rl_train} / 验证 ${data.case_counts.validation} / 同任务测试 ${data.case_counts.eval_seen} / 泛化 ${data.case_counts.eval_generalization}`;
    const benches = data.active_benches || [...new Set(tasks.map(t=>t.bench))];
    $('task-bench').innerHTML = '<option value="">全部 Benchmark</option>' + benches.map(b=>`<option value="${escapeHtml(b)}">${escapeHtml(b)}</option>`).join('');
    $('benchmark-summary').innerHTML = benches.map(b => {
      const rows = tasks.filter(t => t.bench === b);
      return `<button class="benchmark-card" data-bench="${b}"><b>${b}</b><strong>${rows.length}<span class="muted"> 道题</span></strong><small>${rows.filter(t => t.role === 'shared').length} 共同题 + ${rows.filter(t => t.role === 'generalization').length} 泛化题</small></button>`;
    }).join('');
    $('benchmark-summary').querySelectorAll('button').forEach(b => b.onclick = () => { $('task-bench').value=b.dataset.bench; drawList(); });
    drawList();
    const wanted = selected || new URL(location.href).searchParams.get('task');
    await openTask(tasks.find(t => t.id === wanted)?.id || tasks[0].id);
  } catch (error) { $('task-error').textContent = error.message; }
}
['task-bench','task-role'].forEach(id => $(id).onchange=drawList); $('task-search').oninput=drawList;
$('case-phase').onchange = () => openTask(selected,detail.cases.find(c => c.phase === $('case-phase').value).case_id);
$('case-picker').onchange = () => openTask(selected,$('case-picker').value);
$('refresh-tasks').onclick=refresh;
refresh();
