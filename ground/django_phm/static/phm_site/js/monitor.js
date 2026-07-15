  // ====================== 颜色常量（Canvas 不支持 var()，必须用 hex） ======================
  // 与 :root 中的 CSS 变量保持同步
  const C = {
    bgPrimary: '#0b0f1a', bgSecondary: '#131825', bgCard: '#1a1f2f',
    border: '#2a3348', textPri: '#e0e6f0', textSec: '#8e9bb5',
    blue: '#2d8cf0', green: '#19be6b', yellow: '#f5a623',
    red: '#ed3f14', cyan: '#00c9db'
  };

  // ====================== 全局状态 ======================
  const state = {
    currentChannel: null,
    selectedFolderId: null,     // 当前选中的文件夹（新建传感器默认父节点）
    mode: 'realtime',           // 'realtime' | 'frozen'
    deviceTree: [],
    sensors: {},
    folders: {},                // Slice 0：文件夹聚合健康度 {folderId: {name, health, strategy, channels}}
    systemHealth: 0,
    windowData: null,
    alerts: [],
    warnings: [],
    dbStats: {},
    viewEndTs: null,            // 冻结模式右边界（epoch s）
    cacheStartTs: null,         // 当前缓存数据的最早时间戳（epoch s）
    cacheEndTs: null,           // 当前缓存数据的最晚时间戳（epoch s）
    pollingTimers: {},
    linkOk: true,
    linkFailCount: 0,
    hoveredIdx: -1,             // 图表悬停的 raw 索引
    hoveredPredIdx: -1,         // 图表悬停的 prediction 索引
    chartLayout: null,          // 缓存上次 drawChart 的坐标映射，供 tooltip 用
    modalOpen: false,           // 模态框打开时暂停 canvas 重绘（避免输入卡顿）
    diagnosedKeys: new Set(),   // 已诊断告警的 key 集合 "channel|type|ts"
  };

  function formatTime(ts) {
    if (ts == null) return '—';
    // UTC 时间显示（遥测标准）：时间戳是 epoch 秒，用 toISOString 直接转 UTC
    const d = new Date(ts * 1000);
    return d.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
  }
  function fmt(n, d=3) { return n == null ? '—' : n.toFixed(d); }

  // CSS 颜色（用于 DOM style，可以用 var()）
  function colorForScore(s) {
    if (s == null) return 'var(--text-secondary)';
    // 阈值对齐 ANOMALY_THRESHOLD(0.5)：>0.5 红（异常），>0.25 黄（警戒），否则绿
    return s > 0.5 ? 'var(--accent-red)' : (s > 0.25 ? 'var(--accent-yellow)' : 'var(--accent-green)');
  }
  function healthColor(v) {
    if (v < 60) return 'var(--accent-red)';
    if (v < 80) return 'var(--accent-yellow)';
    return 'var(--accent-green)';
  }

  // ====================== API 层 ======================
  const api = {
    async request(method, path, body) {
      const opts = { method, headers: {} };
      if (body) {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(body);
      }
      const res = await fetch(path, opts);
      if (path.includes('/export')) {
        if (!res.ok) throw new Error('导出失败');
        return res.blob();
      }
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || '请求失败');
      return data;
    },
    poll(body) { return this.request('POST','/api/poll',body); },
    forecast(values) { return this.request('POST','/api/forecast',{values}); },
    getConfig() { return this.request('GET','/api/config'); },
    saveConfig(tree) { return this.request('POST','/api/config',{device_tree:tree}); },
    reset() { return this.request('POST','/api/reset'); },
    health(blockSize) { return this.request('GET',`/api/health?block_size=${blockSize||20000}`); },
    alerts(limit) { return this.request('GET',`/api/alerts?limit=${limit||50}`); },
    warnings(limit) { return this.request('GET',`/api/warnings?limit=${limit||50}`); },
    predictScores(ch) { return this.request('GET',`/api/predict-scores?channel=${encodeURIComponent(ch)}`); },
    sensors() { return this.request('GET','/api/sensors'); },
    history(params) { return this.request('GET','/api/history?'+new URLSearchParams(params)); },
    deleteHistory(params) { return this.request('DELETE','/api/history?'+new URLSearchParams(params)); },
    detection(params) { return this.request('GET','/api/detection?'+new URLSearchParams(params)); },
    deleteDetection(params) { return this.request('DELETE','/api/detection?'+new URLSearchParams(params)); },
    dbStats() { return this.request('GET','/api/db-stats'); },
    window(params) { return this.request('GET','/api/window?'+new URLSearchParams(params)); },
    alertsHistory(limit) { return this.request('GET',`/api/alerts/history?limit=${limit||50}`); },
    patchAlert(id, status) { return this.request('PATCH',`/api/alerts/${id}`,{status}); },
    warningVerdict(id, verdict) { return this.request('POST',`/api/warnings/${id}/verdict`,{human_verdict:verdict}); },
    alertVerdict(channel, ts, verdict) { return this.request('POST','/api/alerts/verdict',{channel,alert_ts:ts,human_verdict:verdict}); },
    diagnosisAuto() { return this.request('POST','/api/diagnosis/auto',{}); },
    diagnosisAutoStatus() { return this.request('GET','/api/diagnosis/auto/status'); },
    async exportData(params) {
      const blob = await this.request('GET','/api/export?'+new URLSearchParams(params));
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `export.${params.fmt||'csv'}`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  // ====================== 轮询管理 ======================
  const pollManager = {
    start(name, fn, interval) {
      this.stop(name);
      state.pollingTimers[name] = setInterval(async () => {
        try { await fn(); state.linkFailCount = 0; updateLinkStatus(true); }
        catch(e) { state.linkFailCount++; if(state.linkFailCount>=3) updateLinkStatus(false); }
      }, interval);
    },
    stop(name) {
      if(state.pollingTimers[name]) { clearInterval(state.pollingTimers[name]); delete state.pollingTimers[name]; }
    },
    stopAll() { Object.keys(state.pollingTimers).forEach(n => this.stop(n)); }
  };

  function updateLinkStatus(ok) {
    state.linkOk = ok;
    document.getElementById('linkDot').className = 'status-dot' + (ok ? '' : ' lost');
    document.getElementById('linkText').textContent = ok ? '天地链路 正常' : '天地链路 中断';
  }

  // ====================== 数据获取 ======================
  async function fetchWindow() {
    if(!state.currentChannel) return;
    // 预取缓存：拉 CACHE_COUNT 行（4× 可视区），drawChart 截取可视子段画
    const params = { channel: state.currentChannel, count: CACHE_COUNT };
    if(state.mode === 'frozen' && state.viewEndTs != null) params.end_ts = state.viewEndTs;
    const data = await api.window(params);

    // 冻结模式预取合并：如果已有缓存且通道没变，合并新旧数据（按 timestamp
    // 去重排序），避免向左/向右预取时整体替换导致另一侧边界塌缩（拖不回原位置）。
    // 首次加载/切通道时不合并（旧数据属于别的通道）。
    const old = state.windowData;
    if (old && old.channel === data.channel && old.data && old.data.length > 0
        && old.start_ts != null && data.start_ts != null) {
      // 时间范围有重叠或相邻 → 合并
      const overlap = data.start_ts <= old.end_ts && data.end_ts >= old.start_ts;
      if (overlap) {
        const byTs = new Map();
        // 新数据优先（更新鲜），旧数据补充
        for (const d of old.data) byTs.set(d.timestamp, d);
        for (const d of data.data) byTs.set(d.timestamp, d);
        const merged = Array.from(byTs.values()).sort((a, b) => a.timestamp - b.timestamp);
        data.data = merged;
        data.start_ts = Math.min(old.start_ts, data.start_ts);
        data.end_ts = Math.max(old.end_ts, data.end_ts);
        // gaps 合并去重
        if (old.gaps && old.gaps.length) {
          const gapSet = new Map();
          for (const g of [...(old.gaps||[]), ...(data.gaps||[])]) gapSet.set(g.start + '_' + g.end, g);
          data.gaps = Array.from(gapSet.values());
        }
      }
    }

    // 统一序列模型：data.data[] 每行含 raw_value + predicted_value 同行不同列
    state.windowData = data;
    // 记录缓存时间范围（合并后可能比单次拉取更大），供拖拽边界预取判定用
    state.cacheStartTs = data.start_ts;
    state.cacheEndTs = data.end_ts;
    // 记住窗口边界，用于冻结模式平移
    if (data.end_ts != null && state.mode === 'frozen' && state.viewEndTs == null) {
      state.viewEndTs = data.end_ts;
    }
    markChartDirty();  // 新数据到达 → 标记 chart 待重绘
  }

  async function fetchHealth() {
    const data = await api.health();
    state.systemHealth = data.system;
    // Slice 0：消费后端文件夹聚合健康度（min/mean 策略），供设备树着色
    state.folders = data.folders || {};
    drawHealthRing();
    renderDeviceTree(); // 文件夹健康度变了 → 重绘设备树着色
  }

  async function fetchSensors() {
    const data = await api.sensors();
    state.sensors = {};
    const liveSensors = data.sensors || [];
    if (liveSensors.length) {
      // space 端在跑：用 RingBuffer 实时数据
      liveSensors.forEach(s => state.sensors[s.channel] = s);
    } else {
      // space 端未启动：从 SQLite 拉每个通道的最新点回填仪表
      const channels = getFlatSensors().map(n => n.channelName).filter(Boolean);
      await Promise.all(channels.map(async (ch) => {
        try {
          const w = await api.window({ channel: ch, count: 1 });
          if (w.data && w.data.length) {
            const last = w.data[w.data.length - 1];
            state.sensors[ch] = {
              channel: ch,
              latest_raw: last.raw_value,
              latest_score: last.anomaly_score,
              points: w.count || 0,
              received_at: last.timestamp,
              health: 100,
            };
          }
        } catch(e) {}
      }));
    }
    renderGauges();
    renderDeviceTree();
  }

  async function fetchAlerts() {
    const data = await api.alerts();
    state.alerts = data.alerts || [];
    renderAlertsList();
  }

  async function fetchWarnings() {
    const data = await api.warnings();
    state.warnings = data.warnings || [];
    renderWarningsList();
  }

  async function fetchDiagnosedKeys() {
    try {
      const resp = await fetch('/api/diagnosis/done');
      const data = await resp.json();
      state.diagnosedKeys = new Set(
        (data.done || []).map(d => `${d.channel}|${d.alert_type}|${d.alert_ts}`)
      );
    } catch(e) {}
  }

  function isDiagnosed(channel, alertType, ts) {
    return ts != null && state.diagnosedKeys.has(`${channel}|${alertType}|${ts}`);
  }

  async function fetchDbStats() {
    try { state.dbStats = await api.dbStats(); } catch(e) {}
    if(document.getElementById('dbModalOverlay').classList.contains('active') && dbCurrentTab===0) renderDbOverview();
  }

  // ====================== UI 渲染 ======================
  function renderDeviceTree() {
    markDiagramDirty();  // 设备树任何变化（增删改/拖拽/健康度/选中态）都需同步示意图
    const container = document.getElementById('deviceTree');
    container.innerHTML = '';
    const walk = (nodes, depth=0) => {
      nodes.forEach(node => {
        const div = document.createElement('div');
        div.className = 'tree-item';
        div.style.paddingLeft = (8 + depth*20) + 'px';
        div.draggable = true;
        div.dataset.id = node.id;
        if(node.type === 'sensor' && node.channelName === state.currentChannel) div.classList.add('selected');
        if(node.type === 'folder' && state.selectedFolderId === node.id) div.classList.add('selected');

        const dot = document.createElement('span');
        dot.className = 'item-dot';
        let score = null;
        let folderHealth = null;
        if(node.type === 'sensor' && state.sensors[node.channelName]) {
          score = state.sensors[node.channelName].latest_score;
        } else if(node.type === 'folder' && state.folders && state.folders[node.id]) {
          // Slice 0：文件夹圆点按聚合健康度着色（health∈[0,100]，<60红/60-80黄/>80绿）
          folderHealth = state.folders[node.id].health;
        }
        if(node.type === 'folder') {
          dot.style.background = folderHealth == null ? 'var(--text-secondary)' : healthColor(folderHealth);
        } else {
          dot.style.background = score == null ? 'var(--text-secondary)' : (score>0.5?'var(--accent-red)':(score>0.25?'var(--accent-yellow)':'var(--accent-green)'));
        }
        div.appendChild(dot);

        const icon = document.createElement('span');
        icon.className = 'item-icon';
        icon.textContent = node.type === 'folder' ? '📁' : '📡';
        div.appendChild(icon);

        const name = document.createElement('span');
        name.className = 'item-name';
        name.textContent = node.name;
        div.appendChild(name);

        if(node.type === 'sensor' && score != null) {
          const s = document.createElement('span');
          s.className = 'item-score';
          s.textContent = score.toFixed(3);
          s.style.color = colorForScore(score);
          div.appendChild(s);
        } else if(node.type === 'folder' && folderHealth != null) {
          // Slice 0：文件夹显示聚合健康度（min 策略=最差通道决定）
          const s = document.createElement('span');
          s.className = 'item-score';
          const strat = state.folders[node.id].strategy || 'min';
          s.textContent = `${folderHealth.toFixed(0)}%${strat === 'min' ? ' ⚡' : ''}`;
          s.style.color = healthColor(folderHealth);
          div.appendChild(s);
        }

        const delBtn = document.createElement('button');
        delBtn.className = 'delete-btn';
        delBtn.textContent = '✕';
        delBtn.onclick = (e) => { e.stopPropagation(); deleteNode(node.id); };
        div.appendChild(delBtn);

        div.onclick = (e) => {
          e.stopPropagation();
          // 单击/双击区分：延迟 250ms 判断是否有第二次点击
          if (div._clickTimer) {
            clearTimeout(div._clickTimer);
            div._clickTimer = null;
            // 双击 → 编辑（传感器编辑配置，文件夹重命名）
            openEditDeviceModal(node);
          } else {
            div._clickTimer = setTimeout(() => {
              div._clickTimer = null;
              // 单击 → 选中（传感器选通道，文件夹记录为"新建传感器默认父节点"）
              if(node.type === 'sensor') {
                selectChannel(node.channelName);
              } else if(node.type === 'folder') {
                state.selectedFolderId = (state.selectedFolderId === node.id) ? null : node.id;
                renderDeviceTree(); // 切换高亮（内部 markDiagramDirty）
              }
            }, 250);
          }
        };

        // 拖拽排序
        div.addEventListener('dragstart', (e) => {
          state.draggingId = node.id;
          div.classList.add('dragging');
          e.dataTransfer.effectAllowed = 'move';
        });
        div.addEventListener('dragend', () => { div.classList.remove('dragging'); });
        div.addEventListener('dragover', (e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = 'move';
          // 文件夹高亮"拖入"背景，普通节点用顶部蓝线表示插入位置
          if(node.type === 'folder') div.classList.add('drop-into');
          else div.classList.add('drag-over');
        });
        div.addEventListener('dragleave', () => { div.classList.remove('drag-over'); div.classList.remove('drop-into'); });
        div.addEventListener('drop', (e) => {
          e.preventDefault();
          e.stopPropagation();
          div.classList.remove('drag-over');
          div.classList.remove('drop-into');
          if(state.draggingId && state.draggingId !== node.id) {
            moveNode(state.draggingId, node.id);
            state.draggingId = null;
          }
        });

        container.appendChild(div);
        if(node.children) walk(node.children, depth+1);
      });
    };
    walk(state.deviceTree);
  }

  // 拖拽移动节点：目标是文件夹 → 放入其 children 末尾；否则插到目标前面（同级排序）
  function moveNode(srcId, targetId) {
    const findAndRemove = (nodes) => {
      for(let i=0;i<nodes.length;i++) {
        if(nodes[i].id === srcId) { return nodes.splice(i,1)[0]; }
        if(nodes[i].children) { const r = findAndRemove(nodes[i].children); if(r) return r; }
      }
      return null;
    };
    const findNode = (nodes, id) => {
      for(const n of nodes) {
        if(n.id === id) return n;
        if(n.children) { const r = findNode(n.children, id); if(r) return r; }
      }
      return null;
    };
    const insertBefore = (nodes, target, item) => {
      for(let i=0;i<nodes.length;i++) {
        if(nodes[i].id === target) { nodes.splice(i,0,item); return true; }
        if(nodes[i].children && insertBefore(nodes[i].children, target, item)) return true;
      }
      return false;
    };
    // 防止把文件夹拖进它自己的子孙（会成环）
    const srcNode = findNode(state.deviceTree, srcId);
    if(srcNode && srcNode.type === 'folder') {
      const isDescendant = (parent, id) => {
        if(!parent.children) return false;
        for(const c of parent.children) {
          if(c.id === id) return true;
          if(c.type === 'folder' && isDescendant(c, id)) return true;
        }
        return false;
      };
      if(isDescendant(srcNode, targetId)) return; // 拖进自己子孙 → 拒绝
    }
    const item = findAndRemove(state.deviceTree);
    if(!item) return;
    const target = findNode(state.deviceTree, targetId);
    if(target && target.type === 'folder') {
      // 拖到文件夹上 → 成为它的子节点
      if(!target.children) target.children = [];
      target.children.push(item);
    } else {
      insertBefore(state.deviceTree, targetId, item);
    }
    autoSaveConfig();
    renderDeviceTree();
  }

  function renderGauges() {
    const grid = document.getElementById('gaugesGrid');
    // 从设备树读通道列表（而非依赖 RingBuffer 的 state.sensors）
    // 这样即使 space 端未启动、RingBuffer 空，仪表区也会显示传感器卡片（数据为 —）
    const sensors = getFlatSensors();
    if(!sensors.length) {
      grid.innerHTML = '<div class="empty-state">🚧 请先添加传感器</div>';
      return;
    }
    grid.innerHTML = sensors.map(node => {
      const ch = node.channelName || node.name;
      const s = state.sensors[ch] || {};  // 可能为空（RingBuffer 无数据）
      const sel = ch === state.currentChannel ? ' selected' : '';
      return `<div class="gauge-card${sel}" onclick="selectChannel('${ch}')">
        <div class="card-title">📡 ${node.name} [${ch}]</div>
        <div class="gauge-value">${fmt(s.latest_raw,4)}</div>
        <div class="gauge-score" style="color:${colorForScore(s.latest_score)}">异常分数: ${fmt(s.latest_score,4)}</div>
        <div class="gauge-health" style="color:${healthColor(s.health != null ? s.health : 100)}">健康: ${fmt(s.health,1)}%</div>
      </div>`;
    }).join('');
  }

  function statusLabel(s) {
    const m = {real:'实警', false_alarm:'虚警', uncertain:'待定', pending:'待核验',
               confirmed:'已证实', false:'已证伪', unverifiable:'无法核验', active:'实报'};
    return m[s] || s;
  }
  function verdictButtons(type, id, humanVerdict) {
    // type='measured' → id={channel,ts}; type='predicted' → id=number
    const fn = type==='measured'
      ? (v)=>`submitAlertVerdict('${id.channel}',${id.ts},'${v}')`
      : (v)=>`submitWarningVerdict(${id},'${v}')`;
    const hv = humanVerdict || null;
    const make = (v,label) => `<button class="vbtn ${hv===v?'active-'+v:''}" onclick="${fn(v)}">${label}</button>`;
    return `<span class="verdict-btns">${make('real','实警')}${make('false_alarm','虚警')}${make('uncertain','待定')}</span>`;
  }

  function renderAlertsList() {
    const container = document.getElementById('alertsList');
    if(!state.alerts.length) { container.innerHTML = '<div class="empty-state">✅ 当前无告警</div>'; return; }
    container.innerHTML = state.alerts.map(a => {
      const done = isDiagnosed(a.channel, 'measured', a.time);
      const fs = a.final_status || 'active';
      return `
      <div class="alert-item">
        <span>${a.channel} · ${fmt(a.score)}</span>
        <span class="badge ${fs}">${statusLabel(fs)}</span>
        <span style="font-size:0.7rem;">${formatTime(a.time)}</span>
        ${verdictButtons('measured', {channel:a.channel, ts:a.time}, a.human_verdict)}
        <button class="diag-btn ${done ? 'diag-done' : ''}" onclick="requestDiagnosis('${a.channel}','measured',${a.time},this)">${done ? '✓' : '诊断'}</button>
      </div>`;
    }).join('');
  }

  function renderWarningsList() {
    const container = document.getElementById('warningsList');
    if(!state.warnings.length) { container.innerHTML = '<div class="empty-state">暂无预测预警</div>'; return; }
    container.innerHTML = state.warnings.map(w => {
      const done = isDiagnosed(w.channel, 'predicted', w.created_at);
      const fs = w.final_status || w.verify_status || w.status || 'pending';
      return `
      <div class="warning-item">
        <span>${w.channel} · ${fmt(w.max_predict_score)}</span>
        <span class="badge ${fs}">${statusLabel(fs)}</span>
        <span style="font-size:0.7rem;">${formatTime(w.created_at)}</span>
        ${verdictButtons('predicted', w.id, w.human_verdict)}
        <button class="diag-btn ${done ? 'diag-done' : ''}" onclick="requestDiagnosis('${w.channel}','predicted',${w.created_at},this)">${done ? '✓' : '诊断'}</button>
      </div>`;
    }).join('');
  }

  function selectChannel(ch) {
    state.currentChannel = ch;
    state.viewEndTs = null;
    state.hoveredIdx = -1;
    markChartDirty();    // 切通道 → 清旧 hover，fetchWindow 回来再标一次
    renderDeviceTree();  // 内部会 markDiagramDirty（脉冲点转移到新选中通道）
    renderGauges();
    if(state.mode === 'realtime') {
      pollManager.stop('chart');
      pollManager.start('chart', fetchWindow, 2000);
      fetchWindow();  // 立即拉取，不等 2s 定时器（消除切换等 3s）
    } else {
      // 切换通道时冻结模式也抓一次最新
      state.viewEndTs = null;
      fetchWindow();
    }
  }

  function setMode(mode) {
    state.mode = mode;
    document.getElementById('btnRealTime').classList.toggle('active', mode==='realtime');
    document.getElementById('btnFrozen').classList.toggle('active', mode==='frozen');
    if(mode === 'realtime') {
      state.viewEndTs = null;
      pollManager.start('chart', fetchWindow, 2000);
      fetchWindow();  // 立即拉最新，不等定时器
    } else {
      pollManager.stop('chart');
      // 冻结时抓一次当前最新作为起点
      state.viewEndTs = null;
      fetchWindow();
    }
  }

  async function resetChart() {
    // 调后端清空内存（RingBuffer + AlertStore + WarningStore）
    try { await api.reset(); } catch(e) {}
    state.viewEndTs = null;
    state.alerts = [];
    state.warnings = [];
    renderAlertsList();
    renderWarningsList();
    setMode('realtime');
  }

  // ====================== Canvas 图表（rAF 持续重绘） ======================
  const canvas = document.getElementById('chartCanvas');
  const ctx = canvas.getContext('2d');
  const tooltip = document.getElementById('chartTooltip');
  const timelineBar = document.getElementById('timelineBar');
  const timelineWindow = document.getElementById('timelineWindow');
  let dpr = window.devicePixelRatio || 1;
  let mouseX = -1, mouseY = -1;
  let isDragging = false;
  // dirty flag：只在数据/交互变化时重绘对应 canvas，避免每帧空转双 canvas
  // chart 是纯数据驱动（无时间动画）→ 事件驱动；diagram 有选中态脉冲光晕（正弦）
  // → 有选中通道时仍需每帧重绘，无选中时事件驱动。
  let chartDirty = true, diagramDirty = true;
  const markChartDirty = () => { chartDirty = true; };
  const markDiagramDirty = () => { diagramDirty = true; };
  // 预取缓存：fetchWindow 拉 CACHE_COUNT 行（4× 可视区），drawChart 按 viewEndTs
  // 截取最后 VIEW_COUNT 行画。拖拽时本地截取重画（无 HTTP），仅 viewEndTs 距
  // 缓存边界不足 PREFETCH_THRESHOLD 行时才重新 fetch。
  const CACHE_COUNT = 2048;        // 单次拉取行数（后端上限 10000）
  const VIEW_COUNT = 512;          // 可视区行数
  const PREFETCH_THRESHOLD = 256;  // 距缓存边界多少行触发预取
  // Y 轴范围输入框：drawChart 每次读取，改值后需标记重绘（之前靠 rAF 每帧兜底）
  ['yMin', 'yMax'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', markChartDirty);
  });

  function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    // 减去 timelineBar 高度
    const tlH = 36;
    canvas.width = rect.width * dpr;
    canvas.height = (rect.height - tlH) * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = (rect.height - tlH) + 'px';
  }

  // 主绘制函数：每帧由 rAF 调用
  function drawChart() {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);

    if(!state.windowData || !state.windowData.data || !state.windowData.data.length) {
      ctx.fillStyle = C.textSec;
      ctx.font = `${14*dpr}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText('🚧 等待数据', w/2, h/2);
      state.chartLayout = null;
      updateTimeline();
      return;
    }

    // 统一序列：每行含 timestamp + raw_value/predicted_value 等四列
    // 预取缓存：windowData 含 CACHE_COUNT 行，这里按 viewEndTs 截取可视 VIEW_COUNT 行。
    // 后续所有循环（X 映射、Y 范围、画线、hover）都遍历这个子段。
    const fullData = state.windowData.data;
    let data;
    if (state.mode === 'frozen' && state.viewEndTs != null && fullData.length > VIEW_COUNT) {
      // 二分定位：找 timestamp <= viewEndTs 的最后一个点的下一位（上界）
      const target = state.viewEndTs;
      let lo = 0, hi = fullData.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (fullData[mid].timestamp <= target) lo = mid + 1; else hi = mid;
      }
      const endIdx = Math.min(lo, fullData.length);
      const startIdx = Math.max(0, endIdx - VIEW_COUNT);
      data = fullData.slice(startIdx, endIdx);
    } else {
      // 实时模式 或 数据不足：取最后 VIEW_COUNT 行
      data = fullData.slice(-VIEW_COUNT);
    }
    const topH = h * 0.7, bottomH = h * 0.3;
    const pad = { top: 20*dpr, right: 50*dpr, bottom: 30*dpr, left: 60*dpr };

    // ---- X 轴：时间轴折叠 ----
    // 后端返回 gaps 列表（系统中断），折叠后 gap 两侧数据紧凑排列，
    // gap 位置画竖虚线+"中断Xh"标注。这样 tSpan 不再爆炸，拖拽不卡。
    const tsMs = data.map(d => d.timestamp * 1000);
    const plotW = w - pad.left - pad.right;
    const gapList = state.windowData.gaps || [];

    // 找到每个 gap 在 data 中的分界索引
    const gapIndices = [];
    for (const g of gapList) {
      // g.end 是 gap 后第一个点的 timestamp（秒），转 ms 比较
      const idx = data.findIndex(d => Math.abs(d.timestamp * 1000 - g.end * 1000) < 1);
      if (idx > 0) gapIndices.push({ index: idx, duration_s: g.duration });
    }

    // 计算折叠后的视觉 X 坐标：数据段按真实时间比例分配宽度，
    // gap 占固定 GAP_W 像素。先算各段真实时间跨度，再按比例分配。
    const GAP_W = 40 * dpr;  // 每个缺口占的视觉宽度
    let segCount = gapIndices.length + 1;
    let totalGapW = gapIndices.length * GAP_W;
    let dataW = plotW - totalGapW;  // 数据区总像素
    if (dataW < plotW * 0.3) { dataW = plotW * 0.3; }  // 保证数据区至少30%

    // 每段的 [startIndex, endIndex) 和真实时间跨度
    const segments = [];
    let segStart = 0;
    for (const gi of gapIndices) {
      segments.push({ start: segStart, end: gi.index, dur: tsMs[gi.index - 1] - tsMs[segStart] });
      segStart = gi.index;
    }
    segments.push({ start: segStart, end: data.length, dur: tsMs[data.length - 1] - tsMs[segStart] });

    // 每段真实时间跨度之和（用于按比例分配 dataW）
    const totalDur = segments.reduce((s, seg) => s + Math.max(seg.dur, 1), 0);

    // 为每个点计算折叠后的 X 坐标
    const foldedX = new Array(data.length);
    let cursorX = pad.left;
    let segIdx = 0;
    for (let i = 0; i < data.length; i++) {
      if (segIdx < segments.length && i >= segments[segIdx].end) {
        cursorX += GAP_W;  // 跨过 gap
        segIdx++;
      }
      const seg = segments[segIdx];
      const segW = dataW * Math.max(seg.dur, 1) / totalDur;
      const frac = seg.dur > 0 ? (tsMs[i] - tsMs[seg.start]) / seg.dur : 0;
      foldedX[i] = cursorX + frac * segW;
      // 段内最后一个点时推进 cursorX
      if (i === seg.end - 1) {
        cursorX += segW;
      }
    }

    // 折叠后总跨度（像素→时间）：用于拖拽计算，不含 gap
    const tMin = tsMs[0];
    const tMax = tsMs[tsMs.length - 1];
    // 折叠后的"等效真实时间跨度"=数据段真实时间之和（不含gap）
    const foldedTSpan = segments.reduce((s, seg) => s + Math.max(seg.dur, 1), 1);
    const xOf = (i) => foldedX[i];  // 改为按索引取折叠坐标

    // 缺口检测阈值（3 倍中位数间隔）——用于段内偶然丢包断线
    const gapThreshold = (() => {
      if (tsMs.length < 3) return Infinity;
      const diffs = [];
      for (let i = 1; i < tsMs.length; i++) {
        // 跳过 gap 处（gap 已折叠，不参与中位数计算）
        if (gapIndices.some(gi => gi.index === i)) continue;
        diffs.push(tsMs[i] - tsMs[i-1]);
      }
      if (diffs.length === 0) return Infinity;
      diffs.sort((a,b) => a-b);
      return diffs[Math.floor(diffs.length/2)] * 3;
    })();

    // 悬停索引：在折叠坐标上找最近点
    state.hoveredIdx = -1;
    if (mouseX >= pad.left && mouseX <= pad.left + plotW) {
      let best = 0, bestDist = Infinity;
      for (let i = 0; i < foldedX.length; i++) {
        const d = Math.abs(foldedX[i] - mouseX);
        if (d < bestDist) { bestDist = d; best = i; }
      }
      state.hoveredIdx = best;
    }

    // ---- 绘制子图 ----
    // rawKey: 实测值列名, predKey: 预测值列名
    // fixedYRange: [min,max] 固定 Y 轴（异常分数图用 [0,1]）
    function drawSubChart(y0, height, rawKey, predKey, rawColor, predColor, threshold, fixedYRange) {
      let minV, maxV;
      if (fixedYRange) {
        minV = fixedYRange[0]; maxV = fixedYRange[1];
      } else {
        minV = Infinity; maxV = -Infinity;
        for (const d of data) {
          if (d[rawKey] != null) { if (d[rawKey] < minV) minV = d[rawKey]; if (d[rawKey] > maxV) maxV = d[rawKey]; }
          if (d[predKey] != null) { if (d[predKey] < minV) minV = d[predKey]; if (d[predKey] > maxV) maxV = d[predKey]; }
        }
        const yMinInput = document.getElementById('yMin').value;
        const yMaxInput = document.getElementById('yMax').value;
        if (yMinInput !== '') minV = parseFloat(yMinInput);
        if (yMaxInput !== '') maxV = parseFloat(yMaxInput);
        if (minV === Infinity) minV = -1;
        if (maxV === -Infinity) maxV = 1;
        if (minV === maxV) { minV -= 0.1; maxV += 0.1; }
      }
      const plotH = height - pad.top - pad.bottom;
      const yScale = plotH / (maxV - minV);
      const yOf = (v) => y0 + pad.top + plotH - (v - minV) * yScale;

      // 网格 + 纵轴标签
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 0.5 * dpr;
      ctx.fillStyle = C.textSec;
      ctx.font = `${10*dpr}px monospace`;
      ctx.textAlign = 'right';
      for (let i = 0; i <= 4; i++) {
        const y = y0 + pad.top + plotH * i / 4;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
        ctx.fillText((maxV - (maxV-minV)*i/4).toFixed(3), pad.left - 6*dpr, y + 3*dpr);
      }

      // 实测实线（raw_value 不为 null 的点，缺口断开）
      ctx.beginPath();
      ctx.strokeStyle = rawColor;
      ctx.lineWidth = 1.5 * dpr;
      let rStarted = false, rPrevTs = null;
      for (let i = 0; i < data.length; i++) {
        const v = data[i][rawKey];
        if (v == null) { rStarted = false; continue; }
        const x = xOf(i);
        const y = yOf(v);
        if (!rStarted || (rPrevTs != null && tsMs[i] - rPrevTs > gapThreshold)) {
          ctx.moveTo(x, y); rStarted = true;
        } else {
          ctx.lineTo(x, y);
        }
        rPrevTs = tsMs[i];
      }
      ctx.stroke();

      // 预测虚线（predicted_value 不为 null 的点，缺口断开）
      ctx.beginPath();
      ctx.strokeStyle = predColor;
      ctx.lineWidth = 1.5 * dpr;
      ctx.setLineDash([5*dpr, 3*dpr]);
      let pStarted = false, pPrevTs = null;
      for (let i = 0; i < data.length; i++) {
        const v = data[i][predKey];
        if (v == null) { pStarted = false; continue; }
        const x = xOf(i);
        const y = yOf(v);
        if (!pStarted || (pPrevTs != null && tsMs[i] - pPrevTs > gapThreshold)) {
          ctx.moveTo(x, y); pStarted = true;
        } else {
          ctx.lineTo(x, y);
        }
        pPrevTs = tsMs[i];
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // 阈值线
      if (threshold != null) {
        ctx.beginPath();
        ctx.strokeStyle = C.red;
        ctx.lineWidth = 1 * dpr;
        ctx.setLineDash([4*dpr, 4*dpr]);
        const yTh = yOf(threshold);
        ctx.moveTo(pad.left, yTh); ctx.lineTo(w - pad.right, yTh);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      return { y0, minV, maxV, yScale, plotH, yOf };
    }

    const topLayout = drawSubChart(0, topH, 'raw_value', 'predicted_value', C.blue, C.green, null, null);
    // 异常分数图：anomaly_score(黄实线) + predicted_anomaly_score(绿虚线)，Y 轴固定 0~1
    const botLayout = drawSubChart(topH, bottomH, 'anomaly_score', 'predicted_anomaly_score', C.yellow, C.green, 0.5, [0, 1]);

    // ---- 缺口竖虚线 + "中断Xh" 标注 ----
    for (const gi of gapIndices) {
      const xGap = foldedX[gi.index] - GAP_W / 2;  // gap 中线
      ctx.strokeStyle = C.textSec;
      ctx.lineWidth = 1 * dpr;
      ctx.setLineDash([4*dpr, 4*dpr]);
      ctx.beginPath();
      ctx.moveTo(xGap, pad.top);
      ctx.lineTo(xGap, topH + bottomH - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      // 标注中断时长
      const dur_h = gi.duration_s / 3600;
      const label = dur_h >= 1 ? `中断 ${dur_h.toFixed(1)}h` : `中断 ${(gi.duration_s/60).toFixed(0)}min`;
      ctx.fillStyle = C.textSec;
      ctx.font = `${9*dpr}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(label, xGap, pad.top - 4*dpr);
    }

    // ---- 数据未到达区间标注（有预测值但无真实值的连续段）----
    // 这些是 pred 预测了未来但 raw 还没到的区间（poll 间隔 > pred 覆盖范围）。
    // 用两条红色竖线标注起止，提示该区间数据不可靠。
    const predOnlyRanges = [];
    let poStart = -1;
    for (let i = 0; i < data.length; i++) {
      const hasPred = data[i].predicted_value != null;
      const hasRaw = data[i].raw_value != null;
      if (hasPred && !hasRaw) {
        if (poStart < 0) poStart = i;
      } else {
        if (poStart >= 0 && i - poStart >= 5) {
          predOnlyRanges.push({ start: poStart, end: i - 1 });
        }
        poStart = -1;
      }
    }
    if (poStart >= 0 && data.length - poStart >= 5) {
      predOnlyRanges.push({ start: poStart, end: data.length - 1 });
    }
    for (const r of predOnlyRanges) {
      const x1 = foldedX[r.start];
      const x2 = foldedX[r.end];
      // 红色半透明背景
      ctx.fillStyle = 'rgba(237, 63, 20, 0.08)';
      ctx.fillRect(x1, pad.top, x2 - x1, topH + bottomH - pad.bottom - pad.top);
      // 起止红竖线
      ctx.strokeStyle = C.red;
      ctx.lineWidth = 1.5 * dpr;
      ctx.setLineDash([3*dpr, 3*dpr]);
      ctx.beginPath();
      ctx.moveTo(x1, pad.top); ctx.lineTo(x1, topH + bottomH - pad.bottom);
      ctx.moveTo(x2, pad.top); ctx.lineTo(x2, topH + bottomH - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      // 标注
      ctx.fillStyle = C.red;
      ctx.font = `${9*dpr}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText('数据未到达', (x1 + x2) / 2, pad.top - 4*dpr);
    }

    // 十字准线 + tooltip
    if (state.hoveredIdx >= 0) {
      const i = state.hoveredIdx;
      const d = data[i];
      const x = xOf(i);
      // 竖线
      ctx.strokeStyle = C.textSec;
      ctx.lineWidth = 1 * dpr;
      ctx.setLineDash([3*dpr, 3*dpr]);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, topH + bottomH - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      // 高亮点：有 raw 画蓝点，有 pred 画绿点
      if (d.raw_value != null) {
        ctx.fillStyle = C.blue;
        ctx.beginPath(); ctx.arc(x, topLayout.yOf(d.raw_value), 4*dpr, 0, Math.PI*2); ctx.fill();
      }
      if (d.predicted_value != null) {
        ctx.fillStyle = C.green;
        ctx.beginPath(); ctx.arc(x, topLayout.yOf(d.predicted_value), 4*dpr, 0, Math.PI*2); ctx.fill();
      }
      if (d.anomaly_score != null) {
        ctx.fillStyle = C.yellow;
        ctx.beginPath(); ctx.arc(x, botLayout.yOf(d.anomaly_score), 4*dpr, 0, Math.PI*2); ctx.fill();
      }
      if (d.predicted_anomaly_score != null) {
        ctx.fillStyle = C.green;
        ctx.beginPath(); ctx.arc(x, botLayout.yOf(d.predicted_anomaly_score), 4*dpr, 0, Math.PI*2); ctx.fill();
      }
      // tooltip：同一时间点同时显示 raw 和 pred（如果有）
      tooltip.style.display = 'block';
      let html = `<div class="tt-label">${formatTime(d.timestamp)}</div>`;
      if (d.raw_value != null) html += `<div>遥测: <span class="tt-val">${fmt(d.raw_value,4)}</span></div>`;
      if (d.predicted_value != null) html += `<div style="color:${C.green}">预测: <span class="tt-val">${fmt(d.predicted_value,4)}</span></div>`;
      if (d.anomaly_score != null) html += `<div>分数: <span class="tt-val">${fmt(d.anomaly_score,4)}</span></div>`;
      if (d.predicted_anomaly_score != null) html += `<div style="color:${C.green}">预测分数: <span class="tt-val">${fmt(d.predicted_anomaly_score,4)}</span></div>`;
      tooltip.innerHTML = html;
      const cssX = x / dpr + 12;
      const cssY = mouseY / dpr + 12;
      const wrapW = canvas.parentElement.clientWidth;
      tooltip.style.left = (cssX + 150 > wrapW ? cssX - 160 : cssX) + 'px';
      tooltip.style.top = Math.max(0, cssY) + 'px';
    } else {
      tooltip.style.display = 'none';
    }

    // 存储真实时间的 gap 边界供拖拽跨 gap 跳跃用
    const gapBounds = gapIndices.map(gi => ({
      before: data[gi.index - 1].timestamp,   // gap 前最后一个点
      after: data[gi.index].timestamp,         // gap 后第一个点
    }));
    state.chartLayout = { pad, topH, bottomH, tSpan: foldedTSpan, plotW, gapIndices, tsMs, gapBounds };

    // 左边界红色提示线：拖拽到最早数据时显示"已无更早数据"
    if (state._atLeftBound) {
      ctx.strokeStyle = C.red;
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, topH + bottomH - pad.bottom);
      ctx.stroke();
      ctx.fillStyle = C.red;
      ctx.font = `${10*dpr}px sans-serif`;
      ctx.textAlign = 'left';
      ctx.fillText('◀ 已无更早数据', pad.left + 4*dpr, pad.top + 12*dpr);
    }
    // 右边界红色提示线：拖拽到最新数据时显示"已无更晚数据"
    if (state._atRightBound) {
      ctx.strokeStyle = C.red;
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.moveTo(w - pad.right, pad.top);
      ctx.lineTo(w - pad.right, topH + bottomH - pad.bottom);
      ctx.stroke();
      ctx.fillStyle = C.red;
      ctx.font = `${10*dpr}px sans-serif`;
      ctx.textAlign = 'right';
      ctx.fillText('已无更晚数据 ▶', w - pad.right - 4*dpr, pad.top + 12*dpr);
    }

    updateTimeline();
  }

  // rAF 主循环：dirty flag 驱动，避免每帧空转双 canvas
  function rafLoop() {
    // 模态框打开时跳过所有 canvas 重绘（模态遮挡了 canvas，重绘无意义；
    // 且 backdrop/重绘会与输入框竞争合成线程导致输入卡顿）
    const anyModalOpen = document.querySelector('.modal-overlay.active');
    if (chartDirty && !anyModalOpen) { drawChart(); chartDirty = false; }
    // diagram 脉冲光晕是时间动画；模态打开时暂停，否则 dirty 时重绘
    if (!anyModalOpen && (diagramDirty || state.currentChannel)) {
      drawDeviceDiagram(); diagramDirty = false;
    }
    requestAnimationFrame(rafLoop);
  }

  // ====================== 图表交互：拖拽平移 ======================
  canvas.addEventListener('mousedown', (e) => {
    if (state.mode !== 'frozen' || !state.windowData || !state.windowData.data || !state.windowData.data.length) return;
    isDragging = true;
    const rect = canvas.getBoundingClientRect();
    state._dragStartX = (e.clientX - rect.left) * dpr;
    state._dragStartEndTs = state.viewEndTs;
    canvas.style.cursor = 'grabbing';
  });

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouseX = (e.clientX - rect.left) * dpr;
    mouseY = (e.clientY - rect.top) * dpr;
    markChartDirty();  // hover tooltip / 十字准线随光标变化
    if (isDragging && state.chartLayout) {
      const dx = mouseX - state._dragStartX;
      const { tSpan, plotW, gapBounds } = state.chartLayout;
      // 像素位移 → 时间位移（向右拖 = 看更早 = end_ts 减小）
      let tsDeltaSec = -(dx / plotW) * tSpan / 1000;
      let newEndTs = state._dragStartEndTs + tsDeltaSec;
      // 跨 gap 跳跃：如果新 viewEndTs 落入某个 gap 内部，跳到 gap 的另一侧
      // 这样拖拽不会卡在 gap 的真实时间跨度（小时级）里
      if (gapBounds && gapBounds.length > 0) {
        for (const g of gapBounds) {
          // 向左拖（newEndTs 减小）：如果穿过 gap.after 进入 gap 内部，跳到 gap.before
          if (newEndTs < g.after && newEndTs > g.before) {
            // 判断拖拽方向：如果起点在 gap.after 之后（向左拖），跳到 gap.before
            if (state._dragStartEndTs >= g.after) {
              newEndTs = g.before;
            } else {
              newEndTs = g.after;
            }
            break;
          }
        }
      }
      // 边界保护：viewEndTs 不能超出缓存范围 [cacheStartTs, cacheEndTs]。
      // 到达边界时截断累积位移（重置拖拽起点），避免越界后无效预取导致卡顿。
      state._atLeftBound = false;
      state._atRightBound = false;
      if (state.cacheStartTs != null && newEndTs <= state.cacheStartTs) {
        newEndTs = state.cacheStartTs;
        state._atLeftBound = true;
        state._dragStartX = mouseX;
        state._dragStartEndTs = state.cacheStartTs;
      }
      // 上界保护（向右拖看更晚数据）：newEndTs 超过 cacheEndTs 时截断。
      // 后端 query_window 只返回 end_ts 之前的数据，cacheEndTs 通常已是
      // 数据库最新时间戳，越界后预取拉不回更晚数据——表现为卡顿。
      if (state.cacheEndTs != null && newEndTs >= state.cacheEndTs) {
        newEndTs = state.cacheEndTs;
        state._atRightBound = true;
        state._dragStartX = mouseX;
        state._dragStartEndTs = state.cacheEndTs;
      }
      state.viewEndTs = newEndTs;
      markChartDirty();  // 本地截取重画（无 HTTP），drawChart 按 viewEndTs 在缓存里取子段
      // 边界预取：viewEndTs 距缓存边界不足 PREFETCH_THRESHOLD 行时才 fetch。
      // 但到达边界（_atLeftBound/_atRightBound）时不再预取——左/右边界已是
      // 数据库极限，再 fetch 也拉不回新数据，徒增无效请求。
      if (!state._atLeftBound && !state._atRightBound
          && state.cacheStartTs != null && state.cacheEndTs != null) {
        const cacheSpan = state.cacheEndTs - state.cacheStartTs;
        const marginSec = cacheSpan * (PREFETCH_THRESHOLD / CACHE_COUNT);
        const nearLeft = (newEndTs - state.cacheStartTs) < marginSec;
        const nearRight = (state.cacheEndTs - newEndTs) < marginSec;
        if ((nearLeft || nearRight) && !state._dragFetching) {
          state._dragFetching = true;
          fetchWindow();
          setTimeout(() => { state._dragFetching = false; }, 200);
        }
      }
    }
  });

  canvas.addEventListener('mouseup', () => { isDragging = false; canvas.style.cursor = ''; });
  // 全局 mouseup：鼠标移出 canvas 后松开时也能正确重置 isDragging，
  // 否则 isDragging 卡在 true，回到 canvas 后拖拽状态混乱（卡死现象之一）
  document.addEventListener('mouseup', () => {
    if (isDragging) { isDragging = false; canvas.style.cursor = ''; }
  });
  canvas.addEventListener('mouseleave', () => {
    // 注意：不在 mouseleave 里清 isDragging —— 拖拽时鼠标短暂移出 canvas
    // 再回来应继续拖拽。只清 mouseX/mouseY 让 tooltip 消失。
    mouseX = -1; mouseY = -1;
    markChartDirty();  // 清掉 hover tooltip
  });
  // 冻结模式光标提示
  canvas.addEventListener('mouseenter', () => {
    if (state.mode === 'frozen') canvas.style.cursor = 'grab';
  });

  // ====================== 时间轴 ======================
  function updateTimeline() {
    const wd = state.windowData;
    if (!wd || wd.start_ts == null || wd.end_ts == null) {
      timelineWindow.style.display = 'none';
      return;
    }
    // 时间轴显示整个 buffer 范围，高亮当前窗口
    // 由于后端只返回窗口数据，这里用窗口自身范围近似
    // 窗口高亮宽度 = 100%（因为返回的就是整个可见窗口）
    timelineWindow.style.display = 'block';
    timelineWindow.style.left = '0%';
    timelineWindow.style.width = '100%';
  }

  // 时间轴拖拽（冻结模式下平移）
  timelineBar.addEventListener('mousedown', (e) => {
    if (state.mode !== 'frozen') return;
    isDragging = true;
    state._tlDragStartX = e.clientX;
    state._tlDragStartEndTs = state.viewEndTs;
  });
  document.addEventListener('mousemove', (e) => {
    if (!isDragging || state.mode !== 'frozen' || !state._tlDragStartX) return;
    const rect = timelineBar.getBoundingClientRect();
    const dx = e.clientX - state._tlDragStartX;
    const pct = dx / rect.width;
    const wd = state.windowData;
    if (wd && wd.start_ts != null && wd.end_ts != null) {
      const span = wd.end_ts - wd.start_ts;
      let newEndTs = state._tlDragStartEndTs - pct * span;
      // 边界保护（同 canvas 拖拽）：上下界截断
      state._atLeftBound = false;
      state._atRightBound = false;
      if (state.cacheStartTs != null && newEndTs <= state.cacheStartTs) {
        newEndTs = state.cacheStartTs;
        state._atLeftBound = true;
      }
      if (state.cacheEndTs != null && newEndTs >= state.cacheEndTs) {
        newEndTs = state.cacheEndTs;
        state._atRightBound = true;
      }
      state.viewEndTs = newEndTs;
      markChartDirty();  // 本地截取重画
      // 边界预取（同 canvas 拖拽）：到达边界时不预取
      if (!state._atLeftBound && !state._atRightBound
          && state.cacheStartTs != null && state.cacheEndTs != null) {
        const cacheSpan = state.cacheEndTs - state.cacheStartTs;
        const marginSec = cacheSpan * (PREFETCH_THRESHOLD / CACHE_COUNT);
        const nearLeft = (newEndTs - state.cacheStartTs) < marginSec;
        const nearRight = (state.cacheEndTs - newEndTs) < marginSec;
        if ((nearLeft || nearRight) && !state._dragFetching) {
          state._dragFetching = true;
          fetchWindow();
          setTimeout(() => { state._dragFetching = false; }, 200);
        }
      }
    }
  });
  document.addEventListener('mouseup', () => {
    isDragging = false; state._tlDragStartX = null;
  });

  window.addEventListener('resize', () => { resizeCanvas(); resizeDeviceDiagram(); });

  // ====================== 数据库面板 ======================
  let dbCurrentTab = 0;
  function openDbModal() {
    document.getElementById('dbModalOverlay').classList.add('active');
    switchDbTab(0);
    pollManager.start('dbStats', fetchDbStats, 5000);
    fetchDbStats();
  }
  function closeDbModal() {
    document.getElementById('dbModalOverlay').classList.remove('active');
    pollManager.stop('dbStats');
  }
  function switchDbTab(idx) {
    dbCurrentTab = idx;
    document.querySelectorAll('.modal-tab').forEach((t,i) => t.classList.toggle('active', i===idx));
    if(idx===0) renderDbOverview();
    else if(idx===1) renderDbHistory();
    else if(idx===2) renderDbDetection();
    else if(idx===3) renderDbAlertsHistory();
  }

  function renderDbOverview() {
    const s = state.dbStats;
    document.getElementById('dbModalBody').innerHTML = `
      <div class="stats-grid">
        <div class="stat-card"><div class="stat-value">${s.telemetry||0}</div><div class="stat-label">遥测记录</div></div>
        <div class="stat-card"><div class="stat-value">${s.detection_results||0}</div><div class="stat-label">检测记录</div></div>
        <div class="stat-card"><div class="stat-value">${s.alert_records||0}</div><div class="stat-label">告警记录</div></div>
      </div>
      <div style="margin-top:10px;font-size:0.85rem;">DB路径: ${s.db_path||'data/phm.db'} | WAL: ON | 队列积压: ${s.queue_pending||0}</div>
      <div style="color:var(--text-secondary);margin-top:10px;font-size:0.8rem;">每 5 秒自动刷新</div>`;
  }

  async function renderDbHistory() {
    const channels = getFlatSensors().map(n=>n.channelName).filter(Boolean);
    const opts = channels.map(c=>`<option value="${c}">${c}</option>`).join('');
    document.getElementById('dbModalBody').innerHTML = `
      <div style="display:flex;gap:8px;margin-bottom:12px;align-items:center;">
        <select id="histChannel" style="background:var(--bg-card);color:var(--text-primary);border:1px solid var(--border-color);border-radius:4px;padding:2px 6px;"><option value="">全部通道</option>${opts}</select>
        <input type="datetime-local" id="histStart" style="background:var(--bg-card);color:var(--text-primary);border:1px solid var(--border-color);border-radius:4px;padding:2px 6px;"> <input type="datetime-local" id="histEnd" style="background:var(--bg-card);color:var(--text-primary);border:1px solid var(--border-color);border-radius:4px;padding:2px 6px;">
        <button class="btn-sm" onclick="queryHistory()">查询</button>
      </div>
      <div id="histTable"></div>`;
  }

  async function queryHistory() {
    const ch = document.getElementById('histChannel').value;
    const startVal = document.getElementById('histStart').value;
    const endVal = document.getElementById('histEnd').value;
    const params = { limit: 200 };
    if(ch) params.channel = ch;
    if(startVal) params.start = new Date(startVal).getTime()/1000;
    if(endVal) params.end = new Date(endVal).getTime()/1000;
    try {
      const data = await api.history(params);
      const rows = data.data || [];
      document.getElementById('histTable').innerHTML = rows.length ? `
        <table><tr><th>时间</th><th>通道</th><th>原始值</th><th>异常分数</th><th>操作</th></tr>
        ${rows.map(r => `<tr class="${(r.score!=null && r.score>0.5)?'row-alert':''}">
          <td>${formatTime(r.received_at)}</td><td>${r.channel}</td><td>${fmt(r.raw)}</td><td>${fmt(r.score)}</td>
          <td><button class="btn-sm" onclick="deleteHistory('${r.channel}',${r.received_at})">🗑</button></td>
        </tr>`).join('')}</table>` : '<div class="empty-state">无数据</div>';
    } catch(e) { document.getElementById('histTable').innerHTML = `<div class="empty-state">查询失败: ${e.message}</div>`; }
  }

  async function deleteHistory(ch, ts) {
    if(!confirm(`确认删除 ${ch} 在 ${formatTime(ts)} 的遥测记录？`)) return;
    try {
      await api.deleteHistory({ channel:ch, start:ts, end:ts+0.001 });
      queryHistory();
    } catch(e) { alert('删除失败: ' + e.message); }
  }

  async function renderDbDetection() {
    const channels = getFlatSensors().map(n=>n.channelName).filter(Boolean);
    document.getElementById('dbModalBody').innerHTML = `
      <div style="display:flex;gap:8px;margin-bottom:12px;align-items:center;">
        <select id="detChannel" style="background:var(--bg-card);color:var(--text-primary);border:1px solid var(--border-color);border-radius:4px;padding:2px 6px;"><option value="">全部通道</option>${channels.map(c=>`<option>${c}</option>`).join('')}</select>
        <input type="number" id="detLimit" value="50" min="1" max="500" style="width:80px;background:var(--bg-card);color:var(--text-primary);border:1px solid var(--border-color);border-radius:4px;padding:2px 6px;">
        <button class="btn-sm" onclick="queryDetection()">查询</button>
      </div>
      <div id="detTable"></div>`;
  }

  async function queryDetection() {
    const ch = document.getElementById('detChannel').value;
    const limit = document.getElementById('detLimit').value;
    const params = { limit };
    if(ch) params.channel = ch;
    try {
      const data = await api.detection(params);
      const rows = data.data || [];
      document.getElementById('detTable').innerHTML = rows.length ? `
        <table><tr><th>时间</th><th>通道</th><th>L1决策</th><th>L1分</th><th>L2分</th><th>L3规则</th><th>最终分数</th><th>操作</th></tr>
        ${rows.map(r => `<tr class="${(r.final_score!=null && r.final_score>0.5)?'row-alert':''} ${r.l1_decision==='skip'?'row-skip':''}">
          <td>${formatTime(r.timestamp)}</td><td>${r.channel}</td><td>${r.l1_decision||'—'}</td><td>${fmt(r.l1_score)}</td><td>${fmt(r.l2_score)}</td>
          <td>${(r.l3_rules||[]).join(', ')||'—'}</td><td>${fmt(r.final_score)}</td>
          <td><button class="btn-sm" onclick="deleteDetection('${r.channel}',${r.timestamp})">🗑</button></td>
        </tr>`).join('')}</table>` : '<div class="empty-state">无数据</div>';
    } catch(e) { document.getElementById('detTable').innerHTML = `<div class="empty-state">查询失败: ${e.message}</div>`; }
  }

  async function deleteDetection(ch, ts) {
    if(!confirm(`确认删除 ${ch} 在 ${formatTime(ts)} 的检测记录？`)) return;
    try {
      await api.deleteDetection({ channel:ch, start:ts, end:ts+0.001 });
      queryDetection();
    } catch(e) { alert('删除失败: ' + e.message); }
  }

  async function renderDbAlertsHistory() {
    try {
      const data = await api.alertsHistory(100);
      const alerts = data.alerts || [];
      document.getElementById('dbModalBody').innerHTML = alerts.length ? `
        <table><tr><th>ID</th><th>通道</th><th>类型</th><th>分数</th><th>消息</th><th>时间</th><th>综合状态</th><th>核验</th><th>LLM</th><th>人工标注</th></tr>
        ${alerts.map(a => `<tr>
          <td>${a.id}</td><td>${a.channel}</td><td>${a.alert_type}</td><td>${fmt(a.score)}</td>
          <td>${a.message||'—'}</td><td>${formatTime(a.created_at)}</td>
          <td><span class="badge ${a.final_status||a.status}">${statusLabel(a.final_status||a.status)}</span></td>
          <td><span class="badge ${a.status}" style="font-size:0.6rem;">${statusLabel(a.status)}</span></td>
          <td>${a.llm_verdict ? `<span class="badge ${a.llm_verdict}" style="font-size:0.6rem;">${statusLabel(a.llm_verdict)}</span>` : '—'}</td>
          <td>${verdictButtons('measured', {channel:a.channel, ts:a.created_at}, a.human_verdict)}</td>
        </tr>`).join('')}</table>` : '<div class="empty-state">无告警历史</div>';
    } catch(e) { document.getElementById('dbModalBody').innerHTML = `<div class="empty-state">查询失败: ${e.message}</div>`; }
  }

  async function patchAlert(id, status) {
    try {
      await api.patchAlert(id, status);
      renderDbAlertsHistory();
    } catch(e) { alert('修改失败: ' + e.message); }
  }

  // ====================== 导出 ======================
  function openExportModal() {
    document.getElementById('exportModalOverlay').classList.add('active');
    const container = document.getElementById('exportChannels');
    const sensors = getFlatSensors();
    container.innerHTML = sensors.map(s => `<label><input type="checkbox" value="${s.channelName}"> ${s.channelName}</label>`).join('');
    // 默认时间范围：基于数据实际时间范围（windowData 的 start_ts/end_ts），
    // 用 UTC 填 datetime-local（遥测标准：时间戳和 UTC 时间）
    const wd = state.windowData;
    let endTs = (wd && wd.end_ts) ? wd.end_ts : Date.now()/1000;
    let startTs = (wd && wd.start_ts) ? wd.start_ts : endTs - 600;
    const toUtcInput = (epochSec) => {
      return new Date(epochSec * 1000).toISOString().substring(0, 16);
    };
    document.getElementById('exportStart').value = toUtcInput(startTs);
    document.getElementById('exportEnd').value = toUtcInput(endTs);
  }
  function closeExportModal() { document.getElementById('exportModalOverlay').classList.remove('active'); }
  async function doExport() {
    const checks = [...document.querySelectorAll('#exportChannels input:checked')].map(c=>c.value);
    if(!checks.length) return alert('请选择通道');
    // datetime-local 填的是 UTC 时间，加 'Z' 后缀按 UTC 解析为 epoch 秒
    const start = new Date(document.getElementById('exportStart').value + 'Z').getTime()/1000;
    const end = new Date(document.getElementById('exportEnd').value + 'Z').getTime()/1000;
    if(!start || !end || start>=end) return alert('时间范围无效');
    const fmtVal = document.querySelector('input[name="fmt"]:checked').value;
    try { await api.exportData({ channels: checks.join(','), start, end, fmt:fmtVal }); }
    catch(e) { alert('导出失败: ' + e.message); }
  }

  // ====================== 设备树操作 ======================
  /** sourceId → 通道名（与 stores/deviceTree.ts sourceToChannel 等价）：
   *  virtual:sine → VS-sine；file:NASA-MSL/C-1 → C-1；否则原样返回。 */
  function sourceToChannel(sourceId) {
    if(!sourceId) return null;
    if(sourceId.startsWith('virtual:')) return 'VS-' + sourceId.slice('virtual:'.length);
    if(sourceId.startsWith('file:')) {
      const parts = sourceId.slice('file:'.length).split('/');
      return parts[parts.length - 1];
    }
    return sourceId;
  }

  function getFlatSensors(nodes=state.deviceTree) {
    let res = [];
    (nodes||[]).forEach(n => {
      if(n.type==='sensor') res.push(n);
      if(n.children) res = res.concat(getFlatSensors(n.children));
    });
    return res;
  }

  function addSensor() { openDeviceModal('sensor'); }
  function addFolder() { openDeviceModal('folder'); }

  // ---- 示意图位置编辑辅助（Slice 0，与 Vue 端 SensorModal 等价）----
  /** 递归收集树中所有出现过的 position.module 值（用于模块下拉候选）。 */
  function getAllModules(nodes) {
    nodes = nodes || state.deviceTree;
    const seen = new Set();
    const walk = (arr) => {
      (arr || []).forEach(n => {
        if (n.position && n.position.module) seen.add(n.position.module);
        if (n.children) walk(n.children);
      });
    };
    walk(nodes);
    return Array.from(seen);
  }

  /** 刷新模块下拉候选（datalist）。 */
  function refreshModuleCandidates() {
    const dl = document.getElementById('moduleCandidates');
    if (!dl) return;
    dl.innerHTML = getAllModules().map(m => `<option value="${m}">`).join('');
  }

  /** 复选框切换：显示/隐藏位置编辑字段。 */
  function togglePositionFields() {
    const checked = document.getElementById('devUsePosition').checked;
    document.getElementById('devPositionFields').style.display = checked ? '' : 'none';
    if (checked) { refreshModuleCandidates(); updatePosPreview(); }
  }

  /** mini 预览方框：实时显示当前传感器点位。 */
  function updatePosPreview() {
    const preview = document.getElementById('devPosPreview');
    if (!preview) return;
    const x = parseFloat(document.getElementById('devPosX').value) || 0;
    const y = parseFloat(document.getElementById('devPosY').value) || 0;
    preview.innerHTML = `<div style="position:absolute;width:10px;height:10px;border-radius:50%;background:var(--accent-blue);left:${x*100}%;top:${y*100}%;transform:translate(-50%,-50%);"></div>`;
  }

  /** 打开模态框时重置位置字段到默认值。 */
  function resetPositionFields() {
    document.getElementById('devUsePosition').checked = false;
    document.getElementById('devPositionFields').style.display = 'none';
    document.getElementById('devModule').value = '';
    document.getElementById('devPosX').value = 0.5;
    document.getElementById('devPosY').value = 0.5;
    document.getElementById('devPosXLabel').textContent = '0.50';
    document.getElementById('devPosYLabel').textContent = '0.50';
    updatePosPreview();
  }

  /** 编辑模态框时从节点加载已有位置。 */
  function loadPositionFields(node) {
    const p = node.position;
    if (p && (p.x !== undefined || p.y !== undefined)) {
      document.getElementById('devUsePosition').checked = true;
      document.getElementById('devPositionFields').style.display = '';
      document.getElementById('devModule').value = p.module || '';
      document.getElementById('devPosX').value = p.x !== undefined ? p.x : 0.5;
      document.getElementById('devPosY').value = p.y !== undefined ? p.y : 0.5;
      document.getElementById('devPosXLabel').textContent = parseFloat(document.getElementById('devPosX').value).toFixed(2);
      document.getElementById('devPosYLabel').textContent = parseFloat(document.getElementById('devPosY').value).toFixed(2);
      refreshModuleCandidates();
      updatePosPreview();
    } else {
      resetPositionFields();
    }
  }

  /** 从表单收集 position 对象（未勾选返回 undefined）。 */
  function collectPosition() {
    if (!document.getElementById('devUsePosition').checked) return undefined;
    const pos = {
      x: parseFloat(document.getElementById('devPosX').value),
      y: parseFloat(document.getElementById('devPosY').value),
    };
    const mod = document.getElementById('devModule').value.trim();
    if (mod) pos.module = mod;
    return pos;
  }

  function openDeviceModal(kind) {
    const isSensor = kind === 'sensor';
    document.getElementById('deviceModalTitle').textContent = isSensor ? '新建传感器' : '新建文件夹';
    document.getElementById('devName').value = '';
    document.getElementById('devSensorFields').style.display = isSensor ? '' : 'none';
    if (isSensor) {
      document.getElementById('devSource').value = 'file:NASA-MSL/C-1';
      document.getElementById('devBlockSize').value = '512';
      resetPositionFields();
    }
    state._deviceModalMode = { kind: 'create', type: isSensor ? 'sensor' : 'folder' };
    document.getElementById('deviceModalOverlay').classList.add('active');
    setTimeout(() => document.getElementById('devName').focus(), 50);
  }

  function openEditDeviceModal(node) {
    const isSensor = node.type === 'sensor';
    document.getElementById('deviceModalTitle').textContent = isSensor ? '编辑传感器' : '编辑文件夹';
    document.getElementById('devName').value = node.name || '';
    document.getElementById('devSensorFields').style.display = isSensor ? '' : 'none';
    if (isSensor) {
      // 通道名从数据源自动派生，不再单独编辑；若节点的 sourceId 不在下拉选项里，补一个临时 option
      const sel = document.getElementById('devSource');
      const sid = node.sourceId || 'file:NASA-MSL/C-1';
      if(!Array.from(sel.options).some(o => o.value === sid)) {
        sel.innerHTML = `<option value="${sid}">${sid}</option>` + sel.innerHTML;
      }
      sel.value = sid;
      document.getElementById('devBlockSize').value = node.blockSize || 512;
      loadPositionFields(node);
    }
    state._deviceModalMode = { kind: 'edit', node };
    document.getElementById('deviceModalOverlay').classList.add('active');
    setTimeout(() => document.getElementById('devName').focus(), 50);
  }

  function closeDeviceModal() {
    document.getElementById('deviceModalOverlay').classList.remove('active');
    state._deviceModalMode = null;
  }

  /** 把节点放进选中的文件夹（若有），否则放顶层。返回 true 表示放进了文件夹。 */
  function insertIntoSelectedFolder(node) {
    if (!state.selectedFolderId) return false;
    const findFolder = (nodes) => {
      for(const n of nodes) {
        if(n.id === state.selectedFolderId && n.type === 'folder') return n;
        if(n.children) { const r = findFolder(n.children); if(r) return r; }
      }
      return null;
    };
    const folder = findFolder(state.deviceTree);
    if(folder) {
      if(!folder.children) folder.children = [];
      folder.children.push(node);
      return true;
    }
    return false;
  }

  function submitDeviceModal() {
    const mode = state._deviceModalMode;
    if (!mode) return;
    const name = document.getElementById('devName').value.trim();
    if (!name) { alert('请输入名称'); return; }
    let changed = false;
    if (mode.kind === 'create') {
      if (mode.type === 'sensor') {
        const src = document.getElementById('devSource').value;
        const ch = sourceToChannel(src) || name;  // 通道名从数据源自动派生
        // 唯一性校验：sourceId 决定 channel/表名，重名会导致数据互相覆盖
        const existing = getFlatSensors();
        if (existing.some(n => n.sourceId === src)) {
          alert(`数据源 "${src}" 已被其他传感器使用，请选择不同的数据源。\n（同名数据源会导致通道数据互相覆盖）`);
          return;
        }
        if (existing.some(n => n.name === name)) {
          alert(`名称 "${name}" 已存在，请使用不同的名称。`);
          return;
        }
        const bs = parseInt(document.getElementById('devBlockSize').value) || 512;
        const node = { id:'node-'+Date.now(), name, type:'sensor', sourceId:src, channelName:ch, blockSize:bs };
        const pos = collectPosition();
        if (pos) node.position = pos;
        // 优先放进选中的文件夹，否则放顶层
        if(!insertIntoSelectedFolder(node)) state.deviceTree.push(node);
        changed = true;
      } else {
        // 文件夹重名校验（同层 + 全局）
        const allFolders = [];
        (function collect(nodes) {
          (nodes||[]).forEach(n => { if(n.type==='folder') allFolders.push(n.name); if(n.children) collect(n.children); });
        })(state.deviceTree);
        if (allFolders.includes(name)) {
          alert(`文件夹名称 "${name}" 已存在，请使用不同的名称。`);
          return;
        }
        const folder = { id:'node-'+Date.now(), name, type:'folder', children:[] };
        if(!insertIntoSelectedFolder(folder)) state.deviceTree.push(folder);
        changed = true;
      }
    } else {
      // edit（传感器改配置 / 文件夹重命名）
      const node = mode.node;
      // 编辑时也要校验重名（排除自身）
      if (node.type === 'sensor') {
        const src = document.getElementById('devSource').value;
        const ch = sourceToChannel(src) || node.channelName || name;
        const existing = getFlatSensors().filter(n => n.id !== node.id);
        if (existing.some(n => n.sourceId === src)) {
          alert(`数据源 "${src}" 已被其他传感器使用，请选择不同的数据源。`);
          return;
        }
        if (node.name !== name && existing.some(n => n.name === name)) {
          alert(`名称 "${name}" 已存在，请使用不同的名称。`);
          return;
        }
        if (node.name !== name) { node.name = name; changed = true; }
        const bs = parseInt(document.getElementById('devBlockSize').value) || 512;
        if(node.channelName !== ch) { node.channelName = ch; changed = true; }
        if(node.sourceId !== src) { node.sourceId = src; changed = true; }
        if(node.blockSize !== bs) { node.blockSize = bs; changed = true; }
        const pos = collectPosition();
        if (pos) { node.position = pos; changed = true; }
        else if(node.position) { delete node.position; changed = true; }
      } else {
        // 文件夹重命名
        if (node.name !== name) {
          const allFolders = [];
          (function collect(nodes) {
            (nodes||[]).forEach(n => { if(n.type==='folder' && n.id!==node.id) allFolders.push(n.name); if(n.children) collect(n.children); });
          })(state.deviceTree);
          if (allFolders.includes(name)) {
            alert(`文件夹名称 "${name}" 已存在，请使用不同的名称。`);
            return;
          }
          node.name = name; changed = true;
        }
      }
    }
    closeDeviceModal();
    if (changed) {
      renderDeviceTree();
      renderGauges();
      autoSaveConfig();
    }
  }

  // Enter 键提交（排除 range/checkbox，避免拖滑块/勾选时误提交）
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && document.getElementById('deviceModalOverlay').classList.contains('active')) {
      const tag = e.target.tagName;
      const t = e.target.type;
      if (tag === 'INPUT' && t !== 'range' && t !== 'checkbox') submitDeviceModal();
    }
    if (e.key === 'Escape') {
      const dg = document.getElementById('diagnosisModalOverlay');
      if (dg.classList.contains('active')) closeDiagnosisModal();
    }
  });

  function deleteNode(id) {
    const remove = (nodes) => {
      const idx = nodes.findIndex(n=>n.id===id);
      if(idx!==-1) { nodes.splice(idx,1); return true; }
      return nodes.some(n => n.children && remove(n.children));
    };
    remove(state.deviceTree);
    // 若删除的是当前选中文件夹，清空选中态
    if(state.selectedFolderId === id) state.selectedFolderId = null;
    renderDeviceTree();  // 内部 markDiagramDirty
    autoSaveConfig();
  }

  // 自动保存（静默、防抖 800ms）—— 增删改设备树后保证后端 config 与前端一致，
  // 否则后端 health 聚合 / auto-poll 采集会漏掉前端新建的节点。
  // 安全保护：空树不保存（防止误删全部节点导致配置丢失、auto-poll 停摆）。
  let _autoSaveTimer = null;
  function autoSaveConfig() {
    if(_autoSaveTimer) clearTimeout(_autoSaveTimer);
    _autoSaveTimer = setTimeout(async () => {
      _autoSaveTimer = null;
      if(!state.deviceTree || state.deviceTree.length === 0) {
        console.warn('autoSaveConfig skipped: device tree is empty (safety guard)');
        return;
      }
      try { await api.saveConfig(state.deviceTree); }
      catch(e) { console.warn('自动保存配置失败:', e); }
    }, 800);
  }

  async function saveConfig() {
    if(!state.deviceTree || state.deviceTree.length === 0) {
      if(!confirm('设备树为空，保存后所有通道将停止采集。确定要保存空配置吗？')) return;
    }
    try {
      await api.saveConfig(state.deviceTree);
      alert('配置已保存');
    } catch(e) { alert('保存失败: ' + e.message); }
  }

  // ====================== 设备示意图 Canvas（Slice 1） ======================
  const diagramCanvas = document.getElementById('deviceDiagram');
  const diagramCtx = diagramCanvas.getContext('2d');
  const diagramTooltip = document.getElementById('diagramTooltip');
  let diagramLayout = null;  // { modules:[{name,x,y,w,h}], dots:[{node,x,y,r,color}] } 缓存供 hit-test

  // DPR 感知 sizing（与 resizeCanvas 同范式）
  function resizeDeviceDiagram() {
    const rect = diagramCanvas.parentElement.getBoundingClientRect();
    diagramCanvas.width = rect.width * dpr;
    diagramCanvas.height = rect.height * dpr;
    diagramCanvas.style.width = rect.width + 'px';
    diagramCanvas.style.height = rect.height + 'px';
  }

  // 模块分组：按 position.module 分桶，无 module 的归"未分组"
  function groupSensorsByModule() {
    const sensors = getFlatSensors();
    const groups = {};  // { moduleName: [sensorNodes] }
    const ungrouped = [];
    sensors.forEach(s => {
      const mod = s.position && s.position.module;
      if (mod) { if(!groups[mod]) groups[mod] = []; groups[mod].push(s); }
      else ungrouped.push(s);
    });
    if (ungrouped.length) groups['未分组'] = ungrouped;
    return groups;
  }

  // 网格布局：cols = ceil(sqrt(n)), rows = ceil(n/cols)
  function gridDims(n) {
    if (n <= 0) return { cols: 1, rows: 1 };
    const cols = Math.ceil(Math.sqrt(n));
    return { cols, rows: Math.ceil(n / cols) };
  }

  // 健康度→颜色（canvas 用 hex，不能用 var()）
  function healthColorHex(h) {
    if (h == null) return C.textSec;
    if (h < 60) return C.red;
    if (h < 80) return C.yellow;
    return C.green;
  }

  function drawDeviceDiagram() {
    const w = diagramCanvas.width, h = diagramCanvas.height;
    if (w === 0 || h === 0) return;
    const dc = diagramCtx;
    dc.clearRect(0, 0, w, h);

    // 背景已由 CSS 提供，这里画机箱轮廓
    const pad = 10 * dpr;
    const chassisX = pad, chassisY = pad;
    const chassisW = w - pad * 2, chassisH = h - pad * 2 - 14 * dpr; // 底部留图例空间
    dc.strokeStyle = C.border;
    dc.lineWidth = 1.5 * dpr;
    roundRect(dc, chassisX, chassisY, chassisW, chassisH, 8 * dpr);
    dc.stroke();

    // 模块分组 + 网格布局
    const groups = groupSensorsByModule();
    const moduleNames = Object.keys(groups);
    const { cols, rows } = gridDims(moduleNames.length);
    const gap = 8 * dpr;
    const cellW = (chassisW - gap * (cols - 1)) / cols;
    const cellH = (chassisH - gap * (rows - 1)) / rows;

    const modules = [];
    const dots = [];
    moduleNames.forEach((modName, i) => {
      const col = i % cols, row = Math.floor(i / cols);
      const mx = chassisX + col * (cellW + gap);
      const my = chassisY + row * (cellH + gap);
      modules.push({ name: modName, x: mx, y: my, w: cellW, h: cellH });

      // 模块区背景（半透明）+ 标题栏
      const folderId = findFolderIdByName(modName);
      const fhealth = folderId && state.folders[folderId] ? state.folders[folderId].health : null;
      const isSelFolder = state.selectedFolderId && folderId === state.selectedFolderId;
      dc.fillStyle = isSelFolder ? 'rgba(45,140,240,0.12)' : 'rgba(255,255,255,0.03)';
      roundRect(dc, mx, my, cellW, cellH, 6 * dpr);
      dc.fill();
      dc.strokeStyle = isSelFolder ? C.blue : C.border;
      dc.lineWidth = 1 * dpr;
      dc.stroke();

      // 标题
      dc.fillStyle = C.textSec;
      dc.font = `${11 * dpr}px 'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif`;
      dc.textAlign = 'left';
      dc.textBaseline = 'top';
      const label = modName + (fhealth != null ? ` ${fhealth.toFixed(0)}%` : '');
      dc.fillText(label, mx + 6 * dpr, my + 4 * dpr);

      // 传感器点位
      const sensors = groups[modName];
      const titleH = 18 * dpr;
      const dotAreaY = my + titleH;
      const dotAreaH = cellH - titleH - 6 * dpr;
      sensors.forEach((node, si) => {
        // 有 position 用之，否则在模块内 fallback 均布
        const fx = node.position && node.position.x != null ? node.position.x : ((si + 0.5) / sensors.length);
        const fy = node.position && node.position.y != null ? node.position.y : 0.5;
        const dx = mx + 6 * dpr + fx * (cellW - 12 * dpr);
        const dy = dotAreaY + fy * dotAreaH;
        const ch = node.channelName;
        // 点位颜色按异常分数着色（与设备树圆点一致，阈值对齐 ANOMALY_THRESHOLD 0.5）：
        // score>0.5 红 / >0.25 黄 / 否则绿；无数据灰
        const sc = state.sensors[ch] ? state.sensors[ch].latest_score : null;
        const color = sc == null ? C.textSec : (sc > 0.5 ? C.red : (sc > 0.25 ? C.yellow : C.green));
        const isSel = state.currentChannel === ch;
        const dotR = (isSel ? 7 : 5) * dpr;
        dots.push({ node, x: dx, y: dy, r: dotR, color, isSel });

        // 选中态脉冲光晕
        if (isSel) {
          const pulse = 1 + 0.3 * Math.sin(Date.now() / 200);
          dc.beginPath();
          dc.arc(dx, dy, dotR * 2 * pulse, 0, Math.PI * 2);
          dc.fillStyle = color + '33';
          dc.fill();
        }
        // 点位
        dc.beginPath();
        dc.arc(dx, dy, dotR, 0, Math.PI * 2);
        dc.fillStyle = color;
        dc.fill();
        dc.strokeStyle = isSel ? C.textPri : 'rgba(0,0,0,0.3)';
        dc.lineWidth = isSel ? 1.5 * dpr : 0.5 * dpr;
        dc.stroke();
        // 选中态显示通道名
        if (isSel) {
          dc.fillStyle = C.textPri;
          dc.font = `${10 * dpr}px sans-serif`;
          dc.textAlign = 'center';
          dc.textBaseline = 'bottom';
          dc.fillText(ch, dx, dy - dotR - 2 * dpr);
        }
      });
    });

    // 空状态提示
    if (dots.length === 0) {
      dc.fillStyle = C.textSec;
      dc.font = `${12 * dpr}px sans-serif`;
      dc.textAlign = 'center';
      dc.textBaseline = 'middle';
      dc.fillText('暂无传感器，请在左侧设备树添加', w / 2, h / 2);
    }

    diagramLayout = { modules, dots };
  }

  // 圆角矩形辅助（Canvas 无内置 roundRect，手动绘制）
  function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y);
    c.lineTo(x + w - r, y);
    c.quadraticCurveTo(x + w, y, x + w, y + r);
    c.lineTo(x + w, y + h - r);
    c.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    c.lineTo(x + r, y + h);
    c.quadraticCurveTo(x, y + h, x, y + h - r);
    c.lineTo(x, y + r);
    c.quadraticCurveTo(x, y, x + r, y);
    c.closePath();
  }

  // 按 module 名找对应 folder id（文件夹聚合健康度 key 是 folder.id）
  function findFolderIdByName(name) {
    const walk = (nodes) => {
      for (const n of nodes) {
        if (n.name === name && n.type === 'folder') return n.id;
        if (n.children) { const r = walk(n.children); if (r) return r; }
      }
      return null;
    };
    return walk(state.deviceTree);
  }

  // hit-test：返回鼠标坐标下的传感器节点，无则 null
  function diagramHitTest(mx, my) {
    if (!diagramLayout) return null;
    for (const d of diagramLayout.dots) {
      const dx = mx - d.x, dy = my - d.y;
      if (dx * dx + dy * dy <= (d.r + 3 * dpr) ** 2) return d;
    }
    return null;
  }

  diagramCanvas.addEventListener('mousemove', (e) => {
    const rect = diagramCanvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * dpr;
    const my = (e.clientY - rect.top) * dpr;
    const hit = diagramHitTest(mx, my);
    if (hit) {
      const s = state.sensors[hit.node.channelName] || {};
      diagramTooltip.innerHTML = `<b>${hit.node.name}</b> [${hit.node.channelName}]<br>健康: ${s.health != null ? s.health.toFixed(1) + '%' : '—'}<br>分数: ${s.latest_score != null ? s.latest_score.toFixed(3) : '—'}`;
      diagramTooltip.style.display = 'block';
      // 越界翻转：靠近底部时 tooltip 显示在上方，避免被面板截断
      let tx = e.clientX - rect.left + 12;
      let ty = e.clientY - rect.top + 12;
      const ttH = diagramTooltip.offsetHeight || 60;
      const ttW = diagramTooltip.offsetWidth || 160;
      if (ty + ttH > rect.height) ty = e.clientY - rect.top - ttH - 8;
      if (tx + ttW > rect.width) tx = rect.width - ttW - 8;
      diagramTooltip.style.left = tx + 'px';
      diagramTooltip.style.top = ty + 'px';
      diagramCanvas.style.cursor = 'pointer';
    } else {
      diagramTooltip.style.display = 'none';
      diagramCanvas.style.cursor = 'default';
    }
  });
  diagramCanvas.addEventListener('mouseleave', () => {
    diagramTooltip.style.display = 'none';
  });
  diagramCanvas.addEventListener('click', (e) => {
    const rect = diagramCanvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * dpr;
    const my = (e.clientY - rect.top) * dpr;
    const hit = diagramHitTest(mx, my);
    if (hit) selectChannel(hit.node.channelName);
  });

  // ====================== 健康环绘制 ======================
  function drawHealthRing() {
    const c = document.getElementById('healthRing');
    const cx = c.getContext('2d');
    const w = c.width, h = c.height;
    cx.clearRect(0,0,w,h);
    const val = (state.systemHealth || 0) / 100;
    const color = val >= 0.8 ? C.green : (val >= 0.6 ? C.yellow : C.red);
    const r = 36;  // 半径加大（配合 96×96 画布）
    // 背景轨道
    cx.beginPath();
    cx.arc(w/2, h/2, r, 0, Math.PI*2);
    cx.strokeStyle = '#2a2f3a';
    cx.lineWidth = 8;
    cx.stroke();
    // 填充弧
    cx.beginPath();
    cx.arc(w/2, h/2, r, -Math.PI/2, -Math.PI/2 + Math.PI*2*val);
    cx.strokeStyle = color;
    cx.lineWidth = 8;
    cx.stroke();
    cx.fillStyle = '#fff';
    cx.font = 'bold 16px sans-serif';
    cx.textAlign = 'center';
    cx.textBaseline = 'middle';
    cx.fillText(Math.round(state.systemHealth||0)+'%', w/2, h/2+1);
  }

  // ====================== 初始化 ======================
  async function init() {
    try {
      const cfg = await api.getConfig();
      state.deviceTree = cfg.device_tree || [];
    } catch(e) { state.deviceTree = []; }
    renderDeviceTree();
    renderGauges();  // 即使 RingBuffer 空，也显示设备树里的传感器卡片

    const sensors = getFlatSensors();
    if(sensors.length) selectChannel(sensors[0].channelName || sensors[0].name);

    pollManager.start('health', fetchHealth, 3000);
    pollManager.start('sensors', fetchSensors, 3000);
    await fetchDiagnosedKeys();  // 页面加载即标记已诊断告警（持久化，刷新不丢）
    pollManager.start('alerts', fetchAlerts, 3000);
    pollManager.start('warnings', fetchWarnings, 3000);

    setInterval(() => {
      const now = new Date();
      document.getElementById('utcClock').textContent = now.toLocaleString('zh-CN', {
        timeZone: 'Asia/Shanghai', hour12: false,
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      });
    }, 1000);

    resizeCanvas();
    resizeDeviceDiagram();
    requestAnimationFrame(rafLoop);  // 启动持续重绘
  }

  // ====================== LLM 诊断（Slice 2） ======================
  async function requestDiagnosis(channel, alertType, alertTs, btn) {
    const overlay = document.getElementById('diagnosisModalOverlay');
    const title = document.getElementById('diagnosisTitle');
    const body = document.getElementById('diagnosisBody');
    const origText = btn ? btn.textContent : '';
    if (btn) { btn.disabled = true; btn.textContent = '...'; }
    title.textContent = `🔍 ${channel} — 异常诊断`;
    body.innerHTML = '<div class="diag-loading"><div class="spinner"></div>正在分析检测数据，请稍候...</div>';
    overlay.classList.add('active');
    try {
      const resp = await fetch('/api/diagnosis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel, alert_type: alertType, alert_ts: alertTs }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        body.innerHTML = `<div style="color:var(--accent-red,#ed3f14);">⚠ ${data.error || '诊断失败'}<br><span style="font-size:0.75rem;color:var(--text-secondary);">${data.detail || data.error || ''}</span></div>`;
      } else if (data.error) {
        body.innerHTML = `<div style="color:var(--accent-red,#ed3f14);">⚠ ${data.error}</div>`;
      } else {
        const s = data.context_summary || {};
        const meta = [
          `**通道**：${channel}${s.display_name ? '（' + s.display_name + '）' : ''}`,
          s.device_path ? `**设备位置**：${s.device_path}` : null,
          `**告警类型**：${alertType === 'measured' ? '实测告警' : '预测预警'}`,
          data.cached ? `📁 缓存命中` : `**耗时**：${data.elapsed_sec}s`,
        ].filter(Boolean).join('  ·  ');
        body.innerHTML = `<div style="font-size:0.7rem;color:var(--text-secondary);margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid var(--border-color);">${meta}</div>` + renderMarkdown(data.diagnosis || '(空)');
        // Mark this alert as diagnosed — green ✓ on the button (persistent).
        if (alertTs != null && data.diagnosis) {
          state.diagnosedKeys.add(`${channel}|${alertType}|${alertTs}`);
          if (btn) { btn.classList.add('diag-done'); btn.textContent = '✓'; }
        }
      }
    } catch (e) {
      body.innerHTML = `<div style="color:var(--accent-red,#ed3f14);">⚠ 请求失败：${e.message}</div>`;
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = btn.classList.contains('diag-done') ? '✓' : origText; }
    }
  }

  function closeDiagnosisModal() {
    document.getElementById('diagnosisModalOverlay').classList.remove('active');
  }

  // ====================== 人工标注（verdict） ======================
  async function submitWarningVerdict(id, verdict) {
    try {
      await api.warningVerdict(id, verdict);
      await fetchWarnings();
      // 如果数据库模态框的告警历史 tab 正打开，刷新它使按钮状态更新
      if(document.getElementById('dbModalOverlay').classList.contains('active') && dbCurrentTab===3) renderDbAlertsHistory();
    } catch(e) { alert('标注失败: ' + e.message); }
  }
  async function submitAlertVerdict(channel, ts, verdict) {
    try {
      await api.alertVerdict(channel, ts, verdict);
      await fetchAlerts();
      // 如果数据库模态框的告警历史 tab 正打开，刷新它使按钮状态更新
      if(document.getElementById('dbModalOverlay').classList.contains('active') && dbCurrentTab===3) renderDbAlertsHistory();
    } catch(e) { alert('标注失败: ' + e.message); }
  }

  // ====================== 全自动诊断模式 ======================
  async function toggleAutoDiag() {
    const btn = document.getElementById('autoDiagBtn');
    const statusEl = document.getElementById('autoDiagStatus');
    if (btn.dataset.running === 'true') return;
    btn.disabled = true;
    try {
      const r = await api.diagnosisAuto();
      if (r.error) { alert(r.error); btn.disabled = false; return; }
      btn.dataset.running = 'true';
      btn.textContent = '⏳ 诊断中...';
      if (r.total === 0) {
        statusEl.textContent = '无待诊断项';
        btn.disabled = false; btn.textContent = '🤖 全自动诊断'; btn.dataset.running = 'false';
        return;
      }
      pollAutoDiagStatus();
    } catch(e) { alert('启动失败: ' + e.message); btn.disabled = false; }
  }
  function pollAutoDiagStatus() {
    const btn = document.getElementById('autoDiagBtn');
    const statusEl = document.getElementById('autoDiagStatus');
    const interval = setInterval(async () => {
      try {
        const s = await api.diagnosisAutoStatus();
        statusEl.textContent = `${s.done}/${s.total}` + (s.errors ? ` (${s.errors}错误)` : '');
        if (!s.running) {
          clearInterval(interval);
          btn.disabled = false; btn.textContent = '🤖 全自动诊断'; btn.dataset.running = 'false';
          fetchAlerts(); fetchWarnings();  // 刷新 verdict 显示
        }
      } catch(e) { /* keep polling */ }
    }, 2000);
  }

  // 极简 Markdown 渲染（h2/h3、段落、列表、行内加粗/代码）
  function renderMarkdown(md) {
    const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    // 行内格式：**bold** → <strong>，`code` → <code>（转义后处理）
    const inline = s => esc(s)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.+?)`/g, '<code style="background:var(--bg-card,#1a1f2f);padding:1px 4px;border-radius:3px;font-size:0.85em;">$1</code>');
    const lines = md.split('\n');
    let html = '', inList = false;
    for (let line of lines) {
      if (line.startsWith('### ')) {
        if (inList) { html += '</ul>'; inList = false; }
        html += `<h3 style="font-size:0.9rem;margin:0.6em 0 0.2em;color:var(--text-primary,#e0e6f0);">${inline(line.slice(4))}</h3>`;
      } else if (line.startsWith('## ')) {
        if (inList) { html += '</ul>'; inList = false; }
        html += `<h2>${inline(line.slice(3))}</h2>`;
      } else if (/^\s*[-*]\s/.test(line)) {
        if (!inList) { html += '<ul>'; inList = true; }
        html += `<li>${inline(line.replace(/^\s*[-*]\s/, ''))}</li>`;
      } else if (line.trim() === '') {
        if (inList) { html += '</ul>'; inList = false; }
      } else {
        if (inList) { html += '</ul>'; inList = false; }
        html += `<p>${inline(line)}</p>`;
      }
    }
    if (inList) html += '</ul>';
    return html;
  }

