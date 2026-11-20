'use strict';
const $ = (id) => document.getElementById(id);
const state = { patients: [], patient: null, model: 'logistic_regression', result: null, index: 0, modality: 't1c', zoom: 1 };
const labels = { logistic_regression: 'Logistic Regression', random_forest: 'Random Forest', vqc_2: 'VQC-2', vqc_4: 'VQC-4', vqc_8: 'VQC-8' };
const metricKeys = [['Acc', 'accuracy'], ['Prec', 'precision'], ['Rec', 'recall'], ['F1', 'f1'], ['AUC', 'roc_auc']];

async function api(path) {
  const response = await fetch(path);
  const data = response.headers.get('content-type')?.includes('application/json') ? await response.json() : response;
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}
function setStatus(message, kind = '') {
  const el = $('status'); el.textContent = message; el.className = `status-line ${kind}`;
}
function fmt(value) { return value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(1)}%`; }
function modelConfig(key) {
  if (key.startsWith('vqc_')) return `${key.slice(4)} qubits · variational quantum classifier`;
  return 'Classical baseline';
}
function refreshDisplay() {
  const image = $('mri-image');
  image.style.transform = `scale(${state.zoom})`;
  image.style.filter = `brightness(${$('brightness').value}%) contrast(${$('contrast').value}%)`;
}
function setZoom(value) { state.zoom = Math.max(1, Math.min(3, Math.round(value * 10) / 10)); refreshDisplay(); }
function loadMri() {
  if (!state.patient) return;
  const image = $('mri-image'); image.hidden = true;
  $('image-loading').hidden = false; $('image-loading').textContent = 'Loading MRI slice…';
  image.onload = () => { $('image-loading').hidden = true; image.hidden = false; refreshDisplay(); };
  image.onerror = () => { $('image-loading').hidden = false; $('image-loading').textContent = 'MRI image unavailable for this slice'; };
  image.src = `/api/image/${encodeURIComponent(state.patient)}/${state.modality}/${state.index}?t=${Date.now()}`;
  $('image-slice-tag').textContent = `SLICE ${String(state.index + 1).padStart(3, '0')}`;
  $('slice-count').textContent = `${state.index + 1} / ${state.result?.slice_count ?? '—'}`;
  $('prev').disabled = state.index <= 0;
  $('next').disabled = !state.result || state.index >= state.result.slice_count - 1;
  updateDetection(); updateSegmentation();
}
function drawChart() {
  const canvas = $('chart'), box = canvas.getBoundingClientRect(), scale = window.devicePixelRatio || 1;
  if (!box.width || !box.height) return;
  canvas.width = Math.round(box.width * scale); canvas.height = Math.round(box.height * scale);
  const ctx = canvas.getContext('2d'); ctx.setTransform(scale, 0, 0, scale, 0, 0);
  const w = box.width, h = box.height, p = { l: 36, r: 10, t: 10, b: 20 }, data = state.result?.detections;
  ctx.clearRect(0, 0, w, h);
  if (!data?.length) { ctx.fillStyle = '#97a4a8'; ctx.font = '11px DM Sans, sans-serif'; ctx.fillText('Run analysis to view slice probabilities', p.l, h / 2); return; }
  const x = i => p.l + i / Math.max(1, data.length - 1) * (w - p.l - p.r);
  const y = v => p.t + (1 - v) * (h - p.t - p.b);
  ctx.font = '8px DM Sans, sans-serif'; ctx.strokeStyle = '#e8edef'; ctx.fillStyle = '#829198'; ctx.lineWidth = 1;
  for (let v = 0; v <= 1.001; v += .25) { ctx.beginPath(); ctx.moveTo(p.l, y(v)); ctx.lineTo(w - p.r, y(v)); ctx.stroke(); ctx.fillText(v.toFixed(2), 0, y(v) + 3); }
  ctx.setLineDash([5, 4]); ctx.strokeStyle = '#c95d55'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(p.l, y(state.result.threshold)); ctx.lineTo(w - p.r, y(state.result.threshold)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = '#397fc2'; ctx.lineWidth = 1.6; ctx.beginPath();
  data.forEach((d, i) => i ? ctx.lineTo(x(i), y(d.probability)) : ctx.moveTo(x(i), y(d.probability))); ctx.stroke();
  const d = data[state.index];
  ctx.fillStyle = '#397fc2'; ctx.beginPath(); ctx.arc(x(state.index), y(d.probability), 4, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.fillStyle = '#75858b'; ctx.fillText('1', p.l, h - 4); ctx.fillText(String(data.length), w - 23, h - 4);
  $('threshold-value').textContent = Number(state.result.threshold).toFixed(2);
}
function updateDetection() {
  const d = state.result?.detections[state.index];
  if (!d) { drawChart(); return; }
  $('probability').textContent = fmt(d.probability);
  $('prediction-label').textContent = d.prediction ? 'Positive slice prediction' : 'Negative slice prediction';
  const pill = $('prediction-status'); pill.className = `state-pill ${d.prediction ? 'positive' : 'negative'}`; pill.textContent = d.prediction ? 'POSITIVE' : 'NEGATIVE';
  drawChart();
}
function updateSegmentation() {
  if (!state.result) return;
  const selected = state.result.segmented_slice_indices.includes(state.index), pill = $('seg-status');
  $('seg-content').hidden = true; $('seg-skipped').hidden = false; $('empty-mask-note').hidden = true;
  if (state.result.segmentation_status === 'skipped_no_positive_slice') {
    pill.className = 'state-pill muted'; pill.textContent = 'SEGMENTATION SKIPPED';
    $('seg-skipped').textContent = `No slices were classified positive at the ${Number(state.result.threshold).toFixed(2)} threshold. Conditional segmentation was skipped; no masks are shown.`; return;
  }
  pill.className = 'state-pill positive'; pill.textContent = 'SEGMENTATION AVAILABLE';
  if (!selected) {
    $('seg-skipped').textContent = 'Segmentation was not run for this slice. It is available on positive detections and selected neighboring slices.'; return;
  }
  $('seg-skipped').hidden = true; $('seg-content').hidden = false;
  const segment = state.result.segmented_slices.find(item => item.slice_index === state.index);
  const detection = state.result.detections[state.index];
  $('seg-slice').textContent = `${state.index + 1} / ${state.result.slice_count}`;
  $('seg-detection').textContent = detection.prediction ? 'Positive detection' : 'Neighbor slice';
  $('seg-state').textContent = 'Predicted independently for selected slice';
  $('seg-threshold').textContent = Number(state.result.threshold).toFixed(2);
  $('mask-state').textContent = segment?.mask_nonempty ? 'Tumor pixels predicted' : 'Empty predicted mask';
  $('empty-mask-note').hidden = Boolean(segment?.mask_nonempty);
  const base = `/api/segment/${encodeURIComponent(state.patient)}/${state.model}/${state.index}/`;
  $('seg-original-image').src = `/api/image/${encodeURIComponent(state.patient)}/t1c/${state.index}?t=${Date.now()}`;
  $('overlay-image').src = `${base}overlay?t=${Date.now()}`;
  $('prediction-image').src = `${base}prediction?t=${Date.now()}`;
  $('ground-image').src = `${base}ground_truth_mask?t=${Date.now()}`;
}
async function analyze() {
  if (!state.patient) return;
  state.model = $('model').value; $('analyze').disabled = true; $('analyze').classList.add('is-loading');
  $('analyze').innerHTML = '<span>◌</span> Running inference…'; setStatus('Loading saved model and analyzing all slices…', 'loading');
  try {
    state.result = await api(`/api/analyze/${encodeURIComponent(state.patient)}/${state.model}`);
    state.index = Math.min(state.index, state.result.slice_count - 1);
    $('slice').max = state.result.slice_count - 1; $('slice').value = state.index; $('slice').disabled = false;
    $('positive-count').textContent = state.result.positive_slices; $('negative-count').textContent = state.result.negative_slices;
    $('selected-model').textContent = state.result.model_label; $('threshold-card').textContent = Number(state.result.threshold).toFixed(2);
    $('modality-status').textContent = `4 modalities · ${state.result.slice_count} slices`;
    $('slice-count').textContent = `${state.index + 1} / ${state.result.slice_count}`;
    setStatus(`Analysis ready · ${state.result.slice_count} slices · validation split`); loadMri();
  } catch (error) { setStatus(error.message, 'error'); }
  finally { $('analyze').disabled = false; $('analyze').classList.remove('is-loading'); $('analyze').innerHTML = '<span>▶</span> Run slice analysis'; }
}
function renderComparison(data) {
  const keys = ['logistic_regression', 'random_forest', 'vqc_2', 'vqc_4', 'vqc_8'];
  $('comparison-body').innerHTML = keys.map(key => {
    const m = data.models[key] || {};
    return `<tr><td>${labels[key]}</td><td>${fmt(m.accuracy)}</td><td>${fmt(m.precision)}</td><td>${fmt(m.recall)}</td><td>${fmt(m.f1)}</td><td>${fmt(m.roc_auc)}</td><td>${m.training_slices ?? '—'}</td></tr>`;
  }).join('');
  $('comparison-notes').textContent = (data.notes || []).join(' ');
  $('comparison-chart').innerHTML = keys.map(key => {
    const model = data.models[key] || {};
    return `<div class="compare-model"><div class="compare-model-title">${labels[key]}</div>${metricKeys.map(([label, field]) => {
      const score = Number(model[field]); const valid = Number.isFinite(score);
      return `<div class="compare-bar"><span>${label}</span><div class="compare-track"><div class="compare-fill" style="width:${valid ? Math.max(0, Math.min(100, score * 100)) : 0}%"></div></div><span class="compare-value">${valid ? `${(score * 100).toFixed(0)}%` : '—'}</span></div>`;
    }).join('')}</div>`;
  }).join('');
}
async function init() {
  try {
    const data = await api('/api/patients'); state.patients = data.patients;
    $('patient').innerHTML = state.patients.map(id => `<option value="${id}">${id}</option>`).join('');
    $('patient').disabled = false; state.patient = state.patients[0]; $('patient').value = state.patient;
    $('modality-status').textContent = 'Validation patient ready'; $('analyze').disabled = false;
    setStatus(`${state.patients.length} permitted validation patients available`); loadMri();
  } catch (error) { $('patient').innerHTML = '<option>Validation cohort unavailable</option>'; setStatus(error.message, 'error'); }
  try { renderComparison(await api('/api/comparison')); }
  catch (error) { $('comparison-body').innerHTML = `<tr><td colspan="7">${error.message}</td></tr>`; }
  try {
    const data = await api('/api/segmentation-metrics'); $('metric-slices').textContent = data.slices ?? '—';
    $('dice').textContent = fmt(data.metrics.dice); $('tumor-dice').textContent = fmt(data.metrics.tumor_positive_slice_dice);
    $('global-dice').textContent = fmt(data.metrics.global_dice); $('iou').textContent = fmt(data.metrics.iou);
    $('pixel-precision').textContent = fmt(data.metrics.pixel_precision); $('pixel-recall').textContent = fmt(data.metrics.pixel_recall);
  } catch (error) { document.querySelector('.metrics-foot').textContent = `Validation metrics unavailable: ${error.message}`; }
}

$('patient').addEventListener('change', () => {
  state.patient = $('patient').value; state.result = null; state.index = 0; $('slice').disabled = true; $('slice').max = 0; $('slice').value = 0;
  ['positive-count', 'negative-count', 'probability', 'selected-model', 'threshold-card'].forEach(id => $(id).textContent = '—');
  $('prediction-label').textContent = 'Select a patient and run analysis';
  $('prediction-status').className = 'state-pill muted'; $('prediction-status').textContent = 'Awaiting analysis';
  $('seg-status').className = 'state-pill muted'; $('seg-status').textContent = 'Awaiting analysis';
  $('seg-skipped').hidden = false; $('seg-skipped').textContent = 'Run detection to determine whether conditional segmentation is required.';
  $('seg-content').hidden = true; $('modality-status').textContent = 'Validation MRI selected'; loadMri();
});
$('analyze').addEventListener('click', analyze);
$('model').addEventListener('change', () => { $('vqc-config').textContent = modelConfig($('model').value); if (state.result) analyze(); });
$('modality').addEventListener('change', () => { state.modality = $('modality').value; loadMri(); });
$('slice').addEventListener('input', () => { state.index = Number($('slice').value); loadMri(); });
$('prev').addEventListener('click', () => { if (state.index > 0) { state.index--; $('slice').value = state.index; loadMri(); } });
$('next').addEventListener('click', () => { if (state.result && state.index < state.result.slice_count - 1) { state.index++; $('slice').value = state.index; loadMri(); } });
$('zoom-in').addEventListener('click', () => setZoom(state.zoom + .2));
$('zoom-out').addEventListener('click', () => setZoom(state.zoom - .2));
$('fit').addEventListener('click', () => setZoom(1)); $('reset-zoom').addEventListener('click', () => setZoom(1));
['brightness', 'contrast'].forEach(id => $(id).addEventListener('input', refreshDisplay));
$('reset-display').addEventListener('click', () => { $('brightness').value = 100; $('contrast').value = 100; setZoom(1); });
$('menu-toggle').addEventListener('click', () => { const open = $('sidebar').classList.toggle('open'); $('menu-toggle').setAttribute('aria-expanded', String(open)); });
document.querySelectorAll('.nav').forEach(link => link.addEventListener('click', () => { $('sidebar').classList.remove('open'); $('menu-toggle').setAttribute('aria-expanded', 'false'); document.querySelectorAll('.nav').forEach(item => item.classList.remove('active')); link.classList.add('active'); }));
window.addEventListener('resize', drawChart); init();
