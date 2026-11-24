'use strict';
/* No browser or external JS libraries are needed. Test the actual read-only
   sidebar DOM update script with a minimal observable DOM/fetch environment. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const script = fs.readFileSync(path.resolve(__dirname, '../../src/ml_lab/ui/static/ml_lab_pypique_preflight.js'), 'utf8');

class Element {
  constructor() {
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    this.textContent = '';
    this.title = '';
    this.href = '';
    this.className = '';
  }
  setAttribute(k, v) { this.attributes[k] = v; }
  removeAttribute(k) { delete this.attributes[k]; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = [...children]; }
  querySelector(selector) { return this.selectors?.[selector] || null; }
}
function stage(id) {
  const item = new Element();
  item.dataset.mlwbStage = id;
  item.selectors = { '[data-mlwb-step-summary]': new Element() };
  return item;
}
function fixture() {
  const panel = new Element();
  panel.dataset.mlwbPreflightUrl = '/submodule/api/workbench/preflight';
  panel.selectors = {
    '.mlwb-preflight-state': new Element(),
    '.mlwb-preflight-steps': new Element(),
    '.mlwb-preflight-list': new Element(),
  };
  const chosen = new Element();
  chosen.selectors = {
    '[data-mlwb-dataset-name]': new Element(),
    '[data-mlwb-dataset-detail]': new Element(),
  };
  const stages = Object.fromEntries(['data','profiling','representation','comparison','tools']
    .map(id => [id, stage(id)]));
  let response = null;
  let reject = false;
  const callbacks = {};
  const document = {
    visibilityState: 'visible',
    querySelector(sel) {
      if (sel === '[data-mlwb-preflight-url]') return panel;
      if (sel === '[data-mlwb-dataset-summary]') return chosen;
      return null;
    },
    querySelectorAll(sel) {
      if (sel === '[data-mlwb-stage]') return Object.values(stages);
      return [];
    },
    createElement() { return new Element(); },
    addEventListener(type, fn) { callbacks[`doc:${type}`] = fn; },
  };
  const window = { addEventListener(type, fn) { callbacks[`win:${type}`] = fn; } };
  let calls = 0;
  const sandbox = {
    document, window,
    fetch(url, options) {
      calls++;
      assert.equal(url, panel.dataset.mlwbPreflightUrl);
      assert.equal(options.cache, 'no-store');
      assert.equal(options.credentials, 'same-origin');
      if (reject) return Promise.reject(new Error('offline'));
      return Promise.resolve({ok:true,json:async()=>response});
    },
  };
  function set(report) { response = report; reject = false; }
  function fail() { reject = true; }
  async function tick() { await new Promise(resolve => setImmediate(resolve)); }
  async function mount() { vm.runInNewContext(script, sandbox, {filename:'ml_lab_pypique_preflight.js'}); await tick(); }
  async function refresh() { callbacks['win:pageshow'](); await tick(); }
  function label(id) { return stages[id].selectors['[data-mlwb-step-summary]'].textContent; }
  function state(id) { return stages[id].dataset.state; }
  function selected() { return chosen.selectors['[data-mlwb-dataset-name]'].textContent; }
  function selectedDetail() { return chosen.selectors['[data-mlwb-dataset-detail]'].textContent; }
  return { set, fail, mount, refresh, label, state, selected, selectedDetail,
    calls: () => calls, stage: stages, panel, chosen };
}
const ids = ['data','profiling','representation','comparison','tools'];
function report(active = null, overrides = {}) {
  const selected = !!active;
  return {
    active_status: selected ? 'ready' : 'empty',
    active, saved:[], derived:[], recent_runs:[], profile_cached:false,
    steps: ids.map(id => ({id,label:id,state:selected ? (id === 'data'?'complete':'ready'):'blocked'})),
    ...overrides,
  };
}
(async () => {
  const ui = fixture();
  ui.set(report());
  await ui.mount();
  assert.equal(ui.selected(), 'No active dataset');
  assert.equal(ui.state('data'), 'ready'); // selecting or creating Data is always enabled
  assert.match(ui.label('data'), /create or select a dataset/);
  for (const id of ['profiling','representation','comparison']) {
    assert.equal(ui.state(id), 'blocked');
    assert.equal(ui.label(id), 'Select a dataset in Data first');
  }
  assert.equal(ui.state('tools'), 'partial'); // some synthetic/general tools are independent
  assert.match(ui.label('tools'), /Data-linked tools need dataset/);
  assert.equal(ui.calls(), 1);

  ui.set(report(null, {steps: report().steps.map(s => ({...s, state:s.id==='data'?'ready':s.state})),
    saved:[{name:'experiment',path:'saved.csv',status:'ready',url:'/saved'}]}));
  await ui.refresh();
  assert.equal(ui.label('data'), 'Saved data found · select in Data');
  assert.match(ui.selectedDetail(), /Saved data available/);
  assert.equal(ui.panel.selectors['.mlwb-preflight-list'].children.length, 2);

  ui.set(report({path:'analysis/data/derived/experiment-A.csv',rows:28,columns:6}));
  await ui.refresh();
  assert.equal(ui.selected(), 'experiment-A.csv');
  assert.match(ui.selectedDetail(), /28 rows · 6 columns/);
  assert.equal(ui.state('data'), 'complete');
  assert.equal(ui.label('data'), 'Selected · experiment-A.csv');
  assert.equal(ui.state('representation'), 'ready');
  assert.equal(ui.label('profiling'), 'Dataset ready · run profiling');
  assert.equal(ui.state('tools'), 'ready');

  const cached = report({path:'project\\another.tsv',rows:12,columns:2});
  cached.steps[1].state='complete';
  cached.profile_cached=true;
  ui.set(cached);
  await ui.refresh();
  assert.equal(ui.selected(), 'another.tsv');
  assert.equal(ui.label('profiling'), 'Saved profile available for this dataset');
  assert.equal(ui.state('profiling'), 'complete');

  ui.set(report(null, {active_status:'stale'}));
  await ui.refresh();
  assert.equal(ui.selected(), 'Selected source changed');
  for(const id of ids) assert.equal(ui.state(id), 'blocked');
  assert.match(ui.label('comparison'), /Reselect changed dataset/);

  ui.fail(); await ui.refresh();
  assert.equal(ui.selected(), 'Selection status unavailable');
  for(const id of ids) {
    assert.equal(ui.state(id), 'unknown');
    assert.equal(ui.label(id), 'Readiness check unavailable');
  }
  console.log('ML Lab pyPIQUE-style sidebar readiness: 6 states passed');
})().catch(e => {console.error(e);process.exitCode=1;});
