import test from 'node:test';
import assert from 'node:assert/strict';
import {overlayRows,metricTone,compareMatchedLevels,overlayCsv,HIA11_METRICS}
  from '../src/ml_lab/ui/static/ml_lab_hia_visuals.mjs';
const network=()=>({source_digest:'sha-same',binding_digest:'binding',source_topology:'tree',
  network_information:{status:'ok',support:120,levels:[
    {level:0,mapped_node_ids:['source:root'],mapped_node_count:1,structural_node_count:3,
      level_joint_entropy_bits:1,incremental_joint_entropy_given_prior_bits:1,
      information_shared_with_prior_bits:0,within_level_total_correlation_bits:0,
      target_incremental_information_bits:.75},
    {level:1,mapped_node_ids:['source:child','source:other'],mapped_node_count:2,structural_node_count:15,
      level_joint_entropy_bits:1.5,incremental_joint_entropy_given_prior_bits:.5,
      information_shared_with_prior_bits:1,within_level_total_correlation_bits:.4,
      target_incremental_information_bits:.0},
  ],cross_level_pairs:[{left_level:0,right_level:1,mutual_information_bits:.2,support:120}]}});

test('level-wide metrics are replicated ONLY to explicitly mapped original source IDs',()=>{
  const rows=overlayRows(network(),'network','incremental_joint_entropy_given_prior_bits');
  assert.deepEqual(rows.map(r=>[r.node_id,r.value,r.scope,r.support]),[
    ['source:root',1,'level-aggregate',120],
    ['source:child',.5,'level-aggregate',120],
    ['source:other',.5,'level-aggregate',120],
  ]);
  assert.ok(!rows.some(r=>r.node_id==='unmapped'));
  assert.deepEqual(overlayRows(network(),'network','target_incremental_information_bits').map(r=>r.value),[.75,0,0]);
  assert.ok(Object.hasOwn(HIA11_METRICS.network,'within_level_total_correlation_bits'));
});

test('non-OK / uncomputed numerical evidence is unavailable; never replaced with zeros',()=>{
  const noSupport=network();noSupport.network_information.status='insufficient_support';
  assert.deepEqual(overlayRows(noSupport,'network','level_joint_entropy_bits'),[]);
  const noTarget=network();delete noTarget.network_information.levels[0].target_incremental_information_bits;
  assert.deepEqual(overlayRows(noTarget,'network','target_incremental_information_bits').map(x=>x.node_id),['source:child','source:other']);
});

test('node metrics use HIA-07 per-node estimates, not a level statistic',()=>{
  const report={level:3,level_information:{status:'ok',support:55,
    per_mapped_node:[{node_id:'a',variable_id:'X',entropy_bits:1,
      conditional_entropy_given_other_mapped_members_bits:.25,mi_with_other_mapped_members_bits:.75},
      {node_id:'b',variable_id:'Y',entropy_bits:1.3,
      conditional_entropy_given_other_mapped_members_bits:.65,mi_with_other_mapped_members_bits:.65}]}};
  assert.deepEqual(overlayRows(report,'level','entropy_bits').map(r=>r.value),[1,1.3]);
  assert.deepEqual(overlayRows(report,'level','mi_with_other_mapped_members_bits').map(r=>r.value),[.75,.65]);
  assert.deepEqual(overlayRows(report,'level','support').map(r=>r.support),[55,55]);
});

test('one atomic result is measured only when entropy supports it',()=>{
  const report={node_id:'level3:node',variable_id:'V',node_context:{level:3},support:{observed_scalar_rows:41},entropy:{status:'ok',bits:.75}};
  assert.deepEqual(overlayRows(report,'atomic','entropy_bits').map(x=>x.value),[.75]);
  report.entropy.status='unavailable';
  assert.deepEqual(overlayRows(report,'atomic','entropy_bits'),[]);
  assert.deepEqual(overlayRows(report,'atomic','support').map(x=>x.value),[41]);
});

test('distinct levels are compared on a single matched HIA-10 cohort',()=>{
  const result=compareMatchedLevels(network(),1,0);
  assert.equal(result.cohort_support,120);
  assert.equal(result.first.level,1);
  assert.equal(result.second.level,0);
  assert.equal(result.pairwise_cross_level_mi_bits,.2);
  assert.throws(()=>compareMatchedLevels(network(),0,0));
  assert.throws(()=>compareMatchedLevels(network(),0,2));
  assert.throws(()=>compareMatchedLevels({...network(),network_information:{status:'insufficient_support'}},0,1));
});

test('palette treats measured zero as valid and supports missing metrics',()=>{
  assert.equal(metricTone(0,0).fill,'#eaf2fa');
  assert.equal(metricTone(20,20).fill,'#205e89');
  assert.equal(metricTone(null,1),null);
  assert.equal(metricTone(-1,2),null);
});

test('export keeps original IDs, source fingerprint, metric scope and CSV escaping',()=>{
  const data=overlayCsv([{node_id:'Node, "one"',variable_id:'col',level:7,value:0,
    support:55,scope:'atomic-observation'}],{source_digest:'digest',source_revision:'rev',basis:'level',metric:'entropy_bits'});
  assert.match(data,/"Node, ""one"""/);
  assert.match(data,/"digest"/);
  assert.match(data,/"entropy_bits"/);
  assert.match(data,/"atomic-observation"/);
  assert.ok(data.endsWith('\r\n'));
});

test('malformed or duplicate overlay identities fail rather than silently collapsing',()=>{
  assert.throws(()=>overlayRows({},'unknown','entropy_bits'));
  const bad=network();bad.network_information.levels[0].mapped_node_ids=['a','a'];
  assert.throws(()=>overlayRows(bad,'network','support'));
});
