const {test}=require('node:test');const assert=require('node:assert/strict');
const {containedFile,collectMedia,validateStage}=require('./security.cjs');
test('asset resolution rejects encoded traversal and root escape',()=>{
  assert.equal(containedFile('/app/dist','/assets/main.js'),'/app/dist/assets/main.js');
  assert.throws(()=>containedFile('/app/dist','/%2e%2e/.env'));
  assert.throws(()=>containedFile('/app/dist','/../dist-other/.env'));
});
test('reconstruction media grants only declared previews and masks',()=>{
  const summary={sourceFramePath:'/frame.png',candidatePreviewPath:'/preview.mp4',secret:'/forbidden.png',
    layers:[{maskPreviewPath:'/mask.png',asset:'/secret.png'}],
    frames:[{sourceFramePath:'/keyframe.png',masks:{actor:'/actor.png'},other:'/other.png'}]};
  const values=collectMedia({reconstructions:{18:{candidate:summary,history:[{sourceFramePath:'/old.jpg'}]}}});
  assert.deepEqual(values,['/frame.png','/preview.mp4','/mask.png','/keyframe.png','/actor.png','/old.jpg']);
  assert.equal(validateStage('reconstruct'),'reconstruct');
});
test('media access is explicit and excludes non-media artifacts',()=>{
  assert.deepEqual(collectMedia({source:'/video.mp4',artifacts:{plan:'/secret.env',thumbnails:[{path:'/thumb.jpg'}]}}),['/video.mp4','/thumb.jpg']);
  assert.equal(validateStage('preview'),'preview');assert.equal(validateStage('refine'),'refine');assert.throws(()=>validateStage('shell'));
});
