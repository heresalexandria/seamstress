const {test}=require('node:test');const assert=require('node:assert/strict');
const {containedFile,collectMedia,validateStage}=require('./security.cjs');
test('asset resolution rejects encoded traversal and root escape',()=>{
  assert.equal(containedFile('/app/dist','/assets/main.js'),'/app/dist/assets/main.js');
  assert.throws(()=>containedFile('/app/dist','/%2e%2e/.env'));
  assert.throws(()=>containedFile('/app/dist','/../dist-other/.env'));
});
test('media access is explicit and excludes non-media artifacts',()=>{
  assert.deepEqual(collectMedia({source:'/video.mp4',artifacts:{plan:'/secret.env',thumbnails:[{path:'/thumb.jpg'}]}}),['/video.mp4','/thumb.jpg']);
  assert.equal(validateStage('preview'),'preview');assert.throws(()=>validateStage('shell'));
});
