const {test}=require('node:test');const assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {serveMedia}=require('./media.cjs');
test('media replies have exact lengths and seekable byte ranges',async()=>{
 const folder=fs.mkdtempSync(path.join(os.tmpdir(),'seamstress-range-')),file=path.join(folder,'source.mp4');fs.writeFileSync(file,'0123456789');
 try{
  const request=(range,method='GET')=>new Request('https://local/video',{method,headers:range?{Range:range}:{}});
  let r=serveMedia(file,request());assert.equal(r.status,200);assert.equal(r.headers.get('content-length'),'10');assert.equal(await r.text(),'0123456789');
  r=serveMedia(file,request('bytes=3-5'));assert.equal(r.status,206);assert.equal(r.headers.get('content-range'),'bytes 3-5/10');assert.equal(r.headers.get('content-length'),'3');assert.equal(await r.text(),'345');
  r=serveMedia(file,request('bytes=8-'));assert.equal(await r.text(),'89');
  r=serveMedia(file,request('bytes=-3'));assert.equal(await r.text(),'789');
  r=serveMedia(file,request(null,'HEAD'));assert.equal(r.headers.get('content-length'),'10');assert.equal(await r.text(),'');
  for(const range of ['bytes=10-','bytes=6-3','bytes=-0','bytes=1-2,5-6'])assert.equal(serveMedia(file,request(range)).status,416);
 }finally{fs.rmSync(folder,{recursive:true,force:true});}
});
