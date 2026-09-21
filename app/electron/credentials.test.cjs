const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {createCredentialStore,redactSecrets,workerEnvironment,assertNoRendererSecrets}=require('./credentials.cjs');
const KEY='sk-test_key_1234567890_private';
function harness(t,overrides={},platform='linux'){
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'seamstress-credentials-'));
  t.after(()=>fs.rmSync(directory,{recursive:true,force:true}));
  const file=path.join(directory,'private','openai.enc');
  const safeStorage={isEncryptionAvailable:()=>true,getSelectedStorageBackend:()=> 'gnome_libsecret',
    encryptString:key=>Buffer.from([...Buffer.from(key)].map(v=>v^0xaa)),
    decryptString:bytes=>Buffer.from([...bytes].map(v=>v^0xaa)).toString(),...overrides};
  return {file,store:createCredentialStore({safeStorage,file,platform})};
}
test('API key is encrypted on disk and status never returns a key',t=>{
  const {store,file}=harness(t);
  assert.deepEqual(store.status(),{provider:'openai',configured:false,secureStorageAvailable:true});
  assert.equal(store.set(KEY).configured,true);
  assert.ok(!fs.readFileSync(file).includes(Buffer.from(KEY)));
  assert.equal(fs.statSync(file).mode&0o777,0o600);
  assert.equal(store.get(),KEY);
  assert.equal(JSON.stringify(store.status()).includes(KEY),false);
  assert.equal(store.clear().configured,false);
  assert.throws(()=>store.get(),/Add an OpenAI/);
  assert.equal(store.clear().configured,false);
});
test('credential store fails closed with no secure backend or Linux basic_text',t=>{
  for(const overrides of [{isEncryptionAvailable:()=>false},{getSelectedStorageBackend:()=> 'basic_text'}]){
    const {store,file}=harness(t,overrides);
    assert.equal(store.status().secureStorageAvailable,false);
    assert.throws(()=>store.set(KEY),/not saved/);
    assert.equal(fs.existsSync(file),false);
    assert.throws(()=>store.get(),/unavailable/);
  }
});
test('storage failures and invalid input never echo the key',t=>{
  const {store}=harness(t,{encryptString:()=>{throw new Error(KEY);}});
  assert.throws(()=>store.set(KEY),error=>!error.message.includes(KEY));
  for(const key of ['',42,'sk-short',KEY+'\nAuthorization: value'])assert.throws(()=>store.set(key),/valid OpenAI/);
});
test('macOS and Windows do not call the Linux-only backend API',t=>{
  for(const platform of ['darwin','win32']){
    const {store}=harness(t,{getSelectedStorageBackend:()=>{throw new Error('Linux only');}},platform);
    assert.equal(store.status().secureStorageAvailable,true);
    assert.equal(store.set(KEY).configured,true);
  }
});
test('worker environment excludes inherited provider keys and option credentials are rejected',()=>{
  const clean=workerEnvironment({OPENAI_API_KEY:KEY,OPENAI_BASE_URL:'https://evil',SEAMSTRESS_OPENAI_KEY:KEY,PATH:'/bin',OTHER:'yes'});
  assert.deepEqual(clean,{PATH:'/bin',OTHER:'yes'});
  for(const options of [{apiKey:KEY},{reconstruction:{credentials:{openaiApiKey:KEY}}},{options:{authorization:KEY}}])assert.throws(()=>assertNoRendererSecrets(options));
  assert.doesNotThrow(()=>assertNoRendererSecrets({reconstruction:{allowAI:true,maxAIRequests:1}}));
  assert.equal(redactSecrets('failed '+KEY,[KEY]),'failed [redacted]');
  assert.equal(redactSecrets('failed '+KEY),'failed [redacted]');
});
