// Only the Electron main process may own an API key. The renderer receives a
// status, never the saved key, and workers receive it over their private stdin.
const fs = require('node:fs');
const path = require('node:path');
const {randomUUID} = require('node:crypto');

function validKey(value) {
  if(typeof value!=='string'||!/^sk-[A-Za-z0-9_-]{16,4093}$/.test(value.trim()))
    throw new Error('Enter a valid OpenAI API key.');
  return value.trim();
}
function redactSecrets(value, secrets=[]) {
  let text=String(value);
  for(const secret of secrets)if(typeof secret==='string'&&secret.length)text=text.split(secret).join('[redacted]');
  return text.replace(/sk-[A-Za-z0-9_-]{16,}/g,'[redacted]');
}
function workerEnvironment(environment) {
  const result={...environment};
  // Never leak a shell's API credentials into unrelated media workers.
  for(const key of Object.keys(result))if(/^(OPENAI|SEAMSTRESS_OPENAI)_/i.test(key))delete result[key];
  return result;
}
function assertNoRendererSecrets(value) {
  if(!value||typeof value!=='object')return;
  for(const [key,item] of Object.entries(value)) {
    if(/api.?key|credential|authorization|access.?token/i.test(key))
      throw new Error('Save API credentials in AI settings, not in repair options.');
    assertNoRendererSecrets(item);
  }
}
function createCredentialStore({safeStorage,file,filesystem=fs,platform=process.platform}) {
  function available() {
    try {
      if(!safeStorage.isEncryptionAvailable())return false;
      if(platform!=='linux')return true;
      return ['gnome_libsecret','kwallet','kwallet5','kwallet6'].includes(safeStorage.getSelectedStorageBackend?.());
    }catch{return false;}
  }
  function status() {
    return {provider:'openai',configured:available()&&filesystem.existsSync(file),secureStorageAvailable:available()};
  }
  function get() {
    if(!available())throw new Error('Secure OS credential storage is unavailable.');
    if(!filesystem.existsSync(file))throw new Error('Add an OpenAI API key in AI settings first.');
    try{return validKey(safeStorage.decryptString(filesystem.readFileSync(file)));}
    catch{throw new Error('The saved API key could not be unlocked. Save it again in AI settings.');}
  }
  function set(value) {
    const key=validKey(value);
    if(!available())throw new Error('Secure OS credential storage is unavailable. The key was not saved.');
    const directory=path.dirname(file),temporary=path.join(directory,`.credential-${randomUUID()}`);
    filesystem.mkdirSync(directory,{recursive:true,mode:0o700});
    try {
      const encrypted=safeStorage.encryptString(key);
      filesystem.writeFileSync(temporary,encrypted,{mode:0o600,flag:'wx'});
      filesystem.renameSync(temporary,file);
    }catch{throw new Error('The API key could not be saved securely.');}
    finally{try{filesystem.unlinkSync(temporary);}catch{}}
    return status();
  }
  function clear() {try{filesystem.unlinkSync(file);}catch(error){if(error.code!=='ENOENT')throw new Error('The saved API key could not be removed.');}return status();}
  return {get,set,clear,status};
}
module.exports={createCredentialStore,validKey,redactSecrets,workerEnvironment,assertNoRendererSecrets};
