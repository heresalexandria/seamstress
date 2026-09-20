const {app,BrowserWindow,ipcMain,dialog,protocol,net,shell,Menu,autoUpdater:nativeUpdater} = require('electron');
const smoke=app.commandLine.hasSwitch('smoke');
// Set this before Electron is ready: automated checks must never activate the
// app or interrupt the foreground application. Normal launches are unchanged.
if(smoke&&process.platform==='darwin')app.setActivationPolicy('prohibited');
const {autoUpdater}=require('electron-updater');
const {createUpdater}=require('./updater.cjs');
const path=require('node:path');
const fs=require('node:fs');
const {pathToFileURL}=require('node:url');
const {spawn}=require('node:child_process');
const {randomUUID}=require('node:crypto');
const {containedFile,collectMedia,validateStage}=require('./security.cjs');
const {serveMedia}=require('./media.cjs');
const root=path.resolve(__dirname,'../..');
const explicitUserData=app.commandLine.getSwitchValue('user-data-dir');
if(explicitUserData)app.setPath('userData',path.resolve(explicitUserData));
else if(!app.isPackaged)app.setPath('userData',path.join(root,'.app-data'));
protocol.registerSchemesAsPrivileged([
  {scheme:'seamstress-app',privileges:{standard:true,secure:true,supportFetchAPI:true}},
  {scheme:'seamstress-media',privileges:{standard:true,secure:true,stream:true,supportFetchAPI:true}}
]);
let window,updater;
const registryPath=path.join(app.getPath('userData'),'recent-projects.json');
let recent=[];
try{recent=JSON.parse(fs.readFileSync(registryPath,'utf8'));if(!Array.isArray(recent))recent=[];}catch{}
const projects=new Map(),media=new Set(),reveals=new Set(),exportsAllowed=new Set(),jobs=new Map(),busy=new Set();
function grant(project){
  projects.set(path.resolve(project.projectPath),project);
  recent=[project.projectPath,...recent.filter(v=>v!==project.projectPath)].slice(0,50);
  fs.mkdirSync(path.dirname(registryPath),{recursive:true});
  fs.writeFileSync(registryPath,JSON.stringify(recent));
  for(const file of collectMedia(project))media.add(file);
  reveals.add(path.resolve(project.projectPath));
  for(const file of Object.values(project.artifacts || {}))if(typeof file==='string')reveals.add(path.resolve(file));
  return project;
}
function knownProject(value){
  if(typeof value!=='string'||(!projects.has(path.resolve(value))&&!recent.includes(path.resolve(value))))throw new Error('Open this project before using it');
  return path.resolve(value);
}
function send(event){if(window&&!window.isDestroyed())window.webContents.send('job:event',event);}
function worker(operation,args,{jobId=randomUUID(),stage='import',events=false}={}){
  if(updater?.isInstalling())throw new Error('Seamstress is restarting to install an update.');
  const packaged=app.isPackaged;
  const backend=path.join(process.resourcesPath,'backend');
  const executable=packaged?path.join(backend,'seamstress-worker','seamstress-worker'):path.join(root,'.venv','bin','python');
  const argv=packaged?[]:['-m','seamstress.desktop_worker'];
  const child=spawn(executable,argv,{cwd:packaged?backend:root,detached:process.platform!=='win32',
    env:{...process.env,PYTHONUNBUFFERED:'1',OPENBLAS_NUM_THREADS:'2',OMP_NUM_THREADS:'2',
      PATH:[packaged?path.join(backend,'bin'):'','/opt/homebrew/bin','/usr/local/bin',process.env.PATH||''].filter(Boolean).join(path.delimiter)},
    stdio:['pipe','pipe','pipe']});
  let buffer='',errors='',finalEvent=null,cancelling=false;
  const promise=new Promise((resolve,reject)=>{
    child.stdout.on('data',data=>{
      buffer+=data.toString();let newline;
      while((newline=buffer.indexOf('\n'))>=0){
        const line=buffer.slice(0,newline);buffer=buffer.slice(newline+1);
        try{
          const event=JSON.parse(line);
          if(event.project)grant(event.project);
          if(event.type!=='progress')finalEvent=event;
          if(events)send({...event,jobId,stage:event.stage||stage});
        }catch(error){errors+='\nWorker protocol error: '+error.message;}
      }
      if(buffer.length>32*1024*1024){child.kill();reject(new Error('Worker message exceeded its limit'));}
    });
    child.stderr.on('data',data=>{errors=(errors+data.toString()).slice(-16000);});
    child.on('error',reject);
    child.on('close',code=>{
      jobs.delete(jobId);
      updater?.refreshActivity();
      if(finalEvent?.type==='complete'&&code===0)resolve(finalEvent.project);
      else if(cancelling||finalEvent?.type==='cancelled'){
        if(events&&finalEvent?.type!=='cancelled')send({jobId,type:'cancelled',stage,message:'Operation cancelled'});
        reject(new Error('Operation cancelled'));
      }else{
        const error=finalEvent?.error||errors.trim()||`Processing stopped (${code})`;
        if(events&&finalEvent?.type!=='error')send({jobId,type:'error',stage,error});
        reject(new Error(error));
      }
    });
  });
  child.stdin.on('error',()=>{});
  child.stdin.end(JSON.stringify({operation,args})+'\n');
  jobs.set(jobId,{child,cancel:()=>{
    cancelling=true;
    try{process.platform==='win32'?child.kill('SIGTERM'):process.kill(-child.pid,'SIGINT');}catch{}
    const force=setTimeout(()=>{if(child.exitCode===null){try{process.platform==='win32'?child.kill('SIGKILL'):process.kill(-child.pid,'SIGKILL');}catch{}}},8000);
    force.unref();
  }});
  updater?.refreshActivity();
  return {jobId,promise};
}
function handle(channel,fn){ipcMain.handle(channel,async(event,...args)=>{
  if(!window||event.sender!==window.webContents||event.senderFrame!==window.webContents.mainFrame)throw new Error('Untrusted request');
  return fn(...args);
});}
function requireIdle(projectPath){if(busy.has(projectPath))throw new Error('Wait for the current operation, or cancel it first');}
async function withProject(projectPath,operation){requireIdle(projectPath);busy.add(projectPath);updater?.refreshActivity();try{return await operation();}finally{busy.delete(projectPath);updater?.refreshActivity();}}
app.whenReady().then(()=>{
  if(!app.isPackaged&&!smoke&&process.platform==='darwin')app.dock.setIcon(path.join(__dirname,'../assets/icon.png'));
  updater=createUpdater({app,autoUpdater,nativeUpdater,hasActiveJobs:()=>jobs.size>0||busy.size>0,
    disabled:smoke,openExternal:url=>shell.openExternal(url),
    onState:state=>{if(window&&!window.isDestroyed())window.webContents.send('update:event',state);}});
  handle('update:state',()=>updater.getState());
  handle('update:check',()=>updater.check({manual:true}));
  handle('update:download',()=>updater.download());
  handle('update:install',()=>updater.install());
  handle('update:releases',()=>updater.openReleases());
  protocol.handle('seamstress-app',request=>{
    try{
      const url=new URL(request.url);
      if(url.hostname!=='app')return new Response('Forbidden',{status:403});
      return net.fetch(pathToFileURL(containedFile(path.join(__dirname,'../dist'),url.pathname==='/'?'/index.html':url.pathname)).href);
    }catch{return new Response('Not found',{status:404});}
  });
  protocol.handle('seamstress-media',request=>{
    try{
      const url=new URL(request.url),file=path.resolve(url.searchParams.get('path')||'');
      if(url.hostname!=='local'||!media.has(file))return new Response('Forbidden',{status:403});
      return serveMedia(file,request);
    }catch{return new Response('Not found',{status:404});}
  });
  handle('video:pick',async()=>{
    const result=await dialog.showOpenDialog(window,{title:'Choose your stitched oner',properties:['openFile'],filters:[{name:'Video',extensions:['mp4','mov','m4v','mkv','webm','avi']}]});
    return result.canceled?null:result.filePaths[0];
  });
  handle('project:create',async options=>{
    if(typeof options?.source!=='string'||!fs.statSync(options.source).isFile())throw new Error('Choose an existing video file');
    const source=path.resolve(options.source);
    if(!/\.(mp4|mov|m4v|mkv|webm|avi)$/i.test(source))throw new Error('Choose a supported video file');
    const name=path.basename(source,path.extname(source)).replace(/[^\p{L}\p{N}._-]+/gu,'-').slice(0,80)||'video';
    const folder=path.join(app.getPath('userData'),'projects',`${name}-${randomUUID().slice(0,8)}.seamstress`);
    const jobId=randomUUID();
    send({jobId,type:'progress',stage:'import',progress:0,message:'Opening your stitched shot and finding candidate seams'});
    return worker('create',{source,folder},{jobId,stage:'import',events:true}).promise;
  });
  handle('project:open',async()=>{
    const result=await dialog.showOpenDialog(window,{title:'Open a Seamstress project.json',defaultPath:path.join(app.getPath('userData'),'projects'),properties:['openFile'],filters:[{name:'Seamstress project',extensions:['json']}]});
    if(result.canceled)return null;
    return worker('get',{projectPath:result.filePaths[0]}).promise;
  });
  handle('project:get',value=>worker('get',{projectPath:knownProject(value)}).promise);
  handle('project:seams',options=>{
    const projectPath=knownProject(options?.projectPath);
    return withProject(projectPath,()=>worker('setSeams',{projectPath,seams:options.seams}).promise);
  });
  handle('job:run',args=>{
    const projectPath=knownProject(args?.projectPath),stage=validateStage(args.stage);requireIdle(projectPath);
    const options={...(args.options||{})};
    if(options.exportPath&&!exportsAllowed.has(path.resolve(options.exportPath)))throw new Error('Choose an export destination first');
    busy.add(projectPath);
    let job;
    try{job=worker('run',{projectPath,stage,options},{events:true,stage});}
    catch(error){busy.delete(projectPath);updater?.refreshActivity();throw error;}
    job.promise.catch(()=>{}).finally(()=>{busy.delete(projectPath);updater?.refreshActivity();});
    return {jobId:job.jobId};
  });
  handle('job:cancel',jobId=>{jobs.get(jobId)?.cancel();});
  handle('export:choose',async options=>{
    const name=path.basename(String(options?.suggestedName||'corrected.mp4'));
    const result=await dialog.showSaveDialog(window,{title:'Export your corrected oner',defaultPath:path.join(app.getPath('videos'),name),filters:[{name:'MP4 video',extensions:['mp4']}]});
    if(result.canceled||!result.filePath)return null;
    const file=path.resolve(result.filePath);
    if(fs.existsSync(file))throw new Error('Choose a new filename. Seamstress preserves existing exports.');
    exportsAllowed.add(file);return file;
  });
  handle('file:reveal',value=>{
    const file=path.resolve(String(value));if(!reveals.has(file)&&!media.has(file))throw new Error('File is not part of an open project');shell.showItemInFolder(file);
  });
  window=new BrowserWindow({width:1480,height:980,minWidth:1100,minHeight:740,backgroundColor:'#11191b',title:'Seamstress',
    show:!smoke,focusable:!smoke,skipTaskbar:smoke,
    titleBarStyle:'hiddenInset',trafficLightPosition:{x:22,y:25},
    webPreferences:{preload:path.join(__dirname,'preload.cjs'),contextIsolation:true,sandbox:true,nodeIntegration:false,webSecurity:true,backgroundThrottling:!smoke}});
  window.webContents.setWindowOpenHandler(()=>({action:'deny'}));
  const dev=app.isPackaged?null:process.env.SEAMSTRESS_DEV_URL;
  window.webContents.on('will-navigate',(event,url)=>{
    const allowed=dev?new URL(url).origin===new URL(dev).origin:url.startsWith('seamstress-app://app/');
    if(!allowed)event.preventDefault();
  });
  window.webContents.session.setPermissionRequestHandler((_contents,_permission,callback)=>callback(false));
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    {label:'Seamstress',submenu:[{role:'about'},{label:'Check for Updates…',click:()=>{void updater.check({manual:true});window.webContents.send('update:open');}},{type:'separator'},{role:'hide'},{role:'hideOthers'},{role:'unhide'},{type:'separator'},{role:'quit'}]},
    {label:'Edit',submenu:[{role:'undo'},{role:'redo'},{type:'separator'},{role:'cut'},{role:'copy'},{role:'paste'},{role:'selectAll'}]},
    {label:'View',submenu:[{role:'resetZoom'},{role:'zoomIn'},{role:'zoomOut'},{role:'togglefullscreen'},...(!app.isPackaged?[{role:'toggleDevTools'}]:[])]},
    {label:'Window',submenu:[{role:'minimize'},{role:'zoom'},{role:'front'}]}
  ]));
  window.loadURL(dev||'seamstress-app://app/index.html');
  updater.start();
});
app.on('before-quit',()=>{updater?.dispose();for(const job of jobs.values())job.cancel();});
app.on('window-all-closed',()=>app.quit());
