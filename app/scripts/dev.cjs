const {spawn}=require('node:child_process');
const path=require('node:path');
const appDir=path.resolve(__dirname,'..');
const vite=spawn(process.execPath,[path.join(appDir,'node_modules/vite/bin/vite.js'),'--host','127.0.0.1'],{cwd:appDir,stdio:'inherit'});
let electron;
function stop(){electron?.kill();vite.kill();}
process.on('SIGINT',()=>{stop();process.exit(0)});process.on('SIGTERM',()=>{stop();process.exit(0)});
(async()=>{
  for(let i=0;i<100;i++){
    try{if((await fetch('http://127.0.0.1:5173')).ok)break;}catch{}
    await new Promise(r=>setTimeout(r,100));
  }
  electron=spawn(require('electron'),['.'],{cwd:appDir,env:{...process.env,SEAMSTRESS_DEV_URL:'http://127.0.0.1:5173'},stdio:'inherit'});
  electron.on('close',code=>{vite.kill();process.exit(code||0);});
})();
