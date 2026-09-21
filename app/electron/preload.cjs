const {contextBridge,ipcRenderer,webUtils} = require('electron');
contextBridge.exposeInMainWorld('seamstress', {
  pickVideo:()=>ipcRenderer.invoke('video:pick'),
  pathForFile:file=>webUtils.getPathForFile(file),
  createProject:options=>ipcRenderer.invoke('project:create',options),
  openProject:()=>ipcRenderer.invoke('project:open'),
  getProject:projectPath=>ipcRenderer.invoke('project:get',projectPath),
  setSeams:options=>ipcRenderer.invoke('project:seams',options),
  importSeamCorrection:options=>ipcRenderer.invoke('project:import-correction',options),
  importReconstruction:options=>ipcRenderer.invoke('project:import-reconstruction',options),
  getAISettings:()=>ipcRenderer.invoke('ai:settings'),
  setAIKey:key=>ipcRenderer.invoke('ai:key:set',key),
  clearAIKey:()=>ipcRenderer.invoke('ai:key:clear'),
  getSegmentationStatus:()=>ipcRenderer.invoke('segmentation:status'),
  run:options=>ipcRenderer.invoke('job:run',options),
  cancelJob:jobId=>ipcRenderer.invoke('job:cancel',jobId),
  chooseExportPath:options=>ipcRenderer.invoke('export:choose',options),
  onJobEvent:callback=>{
    const listener=(_event,value)=>callback(value);
    ipcRenderer.on('job:event',listener);
    return ()=>ipcRenderer.removeListener('job:event',listener);
  },
  mediaUrl:filename=>`seamstress-media://local/?path=${encodeURIComponent(filename)}`,
  revealFile:filename=>ipcRenderer.invoke('file:reveal',filename),
  getUpdateState:()=>ipcRenderer.invoke('update:state'),
  checkForUpdates:()=>ipcRenderer.invoke('update:check'),
  downloadUpdate:()=>ipcRenderer.invoke('update:download'),
  installUpdate:()=>ipcRenderer.invoke('update:install'),
  openReleases:()=>ipcRenderer.invoke('update:releases'),
  onUpdateState:callback=>{
    const listener=(_event,value)=>callback(value);
    ipcRenderer.on('update:event',listener);
    return ()=>ipcRenderer.removeListener('update:event',listener);
  },
  onOpenUpdates:callback=>{
    const listener=()=>callback();ipcRenderer.on('update:open',listener);
    return ()=>ipcRenderer.removeListener('update:open',listener);
  },
});
