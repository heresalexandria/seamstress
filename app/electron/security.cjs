const path = require('node:path');
const MEDIA_EXTENSIONS = new Set(['.mp4','.mov','.mkv','.webm','.m4v','.avi','.jpg','.jpeg','.png']);
function containedFile(root, urlPath) {
  const candidate = path.resolve(root, '.' + decodeURIComponent(urlPath));
  if (!candidate.startsWith(path.resolve(root) + path.sep)) throw new Error('Invalid asset path');
  return candidate;
}
function collectMedia(project) {
  const values = [project.source, project.artifacts?.proxy, project.artifacts?.fullPreview, project.artifacts?.export,
    ...(project.artifacts?.thumbnails || []).map(v=>v.path), ...(project.artifacts?.seamPreviews || []).map(v=>v.path)];
  // Reconstruction packages may contain many arbitrary asset paths. Only these
  // display fields are granted to the renderer, including historical previews.
  for(const entry of Object.values(project.reconstructions||{})) {
    for(const summary of [entry?.candidate,entry?.accepted,...(Array.isArray(entry?.history)?entry.history:[])]) {
      if(!summary||typeof summary!=='object')continue;
      values.push(summary.sourceFramePath,summary.candidatePreviewPath);
      for(const layer of Array.isArray(summary.layers)?summary.layers:[])values.push(layer?.maskPreviewPath);
      for(const frame of Array.isArray(summary.frames)?summary.frames:[]) {
        values.push(frame?.sourceFramePath);
        if(frame?.masks&&typeof frame.masks==='object'&&!Array.isArray(frame.masks))values.push(...Object.values(frame.masks));
      }
    }
  }
  return values.filter(v=>typeof v==='string' && MEDIA_EXTENSIONS.has(path.extname(v).toLowerCase())).map(v=>path.resolve(v));
}
function validateStage(stage) {
  if (!['detect','analyze','refine','reconstruct','preview','process','export'].includes(stage)) throw new Error('Unknown workflow stage');
  return stage;
}
module.exports = {containedFile,collectMedia,validateStage};
