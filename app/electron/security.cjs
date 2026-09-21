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
  return values.filter(v=>typeof v==='string' && MEDIA_EXTENSIONS.has(path.extname(v).toLowerCase())).map(v=>path.resolve(v));
}
function validateStage(stage) {
  if (!['detect','analyze','refine','preview','process','export'].includes(stage)) throw new Error('Unknown workflow stage');
  return stage;
}
module.exports = {containedFile,collectMedia,validateStage};
