const fs=require('node:fs');
const path=require('node:path');
const {Readable}=require('node:stream');
const MIME={'.mp4':'video/mp4','.m4v':'video/mp4','.mov':'video/quicktime','.webm':'video/webm','.mkv':'video/x-matroska','.avi':'video/x-msvideo','.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png'};
/** Serve only an already-authorized path. Explicit ranges are essential for Chromium seeking. */
function serveMedia(filename,request,{origin='seamstress-app://app'}={}){
  const method=request.method||'GET';
  if(!['GET','HEAD'].includes(method))return new Response(null,{status:405,headers:{Allow:'GET, HEAD'}});
  let stat;try{stat=fs.statSync(filename);if(!stat.isFile())throw new Error();}catch{return new Response(null,{status:404});}
  const size=stat.size;
  const headers={'Accept-Ranges':'bytes','Content-Type':MIME[path.extname(filename).toLowerCase()]||'application/octet-stream',
                 'Cache-Control':'no-store','Content-Length':String(size),
                 'Access-Control-Allow-Origin':origin};
  const invalid=()=>new Response(null,{status:416,headers:{...headers,'Content-Range':`bytes */${size}`,'Content-Length':'0'}});
  let start=0,end=size-1,status=200;
  const range=request.headers.get('Range');
  if(range&&method==='GET'){
    const match=/^bytes=(\d*)-(\d*)$/.exec(range);
    if(!match||(!match[1]&&!match[2])||!size)return invalid();
    if(!match[1]){
      const suffix=Number(match[2]);if(!Number.isSafeInteger(suffix)||suffix<=0)return invalid();
      start=Math.max(0,size-suffix);
    }else{
      start=Number(match[1]);end=match[2]?Math.min(Number(match[2]),size-1):size-1;
    }
    if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start>=size||end<start)return invalid();
    status=206;headers['Content-Range']=`bytes ${start}-${end}/${size}`;headers['Content-Length']=String(end-start+1);
  }
  const body=method==='HEAD'||!size?null:Readable.toWeb(fs.createReadStream(filename,{start,end}));
  return new Response(body,{status,headers});
}
module.exports={serveMedia};
