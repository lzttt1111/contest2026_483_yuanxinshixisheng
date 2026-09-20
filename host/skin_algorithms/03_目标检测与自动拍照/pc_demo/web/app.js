import {AngleDisplay} from './angle-display.mjs';
const $ = id => document.getElementById(id);
const angles=new AngleDisplay(p=>{for(const k of ['yaw','pitch','roll'])$(k).textContent=p?`${p[k].toFixed(1)}°`:'—';});
let rawDisplayAt=-Infinity;
function readSettings(){
 const settings={yaw_target:Number($('target-angle').value),yaw_tolerance:Number($('angle-tolerance').value),pitch_limit:Number($('pitch-limit').value),roll_limit:Number($('roll-limit').value),stable_ms:Number($('stable-duration').value)};
 if(!$('capture-settings').reportValidity()||settings.yaw_tolerance>=settings.yaw_target||settings.yaw_target+settings.yaw_tolerance>89)throw Error('请检查参数：误差须小于目标角度，目标加误差不得超过89°');
 return settings;
}
async function pushSettings(){
 const values=readSettings();
 if(session)await api(`/v1/session/${session}/settings`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});
 $('settings-status').textContent=`${session?'已应用':'下次开启时应用'}：${values.yaw_target}° ±${values.yaw_tolerance}° · 稳定${values.stable_ms}ms · 俯仰±${values.pitch_limit}° / 歪头±${values.roll_limit}°`;
 return values;
}
$('capture-settings').onsubmit=async e=>{
 e.preventDefault();try{readSettings();}catch(e){$('settings-status').textContent=e.message;return;}
 const resume=running;running=false;generation++;$('apply-settings').disabled=true;
 if(active)await active;
 try{await pushSettings();renderPhotos();clearResult();status('参数已更新，历史照片仍保留在磁盘');}
 catch(e){$('settings-status').textContent='未应用：'+e.message;}
 finally{$('apply-settings').disabled=false;running=resume&&!!stream;if(running)loop();}
};
for(const id of ['target-angle','angle-tolerance','pitch-limit','roll-limit','stable-duration'])$(id).addEventListener('input',()=>{
 const target=Number($('target-angle').value),tol=Number($('angle-tolerance').value);
 $('range-preview').textContent=`左右各${target}° · 接受${target-tol}°–${target+tol}°`;
 $('settings-status').textContent='有未应用修改，点击“应用参数”后生效';
});
let mode='face', session='', stream=null, running=false, active=null, frameId=0, generation=0;
let latest=null, updated=0, lastCameraTime=-1, displayed=-1;
const video=$('video'), canvas=$('display'), ctx=canvas.getContext('2d');
const capture=document.createElement('canvas'), cg=capture.getContext('2d');
const status=(text,error=false)=>{ $('status').textContent=text; $('status').classList.toggle('error',error); };
async function api(path,options={}){
  const r=await fetch(path,options);if(!r.ok){const e=await r.json().catch(()=>({detail:r.statusText}));throw Error(e.detail||r.statusText);}return r.json();
}
function clearResult(){latest=null; angles.clear();rawDisplayAt=-Infinity;$('raw-pose').textContent='等待原始角度';$('progress').value=0;$('region-status').textContent='等待最新人脸结果';}
function renderPhotos(photos={}){
 for(const side of ['left','right']){
   const p=photos[side], im=$(side+'-image'), a=$(side+'-download'); im.hidden=!p; $(side+'-empty').hidden=!!p;
   document.querySelector(`[data-retake="${side}"]`).disabled=!p;
   a.setAttribute('aria-disabled',String(!p));
   if(p){im.src=p.image_url; a.href=p.image_url; a.download=`${side}-${p.frame_id}.jpg`;$(side+'-meta').textContent=`帧 ${p.frame_id} · yaw ${p.pose.yaw.toFixed(1)}°`;}
   else{im.removeAttribute('src');a.removeAttribute('href');$(side+'-meta').textContent='';}
 }
}
async function reset(side='all', shouldResume=true){
 const resume=running; running=false; generation++; if(active)await active;
 try{if(session)await api(`/v1/session/${session}/reset?side=${side}`,{method:'POST'});
  if(side==='all')renderPhotos();else{const im=$(side+'-image'); im.hidden=true;$(side+'-empty').hidden=false;$(side+'-download').setAttribute('aria-disabled','true');$(side+'-meta').textContent='';document.querySelector(`[data-retake="${side}"]`).disabled=true;}
  clearResult();$('guidance').textContent='等待有效人脸';
 }catch(e){status(e.message,true);}
 running=shouldResume&&resume&&!!stream;if(running)loop();
}
async function stop(){running=false;generation++;if(active)await active;stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;
 $('start').disabled=false;$('stop').disabled=true;$('reset').disabled=true;$('camera').disabled=false;clearResult();canvas.hidden=true;$('empty').hidden=false;
 if(session)await api(`/v1/session/${session}/reset`,{method:'POST'}).catch(()=>{});
 session='';status('摄像头已停止');
}
async function devices(){const list=await navigator.mediaDevices.enumerateDevices();const selected=$('camera').value;$('camera').replaceChildren(new Option('默认摄像头',''));for(const d of list.filter(d=>d.kind==='videoinput'))$('camera').add(new Option(d.label||'摄像头',d.deviceId));$('camera').value=selected;}
async function start(){
 if(stream)await stop();
 $('start').disabled=true;status('正在连接摄像头和推理服务…');
 try{
  await api('/health');
  let expired=false, cameraTimer;
  const cameraRequest=navigator.mediaDevices.getUserMedia({video:{deviceId:$('camera').value?{exact:$('camera').value}:undefined,width:{ideal:1280},height:{ideal:720}},audio:false});
  cameraRequest.then(s=>{if(expired)s.getTracks().forEach(t=>t.stop());},()=>{});
  try{stream=await Promise.race([cameraRequest,new Promise((_,reject)=>{cameraTimer=setTimeout(()=>{expired=true;reject(Error('摄像头等待超时，请检查授权或使用 Windows Edge/Chrome 打开本页'));},15000);})]);}finally{clearTimeout(cameraTimer);}
  video.srcObject=stream;await video.play();await devices();
  session=(await api('/v1/session',{method:'POST'})).session_id;
  await pushSettings();
  running=true;frameId=0;generation++;lastCameraTime=-1;renderPhotos();clearResult();
  $('stop').disabled=false;$('reset').disabled=false;$('camera').disabled=true;canvas.hidden=false;$('empty').hidden=true;
  stream.getVideoTracks()[0].addEventListener('ended',()=>stop());status('运行中 · 图片在本机处理');loop();
 }catch(e){await stop();status(`启动失败：${e.message}`,true);}
}
async function sendFrame(gen){
 if(video.readyState<2||video.currentTime===lastCameraTime)return;
 lastCameraTime=video.currentTime;
 capture.width=video.videoWidth;capture.height=video.videoHeight;const at=performance.now();cg.drawImage(video,0,0);
 const blob=await new Promise(resolve=>capture.toBlob(resolve,'image/jpeg',.90));
 if(!blob||gen!==generation)return;
 const fid=++frameId, before=performance.now();
 const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),10000);
 try{
  const r=await api('/v1/stream/frame',{method:'POST',body:blob,signal:controller.signal,headers:{'Content-Type':'image/jpeg','X-Session-Id':session,'X-Mode':mode,'X-Frame-Id':String(fid),'X-Capture-Ms':String(at),'X-Frame-Age-Ms':String(performance.now()-at)}});
  if(gen!==generation||!running)return;
  const elapsed=performance.now()-at;
  $('performance').textContent=`${(1000/elapsed).toFixed(1)} FPS · 往返 ${(performance.now()-before).toFixed(0)} ms`;
  $('provider').textContent=r.runtime_provider;
  if(elapsed>700){clearResult();$('guidance').textContent='结果过期，等待新画面';renderPhotos(r.photos);return;}
  latest=r;updated=performance.now();displayed=fid;
  if(mode==='face'){
    angles.update(r.pose,performance.now());
    if(!r.raw_pose?.valid){$('raw-pose').textContent='原始角度无效';rawDisplayAt=-Infinity;}
    else if(performance.now()-rawDisplayAt>=1000){rawDisplayAt=performance.now();$('raw-pose').textContent=`原始 Yaw ${r.raw_pose.yaw.toFixed(1)}° / Pitch ${r.raw_pose.pitch.toFixed(1)}° / Roll ${r.raw_pose.roll.toFixed(1)}°`;}
    $('progress').value=r.decision.progress;$('guidance').textContent=r.decision.reason;renderPhotos(r.photos);
    status(r.saved?'照片已保存至实验目录 captures/':'运行中 · 请缓慢移动，达标后保持稳定');
  }else{
    const fr=r.face_regions;
    $('region-status').textContent=fr?.valid?`接触点区域：${fr.reason}`:(fr?.reason||'区域不可用');
    $('region-legend').replaceChildren(...(fr?.regions||[]).map(region=>{const s=document.createElement('span');s.textContent='● '+region.name+' ';s.style.color=region.color;return s;}));
    $('yaw').textContent=r.device_bbox?'已检测':'未检测';$('pitch').textContent=r.device_bbox?`${(r.device_score*100).toFixed(1)}%`:'—';$('roll').textContent=r.contact_point?'已定位':'—';$('guidance').textContent=r.device_bbox?'旧仪器检测正常':'请将旧仪器放入画面';
  }
 }catch(e){if(gen===generation){clearResult();$('guidance').textContent='连接或推理失败，暂停拍照';status(e.message,true); if(e.name==='AbortError'){running=false;$('start').disabled=false;}}}
 finally{clearTimeout(timer);}
}
async function loop(){if(!running||active)return;const gen=generation;active=sendFrame(gen);await active;active=null;if(running&&gen===generation)setTimeout(loop,0);}
function draw(){
 if(stream&&video.readyState>=2){
  if(canvas.width!==video.videoWidth||canvas.height!==video.videoHeight){canvas.width=video.videoWidth;canvas.height=video.videoHeight;}
  ctx.save();if($('mirror').checked){ctx.translate(canvas.width,0);ctx.scale(-1,1);}ctx.drawImage(video,0,0);
  if(latest&&performance.now()-updated<300){
   const r=latest;
   if(r.face_regions?.valid){
     if($('regions').checked)for(const region of r.face_regions.regions){
       ctx.beginPath();for(const ring of region.rings){ring.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();}
       ctx.fillStyle=region.color;ctx.globalAlpha=.28;ctx.fill('evenodd');ctx.globalAlpha=1;ctx.strokeStyle=region.color;ctx.lineWidth=1;ctx.stroke();
     }
     if($('landmarks').checked){ctx.fillStyle='#fff';for(const [x,y] of r.face_regions.landmarks){ctx.beginPath();ctx.arc(x,y,1.2,0,Math.PI*2);ctx.fill();}}
   }
   const b=r.face?[r.face.x,r.face.y,r.face.x+r.face.width,r.face.y+r.face.height]:r.device_bbox;
   if(b){ctx.strokeStyle='#29ddd0';ctx.lineWidth=3;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}
   if(r.contact_point&&$('contact').checked){ctx.fillStyle='#ffbb50';ctx.beginPath();ctx.arc(...r.contact_point,6,0,Math.PI*2);ctx.fill();}
  }ctx.restore();
 }
 if(latest&&performance.now()-updated>700){clearResult();$('guidance').textContent='等待最新有效结果';}
 requestAnimationFrame(draw);
}
$('start').onclick=start;$('stop').onclick=stop;$('reset').onclick=()=>reset();
document.querySelectorAll('[data-retake]').forEach(b=>b.onclick=()=>reset(b.dataset.retake));
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=async()=>{
 if(mode===b.dataset.mode)return;document.querySelectorAll('[data-mode]').forEach(x=>x.disabled=true);
 const resume=running;
 try{await reset('all',false);mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x===b));
 $('photos').hidden=mode!=='face';document.querySelectorAll('.face-only').forEach(x=>x.hidden=mode!=='face');
 $('region-panel').hidden=mode!=='device';
 $('panel-title').textContent=mode==='face'?'当前角度':'仪器检测结果';
 ['metric-a','metric-b','metric-c'].forEach((id,i)=>$(id).textContent=(mode==='face'?['Yaw','Pitch','Roll']:['仪器','置信度','接触点'])[i]);clearResult();
 $('guidance').textContent=mode==='face'?'等待有效人脸':'等待旧仪器进入画面';
 }finally{document.querySelectorAll('[data-mode]').forEach(x=>x.disabled=false);running=resume&&!!stream;if(running)loop();}
});
window.addEventListener('pagehide',()=>stream?.getTracks().forEach(t=>t.stop()));
document.addEventListener('visibilitychange',()=>{if(document.hidden&&stream)stop();});
if(navigator.mediaDevices){devices().catch(()=>{});navigator.mediaDevices.addEventListener('devicechange',()=>{if(stream)stop();devices().catch(()=>{});});}
api('/health').then(h=>{status('服务已连接 · 请选择摄像头');$('provider').textContent=`仪器 ${h.providers.device} · 人脸/角度 CPU`;}).catch(e=>status('服务连接失败：'+e.message,true));
draw();
