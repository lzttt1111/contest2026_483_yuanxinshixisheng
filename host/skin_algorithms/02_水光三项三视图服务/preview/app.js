"use strict";
const labels={left:"左侧",front:"正面",right:"右侧"},modules={pores:"毛孔",spots:"可见色斑",surface_gloss:"表面油光"};
const $=id=>document.getElementById(id);
let current=null,selected="pores",generation=0;
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}
function badge(value){const b=el("span",value||"暂不可评分","grade");b.dataset.grade=value||"";return b}
function display(n){return n===null||n===undefined?"未覆盖":String(n)}
function table(headers,rows){const wrap=el("div",undefined,"scroll"),t=el("table"),head=el("thead"),h=el("tr");headers.forEach(v=>h.append(el("th",v)));head.append(h);t.append(head);const body=el("tbody");rows.forEach(r=>{const tr=el("tr",undefined,r.cls);r.cells.forEach(v=>{const td=el("td");if(v instanceof Node)td.append(v);else{td.textContent=v;if(v==="未覆盖"||v==="待补充")td.className="muted"}tr.append(td)});body.append(tr)});t.append(body);wrap.append(t);return wrap}
function render(){
 if(!current)return;
 const data=current.payload,item=data.results[selected];
 document.querySelectorAll("nav button").forEach(b=>b.classList.toggle("active",b.dataset.module===selected));
 $("description").textContent=selected==="spots"?"独立可见色斑检测；原跨检测支持评分已撤回，当前尚无匹配的独立区域评分参考。":current.description;
 $("score").textContent=item.score??"—";$("unit").hidden=item.score==null;
 $("grade").replaceWith(Object.assign(badge(item.severity),{id:"grade"}));
 $("gallery").replaceChildren();$("gallery").classList.toggle("single",Object.keys(current.images).length===1);
 for(const view of ["left","front","right"]){if(!current.images[view])continue;const card=el("figure",undefined,"photo");card.append(el("h3",labels[view]));const img=el("img");img.src=current.detections[selected][view];img.alt=modules[selected]+" "+labels[view]+"检测结果图";img.onclick=()=>{$("zoom-image").src=img.src;$("zoom").showModal()};card.append(img);
 card.append(el("figcaption",item.total_count?("本视角检出 "+display(item.total_count[view])+" 个目标"):"本样本正面原图"));$("gallery").append(card)}
 const rows=[];
 if(item.total_count){rows.push({cls:"total",cells:["本照片全部目标",...["left","front","right"].map(v=>display(item.total_count[v])),"各照片独立统计"]});
 for(const r of item.regions){let note=r.primary_view?labels[r.primary_view]+" · "+r.primary_count+" 个":"无主观察";if(r.primary_count===0&&r.supplementary_views.length)note+="（其他视角有发现）";rows.push({cells:[r.name,...["left","front","right"].map(v=>display(r[v])),note]})}
 rows.push({cells:["未单列 / 归属待定",...["left","front","right"].map(v=>display(item.unassigned_count[v])),"各照片独立统计"]});
 $("counts").replaceChildren(table(["区域 / 数量","左侧","正面","右侧","主观察参考"],rows))
 }else $("counts").replaceChildren(el("div","这套V3样本没有配套的三视图数量结果，暂不展示数量表。","empty"));
 const scored=item.regional_scores;
 const scoreRows=scored.items.map(r=>{const value=el("div",r.score==null?"暂不可评分":String(r.score));if(r.score!=null){const bar=el("div",undefined,"bar"),fill=el("i");fill.style.width=r.score+"%";bar.append(fill);value.append(bar)}
 const reason=r.score_status==="available"?"已评分":r.score_reason==="missing_same_image_v3_result"?"缺少同图V3结果":r.score_reason==="missing_region_score"?"该区域暂无评分":"评分依据不足";
 return {cells:[r.name,value,badge(r.severity)]}});
 $("regional").replaceChildren(table(["V3区域","评分","程度"],scoreRows));
 const publicData=Object.fromEntries(Object.entries(data.results).map(([key,m])=>[key,{
   name:m.name,score:m.score,severity:m.severity,
   regions:m.regional_scores.items.map(r=>({region:r.region,name:r.name,score:r.score,severity:r.severity}))}]));
 $("json").textContent=JSON.stringify(publicData,null,2);
 $("status").textContent="当前数据："+current.title+" · "+modules[selected];
}
async function load(){
 const token=++generation,key="triplet";
 $("status").textContent="读取已有结果…";
 try{const response=await fetch("/data/"+key);if(!response.ok)throw Error("结果读取失败");const data=await response.json();if(token!==generation)return;current=data;$("description").textContent=data.description;$("download").href="/download/"+key;render()}
 catch(e){$("status").textContent=e.message}
}
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{selected=b.dataset.module;render()});
$("close").onclick=()=>$("zoom").close();
load();
