let comparisonGeneration=0;
async function loadComparison(){
 const token=++comparisonGeneration;
 const response=await fetch('/api/comparison?sample='+sample.value+'&module='+module);
 if(!response.ok){document.getElementById('scoreSummary').textContent='评分对照读取失败';return;}
 const data=await response.json();if(token!==comparisonGeneration)return;
 const old=data.before,next=data.after,body=document.getElementById('scoreRows');body.replaceChildren();
 const value=x=>x===null||x===undefined?'不可评分':String(x);
 document.getElementById('scoreSummary').textContent='整体：'+value(old.score)+' 分（'+value(old.severity)+'） → '+value(next.score)+' 分（'+value(next.severity)+'）'+(module==='spots'?'；前后均按独立色斑面积/密度口径，不是历史综合色素分。':'');
 const previous=new Map(old.regions.map(r=>[r.region,r]));
 for(const row of next.regions){const before=previous.get(row.region)||{},tr=document.createElement('tr');for(const text of [row.name,value(before.score),value(before.severity),value(row.score),value(row.severity)]){const td=document.createElement('td');td.textContent=text;tr.append(td);}body.append(tr);}
 document.getElementById('measurements').textContent=JSON.stringify(data.measurements,null,2);
}
sample.addEventListener('change',loadComparison);
document.querySelectorAll('#modules button').forEach(button=>button.addEventListener('click',loadComparison));
loadComparison();
