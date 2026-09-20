import {strict as assert} from 'node:assert';
import {AngleDisplay} from '../web/angle-display.mjs';
const writes=[];const d=new AngleDisplay(p=>writes.push(p));
for(let t=0;t<=2000;t+=20)d.update({valid:true,yaw:t},t);
assert.equal(writes.length,3);
assert.deepEqual(writes.map(p=>p.yaw),[0,1000,2000]);
d.update(null,2010);assert.equal(writes.at(-1),null);
d.update({valid:true,yaw:45},2020);assert.equal(writes.at(-1).yaw,45);
console.log('PASS: 101 inference results -> 3 display updates at 1000ms; invalid result clears immediately');
