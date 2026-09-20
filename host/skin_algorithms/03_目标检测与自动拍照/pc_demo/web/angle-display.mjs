export class AngleDisplay {
  constructor(write, interval=1000){this.write=write;this.interval=interval;this.last=-Infinity;}
  update(pose, now){
    if(!pose?.valid){this.clear();return;}
    if(now-this.last<this.interval)return;
    this.last=now;this.write(pose);
  }
  clear(){this.last=-Infinity;this.write(null);}
}
