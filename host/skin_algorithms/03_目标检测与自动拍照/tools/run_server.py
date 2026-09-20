"""Relocatable launcher; optional test port without editing captured application source."""
import os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'pc_demo'))
from server import app
import uvicorn
port=int(os.environ.get('LAB_PORT','8876'))
if not 1<=port<=65535:raise SystemExit('LAB_PORT must be 1..65535')
uvicorn.run(app,host='127.0.0.1',port=port)
