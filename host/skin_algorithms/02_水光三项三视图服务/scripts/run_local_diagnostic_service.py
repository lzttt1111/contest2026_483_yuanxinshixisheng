"""Local-only persistent diagnostic demo. Production uses the documented Redis setup."""
from pathlib import Path
import argparse,os,sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='本地文件系统队列诊断服务，非生产Redis部署')
parser.add_argument('component',choices=['worker','api'])
args=parser.parse_args()
runtime=ROOT/'runtime/local_diagnostic_service';runtime.mkdir(parents=True,exist_ok=True)
os.chdir(ROOT)
os.environ.update(SHUIGUANG_INPUT_ROOT=str(ROOT/'examples'),SHUIGUANG_RUNTIME_ROOT=str(runtime),SHUIGUANG_CLOUD_DEVICE='cuda:0',SHUIGUANG_PREPROCESS_MODE='bisenet_rgb_diagnostic_v1',SHUIGUANG_LEGACY_SCORE_FALLBACK='1',SHUIGUANG_FACE_PARSER_MODEL=str(ROOT/'vendor/v3/models/face_parsing_resnet18.onnx'),SHUIGUANG_BROKER_URL='filesystem://',SHUIGUANG_RESULT_BACKEND='file://'+str(runtime/'backend'),PYTHONDONTWRITEBYTECODE='1')
if args.component=='worker':
 command=['-m','celery','-A','cloud_three.tasks:app','worker','--pool=solo','--concurrency=1','-Q','shuiguang_diagnostic,shuiguang_scores','--loglevel=INFO','--hostname=local-diagnostic@%h','--logfile='+str(runtime/'worker.log')]
else:
 command=['-m','uvicorn','cloud_three.api:app','--host','127.0.0.1','--port','8897']
os.execv(sys.executable,[sys.executable,*command])
