import sys,platform,json
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import onnxruntime as ort
import cv2
import mediapipe as mp
result={'python':sys.version,'platform':platform.platform(),'providers_available':ort.get_available_providers(),
        'opencv_version':cv2.__version__,'yunet_api':hasattr(cv2,'FaceDetectorYN'),'mediapipe_tasks':hasattr(mp,'tasks')}
for name in ['onnxruntime','onnxruntime-gpu','opencv-python','opencv-python-headless','opencv-contrib-python','mediapipe']:
    try:result[name]=version(name)
    except PackageNotFoundError:pass
print(json.dumps(result,indent=2))
if 'onnxruntime' in result and 'onnxruntime-gpu' in result:
    raise SystemExit('ERROR: CPU/GPU ORT distributions must not coexist in this environment')
if not result['yunet_api'] or not result['mediapipe_tasks']:raise SystemExit('Required model APIs missing')
print('Environment imports OK. Available providers are not proof of successful CUDA model loading; inspect /health after startup.')
