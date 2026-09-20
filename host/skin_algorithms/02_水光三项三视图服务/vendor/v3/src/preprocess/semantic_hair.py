"""Official pretrained categories; no custom dark-line/shape hair classifier.

Only face-ROI intersection and separately specified feature exclusions remain.
This model does not guarantee pixel-perfect sparse-hair or beard segmentation.
"""
import cv2
import numpy as np
import mediapipe as mp

VERSION='semantic_model_only_v2'

def build(image,domain,features,segmenter,landmarks):
 result=segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
 if result.category_mask is None or len(result.confidence_masks)!=6:
  raise ValueError('semantic hair model must provide six validated categories')
 categories=result.category_mask.numpy_view().copy()
 hair_probability=result.confidence_masks[1].numpy_view().copy()
 if categories.shape!=domain.shape or not np.isin(categories,np.arange(6)).all():raise ValueError('semantic segmentation coordinate/category mismatch')
 # Labels verified from the bundled model labels.txt, not inferred from colour.
 semantic_skin=np.isin(categories,(2,3))
 scalp=categories==1
 allowed=(domain>0)&(features==0)
 strands=np.zeros_like(domain)
 expanded=scalp.astype(np.uint8)*255
 # Semantic background/clothes cannot be rescued by the forehead geometry.
 candidate=allowed&semantic_skin
 skin=(candidate&(expanded==0)).astype(np.uint8)*255
 empty=np.zeros_like(domain)
 debug={'semantic_skin_mask':semantic_skin.astype(np.uint8)*255,
  'semantic_categories':categories,'semantic_hair_probability':hair_probability,
  'semantic_candidate_before_hair':candidate.astype(np.uint8)*255,
  'bulk_hair_mask':scalp.astype(np.uint8)*255,'color_hair_mask':empty,
  'strand_hair_mask':strands,'facial_hair_mask':empty.copy(),
  'hair_mask':expanded,'hair_line_response':np.zeros(domain.shape,np.float32),
  'hair_scale_vote':(strands>0).astype(np.uint8),'display_face_mask':skin.copy()}
 return skin,debug
