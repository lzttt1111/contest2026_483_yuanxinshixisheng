"""Fail-closed compatibility contracts for the frozen historical reference dataset."""

def rejection(module,metric,region,trace,profile):
 if not trace.get("reference"):return None
 if trace.get("method")=="historical_named_roi_image_estimate":
  aliases={"left_jaw":"画面左下颌","right_jaw":"画面右下颌"}
  roi=trace.get("historical_roi")
  if roi not in (region,aliases.get(region)):return "invalid_named_roi_proxy"
  project,field=trace.get("historical_project"),trace.get("historical_field")
  suffix="|"+str(project)+"|"+str(roi)+"|"+str(field)
  if not trace["reference"].endswith(suffix):return "invalid_named_roi_proxy_reference"
  if profile=="institution" and module in ("02","03","04") and not trace["reference"].startswith("legacy|"):return "wrong_proxy_population"
  return None
 if trace.get("method")=="historical_v2_report_proxy":
  mapping={"main_count":"count","length_burden":"length"}
  expected=mapping.get(metric)
  if module!="08" or trace.get("proxy_source_metric")!=expected or not expected:return "invalid_historical_proxy_binding"
  clean={**trace,"method":"same_metric_reference"}
  return rejection(module,expected,region,clean,profile)
 if trace.get("method")=="exact_additive_roi_reference":
  from .additive_scoring import recipe_for
  from .merged_reference import REGIONS
  recipe=recipe_for(module,metric,region)
  if not recipe:return "unknown_additive_roi_recipe"
  project,merged,field=recipe
  expected="additive|development1000|"+project+"|"+merged+"|"+field
  if trace["reference"]!=expected or trace.get("region_components")!=list(REGIONS[merged]):return "invalid_additive_reference_contract"
  if profile!="consumer" and module!="01":return "unvalidated_additive_spectral_transfer"
  return None
 reference_region=trace.get("reference_region",region)
 # Historical names and new names may coincide while their masks do not.
 if module=="01" and region in ("forehead","chin","left_inner_cheek","right_inner_cheek","left_outer_cheek","right_outer_cheek"):
  return "new_pore_roi_has_no_validated_same_domain_reference"
 if module=="02" and region in ("forehead","chin"):
  return "merged_oil_roi_is_not_old_forehead_or_chin"
 if module=="03" and region!="full_face":
  return "v3_pigment_region_contract_not_validated_against_old_rois"
 if module=="03" and metric=="brown.p90":
  return "continuous_domain_p90_is_not_full_valid_domain_p90"
 if module=="03" and metric=="uv.p90":
  return "continuous_pixel_p90_is_not_instance_intensity_p90"
 if module=="10" and region in ("forehead","left_jaw","right_jaw"):
  return "merged_texture_roi_is_not_old_forehead_or_jaw"
 if module=="08" and metric in ("main_count","length_burden"):
  return "v3_mainline_or_extension_is_not_legacy_segment_or_length"
 if profile=="institution" and module in ("02","03","04","05"):
  return "spectral_measurement_has_no_validated_rgb_population_transfer"
 if reference_region!=region:
  # These are exact language aliases in the saved V2 summary, not parent ROIs.
  actual_alias={"left_under_eye":"左眼下细纹","right_under_eye":"右眼下细纹",
                "left_jaw":"画面左下颌","right_jaw":"画面右下颌"}
  if actual_alias.get(region)!=reference_region:
   return "unvalidated_reference_region_transfer"
 return None
