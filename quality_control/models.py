from __future__ import annotations
from dataclasses import dataclass,field

@dataclass
class QualityFinding:
    finding_id:str; category:str; severity:str; curve_id:str|None; depth_start:float|None; depth_end:float|None; message:str; evidence:dict=field(default_factory=dict); recommended_action:str|None=None; blocks_approval:bool=False; review_status:str="open"; review:dict=field(default_factory=dict)

@dataclass
class QualityControlResult:
    log_id:str; qc_run_id:str; whole_log_hash:str; status:str; overall_score:float|None; category_scores:dict; curve_results:list; depth_results:list; join_results:list; metadata_results:list; critical_findings:list; warnings:list; informational_findings:list; export_blockers:list; approval_requirements:list; processing_metadata:dict; findings:list[QualityFinding]; schema_version:int=1; approval:object|None=None

@dataclass(frozen=True)
class QualityControlConfig:
    monotonic_tolerance:float=1e-6; duplicate_tolerance:float=1e-6; expected_step_tolerance:float=.05; small_gap_max_samples:int=3; medium_gap_max_depth:float=10.; large_gap_blocks_approval:bool=True; null_value:float=-999.25; low_confidence_threshold:float=.60; critical_confidence_threshold:float=.30; flat_minimum_depth:float=5.; flat_value_tolerance:float=.001; hampel_window:int=11; spike_sigma:float=5.; maximum_normalized_join_jump:float=4.; minimum_wrap_confidence:float=.70; depth_decimals:int=2; value_decimals:int=5; las_version:str="2.0"; las_ascii_wrapped:bool=False; allow_irregular_sampling:bool=False; allow_accepted_blockers:bool=True; required_metadata:tuple[str,...]=("well_name","field","company","date","depth_unit"); curve_rules:dict=field(default_factory=lambda:{"GR":{"units":["API"],"hard_min":-20,"hard_max":500},"RHOB":{"units":["G/C3"],"hard_min":.5,"hard_max":5},"CALI":{"units":["IN"],"hard_min":0,"hard_max":50}})

@dataclass(frozen=True)
class EvidenceCrop:
    curve_id:str; depth_start:float; depth_end:float; source_references:tuple[dict,...]; overlays:dict=field(default_factory=dict); image_path:str|None=None; status:str="references_only"
