from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np

@dataclass
class PageCurve:
    curve_id:str; mnemonic:str; unit:str|None; depth:np.ndarray; values:np.ndarray; confidence:np.ndarray; image_rows:np.ndarray|None=None; wrap_index:np.ndarray|None=None; track_id:str|None=None; scale:dict=field(default_factory=dict); style:dict=field(default_factory=dict); model_version:str|None=None; decoder_version:str|None=None
    def __post_init__(self):
        for name in ("depth","values","confidence"): setattr(self,name,np.asarray(getattr(self,name)))
        if self.image_rows is None:self.image_rows=np.arange(self.depth.size,dtype=float)
        if self.wrap_index is None:self.wrap_index=np.zeros(self.depth.size,dtype=np.int32)

@dataclass
class PageTraceResult:
    page_id:str; source_file_id:str; curves:list[PageCurve]; depth_top:float|None; depth_bottom:float|None; depth_unit:str|None; page_number:int|None=None; reviewed_page_analysis:object|None=None; page_confidence:dict=field(default_factory=dict); metadata:dict=field(default_factory=dict); source_hash:str|None=None

@dataclass
class SampleProvenance:
    output_depth:float; sources:list[dict]; resampled:bool=False; blended:bool=False; interpolated:bool=False; manually_corrected:bool=False

@dataclass
class WholeLogCurve:
    curve_id:str; mnemonic:str; unit:str|None; description:str|None; depth:np.ndarray; values:np.ndarray; confidence:np.ndarray; quality_flags:list[list[str]]; provenance:list[SampleProvenance]; source_pages:list[str]; whole_log_wrap_index:np.ndarray|None=None; warnings:list=field(default_factory=list)

@dataclass
class WholeLogResult:
    log_id:str; ordered_pages:list[str]; curves:list[WholeLogCurve]; joins:list; gaps:list; duplicate_intervals:list; warnings:list; confidence_summary:dict; provenance_index:dict; status:str="draft"; schema_version:int=1; initial_automatic_page_order:list=field(default_factory=list); initial_automatic_joins:list=field(default_factory=list); review:dict=field(default_factory=dict)

@dataclass(frozen=True)
class WholeLogConfig:
    canonical_depth_unit:str="FT"; exact_tolerance:float=.5; small_overlap_max:float=50.; small_gap_max:float=10.; large_gap_warning:float=50.; value_agreement_tolerance:float=.05; maximum_small_gap:float=2.; allow_small_gap_interpolation:bool=False; depth_step:float|None=None; maximum_wrap_offset:int=3
