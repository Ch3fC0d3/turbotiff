from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np

@dataclass
class BoundingBox:
    x1: float; y1: float; x2: float; y2: float
    def width(self): return self.x2-self.x1
    def height(self): return self.y2-self.y1
    def to_dict(self): return asdict(self)

@dataclass
class CoordinateTransform:
    original_to_corrected: np.ndarray=field(default_factory=lambda:np.eye(3))
    corrected_to_original: np.ndarray=field(default_factory=lambda:np.eye(3))
    operations: list[dict]=field(default_factory=list)
    def map_points(self, points, inverse=False):
        matrix=self.corrected_to_original if inverse else self.original_to_corrected
        points=np.asarray(points,dtype=float).reshape(-1,2); homogeneous=np.c_[points,np.ones(len(points))]@matrix.T
        return homogeneous[:,:2]/homogeneous[:,2:3]
    @classmethod
    def from_matrix(cls,matrix,operations=None):
        matrix=np.asarray(matrix,dtype=float); return cls(matrix,np.linalg.inv(matrix),operations or [])
    def to_dict(self): return {"original_to_corrected":self.original_to_corrected.tolist(),"corrected_to_original":self.corrected_to_original.tolist(),"operations":self.operations}

@dataclass
class DepthMapping:
    control_points: list[tuple[float,float]]; interpolation: str="piecewise_linear"; unit: str="FT"; residual_error: float=0.; confidence: float=0.
    def depth_at(self,y):
        points=np.asarray(self.control_points,float); return np.interp(y,points[:,0],points[:,1])

@dataclass
class ScaleDefinition:
    scale_type: str="unknown"; minimum: float|None=None; maximum: float|None=None; cycles: int|None=None; direction: str="increasing_right"; units: str|None=None; control_points: list=field(default_factory=list); confidence: dict=field(default_factory=dict)

@dataclass
class DetectedTrack:
    track_id: str; bounds: BoundingBox; track_type: str|None=None; scale: ScaleDefinition=field(default_factory=ScaleDefinition); curve_candidates: list=field(default_factory=list); confidence: dict=field(default_factory=dict); warnings: list=field(default_factory=list); grid: dict=field(default_factory=dict)

@dataclass
class PageAnalysisResult:
    page_id: str; source_image_hash: str; original_dimensions: tuple[int,int]; corrected_dimensions: tuple[int,int]; transforms: CoordinateTransform; page_classification: str="unknown"; log_body_regions: list=field(default_factory=list); depth_columns: list=field(default_factory=list); tracks: list[DetectedTrack]=field(default_factory=list); header_elements: list=field(default_factory=list); warnings: list=field(default_factory=list); confidence_summary: dict=field(default_factory=dict); model_versions: dict=field(default_factory=dict); processing_metadata: dict=field(default_factory=dict); review_status: str="automatic_draft"; schema_version: int=1
    def to_dict(self):
        def convert(value):
            if hasattr(value,"__dataclass_fields__"): return {k:convert(getattr(value,k)) for k in value.__dataclass_fields__}
            if isinstance(value,np.ndarray): return value.tolist()
            if isinstance(value,(list,tuple)): return [convert(v) for v in value]
            if isinstance(value,dict): return {k:convert(v) for k,v in value.items()}
            return value
        return convert(self)

@dataclass(frozen=True)
class PageAnalysisConfig:
    max_layout_dimension: int=1800; minimum_separator_fraction: float=.55; minimum_track_width: int=40; header_fraction: float=.16; apply_perspective: bool=False; safe_confidence: float=.8
