from __future__ import annotations
import json
from dataclasses import fields,is_dataclass
from pathlib import Path
import numpy as np
def _convert(value):
    if is_dataclass(value):return {field.name:_convert(getattr(value,field.name)) for field in fields(value)}
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,list):return [_convert(item) for item in value]
    if isinstance(value,dict):return {key:_convert(item) for key,item in value.items()}
    return value
def save_result(path,result):path=Path(path);path.write_text(json.dumps(_convert(result),indent=2,allow_nan=False),encoding="utf-8");return path
