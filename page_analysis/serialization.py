import json
from pathlib import Path
def save_analysis(path,result):
    path=Path(path); path.write_text(json.dumps(result.to_dict(),indent=2,allow_nan=False),encoding="utf-8"); return path
