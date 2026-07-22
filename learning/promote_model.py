from __future__ import annotations
import argparse, json
from pathlib import Path
from .model_registry import ModelRegistry
def main():
    p=argparse.ArgumentParser(); p.add_argument("--registry",type=Path,default=Path("models")); p.add_argument("--candidate",required=True); p.add_argument("--approved-by",required=True); p.add_argument("--reason",required=True); p.add_argument("--gates",required=True,type=Path); a=p.parse_args()
    print(json.dumps(ModelRegistry(a.registry).promote(a.candidate,a.approved_by,a.reason,json.loads(a.gates.read_text())),indent=2))
if __name__ == "__main__": main()
