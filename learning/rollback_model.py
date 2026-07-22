from __future__ import annotations
import argparse, json
from pathlib import Path
from .model_registry import ModelRegistry
def main():
    p=argparse.ArgumentParser(); p.add_argument("--registry",type=Path,default=Path("models")); p.add_argument("--model",required=True); p.add_argument("--approved-by",required=True); p.add_argument("--reason",required=True); a=p.parse_args()
    print(json.dumps(ModelRegistry(a.registry).rollback(a.model,a.approved_by,a.reason),indent=2))
if __name__ == "__main__": main()
