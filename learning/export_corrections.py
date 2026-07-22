from __future__ import annotations
import argparse, json
from pathlib import Path
from .datasets import export_approved_corrections

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True, type=Path); parser.add_argument("--output", required=True, type=Path); parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args(); print(json.dumps(export_approved_corrections(args.input, args.output, args.dataset_id), indent=2))
if __name__ == "__main__": main()
