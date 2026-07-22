from __future__ import annotations
import argparse,json
from .serialization import compare_exports

def main(argv=None):
    parser=argparse.ArgumentParser(description="Compare immutable TurboTIFF export manifests");parser.add_argument("--old",required=True);parser.add_argument("--new",required=True);parser.add_argument("--old-las");parser.add_argument("--new-las");args=parser.parse_args(argv)
    print(json.dumps(compare_exports(args.old,args.new,args.old_las,args.new_las),indent=2,default=str))

if __name__=="__main__":main()
