import re
def check_metadata(metadata,curves,add,config=None):
    required=config.required_metadata if config else ("well_name","field","company","date","depth_unit")
    results=[]
    for key in required:
        present=bool(metadata.get(key));results.append({"field":key,"present":present})
        if not present:add("metadata","high",None,f"Required LAS metadata is missing: {key}",evidence={"field":key})
    names=[curve.mnemonic.strip().upper() for curve in curves]
    for curve in curves:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}",curve.mnemonic or ""):add("metadata","high",curve,"Curve mnemonic has invalid LAS formatting",evidence={"mnemonic":curve.mnemonic})
    if len(names)!=len(set(names)):add("metadata","critical",None,"LAS curve mnemonics are not unique",evidence={"mnemonics":names})
    return results
