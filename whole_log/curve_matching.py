from __future__ import annotations
ALIASES={"DEEPRES":"ILD","RILD":"ILD","LLD":"ILD"}
def normalized_name(name):return ALIASES.get(str(name).upper().replace(" ",""),str(name).upper().replace(" ",""))
def match_curves(curves):
    identities=[]
    for page_id,curve in curves:
        name=normalized_name(curve.mnemonic); match=next((item for item in identities if item["mnemonic"]==name and item["unit"]==curve.unit),None)
        conflict=next((item for item in identities if item["mnemonic"]==name and item["unit"]!=curve.unit),None)
        if match:match["members"].append((page_id,curve));match["source_names"].append(curve.mnemonic)
        else:identities.append({"curve_id":name if not conflict else f"{name}_{len(identities)+1}","mnemonic":name,"unit":curve.unit,"members":[(page_id,curve)],"source_names":[curve.mnemonic],"conflicts":["unit mismatch"] if conflict else []})
    return identities
