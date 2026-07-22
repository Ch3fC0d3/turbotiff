from __future__ import annotations
def propose_groups(pages):
    groups={}
    for page in pages:
        identity=tuple(str(page.metadata.get(key,"?")).upper() for key in ("well_id","tool_run","logging_date"))
        groups.setdefault(identity,[]).append(page)
    return [{"group_id":f"candidate_log_{i+1:03d}","page_ids":[p.page_id for p in values],"confidence":.95 if identity[0]!="?" else .55,"evidence":["matching well/run metadata"],"conflicts":[]} for i,(identity,values) in enumerate(sorted(groups.items(),key=lambda item:str(item[0])))]
