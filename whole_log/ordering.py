from __future__ import annotations
from itertools import permutations
from .depth_alignment import normalize_depth
def order_pages(pages,canonical="FT"):
    normalized={p.page_id:(normalize_depth(p.depth_top,p.depth_unit,canonical)["canonical_value"],normalize_depth(p.depth_bottom,p.depth_unit,canonical)["canonical_value"]) for p in pages}
    def cost(order):
        total=0.
        for a,b in zip(order,order[1:]): total+=abs(normalized[b.page_id][0]-normalized[a.page_id][1])-.01*(b.page_number==((a.page_number or 0)+1))
        return total
    candidates=sorted(permutations(pages),key=lambda order:(cost(order),tuple(p.page_id for p in order))) if len(pages)<=8 else [tuple(sorted(pages,key=lambda p:normalized[p.page_id][0]))]
    return list(candidates[0]),[{"page_order":[p.page_id for p in order],"score":float(-cost(order))} for order in candidates[:3]]
