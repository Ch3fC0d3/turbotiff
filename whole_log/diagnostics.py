from __future__ import annotations
def join_report(result):
    return [{"index":index,"pages":[join["page_a"],join["page_b"]],"relationship":join["relationship"],"depth_delta":join["depth_delta"],"confidence":join["confidence"],"status":join["status"]} for index,join in enumerate(result.joins)]
