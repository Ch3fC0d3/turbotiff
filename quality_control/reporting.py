from __future__ import annotations
import csv,html,json
from pathlib import Path
from dataclasses import asdict

def write_reports(output_dir,qc):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True);payload=asdict(qc)
    json_path=output/"qc_report.json";json_path.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")
    csv_path=output/"findings.csv";fields=("finding_id","category","severity","curve_id","depth_start","depth_end","message","blocks_approval","review_status")
    with csv_path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows({key:getattr(finding,key) for key in fields} for finding in qc.findings)
    summary=["# Quality Control Report","",f"Log: `{qc.log_id}`",f"QC run: `{qc.qc_run_id}`",f"Status: **{qc.status}**",f"Overall score: **{qc.overall_score:.1f}/100**" if qc.overall_score is not None else "Overall score: unavailable",f"Export blockers: **{len(qc.export_blockers)}**","","## Category scores",""]
    summary.extend(f"- {name}: {score:.1f}" for name,score in qc.category_scores.items());summary.extend(["","## Findings",""])
    summary.extend(f"- [{finding.severity.upper()}] {finding.message} ({finding.depth_start}–{finding.depth_end})" for finding in qc.findings)
    markdown_path=output/"summary.md";markdown_path.write_text("\n".join(summary)+"\n",encoding="utf-8")
    finding_rows="".join(f"<tr><td>{html.escape(f.severity)}</td><td>{html.escape(f.category)}</td><td>{html.escape(f.message)}</td><td>{f.depth_start}</td><td>{f.depth_end}</td><td>{html.escape(f.review_status)}</td></tr>" for f in qc.findings)
    html_text=f"<!doctype html><html><head><meta charset='utf-8'><title>TurboTIFF QC</title><style>body{{font:14px sans-serif;max-width:1100px;margin:2rem auto}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:.4rem;text-align:left}}</style></head><body><h1>Quality Control Report</h1><p>Log: {html.escape(qc.log_id)} · Status: <strong>{html.escape(qc.status)}</strong> · Score: {qc.overall_score:.1f}</p><p>Export blockers: {len(qc.export_blockers)}</p><table><thead><tr><th>Severity</th><th>Category</th><th>Finding</th><th>From</th><th>To</th><th>Review</th></tr></thead><tbody>{finding_rows}</tbody></table></body></html>"
    html_path=output/"review_report.html";html_path.write_text(html_text,encoding="utf-8")
    return [json_path,csv_path,markdown_path,html_path]

def write_provenance_companion(path,whole_log):
    payload={curve.curve_id:[{"output_depth":item.output_depth,"sources":item.sources,"resampled":item.resampled,"blended":item.blended,"interpolated":item.interpolated,"manually_corrected":item.manually_corrected,"quality_flags":curve.quality_flags[index]} for index,item in enumerate(curve.provenance)] for curve in whole_log.curves}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8");return path
