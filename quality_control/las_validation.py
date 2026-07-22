from __future__ import annotations
from io import StringIO
import numpy as np

def serialize_las(whole_log,metadata,config,approved=False):
    curves=whole_log.curves
    if not curves:raise ValueError("Whole log has no curves")
    if config.las_version!="2.0":raise ValueError("Only LAS 2.0 is currently supported")
    if config.las_ascii_wrapped:raise ValueError("LAS ASCII wrapped mode is not currently supported")
    depth=np.asarray(curves[0].depth,float)
    if len(depth)<2:raise ValueError("At least two depth samples are required for LAS export")
    if any(not np.array_equal(curve.depth,depth) for curve in curves):raise ValueError("All LAS curves must share one depth array")
    null=config.null_value
    for curve in curves:
        values=np.asarray(curve.values,float)
        if np.any(values[np.isfinite(values)]==null):raise ValueError(f"Valid {curve.mnemonic} measurement equals LAS null value")
        if np.any(np.isinf(values)):raise ValueError(f"Curve {curve.mnemonic} contains infinity")
    unit=metadata.get("depth_unit","FT")
    lines=["~Version Information",f" VERS. {config.las_version}: CWLS LOG ASCII STANDARD"," WRAP. NO:","~Well Information",f" STRT.{unit} {depth[0]:.{config.depth_decimals}f}:",f" STOP.{unit} {depth[-1]:.{config.depth_decimals}f}:",f" STEP.{unit} {np.median(np.diff(depth)):.{config.depth_decimals}f}:",f" NULL. {null}:",f" WELL. {metadata.get('well_name','')}:",f" FLD . {metadata.get('field','')}:",f" COMP. {metadata.get('company','')}:",f" DATE. {metadata.get('date','')}:",f"# TURBOTIFF STATUS: {'APPROVED' if approved else 'DRAFT UNAPPROVED'}","~Curve Information",f" DEPT.{unit} : Depth"]
    lines.extend(f" {curve.mnemonic}.{curve.unit or ''} : {curve.description or curve.mnemonic}" for curve in curves);lines.append("~ASCII")
    for row,depth_value in enumerate(depth):
        values=[depth_value]+[curve.values[row] for curve in curves]
        lines.append(" ".join(f"{(null if not np.isfinite(value) else value):.{config.depth_decimals if index==0 else config.value_decimals}f}" for index,value in enumerate(values)))
    return "\n".join(lines)+"\n"

def validate_las_text(text,expected_curves,config,expected_depth=None):
    findings=[];section=None;sections=set();rows=[];null=None;curve_count=0;well={}
    for line in text.splitlines():
        stripped=line.strip()
        if not stripped or stripped.startswith("#"):continue
        if stripped.startswith("~"):section=stripped.upper();sections.add(section.split()[0]);continue
        if section and section.startswith("~WELL") and stripped.upper().startswith("NULL."):
            try:null=float(stripped.split(":")[0].split()[-1]);well["NULL"]=null
            except ValueError:findings.append("invalid null declaration")
        elif section and section.startswith("~WELL") and any(stripped.upper().startswith(key+".") for key in ("STRT","STOP","STEP")):
            try:well[stripped.split(".")[0].upper()]=float(stripped.split(":")[0].split()[-1])
            except ValueError:findings.append("invalid well numeric field")
        elif section and section.startswith("~CURVE"):curve_count+=1
        elif section and section.startswith("~ASCII"):
            parts=stripped.split()
            if len(parts)!=expected_curves+1:findings.append("wrong column count")
            try:values=[float(item) for item in parts]
            except ValueError:findings.append("invalid numeric value");continue
            if any(not np.isfinite(value) for value in values):findings.append("non-finite ASCII value")
            rows.append(values)
    if null!=config.null_value:findings.append("null mismatch")
    if curve_count!=expected_curves+1:findings.append("curve section count mismatch")
    if not {"~VERSION","~WELL","~CURVE","~ASCII"}.issubset(sections):findings.append("missing required section")
    if not text.endswith("\n"):findings.append("missing final newline")
    if rows:
        tolerance=10**(-config.depth_decimals)
        if abs(rows[0][0]-well.get("STRT",rows[0][0]))>tolerance:findings.append("start depth mismatch")
        if abs(rows[-1][0]-well.get("STOP",rows[-1][0]))>tolerance:findings.append("stop depth mismatch")
        if len(rows)>1 and abs(float(np.median(np.diff([row[0] for row in rows])))-well.get("STEP",0))>tolerance:findings.append("step mismatch")
    if expected_depth is not None and rows:
        parsed=np.asarray([row[0] for row in rows]);expected=np.round(np.asarray(expected_depth,float),config.depth_decimals)
        if len(parsed)!=len(expected) or not np.allclose(parsed,expected,atol=10**(-config.depth_decimals)):findings.append("round-trip depth mismatch")
    independent={"ran":False,"parser":None,"error":None}
    try:
        import lasio;lasio.read(StringIO(text));independent={"ran":True,"parser":"lasio","error":None}
    except ImportError:pass
    except Exception as exc:independent={"ran":True,"parser":"lasio","error":str(exc)};findings.append("independent parser failure")
    return {"passed":not findings,"findings":sorted(set(findings)),"row_count":len(rows),"independent_validation":independent,"well":well}
