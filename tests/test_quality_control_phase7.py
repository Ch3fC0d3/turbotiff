import json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
from whole_log import assemble_whole_log,PageCurve,PageTraceResult,WholeLogConfig
from whole_log.review import apply_review
from quality_control import evaluate_whole_log_quality,QualityControlConfig,approve,whole_log_hash,review_finding,create_evidence_crop
from quality_control.approval import REQUIRED,approval_valid
from quality_control.las_validation import serialize_las,validate_las_text
from quality_control.reporting import write_reports,write_provenance_companion
from quality_control.serialization import write_las,create_export_manifest,validate_manifest,compare_exports
from quality_control.units import conversion_proposal,apply_reviewed_conversion

METADATA={"well_name":"Well 1","field":"Field","company":"Company","date":"2026-07-20","depth_unit":"FT"}

def page(page_id,top,bottom,values,wrap=None,confidence=.9):
    depth=np.linspace(top,bottom,len(values));curve=PageCurve(f"{page_id}_GR","GR","API",depth,np.asarray(values,float),np.full(len(values),confidence),image_rows=np.arange(len(values)),wrap_index=wrap,model_version="m1",decoder_version="d1")
    return PageTraceResult(page_id,"source",[curve],top,bottom,"FT",metadata={"well_id":"W1"})

def clean_log():
    first=page("p1",0,10,np.linspace(50,60,11));second=page("p2",10,20,np.linspace(60,70,11));result=assemble_whole_log([first,second],WholeLogConfig(depth_step=1));return apply_review(result,join_edits={0:{"status":"approved"}},reviewer="reviewer")

def checklist():return {key:True for key in REQUIRED}

class Phase7QualityControlTests(unittest.TestCase):
    def test_clean_qc_is_repeatable_and_las_round_trips(self):
        log=clean_log();first=evaluate_whole_log_quality(log,metadata=METADATA);second=evaluate_whole_log_quality(log,metadata=METADATA)
        self.assertEqual(first.status,"reviewed");self.assertFalse(first.export_blockers);self.assertEqual(first.qc_run_id,second.qc_run_id);self.assertEqual([f.finding_id for f in first.findings],[f.finding_id for f in second.findings])
        config=QualityControlConfig();text=serialize_las(log,METADATA,config);parsed=validate_las_text(text,1,config,log.curves[0].depth);self.assertTrue(parsed["passed"],parsed)
        self.assertIn("DRAFT UNAPPROVED",text)

    def test_depth_duplicates_reversal_gaps_and_precision_block(self):
        log=clean_log();curve=log.curves[0];curve.depth[4]=curve.depth[3];curve.depth[8]=curve.depth[7]-.5
        qc=evaluate_whole_log_quality(log,metadata=METADATA);messages={item.message for item in qc.findings};self.assertIn("Duplicate depth samples remain",messages);self.assertIn("Depth is non-monotonic",messages);self.assertEqual(qc.status,"blocked")
        precision=deepcopy(clean_log());precision.curves[0].depth[1]=.004
        qc=evaluate_whole_log_quality(precision,metadata=METADATA);self.assertTrue(any(item.category=="export_precision" for item in qc.findings))
        gap=deepcopy(clean_log());gap.curves[0].depth[10:]+=20
        qc=evaluate_whole_log_quality(gap,metadata=METADATA);self.assertTrue(any(item.category=="depth_gap" for item in qc.findings))

    def test_array_null_infinity_and_provenance_validation(self):
        mismatch=deepcopy(clean_log());mismatch.curves[0].values=mismatch.curves[0].values[:-1];qc=evaluate_whole_log_quality(mismatch,metadata=METADATA);self.assertTrue(any("different lengths" in f.message for f in qc.findings))
        nulls=clean_log();nulls.curves[0].values[3]=np.nan;text=serialize_las(nulls,METADATA,QualityControlConfig());self.assertIn("-999.25000",text);self.assertNotIn("nan",text.lower())
        invalid=clean_log();invalid.curves[0].values[3]=np.inf;qc=evaluate_whole_log_quality(invalid,metadata=METADATA);self.assertTrue(any("infinity" in f.message for f in qc.findings));self.assertTrue(any(f.category=="las_format" for f in qc.findings))
        missing=clean_log();missing.curves[0].provenance[2].sources=[];qc=evaluate_whole_log_quality(missing,metadata=METADATA);self.assertTrue(any(f.category=="provenance" for f in qc.findings))

    def test_spike_flat_grid_and_wrap_detection(self):
        spike=clean_log();spike.curves[0].values[10]=400;qc=evaluate_whole_log_quality(spike,metadata=METADATA);self.assertTrue(any(f.category in {"spike","range"} for f in qc.findings))
        flat=clean_log();flat.curves[0].values[3:12]=55;qc=evaluate_whole_log_quality(flat,metadata=METADATA);self.assertTrue(any(f.category=="flat_line" for f in qc.findings))
        grid=clean_log();grid.curves[0].quality_flags[5].append("GRID_LOCK_SUSPECTED");qc=evaluate_whole_log_quality(grid,metadata=METADATA);self.assertTrue(any(f.category=="grid_lock" for f in qc.findings))
        wrap=clean_log();wrap.curves[0].whole_log_wrap_index[10:]=2;qc=evaluate_whole_log_quality(wrap,metadata=METADATA);self.assertTrue(any(f.category=="wrap" and f.severity=="critical" for f in qc.findings))

    def test_join_identity_unit_and_las_column_checks(self):
        unresolved=assemble_whole_log([page("p1",0,10,np.linspace(1,2,11)),page("p2",10,20,np.linspace(2,3,11))],WholeLogConfig(depth_step=1));qc=evaluate_whole_log_quality(unresolved,metadata=METADATA);self.assertTrue(any(f.category=="join" and f.blocks_approval for f in qc.findings))
        duplicate=clean_log();other=deepcopy(duplicate.curves[0]);other.curve_id="other";duplicate.curves.append(other);qc=evaluate_whole_log_quality(duplicate,metadata=METADATA);self.assertTrue(any("not unique" in f.message for f in qc.findings))
        wrong=clean_log();wrong.curves[0].unit="PERC";qc=evaluate_whole_log_quality(wrong,metadata=METADATA);self.assertTrue(any(f.category=="unit" for f in qc.findings))
        config=QualityControlConfig();text=serialize_las(clean_log(),METADATA,config);lines=text.splitlines();row=lines.index("~ASCII")+1;lines[row]+=" 123";validation=validate_las_text("\n".join(lines)+"\n",1,config);self.assertFalse(validation["passed"]);self.assertIn("wrong column count",validation["findings"])

    def test_review_approval_invalidation_and_role_gate(self):
        log=clean_log();log.curves[0].values[10]=400;qc=evaluate_whole_log_quality(log,metadata=METADATA)
        for finding in list(qc.findings):qc=review_finding(qc,finding.finding_id,"accepted_as_real","expert","confirmed against image")
        self.assertEqual(qc.status,"reviewed")
        with self.assertRaises(PermissionError):approve(qc,qc.whole_log_hash,"operator",{"operator"},checklist())
        record=approve(qc,qc.whole_log_hash,"approver",{"approver"},checklist());self.assertTrue(approval_valid(record,qc.whole_log_hash))
        log.curves[0].values[0]+=1;self.assertFalse(approval_valid(record,whole_log_hash(log)))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(PermissionError):write_las(Path(temporary)/"changed.las",log,qc,METADATA,QualityControlConfig(),approval=record)

    def test_export_manifest_reports_companions_revision_and_evidence(self):
        log=clean_log();qc=evaluate_whole_log_quality(log,metadata=METADATA);record=approve(qc,qc.whole_log_hash,"approver",{"approver"},checklist())
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);las_path,validation=write_las(root/"well.las",log,qc,METADATA,QualityControlConfig(),approval=record);self.assertTrue(validation["passed"]);self.assertIn("APPROVED",las_path.read_text())
            reports=write_reports(root,qc);provenance=write_provenance_companion(root/"provenance.json",log);self.assertEqual(len(reports),4);self.assertTrue(provenance.exists())
            manifest,payload=create_export_manifest(root,las_path,qc,record,"approver",versions={"assembly_version":"phase6"},companion_files=[*reports,provenance]);self.assertTrue(validate_manifest(manifest,las_path));self.assertEqual(payload["assembly_version"],"phase6")
            comparison=compare_exports(payload,{**payload,"approval_id":"new"});self.assertTrue(comparison["approval_changed"])
            with self.assertRaises(PermissionError):write_las(root/"not-approved.las",log,qc,METADATA,QualityControlConfig())
        evidence=create_evidence_crop(log,log.curves[0].curve_id,4,6);self.assertTrue(evidence.source_references)

    def test_unit_conversion_is_explicit(self):
        proposal=conversion_proposal("PERC","V/V",0,10);self.assertTrue(proposal["requires_review"]);self.assertTrue(proposal["original_values_preserved"])
        with self.assertRaises(PermissionError):apply_reviewed_conversion(np.array([10.]),proposal)
        np.testing.assert_allclose(apply_reviewed_conversion(np.array([10.]),proposal,reviewed=True),[.1])

if __name__=="__main__":unittest.main()
