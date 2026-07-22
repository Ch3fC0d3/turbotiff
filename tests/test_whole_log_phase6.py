import json,tempfile,unittest
from pathlib import Path
import numpy as np
from whole_log import assemble_whole_log,PageCurve,PageTraceResult,WholeLogConfig
from whole_log.depth_alignment import fit_alignment,relationship
from whole_log.grouping import propose_groups
from whole_log.review import apply_review,export_status
from whole_log.serialization import save_result

def page(page_id,top,bottom,values,mnemonic="GR",unit="API",depth_unit="FT",page_number=None,source_hash=None,curve_id=None,wrap=None,scale=None,confidence=.9):
    depth=np.linspace(top,bottom,len(values)); curve=PageCurve(curve_id or f"{page_id}_{mnemonic}",mnemonic,unit,depth,np.asarray(values,float),np.full(len(values),confidence),image_rows=np.arange(len(values))*2,wrap_index=wrap,scale=scale or {})
    return PageTraceResult(page_id,"source",[curve],top,bottom,depth_unit,page_number,page_confidence={"overall":confidence},metadata={"well_id":"W1","tool_run":"R1","logging_date":"D1"},source_hash=source_hash)

class Phase6Tests(unittest.TestCase):
    def test_orders_by_depth_not_filename_and_is_deterministic(self):
        pages=[page("z",110,120,[2,3],page_number=2),page("a",100,110,[1,2],page_number=1)]
        first=assemble_whole_log(pages);second=assemble_whole_log(list(reversed(pages)));self.assertEqual(first.ordered_pages,["a","z"]);self.assertEqual(first.ordered_pages,second.ordered_pages);np.testing.assert_array_equal(first.curves[0].values,second.curves[0].values)
    def test_exact_partial_overlap_has_no_duplicate_depth_and_blends_agreement(self):
        a=page("a",100,110,np.linspace(0,10,11));b=page("b",108,118,np.linspace(8,18,11))
        result=assemble_whole_log([b,a],WholeLogConfig(depth_step=1));self.assertEqual(result.joins[0]["relationship"],"small_overlap");self.assertEqual(len(np.unique(result.curves[0].depth)),len(result.curves[0].depth));self.assertIn("blended",result.curves[0].quality_flags[9]);self.assertTrue(result.curves[0].provenance[9].blended);self.assertEqual(len(result.curves[0].provenance[9].sources),2)
    def test_conflicting_overlap_refuses_blend(self):
        a=page("a",100,110,np.zeros(11));b=page("b",108,118,np.full(11,100.))
        result=assemble_whole_log([a,b],WholeLogConfig(depth_step=1));self.assertIn("overlap_conflict",result.curves[0].quality_flags[9]);self.assertFalse(result.curves[0].provenance[9].blended)
    def test_small_and_large_gaps_are_explicit_and_never_silently_interpolated(self):
        small=assemble_whole_log([page("a",0,10,[1,2]),page("b",15,20,[2,3])]);self.assertEqual(small.joins[0]["relationship"],"small_gap")
        large=assemble_whole_log([page("a",0,10,[1,2]),page("b",100,110,[2,3])],WholeLogConfig(depth_step=1));self.assertEqual(large.joins[0]["relationship"],"large_gap");curve=large.curves[0];self.assertTrue(np.all(np.isnan(curve.values[(curve.depth>10)&(curve.depth<100)])));self.assertEqual(export_status(large),"needs_review")
    def test_duplicate_prefers_higher_quality_without_deleting(self):
        a=page("a",0,10,[1,2],source_hash="same",confidence=.4);b=page("b",0,10,[1,2],source_hash="same",confidence=.9)
        result=assemble_whole_log([a,b]);self.assertEqual(result.duplicate_intervals[0]["recommended_source"],"b");self.assertEqual(len(result.ordered_pages),2)
    def test_curve_alias_matches_but_unit_conflict_stays_separate(self):
        merged=assemble_whole_log([page("a",0,10,[1,2],"ILD","OHMM"),page("b",10,20,[2,3],"DEEP RES","OHMM")]);self.assertEqual(len(merged.curves),1)
        separate=assemble_whole_log([page("a",0,10,[1,2],"ILD","OHMM"),page("b",10,20,[2,3],"ILD","API")]);self.assertEqual(len(separate.curves),2);self.assertTrue(any(w["type"]=="curve unit mismatch" for w in separate.warnings))
    def test_offset_and_limited_stretch(self):
        offset=fit_alignment([1,2,3],[3,4,5]);self.assertEqual(offset["model"],"constant_offset");self.assertAlmostEqual(offset["offset"],2)
        stretch=fit_alignment([0,100,200],[1,102,203]);self.assertEqual(stretch["model"],"linear_stretch")
        with self.assertRaises(ValueError):fit_alignment([0,100],[0,200])
    def test_wrap_continues_across_page_boundary(self):
        scale={"cycle_value":10};a=page("a",0,10,[20,25],"ILD","OHMM",wrap=np.array([2,2]),scale=scale);b=page("b",10,20,[5,8],"ILD","OHMM",wrap=np.array([0,0]),scale=scale)
        result=assemble_whole_log([a,b],WholeLogConfig(depth_step=10));curve=result.curves[0];self.assertEqual(int(curve.whole_log_wrap_index[-1]),2);self.assertEqual(curve.provenance[-1].sources[0]["page_wrap_offset"],2)
    def test_grouping_review_and_serialization_preserve_automatic_proposal(self):
        pages=[page("a",0,10,[1,2]),page("b",10,20,[2,3])];self.assertEqual(len(propose_groups(pages)),1);result=assemble_whole_log(pages);reviewed=apply_review(result,join_edits={0:{"status":"approved"}},reviewer="expert");self.assertEqual(reviewed.initial_automatic_joins[0]["status"],"automatic_proposal");self.assertEqual(reviewed.status,"export_ready")
        with tempfile.TemporaryDirectory() as temp:
            payload=json.loads(save_result(Path(temp)/"whole.json",reviewed).read_text());self.assertEqual(payload["schema_version"],1);self.assertTrue(payload["curves"][0]["provenance"][0]["sources"])
    def test_absent_curve_yields_null_outside_coverage(self):
        a=page("a",0,10,[1,2],"GR","API");b=page("b",10,20,[4,5],"ILD","OHMM");result=assemble_whole_log([a,b],WholeLogConfig(depth_step=5));gr=next(c for c in result.curves if c.mnemonic=="GR");self.assertTrue(np.all(np.isnan(gr.values[gr.depth>10])))

if __name__=="__main__":unittest.main()
