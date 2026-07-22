import json,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from training.real_log_dataset import PairCandidate,alignment_training_eligible,assign_split,audit_pair,dataset_inventory,discover_pairs,project_las_curves,render_projection_overlay,review_alignment,score_color_alignment,select_pilot,validate_alignment,write_alignment_bundle,write_pilot_report
from training.real_log_baseline import build_curve_crop_dataset
from training.legacy_pair_ranker import assess_required_evidence,legacy_config_to_alignment,legacy_role,load_legacy_configs,resolve_legacy_configs,score_raster_alignment,select_review_batch

LAS_TEXT="""~Version Information
 VERS. 2.0: CWLS LAS 2.0
 WRAP. NO:
~Well Information
 STRT.FT 1000.0:
 STOP.FT 1010.0:
 STEP.FT 1.0:
 NULL. -999.25:
 WELL. TEST WELL:
~Curve Information
 DEPT.FT : Depth
 GR.GAPI : Gamma Ray
~ASCII
1000 0
1001 10
1002 20
1003 30
1004 40
1005 50
1006 60
1007 70
1008 80
1009 90
1010 100
"""

def make_pair(root,source="kgs",well="well1"):
    directory=Path(root)/source/"pairs"/well;directory.mkdir(parents=True);las=directory/"reference.las";las.write_text(LAS_TEXT,encoding="ascii")
    image=np.full((30,120,3),255,np.uint8)
    for index in range(11):image[5+index,index*10]=(0,200,0)
    tiff=directory/"scan.tif";cv2.imwrite(str(tiff),image)
    return directory,tiff,las

def alignment(pair_id):
    return {"pair_id":pair_id,"review_status":"automatic_draft","depth_control_points":[{"depth":1000.,"row":5.},{"depth":1010.,"row":15.}],"curve_tracks":[{"mnemonic":"GR","x_left":0.,"x_right":100.,"value_left":0.,"value_right":100.,"scale_type":"linear","color":"green"}]}

class RealLogDatasetTests(unittest.TestCase):
    def test_legacy_configs_resolve_as_unapproved_non_test_proposals(self):
        with tempfile.TemporaryDirectory() as temporary:
            _,tiff,las=make_pair(temporary);legacy=Path(temporary)/"legacy.json"
            config={"image_path":rf"F:\\old\\kgs\\pairs\\well1\\{tiff.name}","config":{"depth":{"top_px":5,"bottom_px":15,"top_depth":1000,"bottom_depth":1010,"unit":"FT"},"curves":[{"name":"GR","las_mnemonic":"GR","left_px":0,"right_px":100,"left_value":0,"right_value":100,"mode":"black"}]}}
            legacy.write_text(json.dumps([config]),encoding="utf-8");loaded=load_legacy_configs(legacy);resolved,summary=resolve_legacy_configs(loaded,temporary)
            self.assertEqual(summary["resolved"],1);self.assertEqual(len(resolved),1);self.assertIn(legacy_role("well1"),{"train","validation_diagnostic"})
            proposal=legacy_config_to_alignment(resolved[0],(120,30));self.assertEqual(proposal["review_status"],"automatic_draft");self.assertIn("final_unbiased_test",proposal["prohibited_dataset_roles"])
            records=project_las_curves(las,proposal);metrics,preview,_=score_raster_alignment(tiff,records,proposal,radius_pixels=1.5);self.assertEqual(preview.shape[:2],(30,120));self.assertIn("GR",metrics)
            self.assertIn(assess_required_evidence(metrics,"GR"),{"strong_review_candidate","weak_legacy_seed"})

    def test_legacy_review_selection_preserves_diagnostic_validation(self):
        rankings=[{"pair_id":f"p{i}","score":1-i/10,"status":"alignment_review_required","dataset_role":"train"} for i in range(5)]
        rankings+=[{"pair_id":f"v{i}","score":.4-i/10,"status":"alignment_review_required","dataset_role":"validation_diagnostic"} for i in range(3)]
        selected=select_review_batch(rankings,count=5,minimum_validation=2);self.assertEqual(len(selected),5);self.assertGreaterEqual(sum(row["dataset_role"]=="validation_diagnostic" for row in selected),2)

    def test_discovery_inventory_audit_and_split_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            _,tiff,las=make_pair(temporary);pairs=discover_pairs(temporary);self.assertEqual(len(pairs),1);self.assertEqual(pairs[0].well_id,"well1")
            inventory=dataset_inventory(temporary);self.assertEqual(inventory["paired_directories"],1);self.assertEqual(inventory["tiff_files"],1);self.assertEqual(inventory["las_files"],1)
            audit=audit_pair(pairs[0],content_hashes=True);self.assertEqual(audit.status,"needs_alignment_review");self.assertFalse(audit.training_eligible);self.assertTrue(audit.las_summary["depth_monotonic"]);self.assertEqual(audit.las_summary["curves"][0]["mnemonic"],"GR");self.assertTrue(audit.tiff_summaries[0]["content_sha256"])
            self.assertEqual(assign_split("same-well"),assign_split("same-well"))

    def test_well_level_pilot_selection_keeps_standalone(self):
        candidates=[PairCandidate(str(i),"kgs",f"k{i}",(f"{i}.tif",),f"{i}.las") for i in range(10)]+[PairCandidate("w","wvgs","wv",("w.tif",),"w.las"),PairCandidate("s","standalone","s",("s.tif",),"s.las")]
        selected=select_pilot(candidates,3,1);self.assertEqual(sum(item.source=="kgs" for item in selected),3);self.assertTrue(any(item.source=="standalone" for item in selected))

    def test_projection_validation_color_score_and_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            _,tiff,las=make_pair(temporary);candidate=discover_pairs(temporary)[0];audit=audit_pair(candidate);spec=alignment(candidate.pair_id)
            self.assertEqual(validate_alignment(spec,audit.las_summary),[]);records=project_las_curves(las,spec);self.assertEqual(len(records),11);self.assertAlmostEqual(records[-1]["x"],100.);self.assertAlmostEqual(records[-1]["y"],15.)
            metrics=score_color_alignment(tiff,records,radius=1.1);self.assertGreater(metrics["GR"]["hit_fraction_within_radius"],.99)
            overlay=render_projection_overlay(tiff,records,Path(temporary)/"overlay.png");self.assertTrue(overlay.exists())
            outputs=write_alignment_bundle(Path(temporary)/"bundle",audit,spec);self.assertEqual(len(outputs),4);self.assertTrue(all(path.exists() for path in outputs))
            bundled=json.loads(outputs[0].read_text());reviewed=review_alignment(bundled,audit,"expert","approved","overlay checked");self.assertTrue(alignment_training_eligible(reviewed,audit))
            dataset=build_curve_crop_dataset(tiff,las,reviewed,Path(temporary)/"dataset",track_id="GR",crop_height=2,depth_bands={"train":(1000,1005),"validation":(1005,1008),"test":(1008,1010)});self.assertGreater(dataset["samples"],0);self.assertGreater(dataset["by_split"]["train"],0)
            with self.assertRaises(PermissionError):build_curve_crop_dataset(tiff,las,spec,Path(temporary)/"unapproved",crop_height=2)

    def test_invalid_alignment_and_report(self):
        bad={"depth_control_points":[{"depth":1,"row":5},{"depth":2,"row":4}],"curve_tracks":[]};self.assertTrue(validate_alignment(bad))
        with tempfile.TemporaryDirectory() as temporary:
            make_pair(temporary);audit=audit_pair(discover_pairs(temporary)[0]);outputs=write_pilot_report(Path(temporary)/"report",dataset_inventory(temporary),[audit]);self.assertEqual(len(outputs),3);summary=json.loads(outputs[-1].read_text());self.assertEqual(summary["training_eligible"],0)

if __name__=="__main__":unittest.main()
