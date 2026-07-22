import json,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from page_analysis import analyze_well_log_page,CoordinateTransform
from page_analysis.depth_detection import optimize_depth_sequence,fit_depth_mapping
from page_analysis.header_detection import normalize_mnemonic,normalize_unit
from page_analysis.models import BoundingBox,DetectedTrack,PageAnalysisResult,ScaleDefinition
from page_analysis.review import apply_review,safety_status,to_tracing_requests
from page_analysis.scale_detection import classify_scale
from page_analysis.serialization import save_analysis

def synthetic_page(width=600,height=800):
    image=np.full((height,width,3),255,np.uint8)
    for x in (40,180,320,460,560): cv2.line(image,(x,80),(x,760),(0,0,0),3)
    for y in range(100,761,40): cv2.line(image,(40,y),(560,y),(180,180,180),1)
    cv2.putText(image,"GR API   DEPTH FT   ILD OHM-M",(55,55),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2)
    return image

class Phase5Tests(unittest.TestCase):
    def test_transform_round_trip(self):
        matrix=np.array([[1,.02,3],[-.01,1,4],[0,0,1]],float); transform=CoordinateTransform.from_matrix(matrix)
        points=np.array([[0,0],[25.5,80.25],[500,700]],float); np.testing.assert_allclose(transform.map_points(transform.map_points(points),inverse=True),points,atol=1e-7)
    def test_multiple_major_tracks_ignore_minor_grid(self):
        result=analyze_well_log_page(synthetic_page()); self.assertEqual(result.page_classification,"log_data"); self.assertGreaterEqual(len(result.tracks),3); self.assertLessEqual(len(result.tracks),5)
    def test_depth_sequence_alternatives_and_piecewise_mapping(self):
        labels=[{"row_position":100,"text_candidates":[{"text":"5400","confidence":.9}]},{"row_position":200,"text_candidates":[{"text":"5480","confidence":.2},{"text":"5500","confidence":.8}]},{"row_position":300,"text_candidates":[{"text":"5600","confidence":.9}]}]
        selected=optimize_depth_sequence(labels); self.assertEqual([x["selected_value"] for x in selected],[5400,5500,5600]); mapping=fit_depth_mapping(selected); self.assertAlmostEqual(float(mapping.depth_at(250)),5550,places=4)
    def test_scale_types_reversed_and_multicycle(self):
        linear=classify_scale([0,10,20,30],[100,0]); self.assertEqual(linear.scale_type,"linear"); self.assertEqual(linear.direction,"increasing_left")
        logarithmic=classify_scale([0,2,5,10,20,23,28,38,58,61,66,76,96]); self.assertEqual(logarithmic.scale_type,"logarithmic"); self.assertGreaterEqual(logarithmic.cycles,1)
    def test_units_and_unknown_mnemonics_preserved(self):
        self.assertEqual(normalize_unit("OHM M")["normalized_unit"],"OHMM"); unknown=normalize_mnemonic("ZZQ"); self.assertTrue(unknown["preserved_unknown"]); self.assertEqual(unknown["normalized_mnemonic"],"ZZQ")
    def test_review_provenance_safety_and_approved_tracing(self):
        result=analyze_well_log_page(synthetic_page()); self.assertEqual(safety_status(result),"manual_setup_required")
        result.tracks[0].curve_candidates=[{"candidate_id":"c1","dominant_color":"black"}]; result.tracks[0].scale=ScaleDefinition("linear",0,100,1,"increasing_right",units="API")
        result.depth_columns=[{"mapping":{"control_points":[[100,1000],[700,1600]],"unit":"FT"}}]
        result.confidence_summary["critical_fields"]={key:.95 for key in result.confidence_summary["critical_fields"]}
        reviewed=apply_review(result,{"page_classification":"log_data"},"expert","approved"); self.assertTrue(reviewed.processing_metadata["review"]["edit_history"]); self.assertEqual(len(to_tracing_requests(reviewed)),1)
    def test_unapproved_export_block_and_json_schema(self):
        result=analyze_well_log_page(synthetic_page())
        with self.assertRaises(PermissionError): to_tracing_requests(result)
        with tempfile.TemporaryDirectory() as temp:
            path=save_analysis(Path(temp)/"page.json",result); payload=json.loads(path.read_text()); self.assertEqual(payload["schema_version"],1); self.assertEqual(payload["source_image_hash"],result.source_image_hash)
    def test_rotated_page_reports_coarse_orientation(self):
        rotated=cv2.rotate(synthetic_page(),cv2.ROTATE_90_CLOCKWISE); result=analyze_well_log_page(rotated); self.assertEqual(result.processing_metadata["orientation"]["coarse_rotation"],90)

if __name__=="__main__": unittest.main()
