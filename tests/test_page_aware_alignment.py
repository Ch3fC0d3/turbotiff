import tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from training.page_aware_alignment import detect_colored_log_body,optimize_colored_curve_alignment

LAS="""~Version Information
 VERS. 2.0:
 WRAP. NO:
~Well Information
 STRT.FT 1000:
 STOP.FT 1100:
 STEP.FT 1:
 NULL. -999.25:
~Curve Information
 DEPT.FT:
 GR.GAPI:
~ASCII
"""+"\n".join(f"{depth} {50+35*np.sin((depth-1000)/9):.4f}" for depth in range(1000,1101))+"\n"

class PageAwareAlignmentTests(unittest.TestCase):
    def test_finds_colored_body_and_curve_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);image=np.full((420,220,3),255,np.uint8);depths=np.arange(1000,1101);values=50+35*np.sin((depths-1000)/9);ys=np.rint(110+(depths-1000)*2.6).astype(int);xs=np.rint(20+values).astype(int)
            for first,second in zip(zip(xs[:-1],ys[:-1]),zip(xs[1:],ys[1:])):cv2.line(image,first,second,(0,0,210),2,cv2.LINE_AA)
            image_path=root/"scan.tif";las_path=root/"reference.las";cv2.imwrite(str(image_path),image);las_path.write_text(LAS,encoding="ascii")
            alignment,metrics=optimize_colored_curve_alignment(image_path,las_path,"GR","pair")
            self.assertGreater(metrics["hit_fraction_within_radius"],.60);self.assertGreater(metrics["hit_lift_over_control"],.5);self.assertEqual(metrics["evidence_status"],"strong_review_candidate");self.assertEqual(alignment["review_status"],"automatic_draft")

    def test_rejects_page_without_sustained_color(self):
        masks={name:np.zeros((100,50),bool) for name in ("red","green","blue")}
        with self.assertRaises(ValueError):detect_colored_log_body(masks)

if __name__=="__main__":unittest.main()
