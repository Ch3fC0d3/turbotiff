import tempfile, unittest
from pathlib import Path
import numpy as np

from learning.active_learning import rank_case, mixed_epoch_indices
from learning.calibration import calibration_report
from learning.corrections import CorrectionStore
from learning.datasets import DatasetRegistry, export_approved_corrections, grouped_split, leakage_report
from learning.model_registry import ModelRegistry
from learning.shadow import shadow_result
from learning.train_candidate import preflight_datasets
from learning.train_candidate import create_candidate


def trace(x):
    x=list(x); return {"x_by_row":x,"unwrapped_x_by_row":x,"wrap_index_by_row":[0]*len(x),"confidence_by_row":[.5]*len(x),"model_version":"test-model","decoder_version":"test-decoder","valid_row_mask":[True]*len(x),"wrap_events":[]}


class Phase4GovernanceTests(unittest.TestCase):
    def test_correction_preserves_prediction_edits_and_approval_controls_training(self):
        with tempfile.TemporaryDirectory() as temp:
            store=CorrectionStore(Path(temp)/"corrections")
            record=store.capture(b"image",trace([1,2,3]),{**trace([1,3,3]),"valid_row_mask":[True]*3,"wrap_events":[]},track_dimensions=[3,8],data_use="training_allowed",edit_history=[{"operation":"move points"}])
            self.assertEqual(store.approved_for_training(),[])
            store.review(record["record_id"],"approved","tester")
            approved=store.approved_for_training(); self.assertEqual(len(approved),1)
            self.assertEqual(approved[0]["prediction"]["x_by_row"],[1,2,3]); self.assertEqual(approved[0]["correction"]["x_by_row"],[1,3,3])

    def test_restricted_and_do_not_retain_records_never_train(self):
        with tempfile.TemporaryDirectory() as temp:
            store=CorrectionStore(temp)
            for use in ("evaluation_only","client_restricted","do_not_retain"):
                item=store.capture(b"same",trace([1]),{**trace([1]),"valid_row_mask":[True]},track_dimensions=[1,2],data_use=use)
                store.review(item["record_id"],"approved","tester")
            self.assertEqual(store.approved_for_training(),[])
            # Restricted records may be retained for audit, but none are trainable.
            self.assertEqual(store.approved_for_training(),[])

    def test_dataset_registry_versions_are_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            registry=DatasetRegistry(temp); registry.register("real_v1",[{"id":"a","labels":{"centerline":True}}])
            with self.assertRaises(FileExistsError): registry.register("real_v1",[])

    def test_export_derives_partial_labels_from_only_approved_records(self):
        import cv2
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); store=CorrectionStore(root/"corrections")
            ok,image=cv2.imencode('.png',np.full((3,8,3),255,dtype=np.uint8)); self.assertTrue(ok)
            approved=store.capture(image.tobytes(),trace([1,2,3]),{**trace([1,2,3]),"valid_row_mask":[True]*3,"wrap_events":[]},track_dimensions=[3,8],data_use="training_allowed")
            store.review(approved["record_id"],"approved","tester")
            pending=store.capture(image.tobytes(),trace([3,3,3]),{**trace([3,3,3]),"valid_row_mask":[True]*3,"wrap_events":[]},track_dimensions=[3,8],data_use="training_allowed")
            manifest=export_approved_corrections(root/"corrections",root/"datasets","real_v1")
            self.assertEqual(manifest["sample_count"],1); self.assertEqual(manifest["label_counts"]["stroke"],0)
            self.assertNotIn(pending["record_id"],manifest["source_record_ids"])

    def test_leakage_and_source_grouped_split(self):
        training=[{"id":"a","source_group":"well-1","image_checksum":"x"},{"id":"b","source_group":"well-1"},{"id":"c","source_group":"well-2"}]
        report=leakage_report(training,[{"source_group":"well-9","image_checksum":"x"}]); self.assertTrue(report["blocked"])
        train,valid=grouped_split(training,.5,3); self.assertFalse({s["source_group"] for s in train}&{s["source_group"] for s in valid})

    def test_candidate_preflight_requires_clean_immutable_golden_split(self):
        with tempfile.TemporaryDirectory() as temp:
            registry=DatasetRegistry(temp)
            registry.register("train_v1",[{"id":"t","source_group":"well-a","image_checksum":"train","labels":{}}])
            registry.register("golden_v1",[{"id":"g","source_group":"well-b","image_checksum":"gold","labels":{}}])
            report=preflight_datasets(temp,["train_v1"],["golden_v1"]); self.assertFalse(report["leakage"]["blocked"])
            registry.register("leaky_v1",[{"id":"x","source_group":"well-b","image_checksum":"other","labels":{}}])
            with self.assertRaises(RuntimeError): preflight_datasets(temp,["leaky_v1"],["golden_v1"])

    def test_novelty_raises_priority_and_mixing_replays_hard_cases(self):
        familiar=rank_case({},[0,0],[[0,0]]); novel=rank_case({},[4,4],[[0,0]])
        self.assertGreater(novel["review_priority"],familiar["review_priority"])
        sequence=mixed_epoch_indices({"synthetic":5,"real":3,"hard":1},{"synthetic":.2,"real":.2,"hard":.6},100,7)
        self.assertGreater(sum(name=="hard" for name,_ in sequence),40)

    def test_promotion_gates_rollback_and_shadow_are_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); checkpoint=root/"x.pt"; checkpoint.write_bytes(b"checkpoint")
            registry=ModelRegistry(root/"models"); registry.register_candidate("m1",checkpoint,"lightweight",[],evaluation_completed=True,metrics={"mae":1.0})
            with self.assertRaises(PermissionError): registry.promote("m1","human","test",{})
            gates={"evaluation_completed":1,"no_leakage":1,"thresholds_passed":1,"regression_report":1}
            registry.promote("m1","human","baseline",gates)
            registry.register_candidate("m2",checkpoint,"advanced",[],evaluation_completed=True,metrics={"mae":.9}); registry.promote("m2","human","better",gates)
            registry.rollback("m1","human","regression"); self.assertEqual(registry.active()["model_id"],"m1")
            output=shadow_result({"x":[1]},{"x":[9]},"m1","m2"); self.assertEqual(output["selected_output"],{"x":[1]})

    def test_candidate_training_registers_candidate_without_changing_production(self):
        import cv2, json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); store=CorrectionStore(root/"corrections")
            ok,image=cv2.imencode('.png',np.full((8,8,3),220,dtype=np.uint8)); self.assertTrue(ok)
            item=store.capture(image.tobytes(),trace([3]*8),trace([3]*8),track_dimensions=[8,8],data_use="training_allowed")
            store.review(item["record_id"],"approved","tester"); export_approved_corrections(root/"corrections",root/"datasets","real_v1")
            DatasetRegistry(root/"datasets").register("golden_v1",[{"id":"gold","source_group":"other","image_checksum":"other","labels":{}}])
            evaluation=root/"evaluation.json"; evaluation.write_text(json.dumps({"suites":{"general":{"candidate_better":1}}}))
            report=create_candidate(None,root/"run",root/"models",["real_v1"],datasets_root=root/"datasets",golden_dataset_ids=["golden_v1"],evaluation_report=evaluation)
            self.assertTrue(report["trained"]); registry=ModelRegistry(root/"models"); self.assertIsNone(registry.active()); self.assertEqual(registry.get(report["model_id"])["status"],"candidate")

    def test_calibration_is_finite(self):
        report=calibration_report([.1,.2,.8,.9],[5,4,0,1],1.0); self.assertTrue(np.isfinite(report["expected_calibration_error"]))


class Phase4ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: import torch
        except Exception as exc: raise unittest.SkipTest(str(exc))
        cls.torch=torch

    def test_lightweight_and_advanced_share_outputs_and_prompt_changes_selection(self):
        from curve_model.phase4_model import CurvePhase4UNet
        from curve_model.advanced import CurveAdapterModel
        image=self.torch.randn(2,3,32,32); prompt=self.torch.zeros(2,1,32,32); prompt[:,:,8:12,8:12]=1
        for model in (CurvePhase4UNet(base_channels=4),CurveAdapterModel(base_channels=4)):
            output=model(image,prompt); unprompted=model(image,None); self.assertEqual(output["wrap_logits"].shape,(2,3,32))
            self.assertEqual(set(output),{"stroke_logits","centerline_logits","distance_field","direction","grid_logits","wrap_logits"})
            self.assertFalse(self.torch.equal(output["centerline_logits"],unprompted["centerline_logits"]))

    def test_phase4_inference_exposes_soft_wrap_evidence(self):
        import cv2
        from curve_model.phase4_model import CurvePhase4UNet
        from curve_model.phase4_infer import predict_phase4_geometry
        with tempfile.TemporaryDirectory() as temp:
            model=CurvePhase4UNet(base_channels=4); path=Path(temp)/"candidate.pt"
            self.torch.save({"state_dict":model.state_dict(),"model_config":model.configuration(),"model_version":model.model_version},path)
            prediction=predict_phase4_geometry(np.zeros((16,12,3),dtype=np.uint8),str(path),device="cpu")
            self.assertEqual(prediction["wrap_probability_right_to_left"].shape,(16,))
            total=prediction["wrap_probability_right_to_left"]+prediction["wrap_probability_left_to_right"]
            self.assertTrue(np.all(total<=1.00001))

    def test_partial_wrap_loss_ignores_unavailable_wrap_labels(self):
        from curve_model.phase4_losses import CurvePhase4Loss
        torch=self.torch; loss=CurvePhase4Loss()
        outputs={"stroke_logits":torch.randn(1,1,8,8),"centerline_logits":torch.randn(1,1,8,8),"distance_field":torch.rand(1,1,8,8),"direction":torch.randn(1,2,8,8),"grid_logits":torch.randn(1,1,8,8),"wrap_logits":torch.randn(1,3,8)}
        targets={"centerline_mask":torch.zeros(1,1,8,8),"distance_field":torch.zeros(1,1,8,8),"direction_field":torch.zeros(1,2,8,8),"valid_direction_mask":torch.zeros(1,1,8,8),"label_available":{"stroke":torch.zeros(1),"grid":torch.zeros(1),"centerline":torch.ones(1),"wrap":torch.zeros(1)},"wrap_target":torch.zeros(1,8,dtype=torch.long)}
        parts=loss(outputs,targets); self.assertEqual(float(parts["wrap"]),0.0); self.assertTrue(torch.isfinite(parts["total"]))

    def test_no_labels_produce_zero_loss_instead_of_false_negative_targets(self):
        from curve_model.phase4_losses import CurvePhase4Loss
        torch=self.torch; loss=CurvePhase4Loss()
        outputs={"stroke_logits":torch.randn(1,1,4,4),"centerline_logits":torch.randn(1,1,4,4),"distance_field":torch.rand(1,1,4,4),"direction":torch.randn(1,2,4,4),"grid_logits":torch.randn(1,1,4,4),"wrap_logits":torch.randn(1,3,4)}
        targets={"label_available":{name:torch.zeros(1) for name in ("stroke","grid","centerline","wrap")}}
        parts=loss(outputs,targets); self.assertAlmostEqual(float(parts["total"]),0.0,places=5)

    def test_mixed_batch_ignores_unavailable_centerline_sample(self):
        from curve_model.phase4_losses import CurvePhase4Loss
        torch=self.torch; criterion=CurvePhase4Loss()
        def outputs(batch):
            return {"stroke_logits":torch.zeros(batch,1,4,4),"centerline_logits":torch.zeros(batch,1,4,4),"distance_field":torch.zeros(batch,1,4,4),"direction":torch.ones(batch,2,4,4),"grid_logits":torch.zeros(batch,1,4,4),"wrap_logits":torch.zeros(batch,3,4)}
        center=torch.zeros(1,1,4,4); center[:,:,1,1]=1
        base={"centerline_mask":center,"distance_field":center.clone(),"direction_field":torch.ones(1,2,4,4),"valid_direction_mask":center.clone(),"label_available":{"stroke":torch.zeros(1),"grid":torch.zeros(1),"centerline":torch.ones(1),"wrap":torch.zeros(1)}}
        single=criterion(outputs(1),base)["total"]
        mixed={key:(torch.cat((value,torch.zeros_like(value)),0) if torch.is_tensor(value) else value) for key,value in base.items()}
        mixed["label_available"]={name:torch.tensor([float(name=="centerline"),0.0]) for name in ("stroke","grid","centerline","wrap")}
        combined=criterion(outputs(2),mixed)["total"]
        self.assertAlmostEqual(float(single),float(combined),places=5)

if __name__=="__main__": unittest.main()
