import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ai_tracer
import fast_tracer
import web_app


class ViterbiRegressionTests(unittest.TestCase):
    def test_path_optimizes_through_final_row(self):
        cost = np.full((10, 3), 50.0, dtype=np.float32)
        cost[:8, 0] = 0.0
        cost[8:, 0] = 100.0
        cost[8:, 1] = 0.0

        probability = np.full((10, 3), 0.02, dtype=np.float32)
        probability[:, 0] = 1.0
        probability[:, 1] = 0.2

        runner = getattr(fast_tracer.run_viterbi, 'py_func', fast_tracer.run_viterbi)
        xs, _ = runner(cost, probability, 1, 0.0, 0.0)

        self.assertEqual(xs[-1], 1.0)
        self.assertTrue(np.all(np.isfinite(xs)))

    def test_wrapped_path_crosses_from_right_edge_to_left_edge(self):
        width = 8
        expected = np.array([5, 6, 7, 0, 1, 2], dtype=np.float32)
        cost = np.full((expected.size, width), 20.0, dtype=np.float32)
        probability = np.full_like(cost, 0.01)
        for row, x in enumerate(expected.astype(np.int32)):
            cost[row, x] = 0.0
            probability[row, x] = 1.0

        runner = getattr(fast_tracer.run_viterbi, 'py_func', fast_tracer.run_viterbi)
        xs, confidence = runner(cost, probability, 1, 0.05, 0.0, True)

        np.testing.assert_array_equal(xs, expected)
        self.assertTrue(np.all(confidence > 0.9))

    def test_confidence_excludes_selected_pixel_and_never_goes_negative(self):
        cost = np.full((3, 4), 10.0, dtype=np.float32)
        cost[:, 1] = 0.0
        probability = np.full((3, 4), 0.1, dtype=np.float32)
        probability[:, 1] = 0.4
        probability[:, 2] = 0.9

        runner = getattr(fast_tracer.run_viterbi, 'py_func', fast_tracer.run_viterbi)
        xs, confidence = runner(cost, probability, 1, 0.0, 0.0)

        np.testing.assert_array_equal(xs, np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(confidence, np.zeros(3, dtype=np.float32))


class NeuralTraceRegressionTests(unittest.TestCase):
    def test_probability_map_decodes_to_one_coordinate_per_row(self):
        prediction = np.zeros((3, 5), dtype=np.float32)
        prediction[0, 1] = 1.0
        prediction[1, 3] = 1.0
        prediction[2, 4] = 1.0

        coords = ai_tracer._prediction_to_normalized_x(prediction)

        np.testing.assert_allclose(coords, [0.25, 0.75, 1.0])
        self.assertEqual(coords.shape, (3,))

    def test_coordinate_vector_is_preserved(self):
        prediction = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
        coords = ai_tracer._prediction_to_normalized_x(prediction)
        np.testing.assert_allclose(coords, [0.1, 0.5, 0.9])

    def test_bimodal_probability_row_selects_a_real_peak(self):
        prediction = np.zeros((2, 9), dtype=np.float32)
        prediction[:, 1] = 0.9
        prediction[:, 7] = 0.8

        coords = ai_tracer._prediction_to_normalized_x(prediction)

        np.testing.assert_allclose(coords, [0.125, 0.125])

    def test_failed_ai_trace_falls_back_to_dp(self):
        roi = np.zeros((4, 6, 3), dtype=np.uint8)
        mask = np.zeros((4, 6), dtype=np.uint8)
        expected_x = np.array([1, 2, 3, 4], dtype=np.float32)
        expected_confidence = np.full(4, 0.5, dtype=np.float32)

        with patch.object(
            web_app.ai_tracer,
            'predict_probability_map',
            side_effect=RuntimeError('bad output'),
        ):
            with patch.object(
                web_app,
                'trace_curve_with_dp',
                return_value=(expected_x, expected_confidence),
            ) as dp:
                xs, confidence = web_app._trace_ai_with_dp_fallback(
                    roi, mask, 'GR', 0.0, 100.0, 'GR', 150, 0.001, 0.001, 'right'
                )

        np.testing.assert_array_equal(xs, expected_x)
        np.testing.assert_array_equal(confidence, expected_confidence)
        dp.assert_called_once()


class ProbabilityMapRegressionTests(unittest.TestCase):
    def test_red_hue_mask_spans_both_ends_of_opencv_hue_range(self):
        hsv = np.array([[[179, 200, 200], [1, 200, 200], [90, 200, 200]]], dtype=np.uint8)

        mask = web_app._circular_hue_mask(hsv, center=0.0, band=4.0)

        np.testing.assert_array_equal(mask[0], [255, 255, 0])
        center = web_app._circular_hue_center(np.array([179, 0, 1], dtype=np.uint8))
        self.assertTrue(center <= 2.0 or center >= 178.0)

    def test_pipeline_skeletonization_preserves_soft_noncenter_evidence(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[2:7, 2:7] = 80

        result = web_app.pipeline_skeletonize(mask)

        self.assertGreater(result[2, 2], 0)
        self.assertGreaterEqual(int(result.max()), 80)

    def test_ai_heatmap_is_decoded_by_dp_with_wrap_configuration(self):
        roi = np.zeros((4, 6, 3), dtype=np.uint8)
        mask = np.zeros((4, 6), dtype=np.uint8)
        heatmap = np.zeros((4, 6), dtype=np.float32)
        heatmap[:, 2] = 1.0
        expected_x = np.full(4, 2.0, dtype=np.float32)
        expected_confidence = np.full(4, 0.7, dtype=np.float32)

        with patch.object(web_app.ai_tracer, 'predict_probability_map', return_value=heatmap):
            with patch.object(
                web_app,
                'trace_curve_with_dp',
                return_value=(expected_x, expected_confidence),
            ) as dp:
                xs, confidence = web_app._trace_ai_with_dp_fallback(
                    roi,
                    mask,
                    'RT',
                    0.2,
                    20.0,
                    'RT',
                    3,
                    0.01,
                    0.0,
                    'right',
                    True,
                )

        np.testing.assert_array_equal(xs, expected_x)
        np.testing.assert_array_equal(confidence, expected_confidence)
        self.assertTrue(dp.call_args.kwargs['wrap_enabled'])
        self.assertGreater(int(dp.call_args.args[0][:, 2].min()), 0)


class FusionRegressionTests(unittest.TestCase):
    def test_centroid_stays_in_selected_connected_peak(self):
        row = np.zeros(10, dtype=np.float32)
        row[1:3] = 1.0
        row[7:9] = 1.0

        centroid = web_app._connected_peak_centroid(row, 1, 0.99)

        self.assertAlmostEqual(centroid, 1.5)


class TrainingDataRegressionTests(unittest.TestCase):
    def _job_with_three_curves(self):
        curves = []
        for index, name in enumerate(('A', 'B', 'C')):
            curves.append({
                'name': name,
                'las_mnemonic': name,
                'left_px': 10 + index * 10,
                'right_px': 20 + index * 10,
                'left_value': 1.0,
                'right_value': 100.0,
                'scale_type': 'log' if name == 'B' else 'linear',
            })
        return {
            'config': {
                'depth': {
                    'top_px': 0,
                    'bottom_px': 8,
                    'top_depth': 1000.0,
                    'bottom_depth': 1007.0,
                    'unit': 'FT',
                },
                'curves': curves,
                'global_options': {'downsample': 2, 'null': -999.25},
            }
        }

    def test_three_curve_downsampling_keeps_every_vector_aligned(self):
        job = self._job_with_three_curves()
        trace_points = {}
        for curve in job['config']['curves']:
            left = curve['left_px']
            trace_points[curve['name']] = [[left + row, row] for row in range(8)]

        result = web_app._build_batch_training_result(
            0,
            job,
            {'curve_traces': trace_points},
            np.zeros((8, 50, 3), dtype=np.uint8),
            False,
        )

        self.assertEqual(len(result['depth']['values']), 4)
        for name in ('A', 'B', 'C'):
            self.assertEqual(len(result['curve_traces'][name]), 4)
            self.assertEqual(len(result['curves'][name]), 4)
        self.assertNotIn('image', result)
        self.assertEqual(result['metadata']['pipeline'], 'production')

    def test_scale_mapping_matches_shared_converter(self):
        xs = np.array([0.0, 4.0, 9.0], dtype=np.float32)
        cases = [
            {'name': 'GR', 'left_value': 0.0, 'right_value': 100.0, 'scale_type': 'linear'},
            {'name': 'RT', 'left_value': 1.0, 'right_value': 100.0, 'scale_type': 'log'},
            {'name': 'SP', 'left_value': -50.0, 'right_value': 50.0, 'scale_type': 'centered'},
            {'name': 'RT', 'left_value': 1.0, 'right_value': 100.0, 'scale_type': 'log', 'wrapped': True},
        ]
        for curve in cases:
            actual, meta = web_app._scale_trace_values(curve, xs, 10)
            expected = web_app.scale_detection.pixel_to_value(
                xs,
                10,
                curve['left_value'],
                curve['right_value'],
                curve['scale_type'],
                bool(curve.get('wrapped')),
            )
            np.testing.assert_allclose(actual, expected)
            self.assertEqual(meta['scale_type'], curve['scale_type'])


class BlackGrFinalizationRegressionTests(unittest.TestCase):
    def test_recenter_preserves_rejected_rows(self):
        roi = np.full((9, 21, 3), 255, dtype=np.uint8)
        roi[:, 9:12] = 0
        xs = np.array([10, 10, np.nan, np.nan, 10, 10, 10, 10, 10], dtype=np.float32)

        result = web_app.recenter_black_trace_post_dp(
            roi,
            xs,
            preserve_missing=True,
        )

        self.assertTrue(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[3]))
        self.assertTrue(np.all(np.isfinite(result[[0, 1, 4, 5, 6, 7, 8]])))

    def test_final_continuity_gate_does_not_recreate_rejected_rows(self):
        mask = np.zeros((9, 21), dtype=np.uint8)
        mask[:, 10] = 255
        xs = np.array([10, 10, np.nan, np.nan, 10, 10, 10, 10, 10], dtype=np.float32)

        result = web_app.enforce_local_trace_continuity(mask, xs, max_step=12.0)

        self.assertTrue(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[3]))
        self.assertEqual(float(result[4]), 10.0)

    def test_bounded_gr_snap_moves_only_local_crest_to_connected_evidence(self):
        mask = np.zeros((9, 30), dtype=np.uint8)
        xs = np.array([5, 6, 7, 8, 9, 8, 7, 6, 5], dtype=np.float32)
        for row, x in enumerate(xs.astype(int)):
            mask[row, x] = 255
        mask[4, 9:15] = 255
        mask[3:6, 13:15] = 255

        result = web_app.bounded_gr_crest_snap(
            mask,
            xs,
            hot_side='right',
            max_shift=15,
            candidate_rows_only=True,
        )

        self.assertEqual(float(result[4]), 14.0)
        np.testing.assert_array_equal(result[[0, 1, 2, 6, 7, 8]], xs[[0, 1, 2, 6, 7, 8]])
        self.assertLessEqual(float(np.nanmax(np.abs(result - xs))), 15.0)

    def test_bounded_gr_snap_preserves_missing_rows(self):
        mask = np.full((7, 20), 255, dtype=np.uint8)
        xs = np.array([5, 6, np.nan, np.nan, 7, 6, 5], dtype=np.float32)

        result = web_app.bounded_gr_crest_snap(mask, xs, hot_side='right')

        self.assertTrue(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[3]))


class EndpointSecurityRegressionTests(unittest.TestCase):
    def setUp(self):
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()

    def test_tracing_endpoints_require_authentication(self):
        batch = self.client.post('/api/batch_digitize', json={'jobs': [{}]})
        prediction = self.client.post('/api/ml_predict_curve_trace', json={})

        self.assertEqual(batch.status_code, 302)
        self.assertEqual(prediction.status_code, 302)

    def test_batch_endpoint_requires_admin(self):
        with patch.object(
            web_app,
            '_current_user',
            return_value={'id': 7, 'is_admin': False, 'subscription_status': 'active'},
        ):
            response = self.client.post('/api/batch_digitize', json={'jobs': [{}]})

        self.assertEqual(response.status_code, 403)

    def test_batch_endpoint_rejects_unbounded_job_lists(self):
        jobs = [{}] * (web_app.BATCH_DIGITIZE_MAX_JOBS + 1)
        with patch.object(
            web_app,
            '_current_user',
            return_value={'id': 1, 'is_admin': True, 'subscription_status': 'active'},
        ):
            response = self.client.post('/api/batch_digitize', json={'jobs': jobs})

        self.assertEqual(response.status_code, 400)
        self.assertIn('At most', response.get_json()['error'])

    def test_batch_endpoint_never_reads_arbitrary_server_paths(self):
        job = {
            'image_path': 'C:\\Windows\\win.ini',
            'config': {
                'depth': {'top_px': 0, 'bottom_px': 2, 'top_depth': 0, 'bottom_depth': 1},
                'curves': [],
            },
        }
        with patch.object(
            web_app,
            '_current_user',
            return_value={'id': 1, 'is_admin': True, 'subscription_status': 'active'},
        ):
            with patch.object(web_app.cv2, 'imread') as imread:
                response = self.client.post('/api/batch_digitize', json={'jobs': [job]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['summary']['failed'], 1)
        imread.assert_not_called()

    def test_batch_endpoint_uses_production_pipeline(self):
        image = np.full((16, 16, 3), 255, dtype=np.uint8)
        for row in range(16):
            image[row, 3 + row // 3] = (0, 0, 255)
        encoded_ok, encoded = web_app.cv2.imencode('.png', image)
        self.assertTrue(encoded_ok)
        image_data = 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')

        job = {
            'image': image_data,
            'config': {
                'depth': {
                    'top_px': 0,
                    'bottom_px': 16,
                    'top_depth': 1000,
                    'bottom_depth': 1015,
                    'unit': 'FT',
                },
                'curves': [{
                    'name': 'GR',
                    'las_mnemonic': 'GR',
                    'left_px': 0,
                    'right_px': 16,
                    'left_value': 0,
                    'right_value': 150,
                    'mode': 'red',
                    'enable_viterbi': False,
                    'enable_cc_cleanup': False,
                    'enable_skeletonization': False,
                }],
                'global_options': {'blur': 0, 'downsample': 2},
            },
        }
        admin_user = {
            'id': 1,
            'is_admin': True,
            'subscription_status': 'active',
            'email': 'admin@example.test',
        }
        with patch.object(web_app, '_current_user', return_value=admin_user):
            response = self.client.post('/api/batch_digitize', json={'jobs': [job]})
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body['summary']['success'], 1, body)
        record = body['results'][0]
        self.assertEqual(record['metadata']['pipeline'], 'production')
        self.assertEqual(len(record['depth']['values']), len(record['curve_traces']['GR']))
        self.assertEqual(len(record['depth']['values']), len(record['curves']['GR']))

    def test_ml_endpoint_rejects_client_model_paths(self):
        with patch.object(
            web_app,
            '_current_user',
            return_value={'id': 1, 'is_admin': False, 'subscription_status': 'active'},
        ):
            response = self.client.post(
                '/api/ml_predict_curve_trace',
                json={'image': 'data:image/png;base64,AA==', 'model_path': 'attacker.pt'},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn('configured by the server', response.get_json()['error'])

    def test_checkpoint_loader_uses_weights_only(self):
        class FakeTorch:
            def __init__(self):
                self.load_kwargs = None

            def load(self, path, **kwargs):
                self.load_kwargs = kwargs
                return {
                    'state_dict': {'weight': object()},
                    'input_h': 256,
                    'input_w': 128,
                }

        class FakeModel:
            def load_state_dict(self, state_dict):
                self.state_dict = state_dict

            def eval(self):
                return self

            def to(self, device):
                return self

        fake_torch = FakeTorch()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'model.pt'
            model_path.write_bytes(b'checkpoint')
            with patch.object(web_app, 'torch', fake_torch):
                with patch.object(web_app, '_CurveTraceNet', FakeModel, create=True):
                    with patch.dict(
                        web_app._ML_CURVE_TRACE_MODEL_CACHE,
                        {'model_path': None, 'device': None, 'model': None, 'meta': None},
                        clear=True,
                    ):
                        web_app._ml_load_curve_trace_model(str(model_path), 'cpu')

        self.assertTrue(fake_torch.load_kwargs['weights_only'])
        self.assertEqual(fake_torch.load_kwargs['map_location'], 'cpu')


if __name__ == '__main__':
    unittest.main()
