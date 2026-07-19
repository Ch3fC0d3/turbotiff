import base64
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import track_analysis
import web_app


def _preview_meta():
    return {
        'source_width': 1000,
        'source_height': 2000,
        'region_left': 100,
        'region_right': 900,
        'region_top': 200,
        'region_bottom': 1800,
        'preview_width': 400,
        'preview_height': 800,
    }


def _raw_analysis():
    return {
        'analysis_confidence': 0.91,
        'tracks': [{
            'id': 'track_1',
            'left_x': 50,
            'right_x': 350,
            'top_y': 20,
            'bottom_y': 780,
            'scale_type': 'log',
            'scale_min': 0.2,
            'scale_max': 2000,
            'unit': 'OHMM',
            'horizontal_grid_spacing_px': 20,
            'vertical_grid_spacing_px': 40,
            'wraparound': True,
            'confidence': 0.88,
            'curves': [{
                'id': 'curve_1',
                'mnemonic': 'RT',
                'color': 'red',
                'estimated_start_x': 200,
                'wrap_enabled': True,
                'confidence': 0.9,
                'max_jump_px': 10,
                'seed_points': [
                    {'x': 200, 'y': 100, 'confidence': 0.9},
                    {'x': 260, 'y': 700, 'confidence': 0.8},
                ],
                'low_confidence_sections': [
                    {'y1': 300, 'y2': 340, 'reason': 'overlap'},
                ],
            }],
            'ignore_regions': [{
                'x1': 100,
                'y1': 30,
                'x2': 180,
                'y2': 80,
                'reason': 'header text',
                'confidence': 0.95,
            }],
        }],
        'global_ignore_regions': [],
        'notes': ['Review the overlap.'],
    }


class TrackAnalysisNormalizationTests(unittest.TestCase):
    def test_preview_coordinates_are_clamped_and_mapped_to_source(self):
        result = track_analysis.normalize_analysis(_raw_analysis(), _preview_meta())

        self.assertEqual(result['schema_version'], 1)
        self.assertEqual(len(result['tracks']), 1)
        track = result['tracks'][0]
        self.assertAlmostEqual(track['left_x'], 100 + (50 / 399) * 799)
        self.assertAlmostEqual(track['right_x'], 100 + (350 / 399) * 799)
        self.assertEqual(track['scale_type'], 'log')
        curve = track['curves'][0]
        self.assertEqual(curve['mnemonic'], 'RT')
        self.assertTrue(curve['wrap_enabled'])
        self.assertGreater(curve['max_jump_px'], 10)
        self.assertGreater(curve['seed_points'][0]['y'], 200)
        self.assertLess(curve['seed_points'][-1]['y'], 1800)

    def test_invalid_track_geometry_is_dropped(self):
        raw = _raw_analysis()
        raw['tracks'][0]['left_x'] = 200
        raw['tracks'][0]['right_x'] = 200

        result = track_analysis.normalize_analysis(raw, _preview_meta())

        self.assertEqual(result['tracks'], [])

    def test_guided_jump_is_bounded_and_wrap_keeps_default(self):
        self.assertEqual(
            track_analysis.guided_max_step(150, {'max_jump_px': 12}, 200),
            18,
        )
        self.assertEqual(
            track_analysis.guided_max_step(150, {'max_jump_px': 12, 'wrap_enabled': True}, 200),
            150,
        )
        self.assertEqual(
            track_analysis.guided_max_step(20, {'max_jump_px': 1000}, 50),
            20,
        )


class TrackAnalysisMaskTests(unittest.TestCase):
    def test_seed_color_boosts_evidence_and_ignore_region_suppresses_it(self):
        image = np.full((80, 60, 3), 255, dtype=np.uint8)
        image[:, 25:28] = (0, 0, 220)
        mask = np.zeros((80, 60), dtype=np.uint8)
        guidance = {
            'estimated_start_x': 26,
            'seed_points': [
                {'x': 26, 'y': 10, 'confidence': 0.9},
                {'x': 26, 'y': 70, 'confidence': 0.9},
            ],
            'ignore_regions': [{
                'x1': 20,
                'x2': 35,
                'y1': 35,
                'y2': 45,
            }],
        }

        guided = track_analysis.apply_curve_guidance(mask, image, guidance, 0, 0)

        self.assertGreater(int(guided[20, 26]), 80)
        self.assertLess(int(guided[40, 26]), int(guided[20, 26]) * 0.2)
        self.assertEqual(guided.shape, mask.shape)

    def test_guided_grid_spacing_tunes_grid_removal_kernels(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[:, ::20] = 0
        image[::15, :] = 0
        original = web_app.cv2.getStructuringElement
        kernel_sizes = []

        def capture_kernel(shape, size, *args, **kwargs):
            kernel_sizes.append(tuple(size))
            return original(shape, size, *args, **kwargs)

        with patch.object(web_app.cv2, 'getStructuringElement', side_effect=capture_kernel):
            result = web_app.compute_prob_map(
                image,
                mode='black',
                ui_filters={
                    'enable_grid_suppression': True,
                    'grid_spacing_x_px': 40,
                    'grid_spacing_y_px': 30,
                },
            )

        self.assertEqual(result.shape, image.shape[:2])
        self.assertIn((28, 1), kernel_sizes)
        self.assertIn((1, 21), kernel_sizes)


class OpenAITrackAnalysisContractTests(unittest.TestCase):
    def test_responses_request_uses_image_input_and_strict_schema(self):
        raw = _raw_analysis()

        class FakeResponse:
            status_code = 200

            def json(self):
                return {
                    'output': [{
                        'type': 'message',
                        'content': [{'type': 'output_text', 'text': json.dumps(raw)}],
                    }],
                }

        calls = []

        def fake_post(url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse()

        image = np.full((100, 80, 3), 255, dtype=np.uint8)
        result = track_analysis.analyze_with_openai(
            image,
            api_key='test-key',
            model='test-vision-model',
            post=fake_post,
        )

        self.assertEqual(result['provider'], 'openai')
        self.assertEqual(result['model'], 'test-vision-model')
        self.assertEqual(len(calls), 1)
        _, kwargs = calls[0]
        payload = kwargs['json']
        self.assertFalse(payload['store'])
        self.assertTrue(payload['text']['format']['strict'])
        self.assertEqual(payload['text']['format']['type'], 'json_schema')
        image_part = payload['input'][0]['content'][1]
        self.assertEqual(image_part['type'], 'input_image')
        self.assertTrue(image_part['image_url'].startswith('data:image/jpeg;base64,'))


class TrackAnalysisEndpointTests(unittest.TestCase):
    def setUp(self):
        web_app.app.config.update(TESTING=True)
        self.client = web_app.app.test_client()
        image = np.full((12, 10, 3), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode('.png', image)
        self.assertTrue(ok)
        self.image_data = 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')
        self.user = {'id': 7, 'is_admin': False, 'subscription_status': 'active'}

    def test_endpoint_requires_authentication(self):
        response = self.client.post('/api/analyze-track-preview', json={})
        self.assertEqual(response.status_code, 302)

    def test_endpoint_reports_missing_server_configuration(self):
        with patch.object(web_app, '_current_user', return_value=self.user):
            with patch.object(web_app, 'OPENAI_API_KEY', None):
                response = self.client.post(
                    '/api/analyze-track-preview',
                    json={'image': self.image_data},
                )

        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()['configured'])

    def test_endpoint_rejects_arbitrary_server_paths(self):
        with patch.object(web_app, '_current_user', return_value=self.user):
            with patch.object(web_app, 'OPENAI_API_KEY', 'test-key'):
                with patch.object(web_app.cv2, 'imread') as imread:
                    response = self.client.post(
                        '/api/analyze-track-preview',
                        json={'image_path': 'C:\\Windows\\win.ini'},
                    )

        self.assertEqual(response.status_code, 403)
        imread.assert_not_called()

    def test_endpoint_returns_mocked_analysis(self):
        expected = {'schema_version': 1, 'tracks': []}
        with patch.object(web_app, '_current_user', return_value=self.user):
            with patch.object(web_app, 'OPENAI_API_KEY', 'test-key'):
                with patch.object(
                    web_app.track_analysis,
                    'analyze_with_openai',
                    return_value=expected,
                ) as analyzer:
                    response = self.client.post(
                        '/api/analyze-track-preview',
                        json={'image': self.image_data},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['analysis'], expected)
        analyzer.assert_called_once()


if __name__ == '__main__':
    unittest.main()
