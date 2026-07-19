import base64
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import web_app


def _ocr_entry(text, left, top, right, bottom, confidence=None):
    entry = {
        'text': text,
        'vertices': [
            {'x': left, 'y': top},
            {'x': right, 'y': top},
            {'x': right, 'y': bottom},
            {'x': left, 'y': bottom},
        ],
    }
    if confidence is not None:
        entry['confidence'] = confidence
    return entry


class HeaderLayoutEndpointTests(unittest.TestCase):
    def setUp(self):
        image = np.full((120, 1000, 3), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode('.jpg', image)
        self.assertTrue(ok)
        self.image_data = (
            'data:image/jpeg;base64,'
            + base64.b64encode(encoded.tobytes()).decode('ascii')
        )

    def test_uses_local_ocr_layout_when_text_model_is_unavailable(self):
        detected = {
            'raw': [
                _ocr_entry('GAMMA RAY', 180, 20, 300, 50),
                _ocr_entry('MICROSECONDS PER FOOT', 680, 20, 860, 50),
            ],
            'full_text': 'GAMMA RAY\nMICROSECONDS PER FOOT',
            'numbers': [],
            'suggestions': {},
        }
        client = web_app.app.test_client()

        with patch.object(web_app, 'detect_text_vision_api', return_value=detected):
            with patch.object(web_app, 'call_ai_auto_layout', return_value=None):
                response = client.post('/api/auto_layout', json={
                    'image': self.image_data,
                    'region': {
                        'left_px': 0,
                        'right_px': 1000,
                        'top_px': 0,
                        'bottom_px': 120,
                    },
                    'treat_region_as_header': True,
                })

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual(
            [track['name'] for track in payload['tracks']],
            ['GR', 'DT'],
        )
        self.assertEqual(payload['raw_layout']['fallback'], 'local_ocr_layout')

    def test_easyocr_populates_safe_metadata_and_rejects_noise(self):
        detected = {
            'raw': [
                _ocr_entry('GAMMA RAY', 180, 20, 300, 50, 0.9),
                _ocr_entry('WELL', 100, 70, 180, 90, 0.95),
                _ocr_entry('MENDEL ESTATE NO 1', 200, 70, 480, 90, 0.95),
                _ocr_entry('COUNTY', 100, 95, 200, 115, 0.99),
                _ocr_entry('~STATE', 220, 95, 320, 115, 0.99),
                _ocr_entry('API', 500, 70, 560, 90, 0.99),
                _ocr_entry('UNITS', 580, 70, 660, 90, 0.99),
            ],
            'full_text': '',
            'numbers': [],
            'suggestions': {},
            'engine': 'easyocr',
        }
        client = web_app.app.test_client()

        with patch.object(web_app, 'detect_text_vision_api', return_value=detected):
            with patch.object(web_app, 'call_ai_auto_layout', return_value=None):
                response = client.post('/api/auto_layout', json={
                    'image': self.image_data,
                    'region': {
                        'left_px': 0,
                        'right_px': 1000,
                        'top_px': 0,
                        'bottom_px': 120,
                    },
                    'treat_region_as_header': True,
                })

        self.assertEqual(response.status_code, 200)
        metadata = response.get_json()['header_metadata']
        self.assertEqual(metadata, {'well': 'MENDEL ESTATE NO 1'})


if __name__ == '__main__':
    unittest.main()
