import os
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import web_app


class _PaddlePage:
    def __init__(self, result):
        self.json = {'res': result}


class PaddleOcrTests(unittest.TestCase):
    def setUp(self):
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode('.png', image)
        self.assertTrue(ok)
        self.image_bytes = encoded.tobytes()

    def test_paddle_import_is_deferred_until_first_ocr_request(self):
        self.assertTrue(web_app.PADDLE_OCR_AVAILABLE)
        self.assertIsNone(web_app.PaddleOCR)

    def test_paddle_adapter_returns_common_ocr_payload(self):
        reader = Mock()
        reader.predict.return_value = [_PaddlePage({
            'rec_texts': ['GAMMA RAY', '0', '150'],
            'rec_scores': [0.98, 0.93, 0.94],
            'rec_polys': [
                [[10, 20], [110, 20], [110, 50], [10, 50]],
                [[20, 60], [40, 60], [40, 70], [20, 70]],
                [[20, 80], [60, 80], [60, 90], [20, 90]],
            ],
        })]

        with patch.object(web_app, '_get_paddle_ocr_reader', return_value=reader):
            payload = web_app._detect_text_paddleocr(
                self.image_bytes,
                preserve_detail=True,
            )

        self.assertEqual(payload['engine'], 'paddleocr')
        self.assertEqual([entry['text'] for entry in payload['raw']], ['GAMMA RAY', '0', '150'])
        self.assertAlmostEqual(payload['raw'][0]['confidence'], 0.98)
        self.assertEqual(payload['raw'][0]['vertices'][0], {'x': 10, 'y': 20})
        self.assertEqual([entry['value'] for entry in payload['numbers']], [0.0, 150.0])
        self.assertIn('GAMMA RAY', payload['full_text'])

    def test_local_provider_uses_paddle_without_calling_google(self):
        paddle_result = {
            'raw': [{'text': 'GR'}],
            'numbers': [],
            'suggestions': {},
            'engine': 'paddleocr',
        }

        with patch.dict(os.environ, {'OCR_PROVIDER': 'local'}):
            with patch.object(web_app, '_detect_text_paddleocr', return_value=paddle_result):
                with patch.object(web_app, '_detect_text_google_vision') as google:
                    with patch.object(web_app, '_detect_text_easyocr') as easy:
                        payload = web_app.detect_text_vision_api(self.image_bytes)

        self.assertIs(payload, paddle_result)
        google.assert_not_called()
        easy.assert_not_called()

    def test_local_provider_falls_back_to_easyocr(self):
        paddle_result = {
            'raw': [],
            'numbers': [],
            'suggestions': {},
            'engine': 'paddleocr',
        }
        easy_result = {
            'raw': [{'text': 'DT'}],
            'numbers': [],
            'suggestions': {},
            'engine': 'easyocr',
        }

        with patch.dict(os.environ, {'OCR_PROVIDER': 'local'}):
            with patch.object(web_app, '_detect_text_paddleocr', return_value=paddle_result):
                with patch.object(web_app, '_detect_text_easyocr', return_value=easy_result):
                    payload = web_app.detect_text_vision_api(self.image_bytes)

        self.assertIs(payload, easy_result)


if __name__ == '__main__':
    unittest.main()
