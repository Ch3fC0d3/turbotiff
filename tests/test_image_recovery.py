import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import web_app


class ImageRecoveryEndpointTests(unittest.TestCase):
    def setUp(self):
        image = np.full((16, 20, 3), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode('.jpg', image)
        self.assertTrue(ok)
        self.jpeg = encoded.tobytes()

    def test_restores_browser_jpeg_to_an_authorized_server_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            web_app.config, 'DATA_ROOT', temp_dir
        ):
            client = web_app.app.test_client()
            with client.session_transaction() as session:
                session['admin_override'] = True

            response = client.post(
                '/api/images/recover',
                data={'file': (io.BytesIO(self.jpeg), 'workspace-recovery.jpg')},
                content_type='multipart/form-data',
            )

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertTrue(payload['success'])
            filename = payload['image_path'].rsplit('/', 1)[-1]
            self.assertTrue((Path(temp_dir) / 'images' / filename).is_file())

            fetched = client.get(payload['image_path'])
            self.assertEqual(fetched.status_code, 200)
            self.assertEqual(fetched.data, self.jpeg)
            fetched.close()

    def test_rejects_non_jpeg_recovery_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            web_app.config, 'DATA_ROOT', temp_dir
        ):
            client = web_app.app.test_client()
            with client.session_transaction() as session:
                session['admin_override'] = True
            response = client.post(
                '/api/images/recover',
                data={'file': (io.BytesIO(b'not-a-jpeg'), 'bad.jpg')},
                content_type='multipart/form-data',
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn('must be JPEG', response.get_json()['error'])


if __name__ == '__main__':
    unittest.main()
