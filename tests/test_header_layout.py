import unittest

from app.header_layout import infer_tracks_from_ocr


class HeaderLayoutInferenceTests(unittest.TestCase):
    def test_infers_named_tracks_and_standard_scales(self):
        tracks = infer_tracks_from_ocr([
            {'text': 'GAMMA RAY', 'x': 240, 'y': 50},
            {'text': 'RHOB', 'x': 760, 'y': 52},
        ], 1000)

        self.assertEqual([track['name'] for track in tracks], ['GR', 'RHOB'])
        self.assertEqual(tracks[0]['unit'], 'API')
        self.assertEqual(tracks[0]['scale_min'], 0.0)
        self.assertEqual(tracks[0]['scale_max'], 150.0)
        self.assertEqual(tracks[1]['unit'], 'G/CC')
        self.assertAlmostEqual(tracks[0]['right_x'], tracks[1]['left_x'])

    def test_recognizes_sonic_phrase(self):
        tracks = infer_tracks_from_ocr([
            {'text': 'INTERVAL TRANSIT TIME', 'x': 500, 'y': 80},
        ], 1000)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['name'], 'DT')
        self.assertEqual(tracks[0]['unit'], 'US/F')

    def test_recognizes_fuzzy_microseconds_per_foot_unit(self):
        tracks = infer_tracks_from_ocr([
            {'text': 'MCIOMCONDS FTR FOO1', 'x': 720, 'y': 80},
        ], 1000)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['name'], 'DT')

    def test_prefers_curve_header_over_inventory_reference(self):
        tracks = infer_tracks_from_ocr([
            {'text': 'G.R. CART NO.', 'x': 120, 'y': 200},
            {'text': 'GAMMA RAY API UNITS', 'x': 420, 'y': 40},
        ], 1000)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['name'], 'GR')
        self.assertGreater(tracks[0]['left_x'], 200)

    def test_returns_no_tracks_without_curve_labels(self):
        tracks = infer_tracks_from_ocr([
            {'text': 'COMPANY SKELLY OIL COMPANY', 'x': 500, 'y': 40},
            {'text': 'WELL MENDEL ESTATE', 'x': 500, 'y': 80},
        ], 1000)

        self.assertEqual(tracks, [])


if __name__ == '__main__':
    unittest.main()
