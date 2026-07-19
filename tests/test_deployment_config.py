import pathlib
import tomllib
import unittest


class DeploymentConfigTests(unittest.TestCase):
    def test_nixpacks_exposes_opencv_shared_libraries(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        config = tomllib.loads((root / 'nixpacks.toml').read_text(encoding='utf-8'))
        setup = config['phases']['setup']

        self.assertTrue({'libGL', 'glib'}.issubset(set(setup['nixPkgs'])))
        self.assertTrue({'libGL', 'glib'}.issubset(set(setup['nixLibs'])))


if __name__ == '__main__':
    unittest.main()
