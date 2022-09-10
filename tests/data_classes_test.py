from unittest import TestCase, main
from data_classes import SegyIO3D

SEGY3D_FILENAME = '/Users/ruslan/Downloads/Ichthys 3D seismic for fault competition.segy'
DATA = SegyIO3D(SEGY3D_FILENAME)

class SegyIO3DTest(TestCase):
    def test_vm(self):
        self.assertEqual(DATA.get_vm(), 593.064233398437)

    def test_get_n_ilines(self):
        self.assertEqual(DATA.get_n_ilines(), 2400)

    def test_get_n_xlines(self):
        self.assertEqual(DATA.get_n_xlines(), 1001)

    def test_get_n_zslices(self):
        self.assertEqual(DATA.get_n_zslices(), 1001)

    def test_get_sample_rate(self):
        self.assertEqual(DATA.get_sample_rate(), 4000.0)

if __name__ == '__main__':
    main()
