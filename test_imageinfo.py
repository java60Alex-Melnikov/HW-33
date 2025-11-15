import unittest
from image_info import ImageInfo

class TestImageInfo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img_info = ImageInfo('street.jpg')
    
    def test_boxesClass(self):
        person_indices = self.img_info.boxesClass('person')
        self.assertIsInstance(person_indices, list)
        self.assertGreater(len(person_indices), 0)
        for idx in person_indices:
            info = self.img_info.boxInfo(idx)
            self.assertEqual(info[5], 'person')
    
    def test_boxInfo(self):
        info = self.img_info.boxInfo(0)
        self.assertIsInstance(info, tuple)
        self.assertEqual(len(info), 6)
        xmin, ymin, xmax, ymax, confidence, class_name = info
        self.assertGreater(xmax, xmin)
        self.assertGreater(ymax, ymin)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(class_name, str)
    
    def test_dataFrame(self):
        df = self.img_info.dataFrame()
        import pandas as pd
        self.assertIsInstance(df, pd.DataFrame)
        expected_columns = ['xmin', 'ymin', 'xmax', 'ymax', 'name', 'confidence']
        self.assertListEqual(list(df.columns), expected_columns)
        self.assertGreater(len(df), 0)
    
    def test_suitcaseHandbagPerson(self):
        result = self.img_info.suitcaseHandbagPerson(0.5)
        self.assertIsInstance(result, dict)
        for sh_idx, value in result.items():
            info = self.img_info.boxInfo(sh_idx)
            self.assertIn(info[5], ['suitcase', 'handbag'])
            if value is not None:
                self.assertIsInstance(value, tuple)
                person_idx = value[0]
                person_info = self.img_info.boxInfo(person_idx)
                self.assertEqual(person_info[5], 'person')

if __name__ == '__main__':
    unittest.main()