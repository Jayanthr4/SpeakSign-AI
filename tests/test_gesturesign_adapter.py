import unittest
import numpy as np
from ml.recognition.gesturesign_inference import to_holistic

class AdapterTests(unittest.TestCase):
    def test_coordinates_and_missing_face(self):
        pose=[[0.,0.,0.] for _ in range(33)]
        pose[15]=[20.,20.,0.]
        pose[16]=[80.,20.,0.]
        hand=[[20.,20.,0.] for _ in range(21)]
        frame={"hands":[{"landmarks_xy_pixels_z_relative":hand}],"pose":pose,"face":[]}
        array,cov=to_holistic([frame],100,50)
        self.assertEqual(array.shape,(1,543,3))
        self.assertAlmostEqual(float(array[0,468,0]),0.2)
        self.assertTrue(np.isnan(array[0,0,0]))
        self.assertEqual(cov["face"],0)
        self.assertEqual(cov["hands"],1)
        self.assertTrue(np.isnan(array[0,468,2]))

if __name__=="__main__":
    unittest.main()
