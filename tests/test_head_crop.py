import unittest
from ml.person_tracking.fullbody_landmarks import head_crop_box,face_quality

class HeadCropTests(unittest.TestCase):
    def test_head_box_stays_inside_person(self):
        pts=[[100.,90.,0.] for _ in range(33)]
        pts[7]=[82.,94.,0.]
        pts[8]=[118.,94.,0.]
        pts[11]=[75.,170.,0.]
        pts[12]=[125.,170.,0.]
        box=head_crop_box(pts,[50,40,150,250],200,300)
        self.assertIsNotNone(box)
        self.assertGreaterEqual(box[0],50)
        self.assertLessEqual(box[2],150)

    def test_missing_face_not_usable(self):
        self.assertFalse(face_quality([],200,300)["usable"])

if __name__=="__main__":
    unittest.main()
