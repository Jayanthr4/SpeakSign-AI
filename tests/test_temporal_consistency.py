import unittest
from ml.recognition.temporal_consistency import analyze,evaluate

class TemporalTests(unittest.TestCase):
    def test_tracks_are_independent(self):
        def row(t,s,label):
            return {"track_id":t,"start_frame":s,"end_frame":s+31,"status":"predicted","prediction":label,"top3":[{"sign":label,"probability":0.8}]}
        timeline,summary=analyze([row(1,0,"hello"),row(1,16,"hello"),row(94,0,"hello")],60)
        self.assertEqual(len(timeline),1)
        self.assertEqual(timeline[0]["track_id"],1)
        self.assertEqual(summary["tracks"]["94"]["stable_candidate_intervals"],0)
    def test_evaluation(self):
        pred=[{"track_id":1,"candidate_sign":"hello","start_frame":0,"end_frame":47}]
        truth=[{"track_id":1,"sign":"hello","start_frame":0,"end_frame":47}]
        self.assertEqual(evaluate(pred,truth)["true_positives"],1)
        self.assertEqual(evaluate(pred,[])["false_positives"],1)

if __name__=="__main__":unittest.main()
