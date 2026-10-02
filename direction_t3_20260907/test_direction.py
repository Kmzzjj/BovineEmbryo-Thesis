import unittest
import pandas as pd
from direction_support import history_pick, select_centered

class SelectionTests(unittest.TestCase):
    def test_regular_matched_span(self):
        d=pd.DataFrame({'tmin':list(range(-100,101,10))})
        h=history_pick(d,0);c=select_centered(d,h,0)
        self.assertEqual(d.tmin.iloc[h].tolist(),[-100,-50,0])
        self.assertEqual(d.tmin.iloc[c].tolist(),[-50,0,50])
    def test_no_future_no_invention(self):
        d=pd.DataFrame({'tmin':[-100,-50,0]})
        c=select_centered(d,history_pick(d,0),0)
        self.assertEqual(d.tmin.iloc[c].tolist(),[-50,0])
    def test_no_past_raises(self):
        with self.assertRaises(ValueError):history_pick(pd.DataFrame({'tmin':[10,20]}),0)
    def test_tie_earlier(self):
        d=pd.DataFrame({'tmin':[-100,-50,0,40,60]})
        self.assertEqual(d.tmin.iloc[select_centered(d,history_pick(d,0),0)].tolist(),[-50,0,40])
    def test_short_history(self):
        d=pd.DataFrame({'tmin':[-10,0,10]})
        self.assertEqual(d.tmin.iloc[select_centered(d,history_pick(d,0),0)].tolist(),[-10,0,10])

if __name__=='__main__':unittest.main()
