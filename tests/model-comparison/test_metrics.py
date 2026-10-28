"""Model-free checks for evaluation geometry, pairing and predeclared protocol."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol
import score


class Metrics(unittest.TestCase):
    def fixture(self,direction='forward',side=None):
        rows=[]
        for i in range(10):
            x=i*.1 if direction=='right' else 0;z=i*(-.1 if direction=='backward' else .1) if direction!='right' else 0
            f={'Hips':(x,1,z),'Head':(x,1.8,z),'LeftLeg':(x+.1,1,z),'RightLeg':(x-.1,1,z),
               'RightHand':(x,1,z),'LeftHand':(x,1,z)}
            if direction=='right':f.update(LeftLeg=(x,1,z-.1),RightLeg=(x,1,z+.1))
            for name in ['LeftFoot','RightFoot','LeftToeBase','RightToeBase']:f[name]=(i*.01,.01,0)
            if side:f[side+'Foot']=(0,.8,0)
            rows.append(f)
        return rows

    def evaluate(self,frames,case):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'clip.glb';p.with_suffix('.glb.json').write_text(json.dumps({'model':'nvidia/Kimodo-SOMA-RP-v1.1'}))
            item=dict(duration=len(frames)/30,version='v1.1',case=case)
            with patch.object(score,'positions',return_value=frames),patch.object(score,'angular_steps',return_value=[2,3]):return score.score(p,item)

    def test_heading_and_ground_motion(self):
        for direction,case in [('forward','walk'),('backward','backward'),('right','walk')]:
            row=self.evaluate(self.fixture(direction),case)
            self.assertTrue(row['success'])
            self.assertAlmostEqual(row['slide'],1)
            self.assertEqual(row['contactFraction'],1)
            self.assertEqual(row['penetrationPercent'],0)
        self.assertFalse(self.evaluate(self.fixture('forward'),'backward')['success'])

    def test_laterality_and_finite_rejection(self):
        self.assertTrue(self.evaluate(self.fixture(side='Right'),'rightkick')['success'])
        self.assertFalse(self.evaluate(self.fixture(side='Left'),'rightkick')['success'])
        frames=self.fixture();frames[2]['Hips']=(float('nan'),1,0)
        with self.assertRaises(AssertionError):self.evaluate(frames,'walk')

    def test_protocol_pairing(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'pose.glb').write_bytes(b'authored test placeholder')
            protocol.make(Path(d),Path(d)/'pose.glb')
            doc=json.loads((Path(d)/'protocol.json').read_text());items=doc['items']
            self.assertEqual(len(items),400)
            keys=[(r['case'],r['seed'],r['duration']) for r in items]
            self.assertTrue(all(keys.count(k)==2 for k in set(keys)))
        self.assertEqual(score.interval([0]*8),[0,0])


if __name__=='__main__':unittest.main()
