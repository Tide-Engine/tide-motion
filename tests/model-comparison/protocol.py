"""Generate a fixed, self-authored paired Kimodo evaluation protocol (no datasets)."""
import hashlib
import json
from pathlib import Path
import sys

CASES = [
 ('walk','A person walks forward.'),
 ('backward','A person walks backwards.'),
 ('run','A person runs forward.'),
 ('rightkick','A person kicks forward with the right leg.'),
 ('leftkick','A person kicks forward with the left leg.'),
 ('swordkick','A person holding a sword kicks forward with the right leg.'),
 ('jump','A standing person jumps up and lands on both feet.'),
 ('crouch','A standing person crouches down.'),
 ('wave','A person waves with the right hand.'),
 ('turn','A person turns around.')]


def make(work, pose):
    pose_hash=hashlib.sha256(pose.read_bytes()).hexdigest()
    work.mkdir(parents=True,exist_ok=True)
    items=[]
    for block,(version,seeds) in enumerate([('v1.1',range(101,105)),('v1.2',range(101,105)),('v1.2',range(105,109)),('v1.1',range(105,109))]):
        actions=[]
        def add(case,seed,duration,**fields):
            name=f'{version}-{case}-{duration:g}-{seed}.glb'
            action=dict(action='generate',model=version,seed=seed,steps=50,device='gpu',guidance=2,constraintGuidance=2,output=name)
            action.update(fields)
            if 'prompts' not in fields:action['duration']=duration
            actions.append(action)
            items.append(dict(file=name,case=case,seed=seed,duration=duration,version=version,block=block+1))
        for seed in seeds:
            for case,prompt in CASES:
                for duration in [2.5,5]:add(case,seed,duration,prompt=prompt)
            for duration in [2.5,4]:add('posekick',seed,duration,prompt=CASES[3][1],startPose=str(pose),endPose=dict(file=str(pose),frame=0),placement='world')
            add('path',seed,3,prompt=CASES[0][1],path=[dict(frame=0,x=0,z=0),dict(frame=45,x=.4,z=.75),dict(frame=89,x=0,z=1.5)])
            add('walkwave',seed,3,prompts=[dict(prompt=CASES[0][1],duration=1.5),dict(prompt=CASES[8][1],duration=1.5)],startPose=str(pose))
            add('kicksequence',seed,3,prompts=[dict(prompt=CASES[3][1],duration=1.5),dict(prompt=CASES[4][1],duration=1.5)])
        (work/f'block{block+1}.json').write_text(json.dumps(dict(actions=actions),indent=2))
    (work/'protocol.json').write_text(json.dumps(dict(schema=1,steps=50,guidance=2,seeds=list(range(101,109)),durations=[2.5,5],pose=str(pose),poseSha256=pose_hash,cases=CASES,items=items),indent=2))


if __name__=='__main__':make(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
