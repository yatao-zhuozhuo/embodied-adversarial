"""Regression checks for the v2 data-separation and reproducibility gates."""
import json
from pathlib import Path
import numpy as np
import pytest
from baseline.v2.comparison import robot_equal,pixel_match

V2=Path(__file__).resolve().parent


def test_robot_numeric_tolerance_rejects_real_motion():
    assert robot_equal({'xyz':[1.,2.,3.]},{'xyz':[1.+1e-9,2.,3.]})
    assert not robot_equal({'xyz':[1.,2.,3.]},{'xyz':[1.001,2.,3.]})
    assert not robot_equal({'xyz':[float('nan')]},{'xyz':[float('nan')]})
    assert not robot_equal({'xyz':[1.]},{'xyz':[1.,2.]})


def test_raster_tolerance_rejects_scene_changes():
    a=np.zeros((256,256,3),dtype=np.uint8);b=a.copy();b[0,0,0]=1
    assert pixel_match(a,b,'robocasa')
    b[0,0,0]=2
    assert not pixel_match(a,b,'robocasa')
    assert not pixel_match(a,np.ones_like(a)*2,'metaworld')


def test_manifest_complete_and_disjoint():
    audit=json.loads((V2/'overlap_audit.json').read_text())
    assert audit['status']=='passed' and not audit['canonical_or_definition_overlaps']
    manifest=V2/'manifests/all.json'
    if not manifest.exists():pytest.skip('Manifest not frozen yet; run again after freeze')
    rows=json.loads(manifest.read_text())['episodes']
    profiles=json.loads((V2/'configs/input_profiles.json').read_text())
    states=json.loads((V2/'manifests/initial_states.json').read_text())
    excluded={t['env_id'] for t in json.loads((V2/'training_exclusions.json').read_text())['tasks']}
    assert len({r['episode_id'] for r in rows})==len(rows)
    counts={}
    for row in rows:
        assert row['env_id'] not in excluded
        assert row['env_id'] in profiles and row['episode_id'] in states
        assert row['max_control_steps']==row['metadata']['native_horizon']
        counts[row['metadata']['task_id']]=counts.get(row['metadata']['task_id'],0)+1
        if row['env_id'] in ('openeta/maniskill_PickCube-v1-v0','openeta/maniskill_PickSingleYCB-v1-v0'):
            assert profiles[row['env_id']]['cameras'][0]['id']=='render'
    assert set(counts.values())=={3}
