import numpy as np
import pytest
from sim.envs.robocasa.direct_env import _canonical_counter_regions


def test_seeded_region_selection_survives_xml_element_order():
    left={'offset':[-.79,0.,.46],'size':[2.31,.65]}
    right={'offset':[1.43,0.,.46],'size':[1.03,.65]}
    a=_canonical_counter_regions({'geom_0':left,'geom_1':right})
    b=_canonical_counter_regions({'geom_0':right,'geom_1':left})
    for seed in (20321400,0,7,42):
        i=np.random.default_rng(seed).integers(2)
        assert list(a.values())[i]==list(b.values())[i]
    assert list(a.values())==[left,right]
    assert _canonical_counter_regions(a)==a


def test_region_geometry_is_preserved_and_unknown_ids_fail():
    r={'offset':[1,2,3],'size':[4,5]}
    assert _canonical_counter_regions({'geom_0':r})['geom_0'] is r
    assert _canonical_counter_regions({})=={}
    with pytest.raises(ValueError,match='identifiers'):
        _canonical_counter_regions({'named_surface':r})
