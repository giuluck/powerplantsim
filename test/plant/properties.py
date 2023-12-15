import unittest

import numpy as np
import pandas as pd

from ppsim import Plant
from ppsim.datatypes import Supplier, Machine, Storage, Edge, Customer
from test.datatypes.datatype import VARIANCE_1

PLANT_1 = Plant(horizon=24)
PLANT_1.add_supplier(name='sup', commodity='in', predictions=1.)

PLANT_2 = Plant(horizon=24)
PLANT_2.add_supplier(name='sup', commodity='in', predictions=1.)
PLANT_2.add_machine(name='mac', parents='sup', commodity='in', setpoint={'setpoint': [1.], 'out': [1.]})
PLANT_2.add_storage(name='sto', parents='mac', commodity='out')
PLANT_2.add_client(name='cli', parents=['mac', 'sto'], commodity='out', predictions=1.)

SERIES = np.ones(24)
SETPOINT = pd.DataFrame(data=[1.], columns=['out'], index=[1.])
SUPPLIER = Supplier(name='sup', commodity='in', _predictions=SERIES, _variance_fn=VARIANCE_1, _plant=PLANT_1)
MACHINE = Machine(
    name='mac',
    commodity='in',
    _setpoint=SETPOINT,
    discrete_setpoint=False,
    max_starting=None,
    cost=0.0,
    _plant=PLANT_1
)
STORAGE = Storage(
    name='sto',
    commodity='out',
    capacity=float('inf'),
    dissipation=0.0,
    charge_rate=float('inf'),
    discharge_rate=float('inf'),
    _plant=PLANT_1
)
CLIENT = Customer(name='cli', commodity='out', _predictions=SERIES, _variance_fn=VARIANCE_1, _plant=PLANT_1)

EDGE_1 = Edge(
    _source=SUPPLIER,
    _destination=MACHINE,
    commodity='in',
    min_flow=0.0,
    max_flow=float('inf'),
    integer=False,
    _plant=PLANT_1
)
EDGE_2 = Edge(
    _source=MACHINE,
    _destination=STORAGE,
    commodity='out',
    min_flow=0.0,
    max_flow=float('inf'),
    integer=False,
    _plant=PLANT_1
)
EDGE_3 = Edge(
    _source=MACHINE,
    _destination=CLIENT,
    commodity='out',
    min_flow=0.0,
    max_flow=float('inf'),
    integer=False,
    _plant=PLANT_1
)
EDGE_4 = Edge(
    _source=STORAGE,
    _destination=CLIENT,
    commodity='out',
    min_flow=0.0,
    max_flow=float('inf'),
    integer=False,
    _plant=PLANT_1
)

NEGATIVE_HORIZON_EXCEPTION = lambda v: f"The time horizon must be a strictly positive integer, got {v}"


class TestPlantProperties(unittest.TestCase):
    def test_init(self):
        """Tests an empty plant."""
        # create reference horizon [0, 1, ..., 23]
        horizon = [i for i in range(24)]
        # build five different plants with different input types and check that the horizon is like the reference
        tests = [24, horizon, np.array(horizon), pd.Series(horizon), pd.Index(horizon)]
        for hrz in tests:
            p = Plant(horizon=hrz)
            self.assertIsInstance(p.horizon, pd.Index, msg=f"Horizon should be of type pd.index, got {type(p.horizon)}")
            self.assertListEqual(list(p.horizon), horizon, msg=f"Horizon should be [0, ..., 23], got {list(p.horizon)}")
        # test sanity check for negative integer
        with self.assertRaises(AssertionError, msg="Null time horizon should raise exception") as e:
            Plant(horizon=0)
        self.assertEqual(
            str(e.exception),
            NEGATIVE_HORIZON_EXCEPTION(0),
            msg='Wrong exception message returned for null time horizon'
        )
        with self.assertRaises(AssertionError, msg="Null time horizon should raise exception") as e:
            Plant(horizon=-1)
        self.assertEqual(
            str(e.exception),
            NEGATIVE_HORIZON_EXCEPTION(-1),
            msg='Wrong exception message returned for null time horizon'
        )

    def test_commodities(self):
        """Tests that the commodities in the structure are correctly stored and returned."""
        self.assertSetEqual(PLANT_1.commodities, {'in'}, msg='Wrong commodities returned on plant 1')
        self.assertSetEqual(PLANT_2.commodities, {'in', 'out'}, msg='Wrong commodities returned on plant 2')

    def test_nodes(self):
        """Tests that the nodes in the structure are correctly stored and returned."""
        # test plant 1
        self.assertDictEqual(PLANT_1.suppliers, {'sup': SUPPLIER}, msg='Wrong supplier nodes returned on plant 1')
        self.assertDictEqual(PLANT_1.machines, {}, msg='Wrong machine nodes returned on plant 1')
        self.assertDictEqual(PLANT_1.storages, {}, msg='Wrong storage nodes returned on plant 1')
        self.assertDictEqual(PLANT_1.clients, {}, msg='Wrong client nodes returned on plant 1')
        self.assertDictEqual(
            PLANT_1.nodes(indexed=False),
            {'sup': SUPPLIER},
            msg='Wrong non-indexed nodes returned on plant 1'
        )
        self.assertDictEqual(
            PLANT_1.nodes(indexed=True),
            {'supplier': {'sup': SUPPLIER}},
            msg='Wrong indexed nodes returned on plant 1'
        )
        # test plant 2
        self.assertDictEqual(PLANT_2.suppliers, {'sup': SUPPLIER}, msg='Wrong supplier nodes returned on plant 2')
        self.assertDictEqual(PLANT_2.machines, {'mac': MACHINE}, msg='Wrong machine nodes returned on plant 2')
        self.assertDictEqual(PLANT_2.storages, {'sto': STORAGE}, msg='Wrong storage nodes returned on plant 2')
        self.assertDictEqual(PLANT_2.clients, {'cli': CLIENT}, msg='Wrong client nodes returned on plant 2')
        self.assertDictEqual(PLANT_2.nodes(indexed=False), {
            'sup': SUPPLIER,
            'mac': MACHINE,
            'sto': STORAGE,
            'cli': CLIENT
        }, msg='Wrong non-indexed nodes returned on plant 2')
        self.assertDictEqual(PLANT_2.nodes(indexed=True), {
            'supplier': {'sup': SUPPLIER},
            'machine': {'mac': MACHINE},
            'storage': {'sto': STORAGE},
            'client': {'cli': CLIENT}
        }, msg='Wrong indexed nodes returned on plant 2')

    def test_edges(self):
        """Tests that the edges in the structure are correctly stored and returned."""
        # test plant 1
        self.assertDictEqual(PLANT_1.edges(), {}, msg='Wrong edges returned on plant 1')
        # test plant 2
        self.assertDictEqual(PLANT_2.edges(), {
            ('sup', 'mac'): EDGE_1,
            ('mac', 'sto'): EDGE_2,
            ('mac', 'cli'): EDGE_3,
            ('sto', 'cli'): EDGE_4
        }, msg='Wrong edges returned on plant 2')
        # test edges filtering operations
        self.assertDictEqual(PLANT_2.edges(sources='mac'), {
            ('mac', 'sto'): EDGE_2,
            ('mac', 'cli'): EDGE_3,
        }, msg='Wrong edges returned on filtering by source')
        self.assertDictEqual(PLANT_2.edges(destinations='cli'), {
            ('mac', 'cli'): EDGE_3,
            ('sto', 'cli'): EDGE_4
        }, msg='Wrong edges returned on filtering by destination')
        self.assertDictEqual(PLANT_2.edges(commodities='out'), {
            ('mac', 'sto'): EDGE_2,
            ('mac', 'cli'): EDGE_3,
            ('sto', 'cli'): EDGE_4
        }, msg='Wrong edges returned on filtering by commodity')
        edges = PLANT_2.edges(
            sources=['sup', 'mac'],
            destinations=['mac', 'cli'],
            commodities=['in']
        )
        self.assertDictEqual(edges, {('sup', 'mac'): EDGE_1}, msg='Wrong edges returned on multiple filtering')

    def test_graph(self):
        """Tests that the returned graph is correct."""
        # test graph without attributes
        graph = PLANT_2.graph(attributes=False)
        self.assertDictEqual(
            {name: attr for name, attr in graph.nodes(data=True)},
            {'sup': {}, 'mac': {}, 'sto': {}, 'cli': {}},
            msg='Wrong nodes returned in graph without attributes'
        )
        self.assertDictEqual(
            {(sour, dest): attr for sour, dest, attr in graph.edges(data=True)},
            {('sup', 'mac'): {}, ('mac', 'sto'): {}, ('mac', 'cli'): {}, ('sto', 'cli'): {}},
            msg='Wrong edges returned in graph without attributes'
        )
        # test graph with attributes
        graph = PLANT_2.graph(attributes=True)
        self.assertDictEqual(
            {name: attr.keys() for name, attr in graph.nodes(data=True)},
            {
                'sup': SUPPLIER.dict.keys(),
                'mac': MACHINE.dict.keys(),
                'sto': STORAGE.dict.keys(),
                'cli': CLIENT.dict.keys()
            },
            msg='Wrong nodes returned in graph with attributes'
        )
        self.assertDictEqual(
            {(sour, dest): attr.keys() for sour, dest, attr in graph.edges(data=True)},
            {
                ('sup', 'mac'): EDGE_1.dict.keys(),
                ('mac', 'sto'): EDGE_2.dict.keys(),
                ('mac', 'cli'): EDGE_3.dict.keys(),
                ('sto', 'cli'): EDGE_4.dict.keys()
            },
            msg='Wrong edges returned in graph with attributes'
        )

    def test_copy(self):
        """Tests that the plant copy is correct and immutable."""
        # test correct copy
        p = PLANT_2.copy()
        self.assertSetEqual(p.commodities, {'in', 'out'}, msg='Wrong commodities copy returned')
        self.assertDictEqual(p.nodes(), {
            'sup': SUPPLIER,
            'mac': MACHINE,
            'sto': STORAGE,
            'cli': CLIENT
        }, msg='Wrong nodes copy returned')
        self.assertDictEqual(p.edges(), {
            ('sup', 'mac'): EDGE_1,
            ('mac', 'sto'): EDGE_2,
            ('mac', 'cli'): EDGE_3,
            ('sto', 'cli'): EDGE_4
        }, msg='Wrong edges copy returned')
        # test immutability
        p.add_storage(name='sto_2', commodity='out', parents='mac')
        self.assertIn('sto_2', p.nodes(), msg='New node not added to the copy')
        self.assertIn(('mac', 'sto_2'), p.edges(), msg='New edge not added to the copy')
        self.assertNotIn('sto_2', PLANT_2.nodes(), msg='New node must not be added to the original plant')
        self.assertNotIn(('mac', 'sto_2'), PLANT_2.edges(), msg='New edge must not be added to the original plant')
