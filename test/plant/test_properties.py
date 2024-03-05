import unittest

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from powerplantsim.datatypes import Supplier, Machine, Storage, SingleEdge, Customer, Purchaser
from test.datatypes.test_datatype import VARIANCE_1
from test.test_utils import DummyPlant, SOLVER, IN_GITHUB_ACTIONS

PLANT_1 = DummyPlant(horizon=24)
PLANT_1.add_extremity(kind='supplier', name='sup', commodity='in', predictions=1.)

PLANT_2 = DummyPlant(horizon=24)
PLANT_2.add_extremity(kind='supplier', name='sup', commodity='in', predictions=1.)
PLANT_2.add_machine(
    name='mac',
    parents='sup',
    commodity='in',
    setpoint=[1.],
    inputs=[1.],
    outputs={'out': [1.]}
)
PLANT_2.add_storage(name='sto', parents=['mac'], commodity='out', capacity=100)
PLANT_2.add_extremity(kind='customer', name='cus', parents=['mac', 'sto'], commodity='out', predictions=1.)
PLANT_2.add_extremity(kind='purchaser', name='pur', parents='mac', commodity='out', predictions=1.)

SERIES = np.ones(24)
SETPOINT = pd.DataFrame(
    data=[[1., 1.]],
    columns=pd.MultiIndex.from_tuples([('input', 'in'), ('output', 'out')]),
    index=[1.]
)

SUPPLIER = Supplier(
    name='sup',
    commodity='in',
    _predictions=SERIES,
    _variance_fn=VARIANCE_1,
    _plant=PLANT_1
)
MACHINE = Machine(
    name='mac',
    _setpoint=SETPOINT,
    discrete_setpoint=False,
    max_starting=None,
    cost=0.0,
    _plant=PLANT_1
)
STORAGE = Storage(
    name='sto',
    commodity='out',
    capacity=100,
    dissipation=0.0,
    charge_rate=float('inf'),
    discharge_rate=float('inf'),
    _plant=PLANT_1
)
CUSTOMER = Customer(
    name='cus',
    commodity='out',
    _predictions=SERIES,
    _variance_fn=VARIANCE_1,
    _plant=PLANT_1
)
PURCHASER = Purchaser(
    name='pur',
    commodity='out',
    _predictions=SERIES,
    _variance_fn=VARIANCE_1,
    _plant=PLANT_1
)

EDGE_1 = SingleEdge(
    _source=SUPPLIER,
    _destination=MACHINE,
    commodity='in',
    min_flow=0.0,
    max_flow=float('inf'),
    _plant=PLANT_1
)
EDGE_2 = SingleEdge(
    _source=MACHINE,
    _destination=STORAGE,
    commodity='out',
    min_flow=0.0,
    max_flow=float('inf'),
    _plant=PLANT_1
)
EDGE_3 = SingleEdge(
    _source=MACHINE,
    _destination=CUSTOMER,
    commodity='out',
    min_flow=0.0,
    max_flow=float('inf'),
    _plant=PLANT_1
)
EDGE_4 = SingleEdge(
    _source=STORAGE,
    _destination=CUSTOMER,
    commodity='out',
    min_flow=0.0,
    max_flow=float('inf'),
    _plant=PLANT_1
)
EDGE_5 = SingleEdge(
    _source=MACHINE,
    _destination=PURCHASER,
    commodity='out',
    min_flow=0.0,
    max_flow=float('inf'),
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
            p = DummyPlant(horizon=hrz)
            self.assertIsInstance(p.horizon, pd.Index, msg=f"Horizon should be of type pd.index, got {type(p.horizon)}")
            self.assertListEqual(list(p.horizon), horizon, msg=f"Horizon should be [0, ..., 23], got {list(p.horizon)}")
        # test sanity check for negative integer
        with self.assertRaises(AssertionError, msg="Null time horizon should raise exception") as e:
            DummyPlant(horizon=0)
        self.assertEqual(
            str(e.exception),
            NEGATIVE_HORIZON_EXCEPTION(0),
            msg='Wrong exception message returned for null time horizon'
        )
        with self.assertRaises(AssertionError, msg="Null time horizon should raise exception") as e:
            DummyPlant(horizon=-1)
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
        self.assertDictEqual(PLANT_1.customers, {}, msg='Wrong client nodes returned on plant 1')
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
        self.assertDictEqual(PLANT_2.customers, {'cus': CUSTOMER}, msg='Wrong customer nodes returned on plant 2')
        self.assertDictEqual(PLANT_2.purchasers, {'pur': PURCHASER}, msg='Wrong purchaser nodes returned on plant 2')
        self.assertDictEqual(PLANT_2.nodes(indexed=False), {
            'sup': SUPPLIER,
            'mac': MACHINE,
            'sto': STORAGE,
            'cus': CUSTOMER,
            'pur': PURCHASER
        }, msg='Wrong non-indexed nodes returned on plant 2')
        self.assertDictEqual(PLANT_2.nodes(indexed=True), {
            'supplier': {'sup': SUPPLIER},
            'machine': {'mac': MACHINE},
            'storage': {'sto': STORAGE},
            'customer': {'cus': CUSTOMER},
            'purchaser': {'pur': PURCHASER}
        }, msg='Wrong indexed nodes returned on plant 2')

    def test_edges(self):
        """Tests that the edges in the structure are correctly stored and returned."""
        # test plant 1
        self.assertDictEqual(PLANT_1.edges(), {}, msg='Wrong edges returned on plant 1')
        # test plant 2
        self.assertDictEqual(PLANT_2.edges(), {
            ('sup', 'mac'): EDGE_1,
            ('mac', 'sto'): EDGE_2,
            ('mac', 'cus'): EDGE_3,
            ('sto', 'cus'): EDGE_4,
            ('mac', 'pur'): EDGE_5,
        }, msg='Wrong edges returned on plant 2')
        # test edges filtering operations
        self.assertDictEqual(PLANT_2.edges(sources='mac'), {
            ('mac', 'sto'): EDGE_2,
            ('mac', 'cus'): EDGE_3,
            ('mac', 'pur'): EDGE_5
        }, msg='Wrong edges returned on filtering by source')
        self.assertDictEqual(PLANT_2.edges(destinations='cus'), {
            ('mac', 'cus'): EDGE_3,
            ('sto', 'cus'): EDGE_4
        }, msg='Wrong edges returned on filtering by destination')
        self.assertDictEqual(PLANT_2.edges(commodities='out'), {
            ('mac', 'sto'): EDGE_2,
            ('mac', 'cus'): EDGE_3,
            ('sto', 'cus'): EDGE_4,
            ('mac', 'pur'): EDGE_5
        }, msg='Wrong edges returned on filtering by commodity')
        edges = PLANT_2.edges(
            sources=['sup', 'mac'],
            destinations=['mac', 'cus'],
            commodities=['in']
        )
        self.assertDictEqual(edges, {('sup', 'mac'): EDGE_1}, msg='Wrong edges returned on multiple filtering')

    def test_graph(self):
        """Tests that the returned graph is correct."""
        # test graph without attributes
        graph = PLANT_2.graph(attributes=False)
        self.assertDictEqual(
            {name: attr for name, attr in graph.nodes(data=True)},
            {'sup': {}, 'mac': {}, 'sto': {}, 'cus': {}, 'pur': {}},
            msg='Wrong nodes returned in graph without attributes'
        )
        self.assertDictEqual(
            {(sour, dest): attr for sour, dest, attr in graph.edges(data=True)},
            {('sup', 'mac'): {}, ('mac', 'sto'): {}, ('mac', 'cus'): {}, ('sto', 'cus'): {}, ('mac', 'pur'): {}},
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
                'cus': CUSTOMER.dict.keys(),
                'pur': PURCHASER.dict.keys(),
            },
            msg='Wrong nodes returned in graph with attributes'
        )
        self.assertDictEqual(
            {(sour, dest): attr.keys() for sour, dest, attr in graph.edges(data=True)},
            {
                ('sup', 'mac'): EDGE_1.dict.keys(),
                ('mac', 'sto'): EDGE_2.dict.keys(),
                ('mac', 'cus'): EDGE_3.dict.keys(),
                ('sto', 'cus'): EDGE_4.dict.keys(),
                ('mac', 'pur'): EDGE_5.dict.keys()
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
            'cus': CUSTOMER,
            'pur': PURCHASER
        }, msg='Wrong nodes copy returned')
        self.assertDictEqual(p.edges(), {
            ('sup', 'mac'): EDGE_1,
            ('mac', 'sto'): EDGE_2,
            ('mac', 'cus'): EDGE_3,
            ('sto', 'cus'): EDGE_4,
            ('mac', 'pur'): EDGE_5
        }, msg='Wrong edges copy returned')
        # test immutability
        p.add_storage(name='sto_2', commodity='out', parents='mac', capacity=10)
        self.assertIn('sto_2', p.nodes(), msg='New node not added to the copy')
        self.assertIn(('mac', 'sto_2'), p.edges(), msg='New edge not added to the copy')
        self.assertNotIn('sto_2', PLANT_2.nodes(), msg='New node must not be added to the original plant')
        self.assertNotIn(('mac', 'sto_2'), PLANT_2.edges(), msg='New edge must not be added to the original plant')

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Solver is absent in Github Actions")
    def test_pyomo(self):
        p = PLANT_2.copy()
        p._step += 1
        rng = np.random.default_rng(0)
        flows = {edge: 0.0 for edge in p.edges().values()}
        states = {machine: np.nan for machine in p.machines.values()}
        for node in p.nodes().values():
            node.update(rng=rng, flows=flows, states=states)
        for edge in p.edges().values():
            edge.update(rng=rng, flows=flows, states=states)
        model = p.to_pyomo(mutable=False)
        results = pyo.SolverFactory(SOLVER).solve(model)
        self.assertEqual(
            results.solver.termination_condition,
            'optimal',
            msg=f"Plant should be always feasible when demands can be satisfied"
        )
        edges = PLANT_2.edges()
        for name, node in PLANT_2.nodes().items():
            component = model.component(name)
            in_flows = {commodity: 0.0 for commodity in node.commodities_in}
            out_flows = {commodity: 0.0 for commodity in node.commodities_out}
            for (source, destination), edge in edges.items():
                if source == name:
                    out_flows[edge.commodity] += model.component(edge.name).flow.value
                if destination == name:
                    in_flows[edge.commodity] += model.component(edge.name).flow.value
            for commodity, flow in in_flows.items():
                self.assertAlmostEqual(
                    component.in_flows[commodity].value,
                    flow,
                    msg=f"Mismatch between input flow of commodity {commodity} and flows sums for node {name}"
                )
            for commodity, flow in out_flows.items():
                self.assertAlmostEqual(
                    component.out_flows[commodity].value,
                    flow,
                    msg=f"Mismatch between out flow of commodity {commodity} and flows sums for node {name}"
                )
