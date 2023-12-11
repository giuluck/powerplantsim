import unittest

import numpy as np
import pandas as pd

from ppsim import Plant
from ppsim.datatypes import Purchaser, Customer

PLANT = Plant(horizon=3)
PLANT.add_supplier(name='sup', commodity='in', predictions=1.)
PLANT.add_machine(name='mac', parents='sup', commodity='in', setpoint={'setpoint': [1.], 'out': [1.]})
PLANT.add_storage(name='sto', parents='mac', commodity='out')
PLANT.add_client(name='cli', parents=['mac', 'sto'], commodity='out', predictions=1.)

NAME_CONFLICT_EXCEPTION = lambda k, n: f"There is already a {k} node '{n}', please use another identifier"
PARENT_COMMODITY_EXCEPTION = lambda n, i, o: f"Parent node '{n}' should return commodity '{i}', but it returns {o}"
PARENT_UNKNOWN_EXCEPTION = lambda n: f"Parent node '{n}' has not been added yet"
EMPTY_PARENTS_EXCEPTION = lambda k: f"{k} node must have at least one parent"


class TestPlantBuilding(unittest.TestCase):
    def test_add_supplier(self):
        p: Plant = PLANT.copy()
        # test node name conflicts
        for kind, name in {(k, n) for k, node in p.nodes(indexed=True).items() for n in node.keys()}:
            with self.assertRaises(AssertionError, msg="Name conflict for new supplier should raise exception") as e:
                p.add_supplier(name=name, commodity='in', predictions=1.)
            self.assertEqual(
                str(e.exception),
                NAME_CONFLICT_EXCEPTION(kind, name),
                msg='Wrong exception message returned for node name conflict'
            )
        # test supplier properties (and input)
        prices = [2., 2., 2.]
        tests = [2., prices, np.array(prices), pd.Series(prices)]
        for i, prc in enumerate(tests):
            s = p.add_supplier(name=f'sup_{i}', commodity='in', predictions=prc)
            self.assertEqual(s.name, f'sup_{i}', msg="Wrong name for supplier")
            self.assertIsNone(s.commodity_in, msg="Wrong input commodity for supplier")
            self.assertSetEqual(s.commodities_out, {'in'}, msg="Wrong output commodity for supplier")
            self.assertListEqual(list(s.prices.index), list(p.horizon), msg="Wrong prices index for supplier")
            self.assertListEqual(list(s.prices.values), prices, msg="Wrong prices for supplier")

    def test_add_client(self):
        p: Plant = PLANT.copy()
        # test node name conflicts
        for kind, name in {(k, n) for k, node in p.nodes(indexed=True).items() for n in node.keys()}:
            with self.assertRaises(AssertionError, msg="Name conflict for new client should raise exception") as e:
                p.add_client(name=name, commodity='out', parents='mac', predictions=1.)
            self.assertEqual(
                str(e.exception),
                NAME_CONFLICT_EXCEPTION(kind, name),
                msg='Wrong exception message returned for node name conflict'
            )
        # test parents
        with self.assertRaises(AssertionError, msg="Unknown parent for client should raise exception") as e:
            p.add_client(name='parent_unk', commodity='out', parents='unk', predictions=1.)
        self.assertEqual(
            str(e.exception),
            PARENT_UNKNOWN_EXCEPTION('unk'),
            msg='Wrong exception message returned for unknown parent'
        )
        with self.assertRaises(AssertionError, msg="Wrong parent commodity for client should raise exception") as e:
            p.add_client(name='parent_com', commodity='in', parents='mac', predictions=1.)
        self.assertEqual(
            str(e.exception),
            PARENT_COMMODITY_EXCEPTION('mac', 'in', {'out'}),
            msg='Wrong exception message returned for wrong parent commodity'
        )
        with self.assertRaises(AssertionError, msg="Empty parent list for client should raise exception") as e:
            p.add_client(name='parent_emp', commodity='out', parents=[], predictions=1.)
        self.assertEqual(
            str(e.exception),
            EMPTY_PARENTS_EXCEPTION('Client'),
            msg='Wrong exception message returned for empty parent list'
        )
        p.add_client(name='single', commodity='out', parents='mac', predictions=1.)
        edges = p.edges(destinations='single').set_index('source')['commodity'].to_dict()
        self.assertDictEqual(edges, {'mac': 'out'}, msg='Wrong edges stored for client with single parent')
        p.add_client(name='multiple', commodity='out', parents=['mac', 'sto'], predictions=1.)
        edges = p.edges(destinations='multiple').set_index('source')['commodity'].to_dict()
        self.assertDictEqual(edges, {'mac': 'out', 'sto': 'out'}, msg='Wrong edges stored for client with many parents')
        # test client properties (and input)
        demands = [2., 2., 2.]
        tests = [2., demands, np.array(demands), pd.Series(demands)]
        for i, dmn in enumerate(tests):
            for pur in [True, False]:
                klass, kind = (Purchaser, 'purchaser') if pur else (Customer, 'customer')
                c = p.add_client(name=f'{kind}_{i}', commodity='out', parents='mac', predictions=dmn, purchaser=pur)
                self.assertIsInstance(c, klass, msg=f"Wrong exposed type for {kind}")
                self.assertEqual(c.name, f'{kind}_{i}', msg=f"Wrong name for {kind}")
                self.assertEqual(c.commodity_in, 'out', msg=f"Wrong input commodity for {kind}")
                self.assertSetEqual(c.commodities_out, set(), msg=f"Wrong output commodity for {kind}")
                self.assertListEqual(list(c.predictions.index), list(p.horizon), msg=f"Wrong demands index for {kind}")
                self.assertListEqual(list(c.predictions.values), demands, msg=f"Wrong demands for {kind}")
        # test edge properties
        p.add_client(name='c', commodity='out', parents='mac', predictions=1.)
        e = p.edges(destinations='c')['edge'].values[0]
        self.assertEqual(e.source.name, 'mac', msg="Wrong source name stored in edge built from client")
        self.assertEqual(e.destination.name, 'c', msg="Wrong destination name stored in edge built from client")
        self.assertEqual(e.commodity, 'out', msg="Wrong commodity stored in edge built from client")
        self.assertEqual(e.min_flow, 0.0, msg="Wrong min flow stored in edge built from client")
        self.assertEqual(e.max_flow, float('inf'), msg="Wrong max flow stored in edge built from client")
        self.assertFalse(e.integer, msg="Wrong integer flag stored in edge built from client")

    def test_add_machine(self):
        p: Plant = PLANT.copy()
        # test node name conflicts
        for kind, name in {(k, n) for k, node in p.nodes(indexed=True).items() for n in node.keys()}:
            with self.assertRaises(AssertionError, msg="Name conflict for new machine should raise exception") as e:
                p.add_machine(name=name, commodity='in', parents='sup', setpoint={'setpoint': [1.], 'out': [1.]})
            self.assertEqual(
                str(e.exception),
                NAME_CONFLICT_EXCEPTION(kind, name),
                msg='Wrong exception message returned for node name conflict'
            )
        # test parents
        with self.assertRaises(AssertionError, msg="Unknown parent for machine should raise exception") as e:
            p.add_machine(name='parent_unk', commodity='out', parents='unk', setpoint={'setpoint': [1.], 'out': [1.]})
        self.assertEqual(
            str(e.exception),
            PARENT_UNKNOWN_EXCEPTION('unk'),
            msg='Wrong exception message returned for unknown parent'
        )
        with self.assertRaises(AssertionError, msg="Wrong parent commodity for machine should raise exception") as e:
            p.add_machine(name='parent_com', commodity='out', parents='sup', setpoint={'setpoint': [1.], 'out': [1.]})
        self.assertEqual(
            str(e.exception),
            PARENT_COMMODITY_EXCEPTION('sup', 'out', {'in'}),
            msg='Wrong exception message returned for wrong parent commodity'
        )
        with self.assertRaises(AssertionError, msg="Empty parent list for machine should raise exception") as e:
            p.add_machine(name='parent_emp', commodity='out', parents=[], setpoint={'setpoint': [1.], 'in': [1.]})
        self.assertEqual(
            str(e.exception),
            EMPTY_PARENTS_EXCEPTION('Machine'),
            msg='Wrong exception message returned for empty parent list'
        )
        p.add_machine(name='single', commodity='in', parents='sup', setpoint={'setpoint': [1.], 'out': [1.]})
        edges = p.edges(destinations='single').set_index('source')['commodity'].to_dict()
        self.assertDictEqual(edges, {'sup': 'in'}, msg='Wrong edges stored for machine with single parent')
        p.add_machine(name='multiple', commodity='out', parents=['mac', 'sto'], setpoint={'setpoint': [1.], 'in': [1.]})
        edges = p.edges(destinations='multiple').set_index('source')['commodity'].to_dict()
        self.assertDictEqual(
            edges,
            {'mac': 'out', 'sto': 'out'},
            msg='Wrong edges stored for machine with many parents'
        )
        # test machine properties (and input)
        m1 = p.add_machine(
            name='m1',
            commodity='in',
            parents='sup',
            setpoint={'setpoint': [3., 5.], 'out': [5., 9.]},
            discrete_setpoint=False,
            max_starting=None,
            cost=0.0
        )
        self.assertEqual(m1.setpoint.index.name, 'setpoint', msg="Wrong setpoint name stored for machine")
        self.assertListEqual(list(m1.setpoint.index), [3., 5.], msg="Wrong setpoint stored for machine")
        self.assertListEqual(list(m1.setpoint.columns), ['out'], msg="Wrong output flows names stored for machine")
        self.assertListEqual(list(m1.setpoint['out']), [5., 9.], msg="Wrong output flows stored for machine")
        self.assertFalse(m1.discrete_setpoint, msg="Wrong discrete setpoint flag stored for machine")
        self.assertIsNone(m1.max_starting, msg="Wrong max starting stored for machine")
        self.assertEqual(m1.cost, 0.0, msg="Wrong cost stored for machine")
        m2 = p.add_machine(
            name='m2',
            commodity='in',
            parents='sup',
            setpoint=pd.DataFrame([5., 9.], columns=['out'], index=[3., 5.]),
            discrete_setpoint=True,
            max_starting=(3, 24),
            cost=50.0
        )
        self.assertEqual(m2.setpoint.index.name, 'setpoint', msg="Wrong setpoint name stored for machine")
        self.assertListEqual(list(m2.setpoint.index), [3., 5.], msg="Wrong setpoint stored for machine")
        self.assertListEqual(list(m2.setpoint.columns), ['out'], msg="Wrong output flows names stored for machine")
        self.assertListEqual(list(m2.setpoint['out']), [5., 9.], msg="Wrong output flows stored for machine")
        self.assertTrue(m2.discrete_setpoint, msg="Wrong discrete setpoint flag stored for machine")
        self.assertTupleEqual(m2.max_starting, (3, 24), msg="Wrong max starting stored for machine")
        self.assertEqual(m2.cost, 50.0, msg="Wrong cost stored for machine")
        # test edge properties
        p.add_machine(
            name='m3',
            commodity='in',
            parents='sup',
            setpoint={'setpoint': [1.], 'out': [1.]},
            min_flow=30.0,
            max_flow=50.0,
            integer=True
        )
        e = p.edges(destinations='m3')['edge'].values[0]
        self.assertEqual(e.source.name, 'sup', msg="Wrong source name stored in edge built from machine")
        self.assertEqual(e.destination.name, 'm3', msg="Wrong destination name stored in edge built from machine")
        self.assertEqual(e.commodity, 'in', msg="Wrong commodity stored in edge built from machine")
        self.assertEqual(e.min_flow, 30.0, msg="Wrong min flow stored in edge built from machine")
        self.assertEqual(e.max_flow, 50.0, msg="Wrong max flow stored in edge built from machine")
        self.assertTrue(e.integer, msg="Wrong integer flag stored in edge built from machine")

    def test_add_storage(self):
        p: Plant = PLANT.copy()
        # test node name conflicts
        for kind, name in {(k, n) for k, node in p.nodes(indexed=True).items() for n in node.keys()}:
            with self.assertRaises(AssertionError, msg="Name conflict for new storage should raise exception") as e:
                p.add_storage(name=name, commodity='in', parents='mac')
            self.assertEqual(
                str(e.exception),
                NAME_CONFLICT_EXCEPTION(kind, name),
                msg='Wrong exception message returned for node name conflict'
            )
        # test parents
        with self.assertRaises(AssertionError, msg="Unknown parent for storage should raise exception") as e:
            p.add_storage(name='parent_unk', commodity='out', parents='unk')
        self.assertEqual(
            str(e.exception),
            PARENT_UNKNOWN_EXCEPTION('unk'),
            msg='Wrong exception message returned for unknown parent'
        )
        with self.assertRaises(AssertionError, msg="Wrong parent commodity for storage should raise exception") as e:
            p.add_storage(name='parent_com', commodity='in', parents='mac')
        self.assertEqual(
            str(e.exception),
            PARENT_COMMODITY_EXCEPTION('mac', 'in', {'out'}),
            msg='Wrong exception message returned for wrong parent commodity'
        )
        with self.assertRaises(AssertionError, msg="Empty parent list for storage should raise exception") as e:
            p.add_storage(name='parent_emp', commodity='out', parents=[])
        self.assertEqual(
            str(e.exception),
            EMPTY_PARENTS_EXCEPTION('Storage'),
            msg='Wrong exception message returned for empty parent list'
        )
        p.add_storage(name='single', commodity='out', parents='mac')
        edges = p.edges(destinations='single').set_index('source')['commodity'].to_dict()
        self.assertDictEqual(edges, {'mac': 'out'}, msg='Wrong edges stored for storage with single parent')
        p.add_storage(name='multiple', commodity='out', parents=['mac', 'sto'])
        edges = p.edges(destinations='multiple').set_index('source')['commodity'].to_dict()
        self.assertDictEqual(edges, {'mac': 'out', 'sto': 'out'}, msg='Wrong edges stored for client with many parents')
        # test storage properties and edge
        s = p.add_storage(
            name='s',
            commodity='out',
            parents='mac',
            capacity=25.0,
            dissipation=0.3,
            min_flow=20.0,
            max_flow=40.0,
            integer=True
        )
        e = p.edges(destinations='s')['edge'].values[0]
        self.assertEqual(s.capacity, 25.0, msg="Wrong capacity stored for storage")
        self.assertEqual(s.dissipation, 0.3, msg="Wrong dissipation stored for storage")
        self.assertEqual(e.source.name, 'mac', msg="Wrong source name stored in edge built from storage")
        self.assertEqual(e.destination.name, 's', msg="Wrong destination name stored in edge built from storage")
        self.assertEqual(e.commodity, 'out', msg="Wrong commodity stored in edge built from storage")
        self.assertEqual(e.min_flow, 20.0, msg="Wrong min flow stored in edge built from storage")
        self.assertEqual(e.max_flow, 40.0, msg="Wrong max flow stored in edge built from storage")
        self.assertTrue(e.integer, msg="Wrong integer flag stored in edge built from storage")
