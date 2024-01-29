import unittest

import numpy as np
import pandas as pd

from powerplantsim import Plant
from powerplantsim.datatypes import Purchaser, Customer

PLANT = Plant(horizon=3)
PLANT.add_extremity(kind='supplier', name='sup', commodity='in', predictions=1.)
PLANT.add_machine(name='mac', parents='sup', commodity='in', setpoint=[1.], inputs=[1.], outputs={'out': [1.]})
PLANT.add_storage(name='sto', parents='mac', commodity='out', capacity=100)
PLANT.add_extremity(kind='customer', name='cli', parents=['mac', 'sto'], commodity='out', predictions=1.)

NAME_CONFLICT_EXCEPTION = lambda k, n: f"There is already a {k} node '{n}', please use another identifier"
PARENT_UNKNOWN_EXCEPTION = lambda n: f"Parent node '{n}' has not been added yet"
EMPTY_PARENTS_EXCEPTION = lambda k: f"{k} node must have at least one parent"
HAS_PARENTS_SUPPLIER = lambda n: f"Supplier node {n} cannot accept parents"
NONE_PARENTS_CLIENT = lambda k, n: f"{k} node {n} must have parents"


class TestPlantBuilding(unittest.TestCase):
    def test_add_supplier(self):
        p: Plant = PLANT.copy()
        # test node name conflicts
        for kind, name in {(k, n) for k, node in p.nodes(indexed=True).items() for n in node.keys()}:
            with self.assertRaises(AssertionError, msg="Name conflict for new supplier should raise exception") as e:
                p.add_extremity(kind='supplier', name=name, commodity='in', predictions=1.)
            self.assertEqual(
                str(e.exception),
                NAME_CONFLICT_EXCEPTION(kind, name),
                msg='Wrong exception message returned for node name conflict'
            )
        # test parents
        with self.assertRaises(AssertionError, msg="Passing a parent for supplier should raise exception") as e:
            p.add_extremity(kind='supplier', name='parent_single', commodity='out', parents='parent', predictions=1.)
        self.assertEqual(
            str(e.exception),
            HAS_PARENTS_SUPPLIER('parent_single'),
            msg='Wrong exception message returned when passing a parent to a supplier'
        )
        with self.assertRaises(AssertionError, msg="Passing many parents for supplier should raise exception") as e:
            p.add_extremity(kind='supplier', name='parent_many', commodity='out', parents=['parent'], predictions=1.)
        self.assertEqual(
            str(e.exception),
            HAS_PARENTS_SUPPLIER('parent_many'),
            msg='Wrong exception message returned when passing many parents to a supplier'
        )
        # test supplier properties (and input)
        prices = [2., 2., 2.]
        tests = [2., prices, np.array(prices), pd.Series(prices)]
        for i, prc in enumerate(tests):
            s = p.add_extremity(kind='supplier', name=f'sup_{i}', commodity='in', predictions=prc)
            self.assertEqual(s.name, f'sup_{i}', msg="Wrong name for supplier")
            self.assertSetEqual(s.commodities_in, set(), msg="Wrong input commodity for supplier")
            self.assertSetEqual(s.commodities_out, {'in'}, msg="Wrong output commodity for supplier")

    def test_add_customer(self):
        p: Plant = PLANT.copy()
        # test node name conflicts
        for kind, name in {(k, n) for k, node in p.nodes(indexed=True).items() for n in node.keys()}:
            with self.assertRaises(AssertionError, msg="Name conflict for new customer should raise exception") as e:
                p.add_extremity(kind='customer', name=name, commodity='out', parents='mac', predictions=1.)
            self.assertEqual(
                str(e.exception),
                NAME_CONFLICT_EXCEPTION(kind, name),
                msg='Wrong exception message returned for node name conflict'
            )
        # test parents
        with self.assertRaises(AssertionError, msg="Passing parents=None for customer should raise exception") as e:
            p.add_extremity(kind='customer', name='parent_none', commodity='out', parents=None, predictions=1.)
        self.assertEqual(
            str(e.exception),
            NONE_PARENTS_CLIENT('Customer', 'parent_none'),
            msg='Wrong exception message returned when passing parents=None to a customer'
        )
        with self.assertRaises(AssertionError, msg="Unknown parent for customer should raise exception") as e:
            p.add_extremity(kind='customer', name='parent_unk', commodity='out', parents='unk', predictions=1.)
        self.assertEqual(
            str(e.exception),
            PARENT_UNKNOWN_EXCEPTION('unk'),
            msg='Wrong exception message returned for unknown parent'
        )
        with self.assertRaises(AssertionError, msg="Empty parent list for customer should raise exception") as e:
            p.add_extremity(kind='customer', name='parent_emp', commodity='out', parents=[], predictions=1.)
        self.assertEqual(
            str(e.exception),
            EMPTY_PARENTS_EXCEPTION('Customer'),
            msg='Wrong exception message returned for empty parent list'
        )
        p.add_extremity(kind='customer', name='single', commodity='out', parents='mac', predictions=1.)
        edges = {e.source: e.commodity for e in p.edges(destinations='single').values()}
        self.assertDictEqual(edges, {'mac': 'out'}, msg='Wrong edges stored for customer with single parent')
        p.add_extremity(kind='customer', name='multiple', commodity='out', parents=['mac', 'sto'], predictions=1.)
        edges = {e.source: e.commodity for e in p.edges(destinations='multiple').values()}
        self.assertDictEqual(
            edges,
            {'mac': 'out', 'sto': 'out'},
            msg='Wrong edges stored for customer with many parents'
        )
        # test customer properties (and input)
        demands = [2., 2., 2.]
        tests = [2., demands, np.array(demands), pd.Series(demands)]
        for i, dmn in enumerate(tests):
            c = p.add_extremity(kind='customer', name=f'customer_{i}', commodity='out', parents='mac', predictions=dmn)
            self.assertIsInstance(c, Customer, msg=f"Wrong type for customer")
            self.assertEqual(c.name, f'customer_{i}', msg=f"Wrong name for customer")
            self.assertSetEqual(c.commodities_in, {'out'}, msg=f"Wrong input commodity for customer")
            self.assertSetEqual(c.commodities_out, set(), msg=f"Wrong output commodity for customer")
        # test edge properties
        p.add_extremity(kind='customer', name='c', commodity='out', parents='mac', predictions=1.)
        e = list(p.edges(destinations='c').values())[0]
        self.assertEqual(e.source, 'mac', msg="Wrong source name stored in edge built from customer")
        self.assertEqual(e._destination.name, 'c', msg="Wrong destination name stored in edge built from customer")
        self.assertEqual(e.commodity, 'out', msg="Wrong commodity stored in edge built from customer")
        self.assertAlmostEqual(e.min_flow, 0.0, msg="Wrong min flow stored in edge built from customer")
        self.assertAlmostEqual(e.max_flow, float('inf'), msg="Wrong max flow stored in edge built from customer")

    def test_add_purchaser(self):
        p: Plant = PLANT.copy()
        # test node name conflicts
        for kind, name in {(k, n) for k, node in p.nodes(indexed=True).items() for n in node.keys()}:
            with self.assertRaises(AssertionError, msg="Name conflict for new purchaser should raise exception") as e:
                p.add_extremity(kind='purchaser', name=name, commodity='out', parents='mac', predictions=1.)
            self.assertEqual(
                str(e.exception),
                NAME_CONFLICT_EXCEPTION(kind, name),
                msg='Wrong exception message returned for node name conflict'
            )
        # test parents
        with self.assertRaises(AssertionError, msg="Passing parents=None for purchaser should raise exception") as e:
            p.add_extremity(kind='purchaser', name='parent_none', commodity='out', parents=None, predictions=1.)
        self.assertEqual(
            str(e.exception),
            NONE_PARENTS_CLIENT('Purchaser', 'parent_none'),
            msg='Wrong exception message returned when passing parents=None to a purchaser'
        )
        with self.assertRaises(AssertionError, msg="Unknown parent for purchaser should raise exception") as e:
            p.add_extremity(kind='purchaser', name='parent_unk', commodity='out', parents='unk', predictions=1.)
        self.assertEqual(
            str(e.exception),
            PARENT_UNKNOWN_EXCEPTION('unk'),
            msg='Wrong exception message returned for unknown parent'
        )
        with self.assertRaises(AssertionError, msg="Empty parent list for purchaser should raise exception") as e:
            p.add_extremity(kind='purchaser', name='parent_emp', commodity='out', parents=[], predictions=1.)
        self.assertEqual(
            str(e.exception),
            EMPTY_PARENTS_EXCEPTION('Purchaser'),
            msg='Wrong exception message returned for empty parent list'
        )
        p.add_extremity(kind='purchaser', name='single', commodity='out', parents='mac', predictions=1.)
        edges = {e.source: e.commodity for e in p.edges(destinations='single').values()}
        self.assertDictEqual(edges, {'mac': 'out'}, msg='Wrong edges stored for purchaser with single parent')
        p.add_extremity(kind='purchaser', name='multiple', commodity='out', parents=['mac', 'sto'], predictions=1.)
        edges = {e.source: e.commodity for e in p.edges(destinations='multiple').values()}
        self.assertDictEqual(
            edges,
            {'mac': 'out', 'sto': 'out'},
            msg='Wrong edges stored for purchaser with many parents'
        )
        # test purchaser properties (and input)
        demands = [2., 2., 2.]
        tests = [2., demands, np.array(demands), pd.Series(demands)]
        for i, dmn in enumerate(tests):
            c = p.add_extremity(
                kind='purchaser',
                name=f'purchaser_{i}',
                commodity='out',
                parents='mac',
                predictions=dmn
            )
            self.assertIsInstance(c, Purchaser, msg=f"Wrong type for purchaser")
            self.assertEqual(c.name, f'purchaser_{i}', msg=f"Wrong name for purchaser")
            self.assertSetEqual(c.commodities_in, {'out'}, msg=f"Wrong input commodity for purchaser")
            self.assertSetEqual(c.commodities_out, set(), msg=f"Wrong output commodity for purchaser")
        # test edge properties
        p.add_extremity(kind='purchaser', name='c', commodity='out', parents='mac', predictions=1.)
        e = list(p.edges(destinations='c').values())[0]
        self.assertEqual(e.source, 'mac', msg="Wrong source name stored in edge built from purchaser")
        self.assertEqual(e._destination.name, 'c', msg="Wrong destination name stored in edge built from purchaser")
        self.assertEqual(e.commodity, 'out', msg="Wrong commodity stored in edge built from purchaser")
        self.assertEqual(e.min_flow, 0.0, msg="Wrong min flow stored in edge built from purchaser")
        self.assertEqual(e.max_flow, float('inf'), msg="Wrong max flow stored in edge built from purchaser")

    def test_add_machine(self):
        p: Plant = PLANT.copy()
        # test node name conflicts
        for kind, name in {(k, n) for k, node in p.nodes(indexed=True).items() for n in node.keys()}:
            with self.assertRaises(AssertionError, msg="Name conflict for new machine should raise exception") as e:
                p.add_machine(
                    name=name,
                    parents='sup',
                    commodity='in',
                    setpoint=[1.],
                    inputs=[1.],
                    outputs={'out': [1.]}
                )
            self.assertEqual(
                str(e.exception),
                NAME_CONFLICT_EXCEPTION(kind, name),
                msg='Wrong exception message returned for node name conflict'
            )
        # test parents
        with self.assertRaises(AssertionError, msg="Unknown parent for machine should raise exception") as e:
            p.add_machine(
                name='parent_unk',
                parents='unk',
                commodity='in',
                setpoint=[1.],
                inputs=[1.],
                outputs={'out': [1.]}
            )
        self.assertEqual(
            str(e.exception),
            PARENT_UNKNOWN_EXCEPTION('unk'),
            msg='Wrong exception message returned for unknown parent'
        )
        with self.assertRaises(AssertionError, msg="Empty parent list for machine should raise exception") as e:
            p.add_machine(
                name='parent_emp',
                parents=[],
                commodity='in',
                setpoint=[1.],
                inputs=[1.],
                outputs={'out': [1.]}
            )
        self.assertEqual(
            str(e.exception),
            EMPTY_PARENTS_EXCEPTION('Machine'),
            msg='Wrong exception message returned for empty parent list'
        )
        p.add_machine(
            name='single',
            parents='sup',
            commodity='in',
            setpoint=[1.],
            inputs=[1.],
            outputs={'out': [1.]}
        )
        edges = {e.source: e.commodity for e in p.edges(destinations='single').values()}
        self.assertDictEqual(edges, {'sup': 'in'}, msg='Wrong edges stored for machine with single parent')
        p.add_machine(
            name='multiple',
            parents=['mac', 'sto'],
            commodity='out',
            setpoint=[1.],
            inputs=[1.],
            outputs={'in': [1.]}
        )
        edges = {e.source: e.commodity for e in p.edges(destinations='multiple').values()}
        self.assertDictEqual(
            edges,
            {'mac': 'out', 'sto': 'out'},
            msg='Wrong edges stored for machine with many parents'
        )
        # test machine properties (and input)
        m1 = p.add_machine(
            name='m1',
            parents='sup',
            commodity='in',
            setpoint=[0., 1.],
            inputs=[3., 5.],
            outputs={'out': [5., 9.]},
            discrete=False,
            max_starting=None,
            cost=0.0
        )
        self.assertEqual(m1.setpoint.index.name, 'setpoint', msg="Wrong setpoint name stored for machine")
        self.assertDictEqual(m1.setpoint.to_dict(), {
            ('input', 'in'): {0.0: 3.0, 1.0: 5.0},
            ('output', 'out'): {0.0: 5.0, 1.0: 9.0}
        }, msg="Wrong setpoint stored for machine")
        self.assertFalse(m1.discrete_setpoint, msg="Wrong discrete setpoint flag stored for machine")
        self.assertIsNone(m1.max_starting, msg="Wrong max starting stored for machine")
        self.assertEqual(m1.cost, 0.0, msg="Wrong cost stored for machine")
        # test edge properties
        p.add_machine(
            name='m3',
            parents='sup',
            commodity='in',
            setpoint=[1.],
            inputs=[1.],
            outputs={'out': [1.]},
            min_flow=30.0,
            max_flow=50.0
        )
        e = list(p.edges(destinations='m3').values())[0]
        self.assertEqual(e.source, 'sup', msg="Wrong source name stored in edge built from machine")
        self.assertEqual(e.destination, 'm3', msg="Wrong destination name stored in edge built from machine")
        self.assertEqual(e.commodity, 'in', msg="Wrong commodity stored in edge built from machine")
        self.assertEqual(e.min_flow, 30.0, msg="Wrong min flow stored in edge built from machine")
        self.assertEqual(e.max_flow, 50.0, msg="Wrong max flow stored in edge built from machine")

    def test_add_storage(self):
        p: Plant = PLANT.copy()
        # test node name conflicts
        for kind, name in {(k, n) for k, node in p.nodes(indexed=True).items() for n in node.keys()}:
            with self.assertRaises(AssertionError, msg="Name conflict for new storage should raise exception") as e:
                p.add_storage(name=name, commodity='in', parents='mac', capacity=100)
            self.assertEqual(
                str(e.exception),
                NAME_CONFLICT_EXCEPTION(kind, name),
                msg='Wrong exception message returned for node name conflict'
            )
        # test parents
        with self.assertRaises(AssertionError, msg="Unknown parent for storage should raise exception") as e:
            p.add_storage(name='parent_unk', commodity='out', parents='unk', capacity=100)
        self.assertEqual(
            str(e.exception),
            PARENT_UNKNOWN_EXCEPTION('unk'),
            msg='Wrong exception message returned for unknown parent'
        )
        with self.assertRaises(AssertionError, msg="Empty parent list for storage should raise exception") as e:
            p.add_storage(name='parent_emp', commodity='out', parents=[], capacity=100)
        self.assertEqual(
            str(e.exception),
            EMPTY_PARENTS_EXCEPTION('Storage'),
            msg='Wrong exception message returned for empty parent list'
        )
        p.add_storage(name='single', commodity='out', parents='mac', capacity=100)
        edges = {e.source: e.commodity for e in p.edges(destinations='single').values()}
        self.assertDictEqual(edges, {'mac': 'out'}, msg='Wrong edges stored for storage with single parent')
        p.add_storage(name='multiple', commodity='out', parents=['mac', 'sto'], capacity=100)
        edges = {e.source: e.commodity for e in p.edges(destinations='multiple').values()}
        self.assertDictEqual(edges, {'mac': 'out', 'sto': 'out'}, msg='Wrong edges stored for client with many parents')
        # test storage properties and edge
        s = p.add_storage(
            name='s',
            commodity='out',
            parents='mac',
            capacity=25.0,
            dissipation=0.3,
            min_flow=20.0,
            max_flow=40.0
        )
        e = list(p.edges(destinations='s').values())[0]
        self.assertEqual(s.capacity, 25.0, msg="Wrong capacity stored for storage")
        self.assertEqual(s.dissipation, 0.3, msg="Wrong dissipation stored for storage")
        self.assertEqual(e.source, 'mac', msg="Wrong source name stored in edge built from storage")
        self.assertEqual(e.destination, 's', msg="Wrong destination name stored in edge built from storage")
        self.assertEqual(e.commodity, 'out', msg="Wrong commodity stored in edge built from storage")
        self.assertEqual(e.min_flow, 20.0, msg="Wrong min flow stored in edge built from storage")
        self.assertEqual(e.max_flow, 40.0, msg="Wrong max flow stored in edge built from storage")
