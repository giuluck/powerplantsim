import logging

import numpy as np
import pyomo.environ as pyo
import pytest

from powerplantsim.datatypes import Customer, Purchaser, Supplier
from test.datatypes.test_datatype import TestDataType, SERIES_1, SERIES_2, VARIANCE_1, VARIANCE_2, PLANT, dummy_edge
from test.test_utils import SOLVER, SOLVER_NOT_AVAILABLE

CUSTOMER = Customer(
    name='c',
    commodity='c_com',
    _predictions=SERIES_1,
    _variance_fn=VARIANCE_1,
    _plant=PLANT
)

PURCHASER = Purchaser(
    name='p',
    commodity='p_com',
    _predictions=SERIES_1,
    _variance_fn=VARIANCE_1,
    _plant=PLANT
)

SUPPLIER = Supplier(
    name='s',
    commodity='s_com',
    _predictions=SERIES_1,
    _variance_fn=VARIANCE_1,
    _plant=PLANT
)

EXCEEDING_DEMAND_EXCEPTION = lambda n, d, f: f"Customer node '{n}' can accept at most {d} units, got {f}"


class TestExtremityNodes(TestDataType):

    def test_inputs(self):
        pass

    def test_hashing(self):
        # test customer
        c_equal = Customer(name='c', commodity='c_com_2', _predictions=SERIES_2, _variance_fn=VARIANCE_2, _plant=PLANT)
        self.assertEqual(CUSTOMER, c_equal, msg="Nodes with the same name should be considered equal")
        c_diff = Customer(name='cd', commodity='c_com', _predictions=SERIES_1, _variance_fn=VARIANCE_1, _plant=PLANT)
        self.assertNotEqual(CUSTOMER, c_diff, msg="Nodes with different names should be considered different")
        # test purchaser
        p_equal = Purchaser(name='p', commodity='p_com_2', _predictions=SERIES_2, _variance_fn=VARIANCE_2, _plant=PLANT)
        self.assertEqual(PURCHASER, p_equal, msg="Nodes with the same name should be considered equal")
        p_diff = Purchaser(name='pd', commodity='p_com', _predictions=SERIES_1, _variance_fn=VARIANCE_1, _plant=PLANT)
        self.assertNotEqual(CUSTOMER, p_diff, msg="Nodes with different names should be considered different")
        # test supplier
        s_equal = Supplier(name='s', commodity='s_com_2', _predictions=SERIES_2, _variance_fn=VARIANCE_2, _plant=PLANT)
        self.assertEqual(SUPPLIER, s_equal, msg="Nodes with the same name should be considered equal")
        s_diff = Supplier(name='sd', commodity='s_com', _predictions=SERIES_1, _variance_fn=VARIANCE_1, _plant=PLANT)
        self.assertNotEqual(SUPPLIER, s_diff, msg="Nodes with different names should be considered different")

    def test_properties(self):
        # test customer
        self.assertEqual(CUSTOMER.key, 'c', msg="Wrong customer key name stored")
        self.assertEqual(CUSTOMER.kind, 'customer', msg="Customer node is not labelled as client")
        self.assertSetEqual(CUSTOMER.commodities_in, {'c_com'}, msg="Wrong customer inputs stored")
        self.assertSetEqual(CUSTOMER.commodities_out, set(), msg="Wrong customer outputs stored")
        # test purchaser
        self.assertEqual(PURCHASER.key, 'p', msg="Wrong purchaser key name stored")
        self.assertEqual(PURCHASER.kind, 'purchaser', msg="Purchaser node is not labelled as client")
        self.assertSetEqual(PURCHASER.commodities_in, {'p_com'}, msg="Wrong purchaser inputs stored")
        self.assertSetEqual(PURCHASER.commodities_out, set(), msg="Wrong purchaser outputs stored")
        # test supplier
        self.assertEqual(SUPPLIER.key, 's', msg="Wrong supplier key name stored")
        self.assertEqual(SUPPLIER.kind, 'supplier', msg="Supplier node is not labelled as supplier")
        self.assertSetEqual(SUPPLIER.commodities_in, set(), msg="Wrong supplier inputs stored")
        self.assertSetEqual(SUPPLIER.commodities_out, {'s_com'}, msg="Wrong supplier outputs stored")

    def test_immutability(self):
        # test customer
        CUSTOMER.demands[0] = 5.0
        self.assertEqual(len(CUSTOMER.demands), 0, msg="Customer demands should be immutable")
        # test purchaser
        PURCHASER.prices[0] = 5.0
        self.assertEqual(len(PURCHASER.prices), 0, msg="Purchaser prices should be immutable")
        # test supplier
        SUPPLIER.prices[0] = 5.0
        self.assertEqual(len(SUPPLIER.prices), 0, msg="Supplier prices should be immutable")

    def test_dict(self):
        # test customer
        c_dict = CUSTOMER.dict
        self.assertDictEqual(c_dict, {
            'name': 'c',
            'kind': 'customer',
            'commodity': 'c_com',
            'current_demand': None
        }, msg='Wrong dictionary returned for customer')
        # test purchaser
        p_dict = PURCHASER.dict
        self.assertDictEqual(p_dict, {
            'name': 'p',
            'kind': 'purchaser',
            'commodity': 'p_com',
            'current_price': None
        }, msg='Wrong dictionary returned for purchaser')
        # test_supplier
        s_dict = SUPPLIER.dict
        self.assertDictEqual(s_dict, {
            'name': 's',
            'kind': 'supplier',
            'commodity': 's_com',
            'current_price': None,
        }, msg='Wrong dictionary returned for supplier')

    # noinspection DuplicatedCode
    def test_operation(self):
        # test customer
        c = CUSTOMER.copy()
        rng = np.random.default_rng(0)
        val = 3.0 + np.random.default_rng(0).normal()
        self.assertDictEqual(c.demands.to_dict(), {}, msg=f"Customer demands should be empty before the simulation")
        self.assertIsNone(c.current_demand, msg=f"Customer current demand should be None outside of the simulation")
        c.update(rng=rng, flows={}, states={})
        self.assertDictEqual(c.demands.to_dict(), {}, msg=f"Customer demands should be empty before step")
        self.assertEqual(c.current_demand, val, msg=f"Customer current demand should be stored after update")
        c.step(flows={}, states={})
        self.assertDictEqual(c.demands.to_dict(), {0: val}, msg=f"Customer demands should be filled after step")
        self.assertIsNone(c.current_demand, msg=f"Customer current demand should be None outside of the simulation")
        # test customer exception
        c = CUSTOMER.copy()
        rng = np.random.default_rng(0)
        c.update(rng=rng, flows={}, states={})
        with self.assertRaises(AssertionError, msg="Exceeding demand should raise exception") as e:
            c.step(states={}, flows={
                dummy_edge(source='input_1', destination=c, commodity='c_com'): 1.0,
                dummy_edge(source='input_2', destination=c, commodity='c_com'): 3.0
            })
        self.assertEqual(
            str(e.exception),
            EXCEEDING_DEMAND_EXCEPTION('c', val, 4.0),
            msg='Wrong exception message returned for exceeding demand on customer'
        )
        # test purchaser
        p = PURCHASER.copy()
        rng = np.random.default_rng(0)
        val = 3.0 + np.random.default_rng(0).normal()
        self.assertDictEqual(p.prices.to_dict(), {}, msg=f"Purchaser prices should be empty before the simulation")
        self.assertIsNone(p.current_price, msg=f"Purchaser current price should be None outside of the simulation")
        p.update(rng=rng, flows={}, states={})
        self.assertDictEqual(p.prices.to_dict(), {}, msg=f"Purchaser prices should be empty before step")
        self.assertEqual(p.current_price, val, msg=f"Purchaser current price should be stored after update")
        p.step(flows={}, states={})
        self.assertDictEqual(p.prices.to_dict(), {0: val}, msg=f"Purchaser prices should be filled after step")
        self.assertIsNone(p.current_price, msg=f"Purchaser current price should be None outside of the simulation")
        # test supplier
        s = SUPPLIER.copy()
        rng = np.random.default_rng(0)
        val = 3.0 + np.random.default_rng(0).normal()
        self.assertDictEqual(s.prices.to_dict(), {}, msg=f"Supplier prices should be empty before the simulation")
        self.assertIsNone(s.current_price, msg=f"Supplier current price should be None outside of the simulation")
        s.update(rng=rng, flows={}, states={})
        self.assertDictEqual(s.prices.to_dict(), {}, msg=f"Supplier prices should be empty before step")
        self.assertEqual(s.current_price, val, msg=f"Supplier current price should be stored after update")
        s.step(flows={}, states={})
        self.assertDictEqual(s.prices.to_dict(), {0: val}, msg=f"Supplier prices should be filled after step")
        self.assertIsNone(s.current_price, msg=f"Supplier current price should be None outside of the simulation")

    @pytest.mark.skipif(SOLVER_NOT_AVAILABLE, reason="Solver is absent")
    def test_pyomo(self):
        logging.getLogger('pyomo.core').setLevel(logging.CRITICAL)
        logging.getLogger('pyomo.common.numeric_types').setLevel(logging.CRITICAL)
        # test customer
        c = CUSTOMER.copy()
        rng = np.random.default_rng(0)
        c.update(rng=rng, flows={}, states={})
        model = c.to_pyomo(mutable=False)
        self.assertSetEqual(set(model.in_flows.keys()), {'c_com'}, msg="Wrong in_flows for customer block")
        self.assertSetEqual(set(model.out_flows.keys()), set(), msg="Wrong out_flows for customer block")
        self.assertEqual(
            model.current_demand.value,
            c.current_demand,
            msg="Wrong demand initialized in customer block"
        )
        results = pyo.SolverFactory(SOLVER).solve(model)
        self.assertEqual(
            results.solver.termination_condition,
            'optimal',
            msg="Customer block should always be feasible"
        )
        self.assertEqual(
            model.in_flows[c.commodity].value,
            c.current_demand,
            msg="Customer block should satisfy demand constraint"
        )
        with self.assertRaises(ValueError, msg="Accessing mutable customer parameters should raise exception"):
            pyo.value(c.to_pyomo(mutable=True).current_demand)
        # test purchaser
        p = PURCHASER.copy()
        rng = np.random.default_rng(0)
        p.update(rng=rng, flows={}, states={})
        model = p.to_pyomo(mutable=False)
        self.assertSetEqual(set(model.in_flows.keys()), {'p_com'}, msg="Wrong in_flows for purchaser block")
        self.assertSetEqual(set(model.out_flows.keys()), set(), msg="Wrong out_flows for purchaser block")
        self.assertEqual(
            model.current_price.value,
            p.current_price,
            msg="Wrong price initialized in purchaser block"
        )
        model.flows_value = pyo.Constraint(rule=model.in_flows[p.commodity] == 3.0)
        results = pyo.SolverFactory(SOLVER).solve(model)
        self.assertEqual(
            results.solver.termination_condition,
            'optimal',
            msg="Purchaser block should always be feasible"
        )
        self.assertEqual(
            pyo.value(model.cost),
            -p.current_price * 3.0,
            msg="Wrong cost computed for purchaser block"
        )
        with self.assertRaises(ValueError, msg="Accessing mutable purchaser parameters should raise exception"):
            pyo.value(p.to_pyomo(mutable=True).current_price)
        # test supplier
        s = SUPPLIER.copy()
        rng = np.random.default_rng(0)
        s.update(rng=rng, flows={}, states={})
        model = s.to_pyomo(mutable=False)
        self.assertSetEqual(set(model.in_flows.keys()), set(), msg="Wrong in_flows for supplier block")
        self.assertSetEqual(set(model.out_flows.keys()), {'s_com'}, msg="Wrong out_flows for supplier block")
        self.assertEqual(
            model.current_price.value,
            s.current_price,
            msg="Wrong price initialized in supplier block"
        )
        model.flows_value = pyo.Constraint(rule=model.out_flows[s.commodity] == 3.0)
        results = pyo.SolverFactory(SOLVER).solve(model)
        self.assertEqual(
            results.solver.termination_condition,
            'optimal',
            msg="Supplier block should always be feasible"
        )
        self.assertEqual(
            pyo.value(model.cost),
            s.current_price * 3.0,
            msg="Wrong cost computed for supplier block"
        )
        with self.assertRaises(ValueError, msg="Accessing mutable supplier parameters should raise exception"):
            pyo.value(s.to_pyomo(mutable=True).current_price)
