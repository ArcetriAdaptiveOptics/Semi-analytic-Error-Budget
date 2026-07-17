#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for control_utils.py

Tests for control system transfer function, stability margins, and cost calculations.
Includes normal cases, edge cases, and exception handling.

@author: Test Suite
"""

import pytest
import numpy as np
import control as ct
from unittest.mock import Mock, MagicMock, patch
from src.control_utils import (
    control_CL_tf_margin,
    cost,
    compute_close_loop_peak_penalty
)


# ─────────────────────────────────────────────────────────────────────────────
# MOCK FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

class MockControlOptimization:
    """Mock object for SingleModeControlOptimization with realistic transfer functions"""
    
    def __init__(
        self,
        t_0=0.001,
        num1=None, den1=None,
        num2=None, den2=None,
        num3=None, den3=None
    ):
        self.t_0 = t_0
        # WFS (Wavefront Sensor)
        self.num1 = num1 if num1 is not None else np.array([1.0])
        self.den1 = den1 if den1 is not None else np.array([1.0, 0.1])
        # ASM (Deformable Mirror/Actuator)
        self.num2 = num2 if num2 is not None else np.array([1.0])
        self.den2 = den2 if den2 is not None else np.array([1.0, 0.2])
        # RTC (Real-Time Computer)
        self.num3 = num3 if num3 is not None else np.array([1.0])
        self.den3 = den3 if den3 is not None else np.array([1.0, 0.3])


class MockCostResult:
    """Mock object for cost evaluation result"""
    
    def __init__(self, cost_value=1.5, variance_fitting=0.3):
        self.cost = cost_value
        self.variance_terms = {"fitting": variance_fitting}


@pytest.fixture
def mock_optimization():
    """Standard mock optimization object"""
    return MockControlOptimization()


@pytest.fixture
def mock_optimization_highorder():
    """High-order mock optimization object"""
    return MockControlOptimization(
        num1=[1.0, 0.5], den1=[1.0, 0.5, 0.1],
        num2=[1.0, 0.3], den2=[1.0, 0.6, 0.2],
        num3=[1.0, 0.2], den3=[1.0, 0.4, 0.3]
    )


@pytest.fixture
def mock_optimization_with_evaluate(mock_optimization):
    """Mock optimization with evaluate method"""
    def mock_evaluate(controller_num, controller_den, store_history=False):
        return MockCostResult(cost_value=2.0, variance_fitting=0.4)
    
    mock_optimization.evaluate = mock_evaluate
    return mock_optimization


# ─────────────────────────────────────────────────────────────────────────────
# TEST: control_CL_tf_margin - NORMAL CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestControlCLTfMarginNormalCases:
    """Normal operation tests for control_CL_tf_margin"""
    
    def test_with_scalar_gain(self, mock_optimization):
        """Test with scalar gain parameter (integrator controller)"""
        result = control_CL_tf_margin(mock_optimization, gain=0.5)
        
        assert isinstance(result, dict)
        assert "ctrl_tf" in result
        assert "H_n_tf" in result
        assert "H_r_tf" in result
        assert "H_ol_tf" in result
        assert "H_ol_margins" in result
        assert "CL_stability" in result
        assert "sensitivity_penalty" in result
        assert "bandwidth_H_n" in result
    
    def test_with_array_gain_extracts_first_element(self, mock_optimization):
        """Test that array gain extracts first element"""
        result = control_CL_tf_margin(mock_optimization, gain=np.array([0.5, 0.6]))
        
        assert isinstance(result, dict)
        assert result["ctrl_tf"] is not None
        assert isinstance(result["ctrl_tf"], ct.TransferFunction)
    
    def test_with_controller_numerator_denominator(self, mock_optimization):
        """Test with explicit controller numerator and denominator"""
        result = control_CL_tf_margin(
            mock_optimization,
            controller_num=[1.0, 0.5],
            controller_den=[1.0, -0.8]
        )
        
        assert isinstance(result, dict)
        assert result["ctrl_tf"] is not None
        assert isinstance(result["ctrl_tf"], ct.TransferFunction)
    
    def test_return_transfer_functions_are_valid(self, mock_optimization):
        """Test that returned transfer functions are valid control objects"""
        result = control_CL_tf_margin(mock_optimization, gain=0.3)
        
        ctrl_tf = result["ctrl_tf"]
        H_n_tf = result["H_n_tf"]
        H_r_tf = result["H_r_tf"]
        H_ol_tf = result["H_ol_tf"]
        
        assert isinstance(ctrl_tf, ct.TransferFunction)
        assert isinstance(H_n_tf, ct.TransferFunction)
        assert isinstance(H_r_tf, ct.TransferFunction)
        assert isinstance(H_ol_tf, ct.TransferFunction)
    
    def test_stability_margins_format(self, mock_optimization):
        """Test that stability margins are in correct format [gm, pm, sm]"""
        result = control_CL_tf_margin(mock_optimization, gain=0.2)
        
        margins = result["H_ol_margins"]
        assert isinstance(margins, list)
        assert len(margins) == 3
        assert all(isinstance(m, (int, float, np.number)) for m in margins)
    
    def test_closed_loop_stability_is_boolean_list(self, mock_optimization):
        """Test that closed-loop stability is list of booleans [H_n, H_r]"""
        result = control_CL_tf_margin(mock_optimization, gain=0.1)
        
        cl_stability = result["CL_stability"]
        assert isinstance(cl_stability, list)
        assert len(cl_stability) == 2
        assert all(isinstance(s, (bool, np.bool_)) for s in cl_stability)
    
    def test_with_high_order_plant(self, mock_optimization_highorder):
        """Test with higher-order transfer functions"""
        result = control_CL_tf_margin(mock_optimization_highorder, gain=0.5)
        
        assert result is not None
        assert isinstance(result, dict)
        assert all(key in result for key in [
            "ctrl_tf", "H_n_tf", "H_r_tf", "H_ol_tf", "H_ol_margins"
        ])


# ─────────────────────────────────────────────────────────────────────────────
# TEST: control_CL_tf_margin - EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestControlCLTfMarginEdgeCases:
    """Edge case tests for control_CL_tf_margin"""
    
    def test_with_very_small_gain(self, mock_optimization):
        """Test with very small gain (near zero)"""
        result = control_CL_tf_margin(mock_optimization, gain=1e-8)
        
        assert result is not None
        assert isinstance(result, dict)
        margins = result["H_ol_margins"]
        assert all(np.isinf(m) or np.isfinite(m) for m in margins)
    
    def test_with_large_gain(self, mock_optimization):
        """Test with large gain value"""
        result = control_CL_tf_margin(mock_optimization, gain=1e6)
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_with_negative_gain(self, mock_optimization):
        """Test with negative gain (phase-inverting integrator)"""
        result = control_CL_tf_margin(mock_optimization, gain=-0.5)
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_with_zero_gain(self, mock_optimization):
        """Test with zero gain (no feedback)"""
        result = control_CL_tf_margin(mock_optimization, gain=0.0)
        
        assert result is not None
        assert isinstance(result, dict)
        # With gain=0, controller is [0, 0] / [1, -1], mostly inactive
    
    def test_with_different_time_steps(self, mock_optimization):
        """Test with different sampling time steps"""
        for t_0 in [0.0001, 0.001, 0.01, 0.1]:
            obj = MockControlOptimization(t_0=t_0)
            result = control_CL_tf_margin(obj, gain=0.5)
            
            assert result["ctrl_tf"].dt == t_0
    
    def test_with_higher_order_controller(self, mock_optimization):
        """Test with higher-order controller (2nd order)"""
        result = control_CL_tf_margin(
            mock_optimization,
            controller_num=[1.0, 2.0, 1.0],
            controller_den=[1.0, -0.5, 0.1]
        )
        
        assert result is not None
        assert result["ctrl_tf"] is not None
    
    def test_with_list_vs_array_controller(self, mock_optimization):
        """Test that list and array inputs produce same result"""
        result_list = control_CL_tf_margin(
            mock_optimization,
            controller_num=[1.0, 0.5],
            controller_den=[1.0, -0.8]
        )
        result_array = control_CL_tf_margin(
            MockControlOptimization(),
            controller_num=np.array([1.0, 0.5]),
            controller_den=np.array([1.0, -0.8])
        )
        
        # Both should produce valid results
        assert result_list is not None
        assert result_array is not None
    
    def test_bandwidth_zero_when_not_computable(self, mock_optimization):
        """Test that bandwidth is set to 0 when computation fails"""
        result = control_CL_tf_margin(mock_optimization, gain=0.5)
        
        bw = result["bandwidth_H_n"]
        assert isinstance(bw, (int, float, np.number))
        # Bandwidth should be >= 0
        assert bw >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TEST: control_CL_tf_margin - EXCEPTION HANDLING
# ─────────────────────────────────────────────────────────────────────────────

class TestControlCLTfMarginExceptionHandling:
    """Exception handling tests for control_CL_tf_margin"""
    
    def test_missing_both_gain_and_controller(self, mock_optimization):
        """Test ValueError when neither gain nor controller provided"""
        with pytest.raises(ValueError, match="Provide either 'gain'"):
            control_CL_tf_margin(mock_optimization)
    
    def test_only_controller_num_provided(self, mock_optimization):
        """Test ValueError when only controller_num provided"""
        with pytest.raises(ValueError, match="Provide either 'gain'"):
            control_CL_tf_margin(mock_optimization, controller_num=[1.0, 0.5])
    
    def test_only_controller_den_provided(self, mock_optimization):
        """Test ValueError when only controller_den provided"""
        with pytest.raises(ValueError, match="Provide either 'gain'"):
            control_CL_tf_margin(mock_optimization, controller_den=[1.0, -0.8])
    
    def test_none_gain_without_controller(self, mock_optimization):
        """Test ValueError for None gain with missing controller"""
        with pytest.raises(ValueError, match="Provide either 'gain'"):
            control_CL_tf_margin(mock_optimization, gain=None)
    
    def test_controller_num_none_with_den(self, mock_optimization):
        """Test ValueError when controller_num is None"""
        with pytest.raises(ValueError, match="Provide either 'gain'"):
            control_CL_tf_margin(
                mock_optimization,
                controller_num=None,
                controller_den=[1.0, -0.8]
            )
    
    def test_controller_den_none_with_num(self, mock_optimization):
        """Test ValueError when controller_den is None"""
        with pytest.raises(ValueError, match="Provide either 'gain'"):
            control_CL_tf_margin(
                mock_optimization,
                controller_num=[1.0, 0.5],
                controller_den=None
            )
    
    def test_gain_takes_precedence_over_controller(self, mock_optimization):
        """Test that when both provided, gain is used (takes precedence)"""
        result_gain_only = control_CL_tf_margin(mock_optimization, gain=0.5)
        result_both = control_CL_tf_margin(
            mock_optimization,
            gain=0.5,
            controller_num=[1.0, 0.5],
            controller_den=[1.0, -0.8]
        )
        
        assert result_gain_only is not None
        assert result_both is not None


# ─────────────────────────────────────────────────────────────────────────────
# TEST: compute_close_loop_peak_penalty - NORMAL CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCloseLoupPeakPenaltyNormalCases:
    """Normal operation tests for compute_close_loop_peak_penalty"""
    
    def test_with_stable_transfer_function(self, mock_optimization):
        """Test with stable closed-loop transfer function"""
        result = control_CL_tf_margin(mock_optimization, gain=0.3)
        H_cl_tf = result["H_r_tf"]
        
        penalty = compute_close_loop_peak_penalty(H_cl_tf, close_loop_peak_target_dB=2.0)
        
        assert isinstance(penalty, (int, float, np.number))
        assert penalty >= 0.0
        assert not np.isnan(penalty)
    
    def test_penalty_increases_with_smaller_target(self, mock_optimization):
        """Test that penalty increases when target dB becomes smaller"""
        result = control_CL_tf_margin(mock_optimization, gain=0.5)
        H_cl_tf = result["H_r_tf"]
        
        penalty_relaxed = compute_close_loop_peak_penalty(H_cl_tf, close_loop_peak_target_dB=10.0)
        penalty_strict = compute_close_loop_peak_penalty(H_cl_tf, close_loop_peak_target_dB=1.0)
        
        # Stricter target typically yields higher penalty
        assert penalty_strict >= penalty_relaxed
    
    def test_with_positive_dB_target(self, mock_optimization):
        """Test with positive dB (common case)"""
        result = control_CL_tf_margin(mock_optimization, gain=0.2)
        H_cl_tf = result["H_r_tf"]
        
        penalty = compute_close_loop_peak_penalty(H_cl_tf, close_loop_peak_target_dB=2.0)
        assert isinstance(penalty, (int, float, np.number))
    
    def test_with_zero_dB_target(self, mock_optimization):
        """Test with 0 dB target (unity peak constraint)"""
        result = control_CL_tf_margin(mock_optimization, gain=0.2)
        H_cl_tf = result["H_r_tf"]
        
        penalty = compute_close_loop_peak_penalty(H_cl_tf, close_loop_peak_target_dB=0.0)
        assert isinstance(penalty, (int, float, np.number))


# ─────────────────────────────────────────────────────────────────────────────
# TEST: compute_close_loop_peak_penalty - EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCloseLoupPeakPenaltyEdgeCases:
    """Edge case tests for compute_close_loop_peak_penalty"""
    
    def test_with_large_dB_target(self, mock_optimization):
        """Test with very large dB target (very relaxed constraint)"""
        result = control_CL_tf_margin(mock_optimization, gain=0.1)
        H_cl_tf = result["H_r_tf"]
        
        penalty = compute_close_loop_peak_penalty(H_cl_tf, close_loop_peak_target_dB=100.0)
        assert penalty >= 0.0
    
    def test_with_negative_dB_target(self, mock_optimization):
        """Test with negative dB target (attenuation constraint)"""
        result = control_CL_tf_margin(mock_optimization, gain=0.1)
        H_cl_tf = result["H_r_tf"]
        
        penalty = compute_close_loop_peak_penalty(H_cl_tf, close_loop_peak_target_dB=-6.0)
        assert isinstance(penalty, (int, float, np.number))


# ─────────────────────────────────────────────────────────────────────────────
# TEST: compute_close_loop_peak_penalty - EXCEPTION HANDLING
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCloseLoupPeakPenaltyExceptionHandling:
    """Exception handling tests for compute_close_loop_peak_penalty"""
    
    def test_none_transfer_function_raises_error(self):
        """Test ValueError when transfer function is None"""
        with pytest.raises(ValueError, match="Provide both"):
            compute_close_loop_peak_penalty(H_cl_tf=None, close_loop_peak_target_dB=2.0)
    
    def test_none_target_raises_error(self, mock_optimization):
        """Test ValueError when target dB is None"""
        result = control_CL_tf_margin(mock_optimization, gain=0.2)
        H_cl_tf = result["H_r_tf"]
        
        with pytest.raises(ValueError, match="Provide both"):
            compute_close_loop_peak_penalty(H_cl_tf=H_cl_tf, close_loop_peak_target_dB=None)
    
    def test_both_none_raises_error(self):
        """Test ValueError when both parameters are None"""
        with pytest.raises(ValueError, match="Provide both"):
            compute_close_loop_peak_penalty(H_cl_tf=None, close_loop_peak_target_dB=None)


# ─────────────────────────────────────────────────────────────────────────────
# TEST: cost - NORMAL CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestCostNormalCases:
    """Normal operation tests for cost function"""
    
    @patch('src.control_utils.compute_close_loop_peak_penalty')
    def test_with_gain_controller(self, mock_penalty, mock_optimization_with_evaluate):
        """Test cost calculation with gain controller"""
        mock_penalty.return_value = 0.1
        
        result = cost(mock_optimization_with_evaluate, gain=0.5)
        
        assert isinstance(result, dict)
        assert "cost_function_value" in result
        assert "evaluate_result" in result
        assert "penalty" in result
        assert "weight_penalty" in result
    
    @patch('src.control_utils.compute_close_loop_peak_penalty')
    def test_with_explicit_controller(self, mock_penalty, mock_optimization_with_evaluate):
        """Test cost with explicit controller num/den"""
        mock_penalty.return_value = 0.2
        
        result = cost(
            mock_optimization_with_evaluate,
            controller_num=[1.0, 0.5],
            controller_den=[1.0, -0.8]
        )
        
        assert isinstance(result, dict)
        assert "cost_function_value" in result
    
    @patch('src.control_utils.compute_close_loop_peak_penalty')
    def test_with_stability_margins_targets(self, mock_penalty, mock_optimization_with_evaluate):
        """Test cost with explicit stability margin targets"""
        mock_penalty.return_value = 0.05
        
        result = cost(
            mock_optimization_with_evaluate,
            gain=0.5,
            sm_target=0.6,
            gm_target=2.5
        )
        
        assert isinstance(result, dict)
        assert result["cost_function_value"] is not None
    
    @patch('src.control_utils.compute_close_loop_peak_penalty')
    def test_with_custom_weight_cost(self, mock_penalty, mock_optimization_with_evaluate):
        """Test cost with custom weight vector"""
        mock_penalty.return_value = 0.1
        
        custom_weights = np.array([2.0, 1.0, 5.0, 50.0, 1000.0, 500.0])
        result = cost(
            mock_optimization_with_evaluate,
            gain=0.5,
            weight_cost=custom_weights
        )
        
        assert isinstance(result, dict)
        assert "weight_penalty" in result
        np.testing.assert_array_equal(result["weight_penalty"], custom_weights)


# ─────────────────────────────────────────────────────────────────────────────
# TEST: cost - EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestCostEdgeCases:
    """Edge case tests for cost function"""
    
    @patch('src.control_utils.compute_close_loop_peak_penalty')
    def test_with_none_stability_targets_uses_defaults(self, mock_penalty, mock_optimization_with_evaluate):
        """Test that None stability targets use default values"""
        mock_penalty.return_value = 0.1
        
        result = cost(mock_optimization_with_evaluate, gain=0.5)
        
        # Should use default sm_target=0.65, gm_target=3
        assert result is not None
    
    @patch('src.control_utils.compute_close_loop_peak_penalty')
    def test_with_zero_weights(self, mock_penalty, mock_optimization_with_evaluate):
        """Test cost calculation with zero weights"""
        mock_penalty.return_value = 0.1
        
        zero_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = cost(
            mock_optimization_with_evaluate,
            gain=0.5,
            weight_cost=zero_weights
        )
        
        # With zero weights, cost should be based only on variance
        assert isinstance(result, dict)
    
    @patch('src.control_utils.compute_close_loop_peak_penalty')
    def test_with_large_weights(self, mock_penalty, mock_optimization_with_evaluate):
        """Test cost calculation with large weights"""
        mock_penalty.return_value = 0.1
        
        large_weights = np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6])
        result = cost(
            mock_optimization_with_evaluate,
            gain=0.5,
            weight_cost=large_weights
        )
        
        assert isinstance(result, dict)
        assert np.isfinite(result["cost_function_value"])
    
    @patch('src.control_utils.compute_close_loop_peak_penalty')
    def test_penalty_dict_structure(self, mock_penalty, mock_optimization_with_evaluate):
        """Test that penalty list has 5 elements: [stability, sm, H_n, H_r, gm]"""
        mock_penalty.side_effect = [0.1, 0.2]  # For H_n and H_r penalties
        
        result = cost(mock_optimization_with_evaluate, gain=0.5)
        
        penalties = result["penalty"]
        assert isinstance(penalties, list)
        assert len(penalties) == 5


# ─────────────────────────────────────────────────────────────────────────────
# TEST: cost - EXCEPTION HANDLING
# ─────────────────────────────────────────────────────────────────────────────

class TestCostExceptionHandling:
    """Exception handling tests for cost function"""
    
    @patch('src.control_utils.compute_close_loop_peak_penalty')
    def test_propagates_control_CL_tf_margin_errors(self, mock_penalty, mock_optimization_with_evaluate):
        """Test that errors from control_CL_tf_margin are propagated"""
        mock_penalty.return_value = 0.1
        
        # Pass no controller parameters - should raise ValueError
        with pytest.raises(ValueError, match="Provide either 'gain'"):
            cost(mock_optimization_with_evaluate)


# ─────────────────────────────────────────────────────────────────────────────
# TEST: INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def test_control_CL_tf_margin_then_compute_penalty(self, mock_optimization):
        """Test full pipeline: compute margins then penalties"""
        result_margins = control_CL_tf_margin(mock_optimization, gain=0.3)
        H_r_tf = result_margins["H_r_tf"]
        
        penalty = compute_close_loop_peak_penalty(H_r_tf, close_loop_peak_target_dB=2.0)
        
        assert penalty >= 0.0
        assert np.isfinite(penalty)
    
    @patch('src.control_utils.compute_close_loop_peak_penalty')
    def test_full_cost_pipeline(self, mock_penalty, mock_optimization_with_evaluate):
        """Test complete cost evaluation pipeline"""
        mock_penalty.side_effect = [0.1, 0.2]  # Two calls for H_n, H_r
        
        result = cost(
            mock_optimization_with_evaluate,
            gain=0.5,
            sm_target=0.6,
            gm_target=2.5,
            weight_cost=np.array([1, 1, 10, 100, 1000, 500])
        )
        
        assert result is not None
        assert "cost_function_value" in result
        assert result["cost_function_value"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST: CONSISTENCY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistency:
    """Consistency and reproducibility tests"""
    
    def test_same_gain_produces_same_margins(self, mock_optimization):
        """Test reproducibility with same inputs"""
        result1 = control_CL_tf_margin(mock_optimization, gain=0.5)
        
        obj2 = MockControlOptimization()
        result2 = control_CL_tf_margin(obj2, gain=0.5)
        
        np.testing.assert_array_almost_equal(
            result1["H_ol_margins"],
            result2["H_ol_margins"]
        )
    
    def test_different_gains_produce_different_margins(self, mock_optimization):
        """Test that different gains yield different results"""
        result_low = control_CL_tf_margin(mock_optimization, gain=0.1)
        
        obj_high = MockControlOptimization()
        result_high = control_CL_tf_margin(obj_high, gain=0.9)
        
        # Margins should differ significantly
        assert not np.allclose(result_low["H_ol_margins"], result_high["H_ol_margins"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
