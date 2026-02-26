"""Unit tests for calculator.py"""

import pytest
from calculator import add, subtract, multiply, divide, calculate


class TestBasicOperations:
    def test_add(self):
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(0, 0) == 0

    def test_subtract(self):
        assert subtract(5, 3) == 2
        assert subtract(0, 5) == -5
        assert subtract(-3, -2) == -1

    def test_multiply(self):
        assert multiply(3, 4) == 12
        assert multiply(-2, 5) == -10
        assert multiply(0, 100) == 0

    def test_divide(self):
        assert divide(10, 2) == 5.0
        assert divide(7, 2) == 3.5
        assert divide(-6, 3) == -2.0

    def test_divide_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            divide(5, 0)


class TestCalculate:
    def test_addition_expression(self):
        assert calculate("3 + 4") == 7.0

    def test_subtraction_expression(self):
        assert calculate("10 - 3") == 7.0

    def test_multiplication_expression(self):
        assert calculate("6 * 7") == 42.0

    def test_division_expression(self):
        assert calculate("9 / 3") == 3.0

    def test_float_operands(self):
        assert calculate("1.5 + 2.5") == 4.0

    def test_invalid_expression(self):
        with pytest.raises(ValueError):
            calculate("1 + 2 + 3")

    def test_unsupported_operator(self):
        with pytest.raises(ValueError):
            calculate("5 % 2")

    def test_divide_by_zero_expression(self):
        with pytest.raises(ZeroDivisionError):
            calculate("8 / 0")
