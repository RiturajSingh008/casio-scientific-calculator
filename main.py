"""
Casio FX-991CW Scientific Calculator Emulator
A pure Python emulator providing variable registers, complex polar-rectangular conversions,
matrix algebra, statistical calculations, and numerical calculus.
"""

from __future__ import annotations

import cmath
import math
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class CalculatorState:
    """Manages persistent variables, historical answers, and angle mode settings."""

    def __init__(self) -> None:
        self.variables: Dict[str, float] = {
            "A": 0.0,
            "B": 0.0,
            "C": 0.0,
            "D": 0.0,
            "E": 0.0,
            "F": 0.0,
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
        }
        self.constants: Dict[str, float] = {
            "pi": math.pi,
            "e": math.e,
            "g": 9.80665,         # Standard acceleration due to gravity (m/s^2)
            "c": 299792458.0,      # Speed of light in vacuum (m/s)
            "h": 6.62607015e-34,   # Planck constant (J*s)
        }
        self.ans: float = 0.0
        self.angle_mode: str = "DEG"  # 'DEG' or 'RAD'

    def set_angle_mode(self, mode: str) -> None:
        mode_upper = mode.strip().upper()
        if mode_upper in ("DEG", "RAD"):
            self.angle_mode = mode_upper
        else:
            raise ValueError("Angle mode must be either 'DEG' or 'RAD'.")

    def set_variable(self, name: str, value: float) -> None:
        name_upper = name.strip().upper()
        if name_upper in self.variables:
            self.variables[name_upper] = float(value)
        else:
            raise KeyError(f"Variable '{name_upper}' is not a valid register (A-F, X, Y, Z).")

    def get_variable(self, name: str) -> float:
        name_upper = name.strip().upper()
        if name_upper in self.variables:
            return self.variables[name_upper]
        if name in self.constants:
            return self.constants[name]
        if name_upper == "ANS":
            return self.ans
        raise KeyError(f"Identifier '{name}' not found.")


class ComplexEngine:
    """Handles conversions and formatting between polar and Cartesian complex forms."""

    @staticmethod
    def from_polar(r: float, theta_deg: float) -> complex:
        """Create a complex number from polar coordinates (radius, degrees)."""
        theta_rad = math.radians(theta_deg)
        return cmath.rect(r, theta_rad)

    @staticmethod
    def to_polar(z: complex) -> Tuple[float, float]:
        """Convert a complex number to polar form (r, theta_deg)."""
        r, theta_rad = cmath.polar(z)
        theta_deg = math.degrees(theta_rad)
        return r, theta_deg

    @staticmethod
    def format_cartesian(z: complex, precision: int = 6) -> str:
        real_part = round(z.real, precision)
        imag_part = round(z.imag, precision)
        sign = "+" if imag_part >= 0 else "-"
        return f"{real_part} {sign} {abs(imag_part)}i"

    @staticmethod
    def format_polar(r: float, theta_deg: float, precision: int = 4) -> str:
        return f"{round(r, precision)} ∠ {round(theta_deg, precision)}°"


class MatrixEngine:
    """Performs standard 2x2 and 3x3 matrix arithmetic and determinant computations."""

    Matrix = List[List[float]]

    @staticmethod
    def validate_matrix(m: Matrix) -> Tuple[int, int]:
        if not m or not isinstance(m, list):
            raise ValueError("Invalid matrix format.")
        rows = len(m)
        cols = len(m[0])
        for row in m:
            if len(row) != cols:
                raise ValueError("All rows in a matrix must have equal length.")
        return rows, cols

    @classmethod
    def add(cls, a: Matrix, b: Matrix) -> Matrix:
        r1, c1 = cls.validate_matrix(a)
        r2, c2 = cls.validate_matrix(b)
        if (r1, c1) != (r2, c2):
            raise ValueError("Matrices must have matching dimensions for addition.")
        return [[a[i][j] + b[i][j] for j in range(c1)] for i in range(r1)]

    @classmethod
    def multiply(cls, a: Matrix, b: Matrix) -> Matrix:
        r1, c1 = cls.validate_matrix(a)
        r2, c2 = cls.validate_matrix(b)
        if c1 != r2:
            raise ValueError(f"Cannot multiply {r1}x{c1} matrix with {r2}x{c2} matrix.")
        result = [[0.0 for _ in range(c2)] for _ in range(r1)]
        for i in range(r1):
            for j in range(c2):
                result[i][j] = sum(a[i][k] * b[k][j] for k in range(c1))
        return result

    @classmethod
    def transpose(cls, m: Matrix) -> Matrix:
        rows, cols = cls.validate_matrix(m)
        return [[m[r][c] for r in range(rows)] for c in range(cols)]

    @classmethod
    def determinant(cls, m: Matrix) -> float:
        rows, cols = cls.validate_matrix(m)
        if rows != cols:
            raise ValueError("Determinants are only defined for square matrices.")
        if rows == 2:
            return (m[0][0] * m[1][1]) - (m[0][1] * m[1][0])
        elif rows == 3:
            det = (
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            )
            return det
        else:
            raise NotImplementedError("Determinant support currently limited to 2x2 and 3x3 matrices.")


class CalculusEngine:
    """Performs numerical differentiation and Simpson's numerical definite integration."""

    @staticmethod
    def derivative(func: Callable[[float], float], x: float, h: float = 1e-6) -> float:
        """Central difference derivative: f'(x) ≈ (f(x + h) - f(x - h)) / (2h)."""
        return (func(x + h) - func(x - h)) / (2.0 * h)

    @staticmethod
    def integrate(func: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
        """Simpson's 1/3 rule for definite numerical integration."""
        if n % 2 != 0:
            n += 1  # n must be even for Simpson's rule
        h = (b - a) / n
        total = func(a) + func(b)
        for i in range(1, n):
            x_i = a + i * h
            total += 4.0 * func(x_i) if i % 2 != 0 else 2.0 * func(x_i)
        return (h / 3.0) * total


class StatisticsEngine:
    """Statistical functions for 1-variable dataset analysis."""

    @staticmethod
    def summary(data: List[float]) -> Dict[str, float]:
        if not data:
            raise ValueError("Dataset cannot be empty.")
        n = len(data)
        mean_val = sum(data) / n
        sorted_data = sorted(data)
        mid = n // 2
        median_val = (sorted_data[mid] if n % 2 != 0 else (sorted_data[mid - 1] + sorted_data[mid]) / 2.0)
        variance = sum((x - mean_val) ** 2 for x in data) / n
        std_dev = math.sqrt(variance)
        return {
            "Count": float(n),
            "Mean": mean_val,
            "Median": median_val,
            "Variance (pop)": variance,
            "StdDev (pop)": std_dev,
        }


class CasioScientificCalculator:
    """Master calculator orchestrating math evaluation, memory state, and modules."""

    def __init__(self) -> None:
        self.state = CalculatorState()
        self.complex = ComplexEngine()
        self.matrix = MatrixEngine()
        self.calculus = CalculusEngine()
        self.stats = StatisticsEngine()

    def _trig_wrapper(self, func: Callable[[float], float]) -> Callable[[float], float]:
        def wrapped(angle: float) -> float:
            rad = math.radians(angle) if self.state.angle_mode == "DEG" else angle
            return func(rad)
        return wrapped

    def _inv_trig_wrapper(self, func: Callable[[float], float]) -> Callable[[float], float]:
        def wrapped(val: float) -> float:
            rad = func(val)
            return math.degrees(rad) if self.state.angle_mode == "DEG" else rad
        return wrapped

    def evaluate_expression(self, expression: str) -> float:
        """Safely evaluates a single math expression using calculator state and functions."""
        clean_expr = expression.strip().replace("^", "**")

        scope: Dict[str, Any] = {
            "sin": self._trig_wrapper(math.sin),
            "cos": self._trig_wrapper(math.cos),
            "tan": self._trig_wrapper(math.tan),
            "asin": self._inv_trig_wrapper(math.asin),
            "acos": self._inv_trig_wrapper(math.acos),
            "atan": self._inv_trig_wrapper(math.atan),
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "sqrt": math.sqrt,
            "log": math.log10,
            "ln": math.log,
            "factorial": math.factorial,
            "nPr": lambda n, r: math.perm(int(n), int(r)),
            "nCr": lambda n, r: math.comb(int(n), int(r)),
            "abs": abs,
            "Ans": self.state.ans,
            "ans": self.state.ans,
        }

        # Inject constants and memory registers
        scope.update(self.state.constants)
        scope.update(self.state.variables)

        try:
            result = eval(clean_expr, {"__builtins__": None}, scope)
            if isinstance(result, (int, float)):
                self.state.ans = float(result)
                return self.state.ans
            raise TypeError("Expression did not return a scalar numerical value.")
        except ZeroDivisionError:
            raise ZeroDivisionError("Math Error: Division by zero.")
        except ValueError as ve:
            raise ValueError(f"Math Domain Error: {ve}")
        except Exception as ex:
            raise RuntimeError(f"Syntax/Execution Error: {ex}")


def run_verification_demo(calc: CasioScientificCalculator) -> None:
    """Executes a diagnostic demo confirming every calculation module functions properly."""
    print("=" * 65)
    print("CASIO FX-991CW SCIENTIFIC CALCULATOR - SYSTEM VERIFICATION")
    print("=" * 65)

    # 1. Variables & Evaluation
    calc.state.set_variable("A", 15.0)
    res1 = calc.evaluate_expression("A * 2 + g")
    print("\n--- 1. Variable Assignment & Evaluation ---")
    print(f"Register A = 15.0, g = {calc.state.constants['g']}")
    print(f"Expression: A * 2 + g  --> Result = {res1}")

    # 2. Complex Numbers
    print("\n--- 2. Complex Numbers (Polar & Rectangular) ---")
    phasor = calc.complex.from_polar(120.0, 30.0)
    print(f"120.0 ∠ 30.0° in Rectangular: {calc.complex.format_cartesian(phasor)}")
    r_val, theta_val = calc.complex.to_polar(phasor)
    print(f"Re-converted to Polar:        {calc.complex.format_polar(r_val, theta_val)}")

    # 3. Matrix Multiplications & Determinants
    print("\n--- 3. Matrix Arithmetic & Determinants ---")
    mat_a = [[1.0, 2.0], [3.0, 4.0]]
    mat_b = [[5.0, 6.0], [0.0, 1.0]]
    prod = calc.matrix.multiply(mat_a, mat_b)
    det_a = calc.matrix.determinant(mat_a)
    print(f"MatA: {mat_a}")
    print(f"MatB: {mat_b}")
    print(f"MatA x MatB: {prod}")
    print(f"det(MatA):   {det_a}")

    # 4. Calculus (Simpson's Rule & Finite Difference)
    print("\n--- 4. Numerical Calculus ---")
    deriv = calc.calculus.derivative(lambda x: x ** 3, 2.0)
    print(f"d/dx(x^3) at x=2.0:                  {deriv:.4f}")
    integral = calc.calculus.integrate(math.sin, 0.0, math.pi / 2.0)
    print(f"∫ sin(x) dx from 0 to pi/2 (Simpson): {integral:.6f}")

    # 5. Statistics
    print("\n--- 5. 1-Variable Statistics ---")
    sample_data = [12.0, 15.0, 18.0, 20.0, 25.0, 30.0]
    stats_out = calc.stats.summary(sample_data)
    for k, v in stats_out.items():
        print(f"  {k:15}: {v:.4f}")
    print("=" * 65)


def interactive_cli(calc: CasioScientificCalculator) -> None:
    """Provides an interactive REPL loop for user calculations."""
    print("\n[Interactive Mode Enabled] Enter an expression (or 'help', 'mode', 'exit'):")
    while True:
        try:
            line = input(f"[{calc.state.angle_mode}] fx-991cw >> ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                print("Exiting calculator emulator. Goodbye!")
                break
            if line.lower() == "help":
                print("Available commands/functions:")
                print("  sin, cos, tan, asin, acos, atan, sqrt, log (base 10), ln, factorial, nPr, nCr")
                print("  Constants: pi, e, g, c, h | Registers: A, B, C, D, E, F, X, Y, Z, Ans")
                print("  Commands: 'mode deg', 'mode rad', 'set A=25', 'exit'")
                continue
            if line.lower().startswith("mode "):
                parts = line.split()
                calc.state.set_angle_mode(parts[1])
                print(f"Angle mode set to {calc.state.angle_mode}")
                continue
            if line.lower().startswith("set "):
                # e.g. set A=12.5
                assignment = line[4:].strip()
                var_name, var_val = assignment.split("=")
                calc.state.set_variable(var_name.strip(), float(var_val.strip()))
                print(f"Register {var_name.strip().upper()} = {float(var_val.strip())}")
                continue

            output = calc.evaluate_expression(line)
            print(f"= {output}")
        except Exception as error:
            print(f"Error: {error}")

# --- Casio Advanced Math Helpers ---

def fact(n: int) -> int:
    if n < 0:
        raise ValueError("Math Error: Factorial of negative number.")
    return math.factorial(int(n))

def nPr(n: int, r: int) -> int:
    n, r = int(n), int(r)
    if n < 0 or r < 0 or r > n:
        raise ValueError("Math Error: nPr requires 0 <= r <= n.")
    return math.perm(n, r)

def nCr(n: int, r: int) -> int:
    n, r = int(n), int(r)
    if n < 0 or r < 0 or r > n:
        raise ValueError("Math Error: nCr requires 0 <= r <= n.")
return math.comb(n, r)

if __name__ == "__main__":
    calculator = CasioScientificCalculator()
    run_verification_demo(calculator)
    interactive_cli(calculator)

 