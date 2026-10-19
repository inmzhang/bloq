"""Exact angle parsing for the pinned PyZX QASM frontend."""

import ast
from fractions import Fraction
from importlib.metadata import version
import re
from typing import Any, NoReturn

from pyzx.circuit.gates import qasm_gate_table
from pyzx.circuit.qasmparser import QASMParser

from .topologiq import TopologiqImportError


def _phase(expression: str, parameters: dict[str, Fraction] | None) -> Fraction:
    expression = expression.strip()

    def fail(reason: str) -> NoReturn:
        raise TopologiqImportError("unsupported_angle", f"Angle {expression!r}: {reason}")

    def evaluate(node: ast.AST) -> tuple[Fraction, Fraction]:
        # Keep rational radians separate from rational multiples of pi.
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expression, node)
            if literal is None:
                fail("missing numeric source token")
            try:
                return Fraction(literal), Fraction(0)
            except ValueError:
                fail("unsupported numeric literal")
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return Fraction(0), Fraction(1)
            if parameters is not None and node.id in parameters:
                return Fraction(0), parameters[node.id]
            fail("unbound angle parameter")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            radians, pi = evaluate(node.operand)
            return (radians, pi) if isinstance(node.op, ast.UAdd) else (-radians, -pi)
        if isinstance(node, ast.BinOp):
            left, left_pi = evaluate(node.left)
            right, right_pi = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right, left_pi + right_pi
            if isinstance(node.op, ast.Sub):
                return left - right, left_pi - right_pi
            if isinstance(node.op, ast.Mult):
                if left_pi and right_pi:
                    fail("nonlinear multiples of pi")
                return left * right, left_pi * right + left * right_pi
            if isinstance(node.op, ast.Div):
                if right_pi or not right:
                    fail("division requires a nonzero rational constant")
                return left / right, left_pi / right
        fail("expected a rational multiple of pi")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        fail("unsupported angle syntax")
    radians, pi = evaluate(tree.body)
    if radians:
        fail("nonzero radian constants cannot be represented exactly")
    return pi


class _ExactQASMParser(QASMParser):
    def parse_phase_arg(self, val: str) -> Fraction:
        # PyZX binds custom-gate parameters as normalized pi coefficients.
        return _phase(val, self._param_subst)

    def parse_command(self, c: str, registers: dict[str, tuple[int, int]]) -> list[Any]:
        name, phases, operands = self.extract_command_parts(c)
        if name in ("qreg", "creg"):
            if phases or len(operands) != 1 or registers is not self.registers:
                raise TopologiqImportError("qasm_register", f"Invalid register declaration: {c}")
            register, _ = self._parse_register_declaration(operands[0])
            if register in registers or register in self.cregisters:
                raise TopologiqImportError("qasm_register", f"Register {register!r} is already declared: {c}")
            return super().parse_command(c, registers)
        if name in ("measure", "reset"):
            return super().parse_command(c, registers)
        if name in ("id", "barrier") and phases:
            raise TopologiqImportError("qasm_operand", f"Operation {name!r} takes no angles: {c}")
        if name in self.parametrized_gates:
            arity = len(self.parametrized_gates[name][1])
        elif name in self.custom_gates:
            arity = self.custom_gates[name].qubits
        elif name in qasm_gate_table or name == "id":
            if name in ("ccx", "ccz", "cswap"):
                arity = 3
            elif name in (
                "cx", "CX", "cy", "cz", "ch", "csx", "cp", "cphase", "cu1",
                "crx", "cry", "crz", "rxx", "rzz", "swap", "cu3", "cu",
            ):
                arity = 2
            else:
                arity = 1
        else:
            arity = None
        if arity is not None and len(operands) != arity:
            raise TopologiqImportError(
                "qasm_operand", f"Gate {name!r} requires {arity} qubit operands: {c}",
            )
        ranges: list[tuple[int, int]] = []
        for operand in operands:
            match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)(?:\[([0-9]+)\])?", operand)
            if match is None or match[1] not in registers:
                raise TopologiqImportError("qasm_operand", f"Invalid qubit operand {operand!r}: {c}")
            if match[2] is not None and int(match[2]) >= registers[match[1]][1]:
                raise TopologiqImportError("qasm_operand", f"Qubit operand {operand!r} is out of range: {c}")
            start, size = registers[match[1]]
            if match[2] is not None:
                start, size = start + int(match[2]), 1
            # Register broadcasting repeats indexed operands in every lane.
            if name != "barrier" and any(start < end and begin < start + size for begin, end in ranges):
                raise TopologiqImportError("qasm_operand", f"Gate operands overlap: {c}")
            ranges.append((start, start + size))
        return super().parse_command(c, registers)


def parse_qasm(qasm: str) -> Any:
    """Parse without PyZX's float conversion or denominator rounding."""
    if version("pyzx") != "0.10.7":
        raise TopologiqImportError("pyzx_version", "Exact QASM parsing requires pyzx==0.10.7")
    return _ExactQASMParser().parse(qasm).to_basic_gates()
