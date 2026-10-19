"""Exact source angles must survive the optional PyZX frontend."""

from fractions import Fraction

import pytest

pytest.importorskip("pyzx", reason="opt-in pinned PyZX integration environment")

from bloq.interop import _qasm as frontend  # noqa: E402
from bloq.interop._qasm import QASMImportError  # noqa: E402


def source(gates):
    return f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n{gates}\n'


@pytest.mark.parametrize(
    "expression",
    [
        "0.000000000001", "1e-1000", "pi+1", "pi+1e-1000",
        "0.7853981633974483", "pi/4+0.000000000001",
        "pi*pi", "sin(pi/2)", "1/pi", "pi/0",
    ],
)
def test_lossy_or_unsupported_angles_reject(expression):
    with pytest.raises(QASMImportError) as error:
        frontend.parse_qasm(source(f"rz({expression}) q[0];"))
    assert error.value.code == "unsupported_angle"
    assert expression in str(error.value)


@pytest.mark.parametrize(
    "expression,phase",
    [("0", Fraction(0)), ("pi/2", Fraction(1, 2)), ("-pi/2", Fraction(-1, 2)),
     ("0.25*pi", Fraction(1, 4)), ("(0.1+0.15)*pi", Fraction(1, 4)),
     ("(pi+pi)/8", Fraction(1, 4))],
)
def test_supported_angles_keep_exact_fractions(expression, phase):
    circuit = frontend.parse_qasm(source(f"rz({expression}) q[0];"))
    assert circuit.gates[0].phase == phase
    assert isinstance(circuit.gates[0].phase, Fraction)


def test_decimal_pi_coefficient_is_not_rounded():
    literal = "0.25000000000000000000000000000000001"
    circuit = frontend.parse_qasm(source(f"rz({literal}*pi) q[0];"))
    assert circuit.gates[0].phase == Fraction(literal)
    assert circuit.gates[0].phase != Fraction(1, 4)


def test_custom_gate_parameters_bind_as_exact_angles():
    text = source("""gate inner(a) q { rz(a/2+pi/4) q; }
gate outer(b) q { inner(b) q; }
outer(pi/2) q[0];""")
    circuit = frontend.parse_qasm(text)
    assert circuit.gates[0].phase == Fraction(1, 2)


def test_source_and_global_parser_remain_unchanged():
    from pyzx.circuit.qasmparser import QASMParser

    text = source("rx(pi/2) q[0];")
    original = text
    parser = QASMParser.parse_phase_arg
    circuit = frontend.parse_qasm(text)
    assert circuit.gates[0].phase == Fraction(1, 2)
    assert text == original
    assert QASMParser.parse_phase_arg is parser


def test_version_guard_precedes_source_parsing(monkeypatch):
    monkeypatch.setattr(frontend, "version", lambda _: "0.0.0")
    with pytest.raises(QASMImportError, match="pyzx_version"):
        frontend.parse_qasm("not QASM")


@pytest.mark.parametrize("operation", [
    "x a[1];", "x b[-1];", "x a[2];", "x a[1.0];", "x missing[0];",
    "h a[0],b[0];", "cx a[0];", "ccx a[0],b[0];", "cu1(pi/2) a[0],b[0],a[0];",
    "gate bad q { x q[1]; } bad a[0];",
    "gate pair q,r { x q; x r; } pair a[0];",
    "gate phase(t) q { rz(t) q; } phase(pi/2) a[0],b[0];",
    "id(pi/4) a[0];", "barrier(pi/4) a[0];",
    "cx a[0],a[0];", "cx a,a[0];", "cx a,a;", "ccx a[0],b[0],a[0];",
    "gate pair q,r { x q; x r; } pair a[0],a[0];",
    "gate pair(t) q,r { rz(t) q; x r; } pair(pi/4) a[0],a[0];",
])
def test_invalid_operands_cannot_spill_registers_or_be_dropped(operation):
    text = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg a[1];\nqreg b[1];\n' + operation
    with pytest.raises(QASMImportError) as caught:
        frontend.parse_qasm(text)
    assert caught.value.code == "qasm_operand"


@pytest.mark.parametrize("declarations", [
    "qreg q[1]; qreg q[2];", "creg q[1]; creg q[2];",
    "qreg q[1]; creg q[1];", "creg q[1]; qreg q[1];",
    "qreg q[1],r[1];", "gate bad q { qreg r[1]; } qreg q[1]; bad q;",
])
def test_invalid_register_declarations_reject(declarations):
    text = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' + declarations
    with pytest.raises(QASMImportError, match="qasm_register"):
        frontend.parse_qasm(text)


def test_valid_register_broadcast_and_custom_gate_operands_keep_wire_order():
    text = '''OPENQASM 2.0;
include "qelib1.inc";
qreg a[2];
qreg b[2];
gate pair q,r { cx q,r; }
h a;
pair a,b;
cx b[1],a[0];
cx a[0],b;
cx a[0],a[1];
'''
    circuit = frontend.parse_qasm(text)
    assert circuit.qubits == 4
    assert [str(gate) for gate in circuit.gates] == [
        "HAD(0)", "HAD(1)", "CNOT(0,2)", "CNOT(1,3)", "CNOT(3,0)",
        "CNOT(0,2)", "CNOT(0,3)", "CNOT(0,1)",
    ]
