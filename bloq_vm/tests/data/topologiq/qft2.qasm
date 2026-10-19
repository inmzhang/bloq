OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
// Prepare |0+>, with q[0] the least significant bit.
h q[0];
// Positive-phase QFT, including the final bit-order swap.
h q[1];
t q[0];
t q[1];
cx q[0],q[1];
tdg q[1];
cx q[0],q[1];
h q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
