---
hide-toc: true
---

# Examples

Here we show the construction of some compilable surface code computations.

```{raw} html
<div class="bloq-gallery-filters" role="group" aria-label="Gallery category" hidden>
<button type="button" data-category="all" aria-pressed="true">All</button>
<button type="button" data-category="clifford" aria-pressed="false">Clifford</button>
<button type="button" data-category="non_clifford" aria-pressed="false">Non-Clifford</button>
<button type="button" data-category="factory" aria-pressed="false">Factory</button>
<button type="button" data-category="external_resource" aria-pressed="false">External resource</button>
<button type="button" data-category="adaptive" aria-pressed="false">Adaptive</button>
<button type="button" data-category="arithmetic" aria-pressed="false">Arithmetic</button>
<button type="button" data-category="addition" aria-pressed="false">Addition</button>
</div>
<p class="bloq-gallery-count" aria-live="polite"></p>
<div class="bloq-gallery-grid">
<a class="bloq-gallery-card" href="cnot.html" aria-label="CNOT" data-categories="clifford"><img src="../_static/gallery/cnot.png" alt="CNOT block graph" width="128" height="104" loading="lazy"><span>CNOT</span></a>
<a class="bloq-gallery-card" href="cz_spatial_h.html" aria-label="CZ with a spatial Hadamard" data-categories="clifford"><img src="../_static/gallery/cz_spatial_h.png" alt="CZ with a spatial Hadamard block graph" width="128" height="104" loading="lazy"><span>CZ with a spatial Hadamard</span></a>
<a class="bloq-gallery-card" href="cz_temporal_h.html" aria-label="CZ with a temporal Hadamard" data-categories="clifford"><img src="../_static/gallery/cz_temporal_h.png" alt="CZ with a temporal Hadamard block graph" width="128" height="104" loading="lazy"><span>CZ with a temporal Hadamard</span></a>
<a class="bloq-gallery-card" href="s_gate.html" aria-label="S gate" data-categories="clifford"><img src="../_static/gallery/s_gate.png" alt="S gate block graph" width="128" height="104" loading="lazy"><span>S gate</span></a>
<a class="bloq-gallery-card" href="t_gate.html" aria-label="T gate" data-categories="non_clifford"><img src="../_static/gallery/t_gate.png" alt="T gate block graph" width="128" height="104" loading="lazy"><span>T gate</span></a>
<a class="bloq-gallery-card" href="t_gate_structural_branch.html" aria-label="T gate with a structural branch" data-categories="non_clifford adaptive"><img src="../_static/gallery/t_gate_structural_branch.png" alt="T gate with a structural branch block graph" width="128" height="104" loading="lazy"><span>T gate with a structural branch</span></a>
<a class="bloq-gallery-card" href="t_with_prepared_y.html" aria-label="T gate with a prepared Y state" data-categories="non_clifford"><img src="../_static/gallery/t_with_prepared_y.png" alt="T gate with a prepared Y state block graph" width="128" height="104" loading="lazy"><span>T gate with a prepared Y state</span></a>
<a class="bloq-gallery-card" href="t_comparison.html" aria-label="T-state comparison" data-categories="non_clifford"><img src="../_static/gallery/t_comparison.png" alt="T-state comparison block graph" width="128" height="104" loading="lazy"><span>T-state comparison</span></a>
<a class="bloq-gallery-card" href="phase_gradient.html" aria-label="Phase-gradient sequence" data-categories="non_clifford"><img src="../_static/gallery/phase_gradient.png" alt="Phase-gradient sequence block graph" width="128" height="104" loading="lazy"><span>Phase-gradient sequence</span></a>
<a class="bloq-gallery-card" href="and_4t.html" aria-label="Four-T AND" data-categories="non_clifford"><img src="../_static/gallery/and_4t.png" alt="Four-T AND block graph" width="128" height="104" loading="lazy"><span>Four-T AND</span></a>
<a class="bloq-gallery-card" href="ccz_injected_and.html" aria-label="CCZ-injected AND" data-categories="external_resource non_clifford adaptive arithmetic addition"><img src="../_static/gallery/ccz_injected_and.png" alt="CCZ-injected AND block graph" width="128" height="104" loading="lazy"><span>CCZ-injected AND</span></a>
<a class="bloq-gallery-card" href="ccz_injected_maj.html" aria-label="CCZ-injected MAJ" data-categories="external_resource non_clifford adaptive arithmetic addition"><img src="../_static/gallery/ccz_injected_maj.png" alt="CCZ-injected MAJ block graph" width="128" height="104" loading="lazy"><span>CCZ-injected MAJ</span></a>
<a class="bloq-gallery-card" href="uma.html" aria-label="UMA" data-categories="clifford adaptive arithmetic addition"><img src="../_static/gallery/uma.png" alt="UMA block graph" width="128" height="104" loading="lazy"><span>UMA</span></a>
<a class="bloq-gallery-card" href="three_bit_adder.html" aria-label="Three-bit controlled adder" data-categories="external_resource non_clifford adaptive arithmetic addition"><img src="../_static/gallery/three_bit_adder.png" alt="Three-bit controlled adder block graph" width="128" height="104" loading="lazy"><span>Three-bit controlled adder</span></a>
<a class="bloq-gallery-card" href="ten_bit_adder.html" aria-label="Ten-bit controlled adder" data-categories="external_resource non_clifford adaptive arithmetic addition"><img src="../_static/gallery/ten_bit_adder.png" alt="Ten-bit controlled adder block graph" width="128" height="104" loading="lazy"><span>Ten-bit controlled adder</span></a>
<a class="bloq-gallery-card" href="toffoli_from_and_delayed_cz.html" aria-label="Toffoli from AND" data-categories="non_clifford"><img src="../_static/gallery/toffoli_from_and_delayed_cz.png" alt="Toffoli from AND block graph" width="128" height="104" loading="lazy"><span>Toffoli from AND</span></a>
<a class="bloq-gallery-card" href="ccz_4x3x7_tels.html" aria-label="CCZ factory with TELS" data-categories="factory non_clifford"><img src="../_static/gallery/ccz_4x3x7_tels.png" alt="CCZ factory with TELS block graph" width="128" height="104" loading="lazy"><span>CCZ factory with TELS</span></a>
<a class="bloq-gallery-card" href="ccz_4x3x6.html" aria-label="CCZ factory without TELS" data-categories="factory non_clifford"><img src="../_static/gallery/ccz_4x3x6.png" alt="CCZ factory without TELS block graph" width="128" height="104" loading="lazy"><span>CCZ factory without TELS</span></a>
<a class="bloq-gallery-card" href="ccz_2x3x16.html" aria-label="Serial CCZ factory" data-categories="factory non_clifford external_resource"><img src="../_static/gallery/ccz_2x3x16.png" alt="Serial CCZ factory block graph" width="128" height="104" loading="lazy"><span>Serial CCZ factory</span></a>
<a class="bloq-gallery-card" href="ccz_4x2x12.html" aria-label="CCZ factory with a selective parity check" data-categories="factory non_clifford external_resource"><img src="../_static/gallery/ccz_4x2x12.png" alt="CCZ factory with a selective parity check block graph" width="128" height="104" loading="lazy"><span>CCZ factory with a selective parity check</span></a>
<a class="bloq-gallery-card" href="perfect_3x3x13.html" aria-label="Ten-to-two factory, 3×3" data-categories="factory non_clifford"><img src="../_static/gallery/perfect_3x3x13.png" alt="Ten-to-two factory, 3×3 block graph" width="128" height="104" loading="lazy"><span>Ten-to-two factory, 3×3</span></a>
<a class="bloq-gallery-card" href="perfect_2x3x28.html" aria-label="Ten-to-two factory, 2×3" data-categories="factory non_clifford"><img src="../_static/gallery/perfect_2x3x28.png" alt="Ten-to-two factory, 2×3 block graph" width="128" height="104" loading="lazy"><span>Ten-to-two factory, 2×3</span></a>
<a class="bloq-gallery-card" href="ccz_gate_teleport.html" aria-label="CCZ gate teleportation" data-categories="external_resource non_clifford"><img src="../_static/gallery/ccz_gate_teleport.png" alt="CCZ gate teleportation block graph" width="128" height="104" loading="lazy"><span>CCZ gate teleportation</span></a>
<a class="bloq-gallery-card" href="bell_state.html" aria-label="Bell state" data-categories="clifford"><img src="../_static/gallery/bell_state.png" alt="Bell state block graph" width="128" height="104" loading="lazy"><span>Bell state</span></a>
<a class="bloq-gallery-card" href="ghz.html" aria-label="GHZ state" data-categories="clifford"><img src="../_static/gallery/ghz.png" alt="GHZ state block graph" width="128" height="104" loading="lazy"><span>GHZ state</span></a>
<a class="bloq-gallery-card" href="ghz_slide_then_glide.html" aria-label="GHZ with slides and glides" data-categories="clifford"><img src="../_static/gallery/ghz_slide_then_glide.png" alt="GHZ with slides and glides block graph" width="128" height="104" loading="lazy"><span>GHZ with slides and glides</span></a>
<a class="bloq-gallery-card" href="ghz_patch_rotations.html" aria-label="GHZ with patch rotations" data-categories="clifford"><img src="../_static/gallery/ghz_patch_rotations.png" alt="GHZ with patch rotations block graph" width="128" height="104" loading="lazy"><span>GHZ with patch rotations</span></a>
<a class="bloq-gallery-card" href="1d-yoked.html" aria-label="1D yoked layout" data-categories="clifford"><img src="../_static/gallery/1d-yoked.png" alt="1D yoked layout block graph" width="128" height="104" loading="lazy"><span>1D yoked layout</span></a>
<a class="bloq-gallery-card" href="thth.html" aria-label="T–H–T–H sequence" data-categories="non_clifford"><img src="../_static/gallery/thth.png" alt="T–H–T–H sequence block graph" width="128" height="104" loading="lazy"><span>T–H–T–H sequence</span></a>
<a class="bloq-gallery-card" href="three_cnots.html" aria-label="Compressed three CNOTs" data-categories="clifford"><img src="../_static/gallery/three_cnots.png" alt="Compressed three CNOTs block graph" width="128" height="104" loading="lazy"><span>Compressed three CNOTs</span></a>
<a class="bloq-gallery-card" href="steane_encoding.html" aria-label="Steane encoding" data-categories="clifford"><img src="../_static/gallery/steane_encoding.png" alt="Steane encoding block graph" width="128" height="104" loading="lazy"><span>Steane encoding</span></a>
<a class="bloq-gallery-card" href="x_memory.html" aria-label="X-basis memory" data-categories="clifford"><img src="../_static/gallery/x_memory.png" alt="X-basis memory block graph" width="128" height="104" loading="lazy"><span>X-basis memory</span></a>
<a class="bloq-gallery-card" href="y_memory.html" aria-label="Y-basis memory" data-categories="clifford"><img src="../_static/gallery/y_memory.png" alt="Y-basis memory block graph" width="128" height="104" loading="lazy"><span>Y-basis memory</span></a>
<a class="bloq-gallery-card" href="move_rotation.html" aria-label="Rotation through movement" data-categories="clifford"><img src="../_static/gallery/move_rotation.png" alt="Rotation through movement block graph" width="128" height="104" loading="lazy"><span>Rotation through movement</span></a>
<a class="bloq-gallery-card" href="stability.html" aria-label="Stability experiment" data-categories="clifford"><img src="../_static/gallery/stability.png" alt="Stability experiment block graph" width="128" height="104" loading="lazy"><span>Stability experiment</span></a>
</div>
```

```{toctree}
:hidden:
:maxdepth: 1

cnot
cz_spatial_h
cz_temporal_h
s_gate
t_gate
t_gate_structural_branch
t_with_prepared_y
t_comparison
phase_gradient
and_4t
ccz_injected_and
ccz_injected_maj
uma
three_bit_adder
ten_bit_adder
toffoli_from_and_delayed_cz
ccz_4x3x7_tels
ccz_4x3x6
ccz_2x3x16
ccz_4x2x12
perfect_3x3x13
perfect_2x3x28
ccz_gate_teleport
bell_state
ghz
ghz_slide_then_glide
ghz_patch_rotations
1d-yoked
thth
three_cnots
steane_encoding
x_memory
y_memory
move_rotation
stability
```
