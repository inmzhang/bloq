"""The `GalleryItem` enum: idiomatic access to the built-in block-graph gallery."""

import enum
import functools

from bloq import _core


@functools.cache
def _metadata() -> dict[str, dict]:
    """The compiler's gallery table, keyed by id.

    Built once on first use: the entries are compiled-in constants, so a second
    call would rebuild an identical table.
    """
    return {entry["id"]: entry for entry in _core.gallery_entries()}


class GalleryItem(str, enum.Enum):
    """A built-in example block graph for analysis or compilation.

    Members use the compiler's gallery ids.

    Examples:
        >>> import bloq
        >>> graph = bloq.GalleryItem.CNOT.load()
        >>> graph.block_count
        10
        >>> bloq.GalleryItem("cnot") is bloq.GalleryItem.CNOT
        True
    """

    CNOT = "cnot"
    CZ_SPATIAL_H = "cz_spatial_h"
    CZ_TEMPORAL_H = "cz_temporal_h"
    S_GATE = "s_gate"
    T_GATE = "t_gate"
    T_GATE_STRUCTURAL_BRANCH = "t_gate_structural_branch"
    T_WITH_PREPARED_Y = "t_with_prepared_y"
    T_COMPARISON = "t_comparison"
    PHASE_GRADIENT = "phase_gradient"
    AND_4T = "and_4t"
    CCZ_INJECTED_AND = "ccz_injected_and"
    CCZ_INJECTED_MAJ = "ccz_injected_maj"
    UMA = "uma"
    THREE_BIT_ADDER = "three_bit_adder"
    TEN_BIT_ADDER = "ten_bit_adder"
    TOFFOLI_FROM_AND_DELAYED_CZ = "toffoli_from_and_delayed_cz"
    CCZ_4X3X7_TELS = "ccz_4x3x7_tels"
    CCZ_4X3X6 = "ccz_4x3x6"
    CCZ_GATE_TELEPORT = "ccz_gate_teleport"
    BELL_STATE = "bell_state"
    GHZ = "ghz"
    GHZ_SLIDE_THEN_GLIDE = "ghz_slide_then_glide"
    GHZ_PATCH_ROTATIONS = "ghz_patch_rotations"
    ONE_D_YOKED = "1d-yoked"
    THTH = "thth"
    THREE_CNOTS = "three_cnots"
    STEANE_ENCODING = "steane_encoding"
    X_MEMORY = "x_memory"
    Y_MEMORY = "y_memory"
    MOVE_ROTATION = "move_rotation"
    STABILITY = "stability"

    @property
    def id(self) -> str:
        """The gallery id string, as accepted by the CLI and `.blog` tooling.

        Examples:
            >>> import bloq
            >>> bloq.GalleryItem.X_MEMORY.id
            'x_memory'
        """
        return self.value

    @property
    def description(self) -> str:
        """The human-readable description of this entry.

        Examples:
            >>> import bloq
            >>> "CNOT" in bloq.GalleryItem.CNOT.description
            True
        """
        return _metadata()[self.value]["description"]

    @property
    def categories(self) -> tuple[str, ...]:
        """The entry's categories, for selecting a family of examples.

        Examples:
            >>> import bloq
            >>> clifford = [g for g in bloq.GalleryItem if "clifford" in g.categories]
            >>> bloq.GalleryItem.CNOT in clifford
            True
            >>> bloq.GalleryItem.T_GATE in clifford
            False
        """
        return tuple(_metadata()[self.value]["categories"])

    def load(self) -> _core.BlockGraph:
        """Build this entry's block graph.

        Returns:
            BlockGraph: A complete, valid graph for this gallery entry.

        Examples:
            >>> import bloq
            >>> bloq.GalleryItem.BELL_STATE.load().port_count
            2
        """
        return _core.gallery_load(self.value)

    def source(self) -> str:
        """The `.blog` source text of this entry.

        Examples:
            >>> import bloq
            >>> bloq.GalleryItem.CNOT.source().startswith("BLOG 1.0")
            True
        """
        return _core.gallery_source(self.value)
