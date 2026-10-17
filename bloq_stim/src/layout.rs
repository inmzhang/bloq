use glam::IVec2;
use rustc_hash::FxHashMap;

use crate::StimEmissionError;
use crate::text_utils::push_int;

pub(crate) fn detector_index(index: usize) -> Result<u32, StimEmissionError> {
    u32::try_from(index).map_err(|_| StimEmissionError::IndexOutOfRange {
        space: "detector",
        index,
    })
}

// Stim GateTarget reserves the upper eight bits for target flags.
const MAX_STIM_QUBIT_INDEX: u32 = (1 << 24) - 1;

fn qubit_index(index: usize) -> Result<u32, StimEmissionError> {
    if index > MAX_STIM_QUBIT_INDEX as usize {
        return Err(StimEmissionError::QubitIndexOutOfRange {
            index,
            max: MAX_STIM_QUBIT_INDEX,
        });
    }
    Ok(index as u32)
}

/// Check the generated index range before allocating its coordinate map.
pub(crate) fn coordinate_index(
    coords: impl ExactSizeIterator<Item = IVec2>,
) -> Result<FxHashMap<IVec2, u32>, StimEmissionError> {
    if let Some(last) = coords.len().checked_sub(1) {
        qubit_index(last)?;
    }
    Ok(coords
        .enumerate()
        .map(|(index, coord)| (coord, index as u32))
        .collect())
}

/// Qubit layout for one emission: coordinate → qubit index, plus a cached
/// decimal label per index. Dense labels share one contiguous buffer. Hot emission paths
/// append a qubit target with one lookup and a short `&str` copy instead of
/// hashing the coordinate and re-formatting the index on every instruction
/// line. Lookups go through a dense bounding-box grid when the layout is
/// compact (compiled layouts always are); sparse hand-built layouts fall back
/// to the coordinate map.
#[derive(Debug, Default)]
pub(crate) struct QubitLayout {
    index: FxHashMap<IVec2, u32>,
    grid: Option<CoordGrid>,
    /// Decimal labels of the dense prefix, concatenated in index order.
    labels: String,
    /// `label_starts[q]..label_starts[q + 1]` slices qubit `q`'s label.
    label_starts: Vec<usize>,
    /// Caller-supplied indices outside the dense prefix, cached without filling gaps.
    sparse_labels: FxHashMap<u32, String>,
}

/// Dense qubit-index-plus-one per bounding-box cell (`0` = no qubit). Covers
/// every layout coordinate, so a grid miss is an authoritative "not in layout".
#[derive(Debug)]
struct CoordGrid {
    min: IVec2,
    width: usize,
    height: usize,
    cells: Vec<u32>,
}

impl CoordGrid {
    /// Refuse to densify sparse layouts: the grid must not cost more than a
    /// small constant plus a few words per qubit.
    fn build(index: &FxHashMap<IVec2, u32>) -> Option<Self> {
        let (min, max) = bounding_box(index)?;
        let width = i64::from(max.x) - i64::from(min.x) + 1;
        let height = i64::from(max.y) - i64::from(min.y) + 1;
        let area = u64::try_from(width)
            .ok()?
            .checked_mul(u64::try_from(height).ok()?)?;
        if area > 4096 + 8 * index.len() as u64 {
            return None;
        }
        let width = usize::try_from(width).ok()?;
        let height = usize::try_from(height).ok()?;
        let mut cells = vec![0u32; usize::try_from(area).ok()?];
        for (&coord, &qubit) in index {
            let x = usize::try_from(i64::from(coord.x) - i64::from(min.x)).ok()?;
            let y = usize::try_from(i64::from(coord.y) - i64::from(min.y)).ok()?;
            // QubitLayout checks the 24-bit target range before building the grid.
            cells[y * width + x] = qubit + 1;
        }
        Some(Self {
            min,
            width,
            height,
            cells,
        })
    }

    fn get(&self, coord: IVec2) -> Option<u32> {
        let x = i64::from(coord.x) - i64::from(self.min.x);
        let y = i64::from(coord.y) - i64::from(self.min.y);
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return None;
        }
        let cell = self.cells[(y as usize) * self.width + x as usize];
        cell.checked_sub(1)
    }
}

fn bounding_box(index: &FxHashMap<IVec2, u32>) -> Option<(IVec2, IVec2)> {
    let mut coords = index.keys();
    let first = *coords.next()?;
    Some(coords.fold((first, first), |(min, max), &coord| {
        (min.min(coord), max.max(coord))
    }))
}

impl QubitLayout {
    pub(crate) fn new(index: FxHashMap<IVec2, u32>) -> Result<Self, StimEmissionError> {
        if let Some(&maximum) = index.values().max() {
            qubit_index(maximum as usize)?;
        }
        // Compiler layouts are dense. Caller-supplied plan layouts can have
        // sparse indices, so no allocation may depend on the largest index.
        let count = index.len();
        let mut labels = String::new();
        let mut label_starts = Vec::with_capacity(count + 1);
        for qubit in 0..count {
            label_starts.push(labels.len());
            push_int(&mut labels, qubit);
        }
        label_starts.push(labels.len());
        let sparse_labels = index
            .values()
            .copied()
            .filter(|&qubit| qubit as usize >= count)
            .map(|qubit| (qubit, qubit.to_string()))
            .collect();
        let grid = CoordGrid::build(&index);
        Ok(Self {
            index,
            grid,
            labels,
            label_starts,
            sparse_labels,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.index.len()
    }

    #[cfg(any(test, feature = "verify"))]
    pub(crate) fn coord_index(&self) -> &FxHashMap<IVec2, u32> {
        &self.index
    }

    pub(crate) fn get(&self, coord: IVec2) -> Option<u32> {
        match &self.grid {
            Some(grid) => grid.get(coord),
            None => self.index.get(&coord).copied(),
        }
    }

    /// The pre-rendered decimal label of `qubit`.
    pub(crate) fn label(&self, qubit: u32) -> &str {
        let index = qubit as usize;
        if index < self.index.len() {
            let start = self.label_starts[index];
            let end = self.label_starts[index + 1];
            &self.labels[start..end]
        } else {
            &self.sparse_labels[&qubit]
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::ivec2;

    use super::*;

    #[test]
    fn detector_indices_do_not_wrap() {
        let maximum = u32::MAX as usize;
        assert_eq!(detector_index(maximum), Ok(u32::MAX));
        if let Some(index) = maximum.checked_add(1) {
            assert_eq!(
                detector_index(index),
                Err(StimEmissionError::IndexOutOfRange {
                    space: "detector",
                    index,
                })
            );
        }
    }

    #[test]
    fn generated_qubit_indices_are_checked_before_consuming_coordinates() {
        let count = (1 << 24) + 1;
        let coords = std::iter::repeat_n(IVec2::ZERO, count)
            .map(|_| panic!("an oversized coordinate layout must not be materialized"));
        assert_eq!(
            coordinate_index(coords),
            Err(StimEmissionError::QubitIndexOutOfRange {
                index: 1 << 24,
                max: 16_777_215,
            })
        );
        assert_eq!(
            coordinate_index(std::iter::empty()).unwrap(),
            FxHashMap::default()
        );
        assert_eq!(qubit_index(16_777_215), Ok(16_777_215));
    }

    #[test]
    fn sparse_indices_do_not_expand_the_label_cache_to_the_largest_id() {
        let index = FxHashMap::from_iter([(ivec2(0, 0), 100_000)]);
        let layout = QubitLayout::new(index).unwrap();
        assert_eq!(layout.label(100_000), "100000");
        assert!(layout.label_starts.len() <= 2);
    }

    #[test]
    fn dense_and_sparse_layouts_agree() {
        // Compact layout densifies; the far-away outlier forces the fallback.
        let compact: FxHashMap<IVec2, u32> =
            [(ivec2(0, 0), 0), (ivec2(2, 1), 1), (ivec2(-1, 3), 2)]
                .into_iter()
                .collect();
        let mut sparse = compact.clone();
        sparse.insert(ivec2(1_000_000, -1_000_000), 3);

        let dense_layout = QubitLayout::new(compact.clone()).unwrap();
        assert!(dense_layout.grid.is_some(), "compact layout should densify");
        let sparse_layout = QubitLayout::new(sparse.clone()).unwrap();
        assert!(
            sparse_layout.grid.is_none(),
            "outlier should force fallback"
        );

        for (layout, index) in [(&dense_layout, &compact), (&sparse_layout, &sparse)] {
            for (coord, &qubit) in index {
                assert_eq!(layout.get(*coord), Some(qubit));
                assert_eq!(layout.label(qubit), qubit.to_string());
            }
            assert_eq!(layout.get(ivec2(5, -7)), None);
        }
    }

    #[test]
    fn extreme_coordinates_fall_back_without_overflow() {
        let extreme = FxHashMap::from_iter([
            (ivec2(i32::MIN, i32::MIN), 0),
            (ivec2(i32::MAX, i32::MAX), 1),
        ]);
        let layout = QubitLayout::new(extreme.clone()).unwrap();
        assert!(layout.grid.is_none());
        for (coord, qubit) in extreme {
            assert_eq!(layout.get(coord), Some(qubit));
        }

        let compact = QubitLayout::new(FxHashMap::from_iter([(ivec2(0, 0), 0)])).unwrap();
        assert_eq!(compact.get(ivec2(i32::MIN, i32::MAX)), None);
    }
}
