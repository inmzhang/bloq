//! Shared layered placement and edge routing for dependency graphs.

use std::collections::{HashMap, VecDeque};

/// Cubic Bézier segments, each containing start, two controls, and end.
pub type CubicSpline = Vec<[(f32, f32); 4]>;

/// Cached geometry at zoom 1.0. Renderers apply pan and zoom without relayout.
#[derive(Debug)]
pub struct RelativeLayout {
    /// Normalized top-left coordinates, keyed by node id.
    pub positions: HashMap<u32, (f32, f32)>,
    /// Routed splines in the original input edge order, including parallel edges.
    pub routes: Vec<CubicSpline>,
    /// Bounding box covering nodes and routed edges.
    pub content: (f32, f32),
}

/// Place sized nodes in dependency order and route edges through free gutters.
///
/// Regions are opaque rectangles; callers size their contents bottom-up.
/// Node ids must be unique, sizes and gap finite, and all endpoints present.
///
/// # Panics
///
/// Panics if an edge refers to an absent node.
pub fn compute_relative(
    nodes: &[(u32, (f32, f32))],
    edges: &[(u32, u32)],
    gap: f32,
) -> RelativeLayout {
    let gap = gap.max(1.0);
    let indices = nodes
        .iter()
        .enumerate()
        .map(|(index, &(id, _))| (id, index))
        .collect::<HashMap<_, _>>();
    let edges = edges
        .iter()
        .map(|&(from, to)| (indices[&from], indices[&to]))
        .collect::<Vec<_>>();
    let mut outgoing = vec![Vec::new(); nodes.len()];
    let mut incoming = vec![0; nodes.len()];
    for &(from, to) in &edges {
        outgoing[from].push(to);
        incoming[to] += 1;
    }

    let mut remaining = incoming.clone();
    let mut ready = (0..nodes.len())
        .filter(|&node| remaining[node] == 0)
        .collect::<VecDeque<_>>();
    let mut rank = vec![0_usize; nodes.len()];
    let mut placed = vec![false; nodes.len()];
    // Break cycles in input order. Their backward edges still get routed.
    let mut seeds = 0..nodes.len();
    while let Some(node) = ready.pop_front().or_else(|| seeds.find(|&id| !placed[id])) {
        placed[node] = true;
        for &next in &outgoing[node] {
            if !placed[next] {
                rank[next] = rank[next].max(rank[node] + 1);
                remaining[next] -= 1;
                if remaining[next] == 0 {
                    ready.push_back(next);
                }
            }
        }
    }

    let long_edges = edges
        .iter()
        .filter(|&&(from, to)| rank[from].abs_diff(rank[to]) > 1)
        .count();
    let top = (8.0 * long_edges as f32).min(3.0 * gap);
    let columns = rank.iter().max().map_or(0, |rank| rank + 1);
    let mut widths = vec![0.0_f32; columns];
    let mut bottoms = vec![top; columns];
    let mut tops = Vec::with_capacity(nodes.len());
    for (&(_, (width, height)), &column) in nodes.iter().zip(&rank) {
        widths[column] = widths[column].max(width);
        tops.push(bottoms[column]);
        bottoms[column] += height + gap / 2.0;
    }
    let mut left = 0.0;
    let columns = widths
        .iter()
        .map(|&width| {
            let bounds = (left, left + width);
            left = bounds.1 + gap;
            bounds
        })
        .collect::<Vec<_>>();
    let positions = nodes
        .iter()
        .zip(&rank)
        .zip(&tops)
        .map(|((&(id, _), &column), &y)| (id, (columns[column].0, y)))
        .collect();

    let mut next_out = vec![0; nodes.len()];
    let mut next_in = vec![0; nodes.len()];
    let mut lane = 0;
    let mut right = columns.last().map_or(0.0, |&(_, right)| right);
    let routes = edges
        .into_iter()
        .map(|(from, to)| {
            let (from_rank, to_rank) = (rank[from], rank[to]);
            let (from_left, from_right) = columns[from_rank];
            let (to_left, to_right) = columns[to_rank];
            let (from_width, from_height) = nodes[from].1;
            let (to_width, to_height) = nodes[to].1;
            next_out[from] += 1;
            next_in[to] += 1;
            let anchor = |y, height, index, count| {
                y + height * (0.2 + 0.6 * index as f32 / (count + 1) as f32)
            };
            let sy = anchor(
                tops[from],
                from_height,
                next_out[from],
                outgoing[from].len(),
            );
            let mut ty = anchor(tops[to], to_height, next_in[to], incoming[to]);
            if from == to && sy == ty {
                ty += to_height * 0.15;
            }
            let (sx, entry, exit, tx) = match from_rank.cmp(&to_rank) {
                std::cmp::Ordering::Less => (from_left + from_width, from_right, to_left, to_left),
                std::cmp::Ordering::Greater => (from_left, from_left, to_right, to_left + to_width),
                std::cmp::Ordering::Equal => (
                    from_left + from_width,
                    from_right,
                    from_right,
                    to_left + to_width,
                ),
            };
            let bend = |(x, y), (u, v)| {
                let middle = f32::midpoint(x, u);
                [(x, y), (middle, y), (middle, v), (u, v)]
            };
            let mut route = Vec::with_capacity(5);
            route.push(bend((sx, sy), (entry, sy)));
            match from_rank.abs_diff(to_rank) {
                0 => {
                    let x = entry + gap * 0.7;
                    route.push([(entry, sy), (x, sy), (x, ty), (exit, ty)]);
                    right = right.max(x);
                }
                1 => route.push(bend((entry, sy), (exit, ty))),
                _ => {
                    // shortcut: long edges share the space above the columns;
                    // allocate separate channels if dense graphs need wider separation.
                    lane += 1;
                    let y = top * (lane as f32 / (long_edges + 1) as f32);
                    let offset = if from_rank < to_rank { gap } else { -gap } / 2.0;
                    let over_source = (entry + offset, y);
                    let over_target = (exit - offset, y);
                    route.extend([
                        bend((entry, sy), over_source),
                        bend(over_source, over_target),
                        bend(over_target, (exit, ty)),
                    ]);
                }
            }
            route.push(bend((exit, ty), (tx, ty)));
            route.retain(|[start, first, second, end]| {
                start != end || start != first || start != second
            });
            route
        })
        .collect();
    RelativeLayout {
        positions,
        routes,
        content: (
            right,
            bottoms.into_iter().fold(gap / 2.0, f32::max) - gap / 2.0,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Dependency order advances left to right, including alongside isolated nodes.
    #[test]
    fn chain_places_left_to_right_and_keeps_isolated_nodes() {
        let size = (130.0, 46.0);
        let nodes = [(0, size), (1, size), (2, size), (3, size)];
        let edges = [(0, 1), (1, 2)];

        let layout = compute_relative(&nodes, &edges, 28.0);

        assert_eq!(
            layout.positions.len(),
            4,
            "every node, including the isolated one, places"
        );
        let col = |id: u32| layout.positions[&id].0;
        assert!(col(0) < col(1), "successor sits in a later column");
        assert!(col(1) < col(2), "chain keeps advancing rightwards");
    }

    #[test]
    fn long_edges_route_around_wide_nodes() {
        let small = (180.0, 60.0);
        let large = (400.0, 250.0);
        let nodes = [(0, small), (1, large), (2, small)];
        let edges = [(0, 1), (1, 2), (0, 2)];
        let layout = compute_relative(&nodes, &edges, 40.0);
        let (x, y) = layout.positions[&1];
        let (end_x, _) = layout.positions[&2];
        assert!(layout.positions[&0].0 + small.0 <= x);
        assert!(x + large.0 <= end_x);
        assert!(layout.content.0 >= end_x + small.0);
        assert!(layout.content.1 >= y + large.1);

        for segment in &layout.routes[2] {
            for step in 0..=100 {
                let t = step as f32 / 100.0;
                let u = 1.0 - t;
                let weights = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
                let (px, py) = segment
                    .iter()
                    .zip(weights)
                    .fold((0.0, 0.0), |(px, py), (&(cx, cy), weight)| {
                        (px + cx * weight, py + cy * weight)
                    });
                assert!(
                    !(x < px && px < x + large.0 && y < py && py < y + large.1),
                    "edge enters intervening region at {px},{py}; box={x},{y}; route={:?}",
                    layout.routes[2]
                );
            }
        }
    }

    #[test]
    fn parallel_cycles_and_self_edges_keep_identity_and_bounds() {
        let nodes = [4, 9, 21].map(|id| (id, (180.0, 60.0)));
        let edges = [(4, 9), (4, 9), (9, 21), (21, 4), (9, 9)];
        let first = compute_relative(&nodes, &edges, 40.0);
        let second = compute_relative(&nodes, &edges, 40.0);
        assert_eq!(first.positions, second.positions);
        assert_eq!(first.routes, second.routes);
        assert_ne!(first.routes[0], first.routes[1]);
        assert_eq!(first.routes.len(), edges.len());
        for route in &first.routes {
            assert!(!route.is_empty());
            for &(x, y) in route.iter().flatten() {
                assert!(x.is_finite() && y.is_finite());
                assert!(x >= 0.0 && x <= first.content.0);
                assert!(y >= 0.0 && y <= first.content.1);
            }
        }
    }
}
