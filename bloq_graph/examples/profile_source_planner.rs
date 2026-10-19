//! Compare planning: `profile_source_planner WIDTH|yoke:WIDTH|file:PATH`.
//! Preparation includes module validation; reference timing covers planning only.
//! No physical execution.

use std::time::Instant;

use bloq_graph::{GuardedSurfaceSpace, GuardedTopology, ModuleCertificationLimits};

fn main() {
    let case = std::env::args()
        .nth(1)
        .expect("width, yoke:width, or file:path");
    let (source_kind, bits, authored) = if let Some(path) = case.strip_prefix("file:") {
        let source = std::fs::read_to_string(path).expect("source file");
        (
            "authored_file",
            0,
            bloq_graph::BlockGraph::from_text(&source).expect("authored source"),
        )
    } else if let Some(width) = case.strip_prefix("yoke:") {
        let width = width.parse().expect("yoke width");
        (
            "yoked_memory",
            width,
            bloq_test::benchmark::yoked_memory(width),
        )
    } else {
        let bits = case.parse().expect("adder width");
        assert!(bits >= 3, "benchmark width must be at least three bits");
        (
            "controlled_adder",
            bits,
            bloq_test::benchmark::controlled_adder(bits),
        )
    };
    let linked = bloq_graph::flatten_module_definition(&authored, authored.root(), "")
        .expect("diagnostic source geometry");
    let graph = linked.graph.fix_shadowed_faces();
    let limits = ModuleCertificationLimits::DEFAULT;
    let mut reference_supports = None;
    let mut prepared = None;
    for mode in ["reference", "module_relations", "module_relations_warm"] {
        let preparation_start = Instant::now();
        let preparation_ms = if mode == "module_relations" {
            prepared = Some(
                authored
                    .prepare_module_relations_with_cache(
                        limits,
                        bloq_graph::default_module_jobs(),
                        |_| None,
                    )
                    .expect("prepared module relations"),
            );
            preparation_start.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let (prepared_computed, prepared_reused) = prepared
            .as_ref()
            .map(|prepared| prepared.relation_work())
            .unwrap_or_default();
        let start = Instant::now();
        let topology = GuardedTopology::new(&graph, limits).expect("source topology");
        let topology_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut scope_sites = std::collections::BTreeMap::<String, usize>::new();
        for &position in topology.sites.keys() {
            let scope = linked
                .sites
                .get(&glam::IVec3::from_array(position))
                .map(|site| site.instance_path.as_str())
                .unwrap_or("");
            *scope_sites.entry(scope.to_owned()).or_default() += 1;
        }
        let root_sites = scope_sites.get("").copied().unwrap_or(0);
        let largest_child_scope = scope_sites
            .iter()
            .filter(|(scope, _)| !scope.is_empty())
            .map(|(_, &count)| count)
            .max()
            .unwrap_or(0);
        let relation_start = Instant::now();
        let relation = match &prepared {
            Some(prepared) => prepared.compose(topology, &linked.sites),
            None => GuardedSurfaceSpace::new(topology, &linked.sites, limits),
        }
        .expect("signed source relation");
        let composition_ms = relation_start.elapsed().as_secs_f64() * 1000.0;
        let (relation_computed, relation_reused) = prepared
            .as_ref()
            .map(|prepared| prepared.relation_work())
            .unwrap_or_default();
        let readout_start = Instant::now();
        let plan = relation
            .plan_readouts()
            .expect("complete generic readout plan");
        let readout_planning_ms = readout_start.elapsed().as_secs_f64() * 1000.0;
        let planning_ms = topology_ms + composition_ms + readout_planning_ms;
        let cold_planning_ms = preparation_ms + planning_ms;
        let support_sites = (0..plan.surfaces.len())
            .map(|index| plan.support_sites(index))
            .collect::<Vec<_>>();
        if let Some(reference) = &reference_supports {
            assert_eq!(
                &support_sites, reference,
                "reuse must preserve query supports"
            );
        } else {
            reference_supports = Some(support_sites.clone());
        }
        let supports = support_sites
            .iter()
            .map(std::collections::BTreeSet::len)
            .collect::<Vec<_>>();
        let fragments = (0..plan.surfaces.len())
            .flat_map(|index| plan.query_fragments(index).iter().copied())
            .collect::<std::collections::BTreeSet<_>>();
        let fragment_sites = fragments
            .iter()
            .map(|&id| plan.query_fragment(id).1.len())
            .sum::<usize>();
        println!(
            "{{\"source_kind\":\"{source_kind}\",\"bits\":{bits},\"mode\":\"{mode}\",\"preparation_ms\":{preparation_ms},\"cold_planning_ms\":{cold_planning_ms},\"planning_ms\":{planning_ms},\"topology_ms\":{topology_ms},\"composition_ms\":{composition_ms},\"readout_planning_ms\":{readout_planning_ms},\"prepared_computed\":{prepared_computed},\"prepared_reused\":{prepared_reused},\"relation_computed\":{relation_computed},\"relation_reused\":{relation_reused},\"root_sites\":{root_sites},\"largest_child_scope\":{largest_child_scope},\"queries\":{},\"support_site_visits\":{},\"max_query_support\":{},\"boolean_nodes\":{},\"distinct_fragments\":{},\"fragment_site_visits\":{fragment_sites}}}",
            supports.len(),
            supports.iter().sum::<usize>(),
            supports.iter().max().copied().unwrap_or(0),
            plan.topology.diagram.nodes().len(),
            fragments.len()
        );
    }
}
