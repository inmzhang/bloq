//! Compare planning: `profile_source_planner WIDTH|yoke:WIDTH|file:PATH [--samples N]`.
//! Add `--flat` to run only an explicitly flattened solve with no module-site
//! metadata or prepared relations, e.g. `profile_source_planner 3 --flat --samples 3`.
//! Samples rebuild planners in one process; source construction and flattening run once.
//! Preparation includes module validation; reference timing covers planning only.
//! No physical execution.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use bloq_graph::{GuardedSurfaceSpace, GuardedTopology, ModuleCertificationLimits};

fn main() {
    let mut args = std::env::args().skip(1);
    let case = args.next().expect("width, yoke:width or file:path");
    let mut flat = false;
    let mut samples = 1;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--flat" => flat = true,
            "--samples" => {
                samples = args
                    .next()
                    .expect("sample count after --samples")
                    .parse::<usize>()
                    .expect("positive sample count");
            }
            _ => panic!("unknown argument: {arg}"),
        }
    }
    assert!(samples > 0, "sample count must be positive");
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
    let flattening_start = Instant::now();
    let linked = bloq_graph::flatten_module_definition(&authored, authored.root(), "")
        .expect("diagnostic source geometry");
    let graph = linked.graph.fix_shadowed_faces();
    let flattening_ms = flattening_start.elapsed().as_secs_f64() * 1000.0;
    let source_blocks = graph.blocks().count();
    let source_pipes = graph.pipes().count();
    let empty_sites = HashMap::new();
    let sites = if flat { &empty_sites } else { &linked.sites };
    let limits = ModuleCertificationLimits::DEFAULT;
    let mut reference_supports = None;
    let modes: &[&str] = if flat {
        &["flat"]
    } else {
        &["reference", "module_relations", "module_relations_warm"]
    };
    for sample in 1..=samples {
        let mut prepared = None;
        for &mode in modes {
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
                .map_or((0, 0), |prepared| prepared.relation_work());
            let start = Instant::now();
            let topology = GuardedTopology::new(&graph, limits).expect("source topology");
            let topology_ms = start.elapsed().as_secs_f64() * 1000.0;
            let topology_boolean_steps = topology.diagram.steps();
            let topology_boolean_nodes = topology.diagram.nodes().len();
            let mut scope_sites = BTreeMap::<&str, usize>::new();
            for &position in topology.sites.keys() {
                let scope = sites
                    .get(&glam::IVec3::from_array(position))
                    .map(|site| site.instance_path.as_str())
                    .unwrap_or("");
                *scope_sites.entry(scope).or_default() += 1;
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
                Some(prepared) => prepared.compose(topology, sites),
                None => GuardedSurfaceSpace::new(topology, sites, limits),
            }
            .expect("signed source relation");
            let composition_ms = relation_start.elapsed().as_secs_f64() * 1000.0;
            let relation_boolean_steps = relation.topology.diagram.steps() - topology_boolean_steps;
            let relation_boolean_nodes = relation.topology.diagram.nodes().len();
            let before_readout_steps = relation.topology.diagram.steps();
            let (relation_computed, relation_reused) = prepared
                .as_ref()
                .map_or((0, 0), |prepared| prepared.relation_work());
            let readout_start = Instant::now();
            let plan = relation
                .plan_readouts()
                .expect("complete generic readout plan");
            let readout_planning_ms = readout_start.elapsed().as_secs_f64() * 1000.0;
            let boolean_steps = plan.topology.diagram.steps();
            let readout_boolean_steps = boolean_steps - before_readout_steps;
            let planning_ms = topology_ms + composition_ms + readout_planning_ms;
            let cold_planning_ms = preparation_ms + planning_ms;
            let support_sites = (0..plan.surfaces.len())
                .map(|index| plan.support_sites(index))
                .collect::<Vec<_>>();
            let reference = reference_supports.get_or_insert_with(|| support_sites.clone());
            assert_eq!(
                &support_sites, reference,
                "reuse must preserve query supports"
            );
            let support_site_visits = support_sites.iter().map(BTreeSet::len).sum::<usize>();
            let max_query_support = support_sites.iter().map(BTreeSet::len).max().unwrap_or(0);
            let fragments = (0..plan.surfaces.len())
                .flat_map(|index| plan.query_fragments(index).iter().copied())
                .collect::<BTreeSet<_>>();
            let fragment_sites = fragments
                .iter()
                .map(|&id| plan.query_fragment(id).1.len())
                .sum::<usize>();
            println!(
                "{{\"source_kind\":\"{source_kind}\",\
                 \"bits\":{bits},\
                 \"mode\":\"{mode}\",\
                 \"sample\":{sample},\
                 \"source_blocks\":{source_blocks},\
                 \"source_pipes\":{source_pipes},\
                 \"flattening_ms\":{flattening_ms},\
                 \"preparation_ms\":{preparation_ms},\
                 \"cold_planning_ms\":{cold_planning_ms},\
                 \"planning_ms\":{planning_ms},\
                 \"topology_ms\":{topology_ms},\
                 \"composition_ms\":{composition_ms},\
                 \"readout_planning_ms\":{readout_planning_ms},\
                 \"prepared_computed\":{prepared_computed},\
                 \"prepared_reused\":{prepared_reused},\
                 \"relation_computed\":{relation_computed},\
                 \"relation_reused\":{relation_reused},\
                 \"root_sites\":{root_sites},\
                 \"largest_child_scope\":{largest_child_scope},\
                 \"topology_boolean_steps\":{topology_boolean_steps},\
                 \"relation_boolean_steps\":{relation_boolean_steps},\
                 \"readout_boolean_steps\":{readout_boolean_steps},\
                 \"boolean_steps\":{boolean_steps},\
                 \"topology_boolean_nodes\":{topology_boolean_nodes},\
                 \"relation_boolean_nodes\":{relation_boolean_nodes},\
                 \"queries\":{},\
                 \"support_site_visits\":{},\
                 \"max_query_support\":{},\
                 \"boolean_nodes\":{},\
                 \"distinct_fragments\":{},\
                 \"fragment_site_visits\":{fragment_sites}}}",
                support_sites.len(),
                support_site_visits,
                max_query_support,
                plan.topology.diagram.nodes().len(),
                fragments.len()
            );
        }
    }
}
