//! Print layout metrics of BLOG files as JSON lines, with the compiler's own accounting.
//!
//! usage: measure FILE.blog...

use bloq_graph::BlockGraph;
use serde_json::json;

fn main() {
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).expect("readable BLOG file");
        let row = match BlockGraph::from_blog_text(&text) {
            Ok(graph) => {
                let m = bloq_tblock::metrics(&graph);
                json!({"path": path, "extents": m.extents, "volume": m.volume,
                       "occupied": m.occupied, "t_blocks": m.t_blocks,
                       "selective": m.selective, "blocks": m.blocks})
            }
            Err(e) => json!({"path": path, "error": e.to_string()}),
        };
        println!("{row}");
    }
}
