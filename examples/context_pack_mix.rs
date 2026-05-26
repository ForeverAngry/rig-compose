//! Mix context projected by memory and resource crates into one bounded pack.
//!
//! `rig-compose` owns the neutral `ContextItem` / `ContextPack` contract. It
//! does not depend on `rig-memvid` or `rig-resources`; those crates project
//! their native records into `ContextItem`s at the edge. This example starts
//! at that coordinator boundary, mirroring the output of helpers such as
//! `rig_memvid::projection::search_hits_to_context_items` and
//! `rig_resources::projection::*`.

use rig_compose::{ContextItem, ContextPack, ContextPackConfig, ContextSourceKind};
use serde_json::json;

fn memvid_projection_output() -> Vec<ContextItem> {
    vec![
        ContextItem::new(
            ContextSourceKind::Memory,
            "memvid/frame/42",
            "prior incident: host-7 beaconed to the same ASN after credential spray",
        )
        .with_rank(0)
        .with_score(0.93)
        .with_provenance(json!({
            "resource": "memvid.search",
            "frame_id": 42,
            "uri": "memory://incident/42"
        })),
        ContextItem::new(
            ContextSourceKind::Memory,
            "memvid/card/service-account",
            "memory card: svc-deploy usually logs in from build-runner-2",
        )
        .with_rank(2)
        .with_score(0.74)
        .with_provenance(json!({
            "resource": "memvid.card",
            "entity": "svc-deploy",
            "slot": "normal_login_source"
        })),
    ]
}

fn resources_projection_output() -> Vec<ContextItem> {
    vec![
        ContextItem::new(
            ContextSourceKind::Resource,
            "baseline/host-7/egress_fanout",
            "baseline for host-7 egress_fanout: mean 8, std_dev 2, samples 1440",
        )
        .with_rank(1)
        .with_score(1440.0)
        .with_provenance(json!({
            "resource": "baseline",
            "entity": "host-7",
            "metric": "egress_fanout"
        })),
        ContextItem::new(
            ContextSourceKind::Resource,
            "graph/host-7",
            "graph expansion for host-7: 4 nodes, 3 edges",
        )
        .with_rank(3)
        .with_score(7.0)
        .with_provenance(json!({
            "resource": "graph.subgraph",
            "seed": "host-7"
        })),
    ]
}

fn main() {
    let mut items = memvid_projection_output();
    items.extend(resources_projection_output());

    let pack = ContextPack::pack(
        items,
        ContextPackConfig::new(220)
            .with_max_items(3)
            .with_separator("\n---\n"),
    );

    println!("selected: {}", pack.selected.len());
    println!("omitted: {}", pack.omitted.len());
    println!("\n{}", pack.render_text());
}
