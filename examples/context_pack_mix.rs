//! Mix context projected by memory and resource crates into one bounded pack.
//!
//! `rig-compose` owns the neutral `ContextItem` / `ContextPack` contract. It
//! does not depend on `rig-memvid` or `rig-resources`; those crates project
//! their native records into `ContextItem`s at the edge. This example starts
//! at that coordinator boundary, mirroring the output of helpers such as
//! `rig_memvid::projection::search_hits_to_context_items` and
//! `rig_resources::projection::*`.

use rig_compose::{
    ContextItem, ContextPack, ContextPackConfig, ContextProjectionState, ContextProvenance,
    ContextSourceKind,
};
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
        .with_context_provenance(
            ContextProvenance::new()
                .with_source_uri("memory://incident/42")
                .with_source_frame_id("42")
                .with_confidence(0.93)
                .with_version_key("incident:host-7:asn")
                .with_projection_state(ContextProjectionState::Candidate),
        )
        .with_metadata(json!({ "resource": "memvid.search" })),
        ContextItem::new(
            ContextSourceKind::Memory,
            "memvid/card/service-account",
            "memory card: svc-deploy usually logs in from build-runner-2",
        )
        .with_rank(2)
        .with_score(0.74)
        .with_context_provenance(
            ContextProvenance::new()
                .with_principal("svc-deploy")
                .with_scope("prod")
                .with_confidence(0.74)
                .with_version_key("svc-deploy:normal_login_source"),
        )
        .with_metadata(json!({ "resource": "memvid.card", "slot": "normal_login_source" })),
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
        .with_context_provenance(
            ContextProvenance::new()
                .with_source_uri("baseline://host-7/egress_fanout")
                .with_principal("host-7")
                .with_scope("prod")
                .with_confidence(0.88),
        )
        .with_metadata(json!({ "resource": "baseline", "metric": "egress_fanout" })),
        ContextItem::new(
            ContextSourceKind::Resource,
            "graph/host-7",
            "graph expansion for host-7: 4 nodes, 3 edges",
        )
        .with_rank(3)
        .with_score(7.0)
        .with_context_provenance(
            ContextProvenance::new()
                .with_source_uri("graph://host-7")
                .with_principal("host-7")
                .with_projection_state(ContextProjectionState::Expanded)
                .with_reason("multi_hop_neighborhood"),
        )
        .with_metadata(json!({ "resource": "graph.subgraph" })),
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
