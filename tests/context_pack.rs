//! Integration tests for provider-neutral context packing.
//!
//! These tests intentionally exercise [`ContextPack`] against more than one
//! source kind so the kernel vocabulary is validated against multiple in-crate
//! sources before any cross-crate adapter work begins. See the
//! "Validate Against ≥2 Sources Before Building Adapters" practice in
//! `rig-contributions/docs/development-standards.md`.

use rig_compose::{
    ContextItem, ContextOmissionReason, ContextPack, ContextPackConfig, ContextSourceKind,
    ToolInvocation, ToolInvocationResult,
};
use serde_json::json;

#[test]
fn context_item_defaults_estimated_chars_and_preserves_metadata() {
    let item = ContextItem::new(
        ContextSourceKind::Memory,
        "memory/card/1",
        "alice lives in Berlin",
    )
    .with_rank(2)
    .with_score(7.5)
    .with_provenance(json!({"frame_id": 42, "uri": "memory://alice"}))
    .with_metadata(json!({"entity": "alice", "slot": "location"}));

    assert_eq!(item.source, ContextSourceKind::Memory);
    assert_eq!(item.source_id, "memory/card/1");
    assert_eq!(item.rank, 2);
    assert_eq!(item.score, 7.5);
    assert_eq!(item.estimated_chars, item.text.chars().count());
    assert_eq!(item.provenance["frame_id"], json!(42));
    assert_eq!(item.metadata["slot"], json!("location"));
}

#[test]
fn context_pack_sorts_by_rank_before_packing() {
    let pack = ContextPack::pack(
        vec![
            fixture(2, "third"),
            fixture(0, "first"),
            fixture(1, "second"),
        ],
        ContextPackConfig::new(100),
    );

    assert_eq!(pack.render_text(), "first\nsecond\nthird");
    assert_eq!(
        pack.selected
            .iter()
            .map(|item| item.rank)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}

#[test]
fn context_pack_records_over_budget_omissions() {
    let pack = ContextPack::pack(vec![fixture(0, "too long")], ContextPackConfig::new(3));

    assert!(pack.selected.is_empty());
    assert_eq!(pack.omitted.len(), 1);
    assert_eq!(
        pack.omitted.first().map(|omitted| &omitted.reason),
        Some(&ContextOmissionReason::OverBudget)
    );
    assert!(pack.render_text().is_empty());
}

#[test]
fn context_pack_records_max_item_omissions() {
    let pack = ContextPack::pack(
        vec![fixture(0, "first"), fixture(1, "second")],
        ContextPackConfig::new(100).with_max_items(1),
    );

    assert_eq!(pack.selected.len(), 1);
    assert_eq!(pack.omitted.len(), 1);
    assert_eq!(
        pack.omitted.first().map(|omitted| &omitted.reason),
        Some(&ContextOmissionReason::MaxItems)
    );
}

#[test]
fn context_pack_accounts_for_separator_budget() {
    let config =
        ContextPackConfig::new("alpha".len() + " | ".len() + "bravo".len()).with_separator(" | ");
    let pack = ContextPack::pack(vec![fixture(0, "alpha"), fixture(1, "bravo")], config);

    assert_eq!(pack.render_text(), "alpha | bravo");
    assert_eq!(pack.total_estimated_chars, "alpha | bravo".chars().count());
}

#[test]
fn context_pack_reserves_non_context_budget() {
    let pack = ContextPack::pack(
        vec![fixture(0, "alpha")],
        ContextPackConfig::new(10).with_reserve_chars(5),
    );

    assert_eq!(pack.selected.len(), 1);
    assert_eq!(pack.total_estimated_chars, "alpha".chars().count());
}

#[test]
fn context_pack_saturates_when_reserve_exceeds_budget() {
    let pack = ContextPack::pack(
        vec![fixture(0, "alpha")],
        ContextPackConfig::new(4).with_reserve_chars(40),
    );

    assert!(pack.selected.is_empty());
    assert_eq!(pack.omitted.len(), 1);
    assert_eq!(
        pack.omitted.first().map(|omitted| &omitted.reason),
        Some(&ContextOmissionReason::OverBudget)
    );
}

fn fixture(rank: usize, text: &str) -> ContextItem {
    ContextItem::new(ContextSourceKind::Memory, format!("fixture-{rank}"), text).with_rank(rank)
}

// ── Second-source coverage ──────────────────────────────────────────────────
//
// The following tests pack non-`Memory` sources so `ContextPack` is exercised
// against vocabulary it would have to support for any future cross-crate
// adapter. They are deliberately structured to mirror the memory-source tests
// above so regressions show up symmetrically across source kinds.

fn tool_result(name: &str, output: serde_json::Value) -> ToolInvocationResult {
    let invocation =
        ToolInvocation::new(name, json!({})).expect("tool name is a valid identifier in tests");
    ToolInvocationResult { invocation, output }
}

fn tool_context_item(rank: usize, result: &ToolInvocationResult) -> ContextItem {
    let text = result.output.to_string();
    ContextItem::new(
        ContextSourceKind::ToolResult,
        format!("tool/{}/{rank}", result.invocation.name),
        text,
    )
    .with_rank(rank)
    .with_provenance(json!({
        "tool": result.invocation.name.as_str(),
        "args": result.invocation.args,
    }))
}

fn resource_item(rank: usize, uri: &str, text: &str) -> ContextItem {
    ContextItem::new(ContextSourceKind::Resource, uri.to_owned(), text)
        .with_rank(rank)
        .with_provenance(json!({ "uri": uri }))
}

#[test]
fn context_pack_packs_tool_invocation_results() {
    let weather = tool_result("get_weather", json!({ "city": "Berlin", "tempC": 12 }));
    let calendar = tool_result(
        "list_events",
        json!({ "today": ["standup", "design review"] }),
    );

    let pack = ContextPack::pack(
        vec![
            tool_context_item(1, &calendar),
            tool_context_item(0, &weather),
        ],
        ContextPackConfig::new(1_000),
    );

    assert_eq!(pack.selected.len(), 2);
    assert_eq!(pack.selected[0].source, ContextSourceKind::ToolResult);
    assert_eq!(pack.selected[0].rank, 0);
    assert_eq!(pack.selected[1].rank, 1);
    assert!(pack.render_text().contains("Berlin"));
    assert!(pack.render_text().contains("standup"));
    assert_eq!(pack.selected[0].provenance["tool"], json!("get_weather"));
}

#[test]
fn context_pack_records_tool_result_over_budget_omissions() {
    let bulky = tool_result("dump_state", json!({ "blob": "x".repeat(64) }));
    let pack = ContextPack::pack(
        vec![tool_context_item(0, &bulky)],
        ContextPackConfig::new(8),
    );

    assert!(pack.selected.is_empty());
    assert_eq!(
        pack.omitted.first().map(|omitted| &omitted.reason),
        Some(&ContextOmissionReason::OverBudget)
    );
}

#[test]
fn context_pack_mixes_memory_tool_and_resource_sources() {
    let weather = tool_result("get_weather", json!({ "city": "Berlin" }));

    let pack = ContextPack::pack(
        vec![
            fixture(2, "memory: alice prefers afternoon meetings"),
            resource_item(0, "policy://meetings", "policy: keep meetings under 30m"),
            tool_context_item(1, &weather),
        ],
        ContextPackConfig::new(1_000),
    );

    let kinds: Vec<_> = pack
        .selected
        .iter()
        .map(|item| item.source.clone())
        .collect();
    assert_eq!(
        kinds,
        vec![
            ContextSourceKind::Resource,
            ContextSourceKind::ToolResult,
            ContextSourceKind::Memory,
        ]
    );
    assert!(pack.omitted.is_empty());
}

#[test]
fn context_pack_max_items_applies_uniformly_across_sources() {
    let weather = tool_result("get_weather", json!({ "city": "Berlin" }));

    let pack = ContextPack::pack(
        vec![
            resource_item(0, "doc://intro", "resource one"),
            tool_context_item(1, &weather),
            fixture(2, "memory item"),
        ],
        ContextPackConfig::new(1_000).with_max_items(2),
    );

    assert_eq!(pack.selected.len(), 2);
    assert_eq!(pack.omitted.len(), 1);
    let omitted = pack.omitted.first().expect("one omitted item");
    assert_eq!(omitted.reason, ContextOmissionReason::MaxItems);
    assert_eq!(omitted.item.source, ContextSourceKind::Memory);
}
