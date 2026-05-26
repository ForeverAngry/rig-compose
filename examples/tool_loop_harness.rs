use std::sync::Arc;

use rig_compose::normalizer::{LfmNormalizer, ToolCallNormalizer};
use rig_compose::{
    KernelError, LocalTool, ToolInvocation, ToolInvocationResult, ToolRegistry, ToolSchema,
    dispatch_tool_invocations,
};
use serde_json::json;

#[path = "common/harness_record.rs"]
mod harness_record;

use harness_record::HarnessRun;

#[tokio::main]
async fn main() -> Result<(), KernelError> {
    let tools = ToolRegistry::new();
    tools.register(Arc::new(LocalTool::new(
        ToolSchema {
            name: "get_weather".into(),
            description: "gets weather".into(),
            args_schema: json!({"type": "object"}),
            result_schema: json!({"type": "object"}),
        },
        |args| async move {
            Ok(json!({
                "city": args.get("city").and_then(|value| value.as_str()).unwrap_or("unknown"),
                "forecast": "clear and cool"
            }))
        },
    )));

    let run = run_tool_loop_harness(
        &tools,
        "What is the weather like in Berlin today?",
        "<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>",
    )
    .await?;

    let json = serde_json::to_string_pretty(&run)
        .map_err(|error| KernelError::ToolFailed(error.to_string()))?;
    println!("{json}");

    Ok(())
}

async fn run_tool_loop_harness(
    tools: &ToolRegistry,
    task: &str,
    first_model_output: &str,
) -> Result<HarnessRun, KernelError> {
    let invocations = LfmNormalizer.normalize(first_model_output)?;
    let dispatch_results = dispatch_tool_invocations(tools, &invocations).await?;
    let final_answer = fake_second_model_turn(&dispatch_results);
    let passed_assertions = harness_assertions(&invocations, &dispatch_results, &final_answer);

    Ok(HarnessRun::from_native(
        "rig-compose/examples/tool_loop_harness",
        task,
        first_model_output,
        &invocations,
        &dispatch_results,
        final_answer,
        passed_assertions,
    ))
}

fn harness_assertions(
    invocations: &[ToolInvocation],
    dispatch_results: &[ToolInvocationResult],
    final_answer: &str,
) -> Vec<&'static str> {
    let mut passed = Vec::new();

    if !invocations.is_empty() {
        passed.push("model-output-normalized");
    }
    if dispatch_results
        .iter()
        .any(|result| result.invocation.name == "get_weather")
    {
        passed.push("tool-dispatched");
    }
    if final_answer.contains("Berlin") && final_answer.contains("clear and cool") {
        passed.push("final-answer-grounded");
    }

    passed
}

fn fake_second_model_turn(results: &[ToolInvocationResult]) -> String {
    results
        .first()
        .and_then(|result| {
            let city = result.output.get("city")?.as_str()?;
            let forecast = result.output.get("forecast")?.as_str()?;
            Some(format!("The weather in {city} is {forecast}."))
        })
        .unwrap_or_else(|| "No tool result was available.".to_string())
}
