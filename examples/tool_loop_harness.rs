use std::sync::Arc;

use rig_compose::normalizer::{LfmNormalizer, ToolCallNormalizer};
use rig_compose::{
    KernelError, LocalTool, ToolInvocation, ToolInvocationResult, ToolRegistry, ToolSchema,
    dispatch_tool_invocations,
};
use serde_json::json;

#[derive(Debug)]
struct ToolLoopHarnessRun {
    task: String,
    first_model_output: String,
    invocations: Vec<ToolInvocation>,
    dispatch_results: Vec<ToolInvocationResult>,
    final_answer: String,
    passed_assertions: Vec<&'static str>,
}

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

    println!("task: {}", run.task);
    println!("first model output: {}", run.first_model_output);
    println!("invocations: {:?}", run.invocations);
    println!("tool results: {:?}", run.dispatch_results);
    println!("final answer: {}", run.final_answer);
    println!("passed assertions: {:?}", run.passed_assertions);

    Ok(())
}

async fn run_tool_loop_harness(
    tools: &ToolRegistry,
    task: &str,
    first_model_output: &str,
) -> Result<ToolLoopHarnessRun, KernelError> {
    let invocations = LfmNormalizer.normalize(first_model_output)?;
    let dispatch_results = dispatch_tool_invocations(tools, &invocations).await?;
    let final_answer = fake_second_model_turn(&dispatch_results);
    let passed_assertions = harness_assertions(&invocations, &dispatch_results, &final_answer);

    Ok(ToolLoopHarnessRun {
        task: task.to_string(),
        first_model_output: first_model_output.to_string(),
        invocations,
        dispatch_results,
        final_answer,
        passed_assertions,
    })
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
