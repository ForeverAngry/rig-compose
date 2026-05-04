use std::sync::Arc;

use async_trait::async_trait;
use rig_compose::{
    Agent, GenericAgent, InvestigationContext, KernelError, Skill, SkillOutcome, SkillRegistry,
    ToolRegistry,
};

struct KeywordSkill;

#[async_trait]
impl Skill for KeywordSkill {
    fn id(&self) -> &str {
        "example.keyword"
    }

    fn description(&self) -> &str {
        "raises confidence when the keyword signal is present"
    }

    fn applies(&self, context: &InvestigationContext) -> bool {
        context.has_signal("keyword.match")
    }

    async fn execute(
        &self,
        _context: &mut InvestigationContext,
        _tools: &ToolRegistry,
    ) -> Result<SkillOutcome, KernelError> {
        Ok(SkillOutcome::default().with_delta(0.25))
    }
}

#[tokio::main]
async fn main() -> Result<(), KernelError> {
    let skills = SkillRegistry::new();
    skills.register(Arc::new(KeywordSkill));

    let tools = ToolRegistry::new();
    let agent = GenericAgent::builder("triage")
        .with_skills(["example.keyword"])
        .build(&skills, &tools)?;

    let mut context = InvestigationContext::new("entity-1", "default").with_signal("keyword.match");
    let result = agent.step(&mut context).await?;

    println!(
        "skills run: {:?}; confidence: {}",
        result.skills_run, result.confidence
    );
    Ok(())
}
