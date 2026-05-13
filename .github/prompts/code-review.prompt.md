---
name: code-review
description: 'Conduct a production-grade code and system review with a critical eye for correctness, integration, performance, security, and maintainability.'
argument-hint: What code or system changes should I review?
agent: agent
---

<!-- Tip: Use /create-prompt in chat to generate content with agent assistance -->

Act as a principal-level software engineer performing a production-grade architecture, code, and systems review.

Critically and independently evaluate the implementation without assuming the design or code is correct.

Review for:

- Correctness, edge cases, and failure handling
- API contract consistency and backward compatibility
- Integration risks across services, modules, and dependencies
- Data integrity, concurrency, idempotency, and recovery behavior
- Security vulnerabilities, privilege boundaries, and unsafe assumptions
- Performance, scalability, memory usage, and load behavior
- Test quality, missing coverage, flaky tests, and invalid assertions
- Maintainability, readability, and operational clarity
- Observability gaps (logging, metrics, tracing, alerting)
- Compliance with sound software engineering principles and patterns:
  - DRY (Don't Repeat Yourself)
  - SOLID principles
  - Separation of concerns
  - Encapsulation and abstraction boundaries
  - Cohesion vs coupling
  - YAGNI and avoidance of premature abstraction
  - Consistent error handling and validation patterns

Specifically identify and flag:

- Stubbed, placeholder, mocked, or partially implemented logic
- Methods/functions that only `pass`, `todo`, `panic`, `unimplemented`, or return hardcoded/default values
- Dead or “zombie” code:
  - Unused classes, functions, endpoints, feature flags, configs, or modules
  - Code paths not reachable from any public API, CLI, job, workflow, or runtime execution path
  - Legacy abstractions no longer actively used
- Duplicate or near-duplicate logic that should be consolidated
- Over-engineered or unnecessary abstractions
- Public APIs or interfaces that are implemented but never exercised
- Tests that validate mocks instead of real behavior
- Logic that appears complete but is disconnected from production execution paths

For every issue identified:
1. Describe the issue clearly
2. Explain the technical risk, operational impact, or long-term cost
3. Propose a concrete remediation or refactor
4. Identify severity:
   - Critical
   - High
   - Medium
   - Low

Prioritize findings that would affect:
- Production reliability
- Data correctness
- Security
- Scalability
- Maintainability
- Developer velocity

Be direct, evidence-driven, and technically rigorous. Challenge architectural and implementation decisions where appropriate.
