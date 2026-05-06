---
name: code-review
description: 'Conduct a production-grade code and system review with a critical eye for correctness, integration, performance, security, and maintainability.'
argument-hint: What code or system changes should I review?
agent: ask
---

<!-- Tip: Use /create-prompt in chat to generate content with agent assistance -->

Act as a senior engineer conducting a production-grade code and system review.

Independently and critically evaluate the recent changes with emphasis on:
- Correctness and edge case handling
- System interactions and integration risks
- Data integrity and failure scenarios
- Performance implications under realistic load
- Security considerations and potential vulnerabilities
- Test adequacy and missing coverage
- Long-term maintainability and clarity

For each issue identified:
1. Describe the problem
2. Explain the risk or impact
3. Propose a concrete fix or improvement

Do not assume the changes are correct—challenge decisions where appropriate.