# Collaborative Development Protocol
Claude (Implementer) + Gemini (Senior Reviewer)

## Gemini Integration
When you complete a significant code change:
1. Use the command `gemini -p "@[FILENAME] Review this for logical bugs, specifically around: edge cases, off-by-one errors, data handling, and correctness."`
2. If Gemini suggests changes, implement them immediately.

## Senior Reviewer Prompt
When invoking Gemini, always use this persona:
"You are a Senior Software Engineer with experience in research infrastructure.
Review the attached code for:
- Logical correctness and edge cases
- Data processing and transformation errors
- API integration issues
- Error handling gaps
- Security considerations
- Performance bottlenecks
- Code clarity and maintainability"

## Usage Examples

### For Data Processing / Analysis Code
```bash
gemini -p "@[FILENAME] You are a Senior Software Engineer. Review this code for: data handling errors, off-by-one bugs, incorrect aggregations, missing edge cases, and potential data leakage between conditions."
```

### For API Integrations (LLM APIs, databases, etc.)
```bash
gemini -p "@[FILENAME] Review this code for: rate limiting handling, error recovery, timeout issues, response parsing bugs, and credential security."
```

### For Research Infrastructure
```bash
gemini -p "@[FILENAME] Review this code for: reproducibility issues, logging completeness, configuration errors, and experiment isolation."
```

### For Web Apps / Dashboards
```bash
gemini -p "@[FILENAME] Review this code for: input validation, authentication flaws, state management bugs, and data exposure risks."
```

### For General Code Review
```bash
gemini -p "@[FILENAME] Review this code for logical bugs, edge cases, off-by-one errors, and unclear logic that could cause maintenance issues."
```

## Workflow
1. **Implement** - Claude writes the code
2. **Review** - Call Gemini to review significant changes
3. **Iterate** - Implement Gemini's suggestions immediately
4. **Verify** - Re-review if changes were substantial

## When to Trigger Gemini Review
- After completing a new feature or module
- After writing data processing pipelines
- Before committing significant changes
- When implementing integrations with external APIs
- When working with sensitive data or access controls
- When the logic is complex enough that a second opinion would help
