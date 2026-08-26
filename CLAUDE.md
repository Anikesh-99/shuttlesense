# ShuttleSense — project conventions

## Git
- Never add Claude co-author trailers (or any AI attribution) to commits.

## Agent/model assignments
When delegating work to subagents, use these model assignments:
- **Implementation** (executor agents, code writing): Sonnet
- **Code review** (reviewer agents): Opus
- **Testing** (tester agents): Opus when the testing is substantive; Sonnet is fine for mechanical test runs.

## Key documents
- Design spec: `docs/superpowers/specs/2026-08-26-shuttlesense-design.md`
