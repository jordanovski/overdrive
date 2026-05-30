# Copilot Instructions

When working in this repository:

- Never create a git commit.
- Never amend, rewrite, or otherwise modify commit history.
- Never create or use git worktrees.
- Always work only in the currently active branch.
- Never switch branches unless the user explicitly asks for it.
- Put all source code in the `scr/` folder.
- Put all test code in the `test/` folder.
- Do not create source files outside `scr/` or test files outside `test/` unless the user explicitly asks for an exception.
- Use Python for implementation unless the user explicitly asks for another language.
- Organize Python application code as packages under `scr/` and keep automated tests in `test/`.
- If a task would normally require committing, branching, stashing, or creating a worktree, stop and ask the user how they want to proceed.