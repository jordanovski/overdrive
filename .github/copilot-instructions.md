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
- When a change affects code or assets that ship in the Docker image, always bump the project version.
- Keep the version synchronized in both `pyproject.toml` (`project.version`) and `scr/overdrive/__init__.py` (`__version__`).
- Treat these as Docker-shipped changes that require a version bump: files under `scr/`, the Dockerfile, runtime dependencies in `pyproject.toml`, and web templates/static assets shipped with the package.
- Use semantic versioning for Docker-shipped changes: patch for bug fixes, refactors, dependency updates, or UI tweaks that do not add a new capability; minor for new backward-compatible features, endpoints, commands, pages, options, or workflows; major for breaking changes, removals, incompatible behavior changes, or required migration steps.
- If a change includes multiple categories, apply the highest required semantic version bump.
- Do not bump the version for test-only, documentation-only, CI-only, or repository metadata changes that do not affect what runs inside the Docker container.
- If a task would normally require committing, branching, stashing, or creating a worktree, stop and ask the user how they want to proceed.