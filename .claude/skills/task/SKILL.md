---
name: task
description: Implement one task from docs/tasks by its ID, such as A7, and prove it with the task's Done-when command.
argument-hint: <task-id>
disable-model-invocation: true
---
Implement task $ARGUMENTS.

1. Find the heading that starts with "$ARGUMENTS ·" in docs/tasks/*.md. Read the whole entry, every docs/interfaces.md section it names, and the matching .claude/rules file.
2. Check that each task under "Needs" is ticked in that lane file's Progress list. If one isn't, name it and stop.
3. If the task touches three or more files, show a short plan and wait for my OK.
4. Edit only the files the task lists. If another lane's file or a contract (docs/interfaces.md, schemas/, tests/contract/) must change, stop and explain why instead.
5. Run the Done-when command, fix, and re-run until it passes. If it needs the real arm (--real), ask me and wait until I confirm the arm is clear. Then run `pytest -q`.
6. Paste the passing output as evidence, tick the task in the Progress list, and commit with the message "$ARGUMENTS: <task title>".
