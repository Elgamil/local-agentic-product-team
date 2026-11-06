---
id: cli_tool
title: Command-Line Tool
label: Command-Line Tool
keywords: [cli, command line, command-line, terminal, tool, utility, script, shell, automation tool, converter]
---

Internal guidance for the team. Adapt it to the actual tool; it is not a feature list.

## Considerations

- Who runs it, and how often.
- Commands, arguments and options, with sensible defaults.
- Input sources: arguments, files, standard input.
- Output: readable for people, and parseable where scripts will consume it.
- Exit codes that distinguish success from each kind of failure.
- Help text and usage examples.
- Configuration, only if arguments alone become unwieldy.
- Safe handling of existing files.
- Cross-platform concerns, if it must run beyond one system.

## Typical Development Phases

1. Tool skeleton, help and argument parsing.
2. Core operation.
3. Input and output handling.
4. Error handling and exit codes.
5. Documentation and polish.
6. Final review.

## Quality Requirements

- Core logic can be tested without launching a process.
- Errors go to standard error, results to standard output.
- Failures return a non-zero exit code.
- The tool never destroys user data without an explicit request.

## Common Edge Cases

- No arguments, unknown arguments or conflicting options.
- A missing input file, a directory passed as a file, or missing permissions.
- Empty input and very large input.
- Unicode and spaces in paths.
- An output file that already exists.
- Interruption partway through.

## Suggested Acceptance Checks

- Help output describes every command and option.
- A valid invocation produces the documented output and exit code zero.
- Invalid usage prints a clear message and exits non-zero.
- A missing input file is reported without a stack trace.
- Existing files are not overwritten unless requested.
