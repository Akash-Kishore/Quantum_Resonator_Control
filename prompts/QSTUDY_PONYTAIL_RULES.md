QSTUDY PONYTAIL RULES  (condensed from the user's ponytail skill, for the later refactor phase; level: full)
These rules govern every code change the executor makes during the restructuring. They are NOT used while z1 is running (pipeline code is frozen then).

THE GOAL OF A CHANGE: the smallest complete change that fully solves the task, with IDENTICAL numerical results.
1. Does it need to exist? Skip features, options and flexibility nobody asked for. Deletion beats addition.
2. Use what exists first: a helper already in the project, then the standard library, then an installed package. Never add a dependency for a few lines.
3. No abstraction, wrapper, class, option or config that nobody asked for. Keep the existing structure, interfaces and conventions of the project.
4. One line a reader gets at a glance beats ten lines. A one-liner that needs decoding is not short.
5. Never cut: validation at trust boundaries, error handling that prevents data loss, security, the calibration that real hardware needs, anything the user asked for.
6. Code that is moved or merged keeps its error handling and validation.
7. Bug fix: first grep every caller of the function; fix the root cause once, in the shared code.
8. New non-trivial logic (a branch, a loop, a parser) gets one small test or assert. Trivial changes need none.
9. Comment only the WHY that the code cannot show, in one line. A known limit gets:  # shortcut: <the limit>, <when to upgrade>
10. Finish the change: update every caller, test and fixture that the change touches.

PROJECT-SPECIFIC RULES (they come first when they conflict with the list above)
P1  BEHAVIOUR-PRESERVING ONLY. A refactor may not change any number the code produces. After each packet run the golden comparison (qs8_compare.py as-run versus new).
    Any difference above the agreed tolerance means: revert that packet, report it, go on.
P2  Never touch: PREREG*, configs\frozen content, results\sealed, the as-run archive AS_RUN_v1.zip.
P3  No absolute path and no fixed seed or hyper-parameter inside code. Paths come from one place (the project root, found from the file's own location), seeds and hyper-parameters from
    configs or command-line arguments.
P4  One packet = one small group of files. Backup first, py_compile, nearest test, golden comparison, log line. Never edit more than 5 files per packet.
P5  Every reply ends with one or two lines: what was skipped or not checked, and any risk.
