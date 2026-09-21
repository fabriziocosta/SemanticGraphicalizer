# Scratchpad

## To do

## Working on

## Done

- Replaced the notebook `DemoModel` with the default OpenAI `gpt-4.1-mini` path, executed the two-story notebook successfully, and constrained live triple schemas to the configured ontology. [2026-09-21 11:20]
- Verified the trusted notebook visually in Safari: the inline D3 graph renders with ontology-term text nodes, proposition labels, and thin gray edges; all 20 tests pass. [2026-09-21 11:05]
- Added persistent parsed-story caching to `load_aesop_fables`; repeated calls now avoid downloading, with 20 tests passing. [2026-09-21 10:58]
- Switched notebook graph rendering to an executable IPython JavaScript display that loads D3 explicitly; notebook output now contains `application/javascript` and 18 tests pass. [2026-09-21 10:45]
- Added inline D3 force-directed graph display with text-only ontology nodes, thin gray edges, relation labels, notebook integration, and 20 passing tests. [2026-09-21 10:49]
- Added reusable `load_aesop_fables` download/cache module; notebook now loads exactly the first two stories and all 14 tests pass. [2026-09-21 10:30]
- Made the default model OpenAI `gpt-4.1-mini` via the Responses API using `OPENAI_API_KEY`; added structured-output tests and passed 11 tests plus compilation and notebook validation. [2026-09-21 10:24]
- Implemented the staged sklearn transformer, strict YAML configs, Aesop notebook, documentation, and test suite; unit tests, notebook execution, imports, compilation, and diff checks pass. [2026-09-21 10:11]
- Switched verification to the existing `~/.venvs/py312` Python 3.12 environment; removed the project-local `.venv`. [2026-09-21 10:03]
