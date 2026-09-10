# Roadmap

## Phase 1: Project Setup and Planning
- [x] Inspect current repository and files
- [x] Create proposed project structure
- [x] Define ROADMAP.md
- [ ] Initialize Python environment and `requirements.txt`
- [ ] Set up `.gitignore` and `.env.example`
- [ ] Create DECISIONS.md

## Phase 2: Data Preparation
- [x] Acquire the "Customer Support on Twitter" dataset.
- [x] Create a script to extract a small, representative subset (to ensure the pipeline runs in under 15 minutes).
- [x] Implement data loading and preprocessing logic.

## Phase 3: Core AI Agent Implementation
- [ ] Set up LLM integration using environment variables for API keys.
- [ ] Implement the prompt/context strategy to answer queries based on support context.
- [ ] Build a simple interface/function to pass a user query and get an agent response.

## Phase 4: Pipeline and Evaluation
- [ ] Create a main execution script that runs the agent over the dataset subset.
- [ ] Implement an evaluation strategy (e.g., LLM-as-a-judge to compare against real responses).
- [ ] Run the end-to-end pipeline and optimize if necessary.

## Phase 5: Final Polish
- [ ] Refine README.md with clear "How to run" instructions.
- [ ] Ensure all decisions are documented in DECISIONS.md.
- [ ] Code review and final test from a fresh environment.
