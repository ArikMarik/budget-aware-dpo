#!/bin/bash
# Retroactive commits for implementation progress.
# Run from project root: ./scripts/do_retroactive_commits.sh
# If git commit fails with "trailer" error, fix your git config (e.g. remove commit template) and run again.

set -e
cd "$(dirname "$0")/.."

# Ensure git user is set
git config user.email 2>/dev/null || git config user.email "project@local"
git config user.name 2>/dev/null || git config user.name "Budget-Aware DPO"

# 1. Phase 0
git add .cursorrules .gitignore docs/feature_reports/report_phase0_setup_2026-03-02.md
git commit -m "chore: setup virtual environment and cursor rules"

# 2. Implementation plan
git add implementation_plan.md
git commit -m "docs: add implementation plan"

# 3. Phase 1
git add requirements.txt src/__init__.py src/utils.py src/config.py \
    scripts/generate_dummy_data.py scripts/check_model_load.py scripts/run_all_dummy.sh \
    data/dummy_openmathinstruct.jsonl
git commit -m "chore: setup environment, random seeds, and dummy data generation"

# 4. Phase 2
git add src/data/ scripts/preprocess_dpo_data.py data/processed_dpo_dataset/
git commit -m "feat: implement 4-way augmentation and complexity classification pipeline"

# 5. Phase 3 (before sanity check)
git add src/models/ scripts/train_sanity_check.py
git commit -m "feat: implement budget-aware DPO loss and sanity check script"

# 6. Reports and rules update
git add docs/feature_reports/report_phase1_environment_2026-03-02.md \
    docs/feature_reports/report_phase2_preprocessing_2026-03-02.md \
    implementation_plan.md .cursorrules docs/GIT_SETUP.md scripts/do_retroactive_commits.sh
git commit -m "docs: add Phase 1/2 reports, update progress, cursor rules, and git setup"

echo "Done. Run 'git log --oneline' to verify."
