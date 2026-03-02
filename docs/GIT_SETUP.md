# Git Setup and Remote Repository

## Current Status

- Git is initialized in this project.
- Commits may fail with `error: unknown option 'trailer'` if your environment has a git commit template or hook that uses trailer syntax. Fix by removing or adjusting that config.

## Making the Retroactive Commits

Run the script:

```bash
./scripts/do_retroactive_commits.sh
```

Or commit manually in this order:

1. `chore: setup virtual environment and cursor rules` — .cursorrules, .gitignore, report_phase0
2. `docs: add implementation plan` — implementation_plan.md
3. `chore: setup environment, random seeds, and dummy data generation` — Phase 1 files
4. `feat: implement 4-way augmentation and complexity classification pipeline` — Phase 2 files
5. `feat: implement budget-aware DPO loss and sanity check script` — Phase 3 files
6. `docs: add Phase 1/2 reports, update progress and cursor rules` — reports, progress, rules

## Setting Up a Remote Repository

### Option A: GitHub

1. Create a new repository on GitHub (e.g. `budget-aware-dpo`). Do **not** initialize with README.
2. Add the remote and push:

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/budget-aware-dpo.git
   git branch -M main   # optional: rename master to main
   git push -u origin main
   ```

### Option B: GitLab

1. Create a new project on GitLab.
2. Add the remote and push:

   ```bash
   git remote add origin https://gitlab.com/YOUR_USERNAME/budget-aware-dpo.git
   git push -u origin main
   ```

### Option C: Other (Bitbucket, self-hosted, etc.)

1. Create an empty repository on your host.
2. Add the remote and push:

   ```bash
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

## Fixing the "trailer" Commit Error

**Cause:** Cursor injects `--trailer 'Made-with: Cursor'` into `git commit`. Git 2.30.2 does not support the `--trailer` option (added in newer versions).

**Fix:** Run commits **outside Cursor** (e.g. in a regular terminal):

```bash
cd /storage/arik/nlp_final_project
./scripts/do_retroactive_commits.sh
```

Or upgrade Git to 2.36+ which supports `--trailer`.
