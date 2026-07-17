"""Verification dashboard: replay the converted dataset (or the work-dir
stage outputs) through the deployment-shaped closed loop and render
everything a human needs to judge "is this data faithful and deployable?".

Entry points (run from the repo root with the repo venv):

    uv run python -m ego2g1.data.dashboard episode_1 -o report.html
    uv run python -m ego2g1.data.dashboard --batch --limit 3
    uv run mjpython -m ego2g1.data.dashboard.viewer episode_1
"""
