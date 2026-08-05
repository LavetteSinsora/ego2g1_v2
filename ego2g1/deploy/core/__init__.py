"""The mode-blind execution spine (docs/deploy_refactor_plan.md §1): the
runner loop, the five consumption strategies, safety, latency budgeting,
the vendored-executor wrapper, ExecutorSession, kinematics, and the
policy-server client. Nothing in here knows what kind of policy is
running — modes/ owns that."""
