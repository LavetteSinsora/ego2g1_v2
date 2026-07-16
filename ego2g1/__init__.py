"""ego2g1: egocentric human recordings -> pi0.5 policy -> Unitree G1-D.

Layering (imports only point down):
    core   pure math + physical constants; numpy-only
    kin    the G1+Revo2 model: FK, IK, self-collision  (adds mujoco/mink)
    data   Pico recordings -> LeRobot dataset pipeline (adds lerobot)
    train  pi0.5 fine-tune on top of stock openpi      (adds jax; openpi as library)
    serve  websocket policy server                     (runs where train runs)
    deploy joint-chunk execution on the real robot     (adds unitree_deploy/DDS)

The policy side ends at "timestamped joint chunks"; the execution side starts
there. Everything upstream of that boundary is testable offline.
"""
