"""core: pure math + physical constants. numpy-only by contract.

frames/rot6d      SE3 + 6D-rotation conventions shared by every layer
episode           Pico hdf5 recording loading + resampling
chunks            chunk-relative action representation (+ selftest_identity)
chunk_math        loader math: absolute poses -> anchor-relative deltas,
                  boundary-aware datapoint indexing (training + deploy share it)
hand/             BrainCo Revo2 retarget: constants, FK tables, fingertip
                  retargeter, self-collision screen (screen's sim lazily
                  imports mujoco; everything else stays numpy)
paths             the one place that knows where assets/data/work live
"""
