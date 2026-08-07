(ego2g1) unitree@unitree:~/jc/ego2g1_v2$ uv run python -m ego2g1.deploy.check latency --host 127.0.0.1
INFO:root:Waiting for server at ws://127.0.0.1:8000...
INFO:root:policy: horizon=50 dim=14 fps=30 hands=('left', 'right') mode=relation_eef
INFO:root:checkpoint config hash: ae4c882152ebe073
INFO:root:RTC: {'enabled': True, 'overlap': 10, 'max_guidance_weight': 10.0, 'schedule': 'exp', 'use_vjp': True, 'num_steps': 10} (checkpoint rtc_training=False)

server 127.0.0.1:8000 | horizon 50 dim 14 fps 30 control_mode relation_eef
sending observation/state as (56,) for control_mode 'relation_eef'
first call (includes XLA compile): 11.5 s
steady: mean 123 ms   p95 124 ms   max 126 ms
        of which server-side 57 ms, wire+encode 66 ms (54% of the round trip)
        -> if wire dominates, move the server closer or shrink the image; if server-side dominates, profile the policy.

  sync                 budget —   no hard budget (holds during inference)
  async                budget  500 ms   OK, 357 ms headroom
  temporal_smoothing   budget  267 ms   OK, 124 ms headroom

  (ego2g1) unitree@unitree:~/jc/ego2g1_v2$ uv run --group perception-v2 python -m ego2g1.deploy.perception_v2_latency --prompts "red cube,yellow cube,black pen holder"
      Built ego2g1 @ file:///home/unitree/jc/ego2g1_v2
Uninstalled 4 packages in 22ms
Installed 4 packages in 17ms
[setup] rembg absent -> import stub installed (never called)
[setup] pre-fetching facebook/sam3 (kept out of the timed stages)
Fetching 12 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 2888.80it/s]
Download complete: :                                                                                                                                                      |  0.00B            [setup] stereo calibration: /home/unitree/jc/ego2g1_v2/stereo_calib.npz                                                                                           |  0.00B /  0.00B            
========================================================================
device : NVIDIA GeForce RTX 4090 (23.5 GB)
torch  : 2.7.1+cu126   prompts: ['red cube', 'yellow cube', 'black pen holder']
========================================================================
Download complete: :                                                                                                                                                      |  0.00B            
Reconstruction complete: |                                                                                                                                       |  0.00B /  0.00B            
Drop frame2
Drop frame2
Drop frame2
16:02:37.123819 INFO     Received camera config from server 192.168.123.164:60000                                                                                            imageclient.py:551
16:02:37.125040 INFO     Saved camera config to local /home/unitree/jc/ego2g1_v2/third_party/unitree_deploy/unitree_deploy/cam_config_client.yaml                            imageclient.py:554
Drop frame2
Drop frame2
Drop frame2
frame  : 640x480

--- sgbm  (CPU) ---
  warmup (5): 22, 15, 15, 15, 15 ms (CUDA ctx / autotune / weight upload)
  steady (n=30): mean 14.8  p50 14.5  p95 16.2  p99 17.4  max 17.8 ms
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1797/1797 [00:00<00:00, 9270.39it/s]
[transformers] kernels library is not installed. NMS post-processing, hole filling, and sprinkle removal will be skipped. Install it with `pip install kernels` for better mask quality.

[detect] 2 object(s) on the first frame -- {'red cube': [0], 'yellow cube': [1]}

--- sam3  (GPU, one session, 3 prompts, detect+track) ---
  pushing 300 frames to expose memory-bank growth...
  first 10 : mean 115.2 ms
  deciles  : 119  125  128  128  128  129  129  129  129  129 ms
  steady   (n=150): mean 128.9  p50 128.9  p95 130.0  p99 130.5  max 132.4 ms
  vram     : 0:2123MB  25:2494MB  50:2835MB  75:3165MB  100:3466MB  125:3767MB  150:4073MB  175:4373MB ...
  [warn] VRAM 2123 -> 5575 MB and still climbing. Long episodes will OOM -- re-run with --frames 3000 before trusting a multi-minute rollout.

--- join  (CPU) ---
  warmup (5): 2, 1, 1, 1, 1 ms (CUDA ctx / autotune / weight upload)
  steady (n=150): mean 1.2  p50 1.2  p95 1.2  p99 1.2  max 1.3 ms
[setup] resolving Viglong/OriAnyV2_ckpt/demo_ckpts/rotmod_realrotaug_best.pt (~5 GB)

[setup] Orient Anything V2 loaded in 7.7 s, vram 10227 MB

--- orient  (GPU, 2 crops, one batched forward) ---
  warmup (5): 90, 82, 82, 82, 82 ms (CUDA ctx / autotune / weight upload)
  steady (n=10): mean 82.4  p50 82.4  p95 82.6  p99 82.6  max 82.6 ms
  vram peak 12575 MB

--- orient  (1 crop, for scaling) ---
  warmup (2): 55, 53 ms (CUDA ctx / autotune / weight upload)
  steady (n=10): mean 53.0  p50 52.9  p95 53.7  p99 53.8  max 53.8 ms
  -> 1.54x for 2 crops vs 1. Near 1.0 = batching works; near 2.0 = compute-bound, batching buys nothing.

========================================================================

--- perception_step  (sam3||sgbm -> join -> orient) ---
  warmup (3): 221, 219, 219 ms (CUDA ctx / autotune / weight upload)
  steady (n=15): mean 219.4  p50 219.3  p95 220.9  p99 221.5  max 221.7 ms
  vram peak 12800 MB

--- perception_step  (same, SERIAL) ---
  warmup (3): 233, 232, 230 ms (CUDA ctx / autotune / weight upload)
  steady (n=15): mean 231.5  p50 231.5  p95 233.3  p99 233.6  max 233.6 ms

========================================================================
VERDICT

  [async loop] one full perception iteration = 221 ms p95 -> free-running at 4.5 Hz.
  A policy tick consumes the newest COMPLETED perception, so state age spans 221 ms (just finished) to ~442 ms (one had just started). Budget the worst case, not the mean.
  That is 22% of the 1000 ms policy period.

  [overlap] running SAM 3 and SGBM concurrently saves 12 ms/iteration (233 -> 221). Worth threading the deploy loop.

  [bottleneck] SAM 3 (130 ms) dominates SGBM (16 ms), so depth hides inside the GPU stage and is effectively free.

  [orientation] 83 ms = 37% of the iteration. Moving it to a slower loop of its own would cut position-state age to ~138-277 ms.

  3 prompt(s), ONE session, ONE backbone pass per frame shared by every prompt head and the tracker. Re-run with more --prompts to see the (small) per-prompt head cost directly.
  This script only measures. It changes no deploy default.
16:03:42.420927 INFO     Image client has been closed.      