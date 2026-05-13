
## Core Question

When is a single egocentric/global camera view sufficient for Vision-Language-Action (VLA) robot manipulation, and when does the lack of a wrist camera create systematic failures?

The key issue is not simply whether a wrist camera improves performance. The deeper question is whether the policy has access to the necessary task state from egoview alone.

---

## Main Hypothesis

Egoview-only VLA can be sufficient for tasks dominated by semantic grounding, object localization, and coarse reaching. However, it may fail systematically in contact-rich phases where local gripper-object geometry, occlusion, slip, or fine alignment matters.

In short:

> Egoview is good for **what and where**.  
> Wrist view is useful for **how exactly at contact**.

---

## Research Questions

### RQ1: When is egoview sufficient?

Which task categories can be solved reliably with egoview-only observations?

Useful task split:

- Object localization
- Coarse reaching
- Pre-grasp alignment
- Contact / grasp execution
- Post-grasp transport
- Placement / release
- Recovery from failure

A useful analysis is not only total success rate, but also **which phase fails**.

---

### RQ2: Does egoview-only VLA learn real spatial manipulation or camera-specific shortcuts?

A fixed egoview setup may allow the policy to learn pixel-to-action shortcuts instead of robust 3D relations.

Test with:

- Camera height changes
- Camera yaw / pitch changes
- Image crop or translation
- Object position shifts
- Background changes
- Robot self-occlusion variations

If performance drops sharply, the model may be overfitting to view-specific visual patterns.

---

### RQ3: Are failures caused by missing local contact information?

The most likely weakness of egoview-only VLA is near-contact observability.

Stress-test with:

- Small objects
- Occluded targets
- Objects inside containers or drawers
- Gripper blocking the target
- Insertion or alignment tasks
- Distractor objects near the target
- Tasks requiring recovery after failed contact

This can reveal whether the missing wrist view mainly hurts the final few centimeters of manipulation.

---

### RQ4: Can temporal egoview compensate for the missing wrist camera?

A single egoview frame may be insufficient, but video history may recover motion and contact cues.

Compare:

- Single egoview frame
- Egoview + proprioception
- Short egoview frame stack
- Video encoder over egoview history
- Egoview + predicted object / hand state
- Egoview + wrist view as an upper bound

If temporal egoview closes the gap to wrist view, the issue may be dynamics rather than viewpoint. If not, the local viewpoint itself may be necessary.

---

### RQ5: Can stronger spatial representations replace wrist cameras?

Instead of adding a wrist camera, another path is to improve spatial reasoning from egoview.

Possible directions:

- RGB-D instead of RGB
- Explicit camera intrinsics / extrinsics
- 3D point-cloud or voxel features
- Object-centric crops
- Predicted grasp pose or contact point
- End-effector pose tokens
- Spatial affordance maps

This leads to a more interesting question:

> Which information truly requires a wrist camera, and which can be recovered from better egoview representations?

---

## Minimal Experiment Set

### 1. Reproduce the egoview-only baseline

Measure success rate, but also log failure phase and qualitative failure cases.

### 2. Viewpoint perturbation test

Change camera pose, crop, background, and object distribution to test shortcut learning.

### 3. Contact-observability stress test

Use tasks where the target becomes small, occluded, or close to the gripper.

### 4. Temporal ablation

Compare single-frame egoview against frame-stack or video-history input.

### 5. Optional upper bound: egoview + wrist

Train or evaluate a small multi-view version to estimate the information gap.

---

## Possible Contribution

A strong project framing could be:

> We study the capability boundary of egoview-only VLA policies. We show that egoview is sufficient for semantic and coarse manipulation, but failures emerge systematically during contact-rich phases. We further test whether these failures can be mitigated by temporal history or spatial representations, instead of directly adding wrist cameras.

This contribution is stronger than simply saying “multi-view is better,” because it identifies **what information is missing**, **when it matters**, and **how it might be recovered**.

---

## Key Takeaway

Egoview-only VLA is not necessarily a weaker setup. It may be more scalable and better aligned with human egocentric video data. But its limitation is likely partial observability during fine contact.

The most valuable research direction is to characterize this boundary clearly.
