

## Definition

A **backbone** is the **main feature extractor** in a model.

It takes raw input and converts it into a useful internal representation for downstream tasks.

```text
Input → Backbone → Features → Head → Output
```

---

## Core idea

- **Backbone** = understands the input
- **Head** = solves the task using those features

Example:

```text
Image → Backbone → Visual features → Classifier → Label
```

---

## What the backbone outputs

The output of a backbone is usually:

- **features**
- **embeddings**
- **feature maps**

These are **not final predictions**.  
They are compact representations that keep useful information and discard noise.

Examples:

```text
Image → feature map / feature vector
Text  → token embeddings
State → latent representation
```

---

## Why it matters

A good backbone learns representations that are:

- informative
- reusable
- robust
- transferable across tasks

This is why the same backbone can often be reused for multiple tasks.

---

## How it is trained

### 1. From scratch
Train the backbone and task head together on the target task.

```text
Input → Backbone → Head → Loss
```

### 2. Pretraining + fine-tuning
First train on a large dataset, then adapt to a new task.

This is the most common approach.

### 3. Self-supervised learning
Train without manual labels by using structure inside the data itself.

This is especially important in robotics and representation learning.

---

## Freeze vs fine-tune

### Freeze the backbone when:
- data is limited
- the new task is similar to the pretraining task
- training stability is important

### Fine-tune the backbone when:
- you have enough data
- the new domain is very different
- the task needs specialized features

---

## In robotics

A backbone often acts as the **perception module**.

```text
Camera / sensors → Backbone → State features → Policy → Action
```

So:

- **backbone** learns how to represent the world
- **policy head** learns what to do

If the backbone gives poor features, the policy usually performs poorly too.

---

## CNN vs ViT

### CNN
Better when:
- data is small
- training stability matters
- local patterns are important

### ViT
Better when:
- data is large
- global context matters
- scaling is important

---
