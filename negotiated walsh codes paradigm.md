  SUMMARY: The Complete Pipeline

  INITIALIZATION:
  ├── Create Walsh codebook (128 orthogonal codes)
  ├── Initialize tracker (empty class→code mapping)
  └── Set α = 0.5

  FOR EACH TASK:
  ├── STEP 1: Collect all training data
  ├── STEP 2: Assign Walsh codes to NEW classes
  │   └── Based on BCE distance between predictions and codes
  ├── STEP 3: Create negotiated targets
  │   └── y_neg = (1-α) × walsh_code + α × initial_pred
  ├── STEP 4: Train for N epochs
  │   ├── BCE loss on Walsh layer (learn representations)
  │   └── CE loss on task head (learn classification)
  └── STEP 5: Update α using plasticity formula

  INFERENCE:
  ├── Image → Features → Walsh Layer → sigmoid → Walsh Repr
  └── Walsh Repr → Task Head → Softmax → Class Prediction

────────────────────────────────────────────────────────────────


The Double Training Mechanism

  Looking at _train_epoch_walsh (lines 751-833):

  for x, y_walsh, y_true in train_loader:

      # ═══════════════════════════════════════════════════════════
      # PHASE 1: Train Walsh layer with BCE loss
      # ═══════════════════════════════════════════════════════════
      walsh_logits = model.get_walsh_features(x)  # Features → Walsh layer
      walsh_loss = F.binary_cross_entropy_with_logits(walsh_logits, y_walsh)

      # ═══════════════════════════════════════════════════════════
      # PHASE 2: Train classifier head with CE loss
      # ═══════════════════════════════════════════════════════════
      with torch.no_grad():                        # ← DETACH HERE!
          walsh_repr = torch.sigmoid(walsh_logits)

      classifier_logits = model.heads[task_id](walsh_repr)
      classifier_loss = F.cross_entropy(classifier_logits, y_true)

      # ═══════════════════════════════════════════════════════════
      # Combined loss and single backward pass
      # ═══════════════════════════════════════════════════════════
      loss = walsh_loss + classifier_loss
      loss.backward()
      optimizer.step()

  Gradient Flow Diagram

                            FORWARD PASS
                                 │
      ┌──────────────────────────┼──────────────────────────┐
      │                          │                          │
      │    Input x               │                          │
      │        │                 │                          │
      │        ▼                 │                          │
      │  ┌───────────┐           │                          │
      │  │ Features  │ ◄─────────┼── Gradients from BCE     │
      │  │ Extractor │           │                          │
      │  └─────┬─────┘           │                          │
      │        │                 │                          │
      │        ▼                 │                          │
      │  ┌───────────┐           │                          │
      │  │  Walsh    │ ◄─────────┼── Gradients from BCE     │
      │  │  Layer    │           │                          │
      │  └─────┬─────┘           │                          │
      │        │                 │                          │
      │        ▼                 │                          │
      │   walsh_logits ──────────┼──► BCE Loss (y_walsh)    │
      │        │                 │         │                │
      │        ▼                 │         │                │
      │    sigmoid()             │         │                │
      │        │                 │         │                │
      │   ─────┼─────            │         │                │
      │   │ DETACH │  ◄──────────┼── NO gradients flow back │
      │   ─────┼─────            │         │                │
      │        │                 │         │                │
      │        ▼                 │         │                │
      │  ┌───────────┐           │         │                │
      │  │Task Head  │ ◄─────────┼── Gradients from CE      │
      │  │(Head k)   │           │         │                │
      │  └─────┬─────┘           │         │                │
      │        │                 │         │                │
      │        ▼                 │         │                │
      │   classifier_logits ─────┼──► CE Loss (y_true)      │
      │                          │         │                │
      └──────────────────────────┼─────────┼────────────────┘
                                 │         │
                                 ▼         ▼
                            total_loss = walsh_loss + classifier_loss
                                 │
                                 ▼
                            loss.backward()

  What Gets Updated by Which Loss?

  | Component         | BCE Loss (Walsh) | CE Loss (Classifier) |
  |-------------------|------------------|----------------------|
  | Feature Extractor |    ✅ Updated     |    ❌ No gradient     |
  | Walsh Layer       |    ✅ Updated     |    ❌ No gradient     |
  | Task Head k       |  ❌ No gradient   |      ✅ Updated       |

  Why the Detach is Critical

  # WITHOUT detach (BAD):
  walsh_repr = torch.sigmoid(walsh_logits)
  classifier_logits = model.heads[task_id](walsh_repr)
  classifier_loss = F.cross_entropy(classifier_logits, y_true)

  # Gradients would flow: CE → Head → walsh_repr → walsh_logits → Walsh Layer → 
  Features
  # This means CE loss would ALSO modify Walsh layer!
  # Problem: CE loss optimizes for current task classification, not Walsh codes

  # WITH detach (GOOD):
  with torch.no_grad():
      walsh_repr = torch.sigmoid(walsh_logits)

  classifier_logits = model.heads[task_id](walsh_repr)
  classifier_loss = F.cross_entropy(classifier_logits, y_true)

  # Gradients flow: CE → Head → STOP
  # Walsh layer is ONLY trained by BCE loss on negotiated targets

  The Two Training Objectives

  1. BCE Loss (Walsh Layer Training)

  walsh_loss = BCE_with_logits(walsh_logits, y_negotiated)
  - Goal: Learn to produce Walsh code representations
  - Target: Negotiated codes y_neg = (1-α) * walsh_code + α * initial_pred
  - Updates: Feature extractor + Walsh layer
  - Why BCE? Walsh codes are binary {0,1}, BCE is appropriate for multi-label binary
  targets

  2. CE Loss (Classifier Head Training)

  classifier_loss = CrossEntropy(classifier_logits, y_true)
  - Goal: Learn to classify from Walsh representations
  - Target: True class labels (0, 1, 2, ... within task)
  - Updates: Only the current task head
  - Why CE? Standard classification loss for mutually exclusive classes

  Alternative: Could We Use a Single Loss?

  Yes, but the current design has advantages:

  Option A: Only CE loss (no BCE)
  # Just train end-to-end with CE
  logits = model(x, task_id)
  loss = CrossEntropy(logits, y_true)
  - Problem: Walsh layer would learn task-specific features, not stable codes
  - The whole point of Walsh codes is lost

  Option B: Only BCE loss (no CE)
  # Just train Walsh layer
  walsh_logits = model.get_walsh_features(x)
  loss = BCE_with_logits(walsh_logits, y_walsh)
  - Problem: Task heads never learn to classify
  - At inference, heads would output random predictions

  Current Design: BCE + CE with detach
  - Walsh layer learns stable, orthogonal representations (BCE)
  - Task heads learn to classify from these representations (CE)
  - The detach ensures they don't interfere with each other

  Training Flow Summary

  Each Batch:
      1. Forward: x → Features → Walsh logits
      2. Compute BCE loss on walsh_logits vs negotiated targets
      3. Forward: sigmoid(walsh_logits) → DETACH → Task Head → classifier_logits
      4. Compute CE loss on classifier_logits vs true labels
      5. total_loss = BCE + CE
      6. Backward: Gradients flow to (Features, Walsh) from BCE
                   Gradients flow to (Task Head) from CE
      7. Optimizer step: All parameters updated in single step


