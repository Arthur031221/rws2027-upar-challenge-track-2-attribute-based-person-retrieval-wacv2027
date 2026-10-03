# UPAR Challenge Track 2

This code ranks pedestrian gallery images for a query with 40 binary attributes. Exact matches agree on all attributes. The primary metric is mean average degree of match, or mADM, which rewards early exact matches and partial agreement on near matches.

## Selected model

Submission 958124 scored 0.630940 mADM and ranked first on the leaderboard when checked on 2026-10-03. It uses a CLIP ViT-B/16 student distilled from a CLIP ViT-L teacher. The student was trained for 16 epochs on organizer-provided Track 1 train and validation data with a distillation weight of 0.9 and a horizontally flipped teacher.

For the final weights, the KD2 seed 0 checkpoint was averaged with three copies of the existing three-seed ViT-B/16 Track 1 soup. WiSE-FT then blended 80 percent of the averaged backbone with the pretrained CLIP weights. The trained attribute head was retained. The submission stores the result in float16.

## Inference

The runner predicts the 40 attributes for each gallery image and ranks images by Bernoulli log likelihood. It uses temperature 1.5, an exclusive-group weight of 0.5, and query normalisation with weight 0.5. It averages predictions for the image and its horizontal flip while the projected run time stays within its budget. The flip pass is disabled if the runner projects that it will exceed 2700 seconds.

The package includes its weights and runs without network access. It uses the organizer's submission interface and does not read hidden test labels.

## Training and packaging

Training code is in `train_src/`. The Track 1 teacher checkpoint and three-seed soup are inputs from the companion UPAR Track 1 work and are not included here. Train the KD2 student with the recorded settings:

```bash
python train_src/train_kd.py \
  --arch vit_b16_clip --h 256 --w 128 --epochs 16 --bs 64 \
  --lr 2e-5 --wd 0.05 --loss wbce --ls 0.05 --ema 0.9995 \
  --seed 0 --trainval --lr_head 0.0005 --layer_decay 0.8 \
  --drop_path 0.1 --clip_grad 1.0 \
  --teacher <track1-vit-l-checkpoint> --teacher_arch vit_l14_clip \
  --kd 0.9 --teacher_flip --lowres 0.0 --out ../runs/kd2_tv_s0
```

Average the KD2 checkpoint with three copies of the Track 1 soup checkpoint. Apply the WiSE-FT blend to the backbone and retain the classifier head. Package the resulting checkpoint with the fixed scorer settings:

```bash
python src/make_submission.py \
  --ckpt <four-way-wise-checkpoint> --arch vit_b16_clip \
  --h 256 --w 128 --temp 1.5 --group-weight 0.5 \
  --query-norm 0.5 --out ../runs/submission
```

## Validation

Model selection used leave-one-source-out folds from the public Track 2 validation data. The four-way WiSE-FT variant scored mADM 0.7251476118 on Market1501 and 0.6477457667 on PETA. Its mAP exceeded the three-seed WiSE-FT baseline on both folds. The submission package was checked with synthetic images for finite scores, checkpoint parity, CPU run time, and memory use.
