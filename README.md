# UPAR Challenge Track 2: attribute-based person retrieval

This code ranks pedestrian images for a query containing 40 binary attributes. A gallery image is an exact match when all 40 attributes agree with the query. The leaderboard metric is mean average degree of match (mADM), which rewards early exact matches and gives partial credit to near matches.

## Selected model

The current selected package, submission 950922, uses a CLIP ViT-B/16 attribute model. Its weights are the WiSE-FT 0.8 blend of a three-seed weight average trained on organizer-provided Track 1 data. The trained attribute head is kept unchanged while the backbone is blended with the original CLIP weights.

For each gallery image, inference averages the model logits for the image and its horizontal flip. The scorer ranks each query with the Bernoulli log-likelihood of its 40 attributes, using temperature 1.5. For a query with one positive attribute in an exclusive group, the scorer blends in a categorical log-probability with weight 0.5. Query normalisation subtracts 0.5 times the log-sum-exp across queries for each gallery image.

The inference package runs without network access. It stores model weights in float16 and leaves CPU linear layers unquantised by default. The entry point can drop flip averaging if its runtime estimate approaches the one-hour limit.

## Train and package

The training scripts are in `train_src/`. They use the organizer-provided training data and the CLIP ViT-B/16 foundation weights. Train three seeds with the same recipe. Average matching tensors across the three checkpoints, then blend each backbone tensor as `0.8 * soup + 0.2 * pretrained`. Keep the soup classifier head. Use the resulting checkpoint with:

```bash
python code/src/make_submission.py \
    --ckpt <wise-checkpoint> \
    --arch vit_b16_clip --h 256 --w 128 \
    --temp 1.5 --group-weight 0.5 --query-norm 0.5 \
    --out runs/sub_wise
```

The command writes a zip whose root contains `run.py`, `metadata.yaml`, the model code, the configuration and the float16 checkpoint. The package enables flip averaging and disables CPU int8 quantisation by default. Pass `--cpu-int8` only when the quantised CPU path has been validated. Pass `--no-flip` only when the runtime limit requires it.

`train_src/train.py` trains the attribute model. `train_src/train_kd.py` trains a student from a frozen teacher. `src/make_submission.py` builds the inference package. `submission/run.py` implements `load_model()` and `rank_gallery(sample)`.

## Validation

Local retrieval measurements use the official Track 2 validation annotations. Model selection uses leave-one-source-out folds and the same frozen scorer settings as the submitted package. Test images are not used for training or model selection.
