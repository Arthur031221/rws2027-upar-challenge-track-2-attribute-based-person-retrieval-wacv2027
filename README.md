# UPAR Challenge Track 2: attribute-based person retrieval

Entry for Track 2 of the UPAR Challenge at the RWS workshop, WACV 2027
(Codabench competition 18196). A query is a full 40-bit binary attribute
vector (age, gender, hair, clothing length, colour and type, accessories); the
task is to rank a gallery of pedestrian crops from an unseen domain so that
people whose 40 attributes all equal the query come first. The ranking metric
is mADM (mean average degree of match), which keeps the structure of average
precision but credits images ranked early that agree with the query on most
attributes.

## Approach

1. Attribute model: the Track 1 pedestrian attribute classifier (torchvision
   ConvNeXt-Small, ImageNet-1k weights, 40-way head, 256x128 input, weighted
   BCE, EMA, 10 epochs on the organisers' train split). Training code:
   the Track 1 repository (`train.py`, `model.py`).
2. Gallery: attribute logits for every image, averaged over the image and its
   horizontal flip.
3. Query score: Bernoulli log-likelihood of the query vector,
   `sum_a q_a log p_a + (1 - q_a) log(1 - p_a)`, computed as two matrix
   products (no Q x G x 40 tensor), with logits divided by a temperature of
   1.5. For a query with exactly one positive in a mutually exclusive group
   (age, hair length, upper colour, lower colour, lower-body type), half of
   that group's score is the categorical log-probability of the positive,
   with the sigmoids renormalised over the group. Returned as `similarities`.
4. Optional prior shift (`--prior-shift 0.5`): before scoring, subtract
   `lam * (1 - 2 r_a)` from each logit, where `r_a` is attribute a's rate in
   the training labels. It reduces the pull of frequent attributes toward
   gallery images that match the training prior rather than the query. On
   held-out sources with a ViT-B/16 model it moved mADM from 0.7010 to 0.7020
   (Market1501) and 0.6253 to 0.6269 (PETA).

## Local validation

The attribute model never saw the val split (33,407 images, 3,462 distinct
attribute vectors). `src/val_eval.py` treats every distinct vector in a gallery
as a query and reproduces the metrics of the Evaluation page (`src/retrieval.py`).

| scorer | val all | Market1501 | PA100k | PETA |
|---|---|---|---|---|
| L1 on probabilities (organiser recipe) | 0.6568 | 0.6782 | 0.6369 | 0.6825 |
| log-likelihood | 0.6985 | 0.7161 | 0.6664 | 0.7395 |
| log-likelihood, T 1.5, group weight 0.5 (submitted) | 0.7012 | 0.7223 | 0.6740 | 0.7402 |

The val split shares domains with the training data, so the temperature and
group weight were chosen on leave-one-source-out models instead (a ConvNeXt-S
trained without Market1501, and one without PETA, each scored on the held-out
source's val gallery). There the change moves mADM from 0.6422 to 0.6488 on
Market1501 and from 0.5277 to 0.5386 on PETA. Per-attribute temperatures and a
full group softmax (weight 1) lowered both held-out scores.

## Reproduce

```bash
cd src
python val_eval.py --ckpt <track1 checkpoint> --arch convnext_small \
    --cache <val image cache 256x128 .npy> --ann <starter>/data/annotations/task2/val --out ../runs/val
python make_submission.py --ckpt <track1 checkpoint> --arch convnext_small --out ../runs/sub
# with the log-likelihood settings above and the prior shift
python make_submission.py --ckpt <track1 checkpoint> --arch convnext_small --temp 1.5 --group-weight 0.5 \
    --prior-shift 0.5 --train-csv <starter>/data/annotations/task1/{train,val}/gt.csv --out ../runs/sub
# upload ../runs/sub.zip (run.py at the archive root)
```

`submission/run.py` exposes `load_model()` and `rank_gallery(sample)` and
needs only torch, torchvision, numpy and Pillow (pytorch/pytorch:2.4.1 image).
