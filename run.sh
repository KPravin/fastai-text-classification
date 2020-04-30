swift build
.build/debug/fastai-text-classification \
    --dataset-key data \
    --min-lm-epochs 2 \
    --max-lm-epochs 7 \
    --min-classifier-epochs 5 \
    --max-classifier-epochs 10 \
    --max-unfrozen-layers 5