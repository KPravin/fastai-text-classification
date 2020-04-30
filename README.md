# Fastai text classification
Possible usage:
```sh
( \
START_TIME=$SECONDS && \
swift build &&
.build/debug/fastai-text-classification \
    --dataset-key data \
    --min-lm-epochs 2 \
    --max-lm-epochs 7 \
    --min-classifier-epochs 5 \
    --max-classifier-epochs 10 \
    --max-unfrozen-layers 5 \
&& echo "Finished in $((SECONDS - START_TIME)) seconds" \
) |& tee log.txt
```
