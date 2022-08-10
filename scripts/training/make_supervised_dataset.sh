DATASET=$1

for FILE in train dev ; do

    python make_supervised_hotpotqa_dataset.py \
        $DATASET/hotpot_$FILE.json $DATASET/$FILE \
        --target title \
        --mark_target \
        --mark_silver \
        --n_samples 3 \
        --mode a

    python make_supervised_hotpotqa_dataset.py \
        $DATASET/hotpot_$FILE.json $DATASET/$FILE \
        --target span \
        --mark_target \
        --mark_silver \
        --n_samples 10 \
        --mode a

done