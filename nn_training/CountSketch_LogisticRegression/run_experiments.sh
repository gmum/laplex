


for i in {1..5}; do

python logistic_regression.py \
          --root_dataset path_to_ImageNet \
          --dims 25 50 100 250 500 750 1000 \
          --batch_size 512 \
          --iters 1000 \
          --num_workers 8 \
          --outputs /results/experiments_I

python laplex.py \
          --root_dataset path_to_ImageNet \
          --dims 25 50 100 250 500 750 1000 \
          --batch_size 512 \
          --lr 1.e-3 \
          --epochs 100 \
          --num_workers 8 \
          --save_model 1 \
          --outputs /results/experiments_II \
          --patience 20

python dense.py \
          --root_dataset path_to_ImageNet \
          --dims 25 50 100 250 500 750 1000 \
          --batch_size 512 \
          --lr 1.e-3 \
          --epochs 100 \
          --num_workers 8 \
          --save_model 1 \
          --outputs /results/experiments_III \
          --patience 20

done

