export TORCH_CUDA_ARCH_LIST="8.9;9.0"


unset CC; unset CXX; unset CUDAHOSTCXX
export CC=$(which gcc); export CXX=$(which g++)


python exp_fft_combined.py
python exp_rff_value.py
python exp_rff_gradient.py


python exp1_gpu_time_memory.py
python exp2_accuracy.py
python exp3_batch.py
python exp4_asymm.py
python exp5_gram.py


python exp6_cpu.py
python exp7_toeplitz_accuracy.py
python exp8_cpu_fft.py
