export TORCHINDUCTOR_TRACE=1
python -m intel_extension_for_pytorch.cpu.launch --no_python --node_id 0 python -m pt.sum_test --tag_filter short --flush_cache --iterations 10