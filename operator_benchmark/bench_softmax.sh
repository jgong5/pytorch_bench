export TORCHINDUCTOR_TRACE=1
python -m intel_extension_for_pytorch.cpu.launch --no_python --node_id 0 python -m pt.softmax_test --iterations 10 --tag_filter $@
