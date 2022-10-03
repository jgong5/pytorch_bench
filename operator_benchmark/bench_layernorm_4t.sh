export TORCHINDUCTOR_TRACE=1
python -m intel_extension_for_pytorch.cpu.launch --no_python --ninstances 1 --ncore_per_instance 4 python -m pt.layernorm_test --iterations 100 --tag_filter $@
