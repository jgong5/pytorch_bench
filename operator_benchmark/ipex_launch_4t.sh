export TORCHINDUCTOR_TRACE=1
#python -m intel_extension_for_pytorch.cpu.launch --node_id 0 $@
python -m intel_extension_for_pytorch.cpu.launch --ninstances 1 --ncore_per_instance 4 $@
