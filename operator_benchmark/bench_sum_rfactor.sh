python -m intel_extension_for_pytorch.cpu.launch --no_python --node_id 0 python -m pt.sum_test --tag_filter rfactor --flush_cache --iterations 10
python -m intel_extension_for_pytorch.cpu.launch --no_python --node_id 0 python -m pt.sum_test --tag_filter rfactor_unaligned --flush_cache --iterations 10
python -m intel_extension_for_pytorch.cpu.launch --no_python --node_id 0 python -m pt.sum_test --tag_filter rfactor --iterations 10
python -m intel_extension_for_pytorch.cpu.launch --no_python --node_id 0 python -m pt.sum_test --tag_filter rfactor_unaligned --iterations 10