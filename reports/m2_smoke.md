# M2 frozen-feature expert smoke

- Status: **SUCCEEDED**
- Device: `cpu`
- Backbone: `resnet18` / `IMAGENET1K_V1` (frozen)
- Feature shape: `[100, 512]`
- Cache key: `11c994b70bf90665b8a4ba50b3cd8f384fb6883312019d6f411317c6c42efd3e`
- Adapter rank: `16`
- Training steps: `10`
- Save/load output equality: `true`
- Resource counts: `{'total_parameters': 11192896, 'trainable_parameters': 16384, 'activated_parameters_per_query': 11192896, 'approximate_adapter_flops_per_query': 32768, 'expert_bank_size': 1}`
