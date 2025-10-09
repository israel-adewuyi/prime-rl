# Gradient Masks with Hugging Face Hub Integration

This document describes how to use gradient masking with automatic upload to Hugging Face Hub for easy sharing and reuse of parameter importance masks.

## Overview

Gradient masks are boolean tensors that indicate which parameters have significant gradients during training. These masks can be used to:
- Identify important parameters for fine-tuning
- Create sparse models by freezing unimportant parameters
- Analyze parameter importance across training steps
- Share parameter importance information across teams

## Features

- **Space Efficient**: Uses boolean tensors (~32x more efficient than float tensors)
- **Automatic Upload**: Seamlessly uploads masks to Hugging Face Hub
- **Rich Metadata**: Includes comprehensive metadata for each mask set
- **Easy Integration**: Works with existing training code without modifications
- **Version Control**: Each training step's masks are versioned and accessible

## Configuration

### Basic Configuration

```toml
[grad_acc]
beta = 0.99
epsilon = 1e-8
save_interval = 1000
tolerance = 1e-5
save_masks = true
mask_save_interval = 1000
```

### Hugging Face Hub Integration

```toml
[grad_acc]
# ... basic settings ...

# HF Hub settings
upload_to_hf = true
hf_repo_id = "username/model-gradient-masks"
hf_upload_interval = 2000
hf_private = true
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `beta` | EMA decay rate for gradient accumulation | 0.99 |
| `epsilon` | Small value for numerical stability | 1e-8 |
| `save_interval` | How often to save gradient accumulations | 1 |
| `tolerance` | Threshold for determining active parameters | 1e-5 |
| `save_masks` | Whether to save boolean masks locally | true |
| `mask_save_interval` | How often to save masks | `save_interval` |
| `upload_to_hf` | Whether to upload masks to HF Hub | false |
| `hf_repo_id` | HF repository ID for masks | null |
| `hf_upload_interval` | Upload interval for HF Hub | `mask_save_interval` |
| `hf_private` | Whether to make HF repo private | true |

## Repository Structure

When uploaded to Hugging Face Hub, masks are organized as follows:

```
{repo_id}/
├── README.md                           # Auto-generated documentation
├── masks/
│   ├── step_1000.pt                   # Boolean masks for step 1000
│   ├── step_2000.pt                   # Boolean masks for step 2000
│   └── ...
└── metadata/
    ├── step_1000_info.json            # Metadata for step 1000
    ├── step_2000_info.json            # Metadata for step 2000
    └── ...
```

## Usage Examples

### Loading Masks from Hugging Face Hub

```python
from prime_rl.trainer.utils import load_masks_from_hf, apply_masks_to_model

# Load masks for a specific step
masks = load_masks_from_hf("username/model-gradient-masks", 1000)

# Apply masks to your model
apply_masks_to_model(model, masks)

# Check which steps are available
from prime_rl.trainer.utils import list_available_mask_steps
steps = list_available_mask_steps("username/model-gradient-masks")
print(f"Available steps: {steps}")
```

### Loading Metadata

```python
from prime_rl.trainer.utils import load_mask_metadata_from_hf

# Load metadata for a specific step
metadata = load_mask_metadata_from_hf("username/model-gradient-masks", 1000)

print(f"Active fraction: {metadata['active_fraction']:.2%}")
print(f"Sparsity: {metadata['sparsity']:.2%}")
print(f"Total parameters: {metadata['total_parameters']:,}")
print(f"Active parameters: {metadata['active_parameters']:,}")
```

### Manual Mask Application

```python
import torch
from prime_rl.trainer.utils import load_masks_from_hf

# Load masks
masks = load_masks_from_hf("username/model-gradient-masks", 1000)

# Apply masks manually
for name, param in model.named_parameters():
    if name in masks:
        mask = masks[name].to(param.device)
        param.requires_grad = mask
        print(f"Applied mask to {name}: {mask.sum().item()}/{mask.numel()} active")
```

## Metadata Schema

Each mask set includes comprehensive metadata:

```json
{
    "step": 1000,
    "timestamp": "2025-01-09T20:00:00Z",
    "tolerance": 1e-5,
    "total_parameters": 500000000,
    "active_parameters": 450000000,
    "active_fraction": 0.9,
    "sparsity": 0.1,
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "beta": 0.99,
    "epsilon": 1e-8
}
```

## Logging

During training, you'll see logs like:

```
Step 1000 | EMA RMS Mean: 0.0001
Step 1000 | Active Fraction: 0.9000 | Sparsity: 0.1000
Saved gradient masks to outputs/grad_acc/grad_mask_step_1000.pt
Uploading masks to username/model-gradient-masks/masks/step_1000.pt
Uploading metadata to username/model-gradient-masks/metadata/step_1000_info.json
Successfully uploaded masks for step 1000 to HF Hub
```

## Best Practices

### Repository Naming
- Use descriptive names: `{base_model}-gradient-masks`
- Include model version: `Qwen2.5-0.5B-Instruct-v1-gradient-masks`
- Use consistent naming across experiments

### Upload Intervals
- Set `hf_upload_interval` higher than `mask_save_interval` to reduce upload frequency
- Consider network bandwidth and storage costs
- Upload at meaningful training milestones

### Privacy Settings
- Use `hf_private = true` for sensitive experiments
- Make repositories public for sharing and collaboration
- Consider data sensitivity when choosing privacy settings

### Storage Management
- Monitor repository size as it grows with more steps
- Consider periodic cleanup of old mask versions
- Use appropriate cache directories for downloads

## Troubleshooting

### Authentication Issues
```bash
# Login to Hugging Face Hub
huggingface-cli login
```

### Import Errors
```bash
# Install required dependencies
pip install huggingface_hub
```

### Upload Failures
- Check network connectivity
- Verify repository permissions
- Ensure sufficient disk space for temporary files

### Memory Issues
- Masks are automatically moved to CPU to save GPU memory
- Boolean tensors are very memory efficient
- Consider reducing upload frequency if memory is limited

## Integration with Other Tools

### Weights & Biases
Masks integrate seamlessly with W&B logging:
- Mask statistics are automatically logged
- Metadata is included in run summaries
- Easy comparison across different experiments

### Model Serving
Use masks to create efficient serving models:
```python
# Load masks and apply to create sparse model
masks = load_masks_from_hf("username/model-masks", latest_step)
apply_masks_to_model(model, masks)

# Save sparse model for serving
torch.save(model.state_dict(), "sparse_model.pt")
```

### Fine-tuning
Use masks to guide fine-tuning:
```python
# Load masks from pre-training
masks = load_masks_from_hf("username/pretrained-masks", 10000)

# Only fine-tune important parameters
apply_masks_to_model(model, masks)

# Fine-tune with frozen unimportant parameters
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
```

## Future Enhancements

- **Mask Comparison**: Tools to compare masks across different training runs
- **Visualization**: Generate plots showing mask evolution over time
- **Compression**: Further compression of masks for smaller uploads
- **Differential Updates**: Only upload changes between steps
- **Mask Merging**: Combine masks from multiple training runs

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify HF Hub authentication and permissions
3. Ensure all dependencies are installed
4. Review configuration parameters for correctness
