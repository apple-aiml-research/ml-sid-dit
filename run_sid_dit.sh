#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

#!/bin/bash
# SiD-DiT Training Script
# Model choice: 600m, sprint_0.6b, sprint_1.6b, 1600m_1024, 1600m_512

# SANA teachers based on TrigFlows + DiT:
# Sana_Sprint_0.6B_1024px_teacher_diffusers (sprint_0.6b)
# SANA_Sprint_1.6B_1024px_teacher_diffusers (sprint_1.6b)

# SANA teachers based on Flow Matching + DiT:
# Sana_600M_512px_diffusers (1600m_1024)
# Sana_1600M_512px_diffusers (600m)

set -e  # Exit on any error

# Default values
WEIGHTING_SCHEME=${1:-"1_minus_sigma"}  # First parameter
NUM_GPUS=${2:-4}                        # Second parameter  
NOISE_TYPE=${3:-"fresh"}                # Third parameter
RUN_DIR=${4:-"/data/image_experiment/sid_flow"}  # Last parameter (can be omitted)

if [ -d "/data/datasets/Sana_600M_512px_diffusers" ]; then
    MODEL="/data/datasets/Sana_600M_512px_diffusers"
else
    MODEL="Efficient-Large-Model/Sana_600M_512px_diffusers"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get total number of available GPUs
get_total_gpus() {
    if ! command_exists nvidia-smi; then
        print_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit 1
    fi
    
    local total_gpus
    total_gpus=$(nvidia-smi --list-gpus | wc -l)
    echo $total_gpus
}

# Function to get GPU memory usage and select least used GPUs
select_least_used_gpus() {
    if ! command_exists nvidia-smi; then
        print_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit 1
    fi
    
    local total_gpus
    total_gpus=$(get_total_gpus)
    
    # Validate requested number of GPUs
    if [ "$NUM_GPUS" -gt "$total_gpus" ]; then
        print_warning "Requested $NUM_GPUS GPUs but only $total_gpus available. Using $total_gpus GPUs."
        NUM_GPUS=$total_gpus
    fi
    
    if [ "$NUM_GPUS" -lt 1 ]; then
        print_error "Number of GPUs must be at least 1"
        exit 1
    fi
    
    print_info "Analyzing GPU usage to select $NUM_GPUS least used GPUs..."
    
    # Get GPU memory usage and sort by memory utilization (least used first)
    # Format: GPU_ID MEMORY_USED_MB
    local gpu_usage
    gpu_usage=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
                     awk -F', ' '{print $1, $2}' | \
                     sort -k2,2n | \
                     head -$NUM_GPUS)
    
    if [ -z "$gpu_usage" ]; then
        print_error "Failed to get GPU usage information"
        exit 1
    fi
    
    # Extract GPU IDs
    local selected_gpus
    selected_gpus=$(echo "$gpu_usage" | awk '{print $1}' | tr '\n' ',' | sed 's/,$//')
    
    # Get memory usage for display
    print_info "Selected GPUs (least memory usage):"
    echo "$gpu_usage" | while read gpu_id mem_used; do
        print_info "  GPU $gpu_id: ${mem_used}MB used"
    done
    
    # Set CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES="$selected_gpus"
    
    print_success "Using $NUM_GPUS GPUs: $selected_gpus"
}

# Function to check CUDA availability
check_cuda() {
    if ! command_exists python; then
        print_error "Python not found"
        exit 1
    fi
    
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        print_success "CUDA is available"
    else
        print_error "CUDA is not available. Please check PyTorch installation."
        exit 1
    fi
}

# Function to create output directory
create_output_dir() {
    # Check if parent directory is writable
    local parent_dir=$(dirname "$RUN_DIR")
    if [ ! -w "$parent_dir" ] 2>/dev/null; then
        print_error "Cannot write to parent directory: $parent_dir"
        exit 1
    fi
    
    if [ ! -d "$RUN_DIR" ]; then
        mkdir -p "$RUN_DIR"
        print_info "Created output directory: $RUN_DIR"
    fi
}

# Function to check dataset paths
check_datasets() {
    print_info "Checking dataset paths..."
    
    local dataset1="/data/datasets/MS-COCO-256/val"
    local dataset2="/data/datasets/aesthetics_6_plus"
    local dataset3="/data/datasets/midjourney-v6-llava/data"
    
    local missing=0

    if [ ! -d "$dataset1" ]; then
        print_warning "Dataset path not found: $dataset1"
        print_info "If you need to compute metrics during training, please provide MS-COCO-256/val."
        missing=1
    fi
    
    if [ ! -d "$dataset2" ]; then
        print_warning "Dataset path not found: $dataset2"
        print_info "If running data-free, you may provide either $dataset2 or $dataset3."
        missing=1
    fi
    
    if [ ! -d "$dataset3" ]; then
        print_warning "Dataset path not found: $dataset3"
        print_info "If using a Diffusion-GAN that needs real images, $dataset3 is required."
        print_info "If running data-free, you may provide either $dataset2 or $dataset3."
        missing=1
    fi
    
    if [ "$missing" -eq 1 ]; then
        print_warning "Some datasets are missing. Training may fail if they are required."
        print_info "Continuing anyway..."
    else
        print_success "All datasets found"
    fi
}

# Function to show configuration
show_config() {
    print_info "=== SiD-SANA Training Configuration ==="
    print_info "Model: $MODEL"
    print_info "GPUs: $NUM_GPUS (automatically selected)"
    print_info "Selected GPU IDs: $CUDA_VISIBLE_DEVICES"
    print_info "Output directory: $RUN_DIR"
    print_info "Training data: /data/datasets/midjourney-v6-llava/data"
    print_info "Prompt data: /data/datasets/aesthetics_6_plus"
    print_info "Validation data: /data/datasets/MS-COCO-256/val"
    print_info "Noise type: $NOISE_TYPE"
    print_info "Weighting scheme: $WEIGHTING_SCHEME"
    echo
}

# Function to setup environment
setup_environment() {
    print_info "Setting up environment..."
    
    # Set PyTorch environment variables for better performance
    #export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    #export CUDA_LAUNCH_BLOCKING=0
    
    print_success "Environment setup complete"
}

# Function to start training
start_training() {
    print_info "Starting distributed training on selected GPUs..."
    

    
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        sid_sdxl_train_fsdp.py \
        --outdir $RUN_DIR \
        --resume $RUN_DIR \
        --data "/data/datasets/MS-COCO-256/val" \
        --data_prompt_text "/data/datasets/aesthetics_6_plus" \
        --dit_model $MODEL \
        --optimizer 'adam' \
        --resolution 512 \
        --batch 256 \
        --batch-gpu 8 \
        --lr 0.000005 \
        --glr 0.000005 \
        --cfg_train_fake 4.5 \
        --cfg_eval_fake 4.5 \
        --cfg_eval_real 4.5 \
        --alpha 1.0 \
        --init_timestep 999 \
        --num_steps 4 \
        --fp16 0 \
        --bf16 0 \
        --autocast_bf16 1 \
        --gradient_checkpointing 1 \
        --tick 2 \
        --snap 25 \
        --dump 25 \
        --duration 1 \
        --metrics "fid_clip_10k_full" \
        --noise_type $NOISE_TYPE \
        --weighting_scheme $WEIGHTING_SCHEME \
        --nosubdir 
}

# Function to handle cleanup on exit
cleanup() {
    print_info "Cleaning up..."
    # Add any cleanup tasks here if needed
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    print_info "Starting SiD-SANA training with Sana_600M_512px_diffusers"
    
    # Validate weighting scheme first (since it's the first parameter)
    case $WEIGHTING_SCHEME in
        "sid_legacy"|"snr_sqrt"|"snr"|"1_over_sigma2"|"1_over_sigma"|"1_minus_sigma_squared"|"1_minus_sigma")
            ;;
        *)
            print_error "Invalid weighting_scheme: $WEIGHTING_SCHEME. Must be one of: sid_legacy, snr_sqrt, snr, 1_over_sigma2, 1_over_sigma, 1_minus_sigma_squared, 1_minus_sigma"
            exit 1
            ;;
    esac
    
    # Validate number of GPUs
    if ! echo "$NUM_GPUS" | grep -qE '^[0-9]+$'; then
        print_error "Number of GPUs must be a positive integer"
        exit 1
    fi
    
    # Validate noise type
    case $NOISE_TYPE in
        "fresh"|"fixed"|"ddim")
            ;;
        *)
            print_error "Invalid noise_type: $NOISE_TYPE. Must be one of: fresh, fixed, ddim"
            exit 1
            ;;
    esac
    
    # Pre-flight checks
    select_least_used_gpus
    check_cuda
    check_datasets
    create_output_dir
    
    # Setup and start
    show_config
    setup_environment
    start_training
    
    print_success "Training completed successfully!"
}

# Show usage if help requested
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [WEIGHTING_SCHEME] [NUM_GPUS] [NOISE_TYPE] [RUN_DIR]"
    echo "  WEIGHTING_SCHEME: Loss weighting scheme (default: 1_minus_sigma)"
    echo "    Options: sid_legacy, snr_sqrt, snr, 1_over_sigma2, 1_over_sigma, 1_minus_sigma_squared, 1_minus_sigma"
    echo "  NUM_GPUS: Number of GPUs to use (default: 4)"
    echo "  NOISE_TYPE: Noise type for generation (default: fresh)"
    echo "    Options: fresh, fixed, ddim"
    echo "  RUN_DIR: Output directory (default: /data/image_experiment/sid_flow) - can be omitted"
    echo
    echo "This script automatically selects the N GPUs with the least memory usage."
    echo "Example: $0 1_minus_sigma"
    echo "Example: $0 sid_legacy 2"
    echo "Example: $0 snr_sqrt 8 ddim"
    echo "Example: $0 1_minus_sigma 1 fresh /path/to/output"
    exit 0
fi

# Run main function
main "$@"
```

**Usage examples:**
```bash
# Use all defaults
run_sid_dit.sh

# Specify only weighting scheme
run_sid_dit.sh sid_legacy

# Specify weighting scheme and number of GPUs
run_sid_dit.sh 1_minus_sigma 2

# Specify weighting scheme, GPUs, and noise type
run_sid_dit.sh snr_sqrt 8 ddim

# Specify all parameters
run_sid_dit.sh sid_legacy 4 fixed /path/to/output

# Specify weighting scheme and custom output directory
run_sid_dit.sh 1_minus_sigma 1 fresh /path/to/output
```