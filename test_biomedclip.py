import os
import warnings
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from Model.biomedclip import create_model
from Model.transformer import TimmModel, HFTextEncoder
from dataset.medical_few import MedDataset
from utils import cos_sim, encode_text_with_prompt_ensemble
from sklearn.metrics import roc_auc_score
from prompt import REAL_NAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain': 3, 'Liver': 2, 'Retina_RESC': 1, 'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_features(model, image, device):
    """
    Extract multi-scale features from BiomedCLIP vision encoder.
    Returns features from multiple layers.
    """
    # Get the vision encoder
    vision_encoder = model.visual
    
    # Process image through vision encoder
    with torch.no_grad():
        # Patch embedding
        if hasattr(vision_encoder, 'trunk'):
            # TimmModel path
            trunk = vision_encoder.trunk
            if hasattr(trunk, 'patch_embed'):
                x = trunk.patch_embed(image)
            else:
                x = trunk.conv_proj(image) if hasattr(trunk, 'conv_proj') else trunk.stem(image)
                x = x.flatten(2).transpose(1, 2)
            
            # Add position embeddings and CLS token
            if hasattr(trunk, 'cls_token'):
                cls_token = trunk.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)
            
            if hasattr(trunk, 'pos_embed'):
                x = x + trunk.pos_embed
            
            if hasattr(trunk, 'pos_drop'):
                x = trunk.pos_drop(x)
            
            # Extract features from layers [3, 6, 9, 12]
            blocks = trunk.blocks if hasattr(trunk, 'blocks') else trunk.layers
            layer_features = []
            
            for i, block in enumerate(blocks):
                x = block(x)
                if (i + 1) in [3, 6, 9, 12]:  # Extract at these layers
                    # Remove CLS token, keep only patch tokens
                    patch_features = x[:, 1:, :]  # [B, num_patches, dim]
                    layer_features.append(patch_features)
            
            return layer_features
        
        else:
            # Custom VisionTransformer path
            # Patch embedding
            if vision_encoder.input_patchnorm:
                x = image.reshape(image.shape[0], image.shape[1], vision_encoder.grid_size[0],
                                vision_encoder.patch_size[0], vision_encoder.grid_size[1],
                                vision_encoder.patch_size[1])
                x = x.permute(0, 2, 4, 1, 3, 5)
                x = x.reshape(image.shape[0], vision_encoder.grid_size[0] * vision_encoder.grid_size[1], -1)
                x = vision_encoder.patchnorm_pre_ln(x)
                x = vision_encoder.conv1(x)
            else:
                x = vision_encoder.conv1(image)
                x = x.reshape(image.shape[0], image.shape[1], -1)
                x = x.permute(0, 2, 1)
            
            # Add class embedding and positional embedding
            x = torch.cat(
                [vision_encoder.class_embedding.to(x.dtype) + 
                 torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + vision_encoder.positional_embedding.to(x.dtype)
            
            # Apply patch dropout and pre-norm
            x = vision_encoder.patch_dropout(x)
            x = vision_encoder.ln_pre(x)
            
            # Permute for transformer
            x = x.permute(1, 0, 2)  # NLD -> LND
            
            # Extract features from transformer blocks
            layer_features = []
            num_blocks = len(vision_encoder.transformer.resblocks)
            
            for i in range(num_blocks):
                x, _ = vision_encoder.transformer.resblocks[i](x, attn_mask=None)
                
                if (i + 1) in [3, 6, 9, 12]:  # Extract at these layers
                    # Permute back and remove CLS token
                    x_temp = x.permute(1, 0, 2)  # LND -> NLD
                    patch_features = x_temp[:, 1:, :]  # Remove CLS token
                    layer_features.append(patch_features)
            
            return layer_features


def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP Zero-Shot Testing (No Training)')
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16')
    parser.add_argument('--text_encoder', type=str, 
                       default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    parser.add_argument('--pretrain', type=str, default='microsoft')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=1)  # Use batch_size=1 for simplicity
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Testing BiomedCLIP (Zero-Shot, No Training)")
    print(f"{'='*70}")
    print(f"Dataset: {args.obj}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    setup_seed(args.seed)

    # Load BiomedCLIP model (no adapters)
    print("Loading BiomedCLIP model...")
    biomedclip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True
    )
    biomedclip_model.eval()
    print("✅ BiomedCLIP loaded successfully\n")

    # Load test dataset
    print("Loading dataset...")
    dl_kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        **dl_kwargs
    )
    print(f"✅ Dataset loaded: {len(test_dataset)} samples\n")

    # Generate text features
    print("Generating text prompts...")
    with torch.cuda.amp.autocast(), torch.no_grad():
        # Create a wrapper to pass biomedclip_model directly
        class ModelWrapper:
            def __init__(self, model):
                self.clipmodel = model
        
        wrapper = ModelWrapper(biomedclip_model)
        text_features = encode_text_with_prompt_ensemble(wrapper, REAL_NAME[args.obj], device)
        text_features = text_features.to(dtype=torch.float16)
    print(f" Text features generated: {text_features.shape}\n")

    # Build memory bank from support set (few-shot examples)
    print("Building memory bank from support set...")
    support_images = test_dataset.fewshot_norm_img  # Normal images for reference
    support_dataset = torch.utils.data.TensorDataset(support_images)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=False, **dl_kwargs)
    
    mem_features = [[] for _ in range(4)]  # 4 layers: [3, 6, 9, 12]
    
    for image in tqdm(support_loader, desc="Extracting support features"):
        image = image[0].to(device)
        with torch.no_grad():
            layer_features = extract_features(biomedclip_model, image, device)
            for i, features in enumerate(layer_features):
                mem_features[i].append(features[0].cpu())  # Remove batch dim
    
    # Concatenate memory features
    mem_features = [torch.cat(mem_features[i], dim=0).to(device) for i in range(4)]
    print(f" Memory bank built with {len(support_images)} samples\n")

    # Test
    print("Running inference on test set...")
    result = test(args, biomedclip_model, test_loader, text_features, mem_features, device)
    
    print(f"\n{'='*70}")
    print(f"Final Result: {result:.4f}")
    print(f"{'='*70}\n")


def test(args, model, test_loader, text_features, mem_features, device):
    """Test function without adapters"""
    gt_list = []
    gt_mask_list = []
    
    seg_score_map_zero = []
    seg_score_map_few = []
    det_image_scores_zero = []
    det_image_scores_few = []

    for (image, y, mask) in tqdm(test_loader, desc="Testing"):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Extract features from BiomedCLIP
            layer_features = extract_features(model, image, device)
            
            # Remove batch dimension and CLS token (already done in extract_features)
            patch_tokens = [f[0] for f in layer_features]  # [num_patches, dim] for each layer

            if CLASS_INDEX[args.obj] > 0:
                # SEGMENTATION TASK
                
                # Few-shot scoring (cosine similarity with memory bank)
                anomaly_maps_few_shot = []
                for idx, p in enumerate(patch_tokens):
                    cos = cos_sim(mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map = F.interpolate(
                        anomaly_map,
                        size=args.img_size,
                        mode='bilinear',
                        align_corners=True
                    )
                    anomaly_maps_few_shot.append(anomaly_map[0].cpu().numpy())
                
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                seg_score_map_few.append(score_map_few)

                # Zero-shot scoring (text-image similarity)
                anomaly_maps_zero = []
                for p in patch_tokens:
                    p_norm = p / p.norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * p_norm @ text_features).unsqueeze(0)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(
                        anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                        size=args.img_size,
                        mode='bilinear',
                        align_corners=True
                    )
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                    anomaly_maps_zero.append(anomaly_map.cpu().numpy())
                
                score_map_zero = np.sum(anomaly_maps_zero, axis=0)
                seg_score_map_zero.append(score_map_zero)
                
                # Ground truth
                gt_mask_list.append(mask[0].cpu().numpy())
                gt_list.append(y[0].item())

            else:
                # DETECTION TASK
                
                # Few-shot scoring
                anomaly_maps_few = []
                for idx, p in enumerate(patch_tokens):
                    cos = cos_sim(mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map = F.interpolate(
                        anomaly_map,
                        size=args.img_size,
                        mode='bilinear',
                        align_corners=True
                    )
                    anomaly_maps_few.append(anomaly_map[0].cpu().numpy())
                
                score_few = np.sum(anomaly_maps_few, axis=0).mean()
                det_image_scores_few.append(score_few)

                # Zero-shot scoring
                anomaly_score = 0
                for p in patch_tokens:
                    p_norm = p / p.norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * p_norm @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                
                det_image_scores_zero.append(anomaly_score.cpu().item())
                gt_list.append(y[0].item())

    # Compute metrics
    gt_list = np.array(gt_list)

    if CLASS_INDEX[args.obj] > 0:
        # SEGMENTATION METRICS
        gt_mask_list = np.array(gt_mask_list)
        gt_mask_list = (gt_mask_list > 0).astype(np.int_)

        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        # Normalize
        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / \
                             (seg_score_map_zero.max() - seg_score_map_zero.min() + 1e-8)
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / \
                           (seg_score_map_few.max() - seg_score_map_few.min() + 1e-8)

        # Combine scores
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few

        # Pixel-level AUC (pAUC)
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'\n{args.obj} Pixel-level AUC (pAUC): {round(seg_roc_auc, 4)}')

        # Image-level AUC
        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} Image-level AUC: {round(roc_auc_im, 4)}')

        return seg_roc_auc + roc_auc_im

    else:
        # DETECTION METRICS
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        # Normalize
        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / \
                                (det_image_scores_zero.max() - det_image_scores_zero.min() + 1e-8)
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / \
                              (det_image_scores_few.max() - det_image_scores_few.min() + 1e-8)

        # Combine scores
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        
        # Image-level AUC
        img_roc_auc = roc_auc_score(gt_list, image_scores)
        print(f'\n{args.obj} Image-level AUC: {round(img_roc_auc, 4)}')

        return img_roc_auc


if __name__ == '__main__':
    main()