import os
import warnings
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from Model.biomedclip import create_model
from Model.transformer import TimmModel , HFTextEncoder
from Model.adapter import BiomedCLIP_Inplanted
from dataset.medical_few import MedDataset
from utils import augment, cos_sim, encode_text_with_prompt_ensemble
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from prompt import REAL_NAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"


warnings.filterwarnings("ignore")

# Device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Biomed datasets indexing (you can keep as is if relevant for your task)
CLASS_INDEX = {'Brain': 3, 'Liver': 2, 'Retina_RESC': 1, 'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3}
CLASS_INDEX_INV = {v: k for k, v in CLASS_INDEX.items()}

# Global container for convenience (if you need later)
global_vars = {}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='BiomedCLIP Testing')
    # General defaults
    parser.add_argument('--model_name', type=str, default='BiomedCLIP-PubMedBERT-ViT-B-16',
                        help="BiomedCLIP model version")    
    parser.add_argument('--text_encoder', type=str, default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
                        help="Text encoder used for BiomedCLIP" )

    parser.add_argument('--pretrain', type=str, default='microsoft',
                            help="pretrained checkpoint source")
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help="path to dataset"  )
    #parser.add_argument('--data_path', type=str, default='/kaggle/input/preprocessed/Liver')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224, 
                        help="BiomedCLIP trained with 224x224 resolution")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12],
                        help="layer features used for adapters")    
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    args, _ = parser.parse_known_args()

    return args


def main():
    args = parse_args()

    # Save args in global_vars and globals for convenience
    global_vars.update(vars(args))
    global_vars['args'] = args
    for k, v in global_vars.items():
        globals()[k] = v

    # Print device and parsed args (optional)
    print(f"Using device: {device}")
    print("Parsed arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # set seed
    setup_seed(args.seed)

    # ---- Prepare models (user must implement / provide these functions/classes) ----
    # create_model, CLIP_Inplanted, MedDataset, augment, encode_text_with_prompt_ensemble,
    # REAL_NAME, FocalLoss, BinaryDiceLoss, cos_sim must be available in the environment.

    # fixed feature extractor (clip_model)
    biomedclip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device,
                              pretrained=args.pretrain, require_pretrained=True)
    biomedclip_model.eval()

    model = BiomedCLIP_Inplanted(biomedclip_model=biomedclip_model, features=args.features_list).to(device)
    model.eval()

    # make sure adapter params require grad (we'll optimize adapters)
    for name, param in model.named_parameters():
        param.requires_grad = True

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # load test dataset
    dl_kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    # few-shot image augmentation (user-provided augment function expected)
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)

    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **dl_kwargs)

    # memory bank construction (support set)
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **dl_kwargs)

    # losses (user must provide these classes / functions)
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # text prompt features
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(model, REAL_NAME[args.obj], device)

    best_result = 0.0

    for epoch in range(args.epoch):
        print('epoch ', epoch, ':')

        loss_list = []
        model.train()
        for (image, gt, label) in train_loader:
            image = image.to(device)
            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                # Convert tokens to expected shapes
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

                # detection loss
                det_loss = 0.0
                image_label = label.to(device)
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)

                if CLASS_INDEX[args.obj] > 0:
                    # pixel level segmentation loss
                    seg_loss = 0.0
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(seg_patch_tokens)):
                        seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features).unsqueeze(0)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)

                    loss = seg_loss + det_loss
                    loss.requires_grad_(True)
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    seg_optimizer.step()
                    det_optimizer.step()
                else:
                    loss = det_loss
                    loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    loss.backward()
                    det_optimizer.step()

                loss_list.append(loss.item())

        print("Loss: ", np.mean(loss_list))

        # Build memory bank features from support loader
        model.eval()
        seg_features = []
        det_features = []
        for image in support_loader:
            image = image[0].to(device)
            with torch.no_grad():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
                det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
                seg_features.append(seg_patch_tokens)
                det_features.append(det_patch_tokens)

        seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
        det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]

        # Evaluate on test set (user-provided test function expected)
        result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
        if result > best_result:
            best_result = result
            print("Best result:", best_result)
            if args.save_model == 1:
                os.makedirs(args.save_path, exist_ok=True)
                ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')
                torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                            'det_adapters': model.det_adapters.state_dict()},
                           ckp_path)


                           
def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    gt_list = []
    gt_mask_list = []
    det_image_scores_few = []
    seg_score_map_few = []
    det_image_scores_zero = []
    seg_score_map_zero = []

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        # ✅ FIX: Extend the ground truth lists with each item from the batch
        gt_list.extend(y.cpu().detach().numpy())
        # Detach and move to CPU before converting to list
        gt_mask_list.extend(list(mask.cpu().detach().numpy()))

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            # No change needed here, these are correctly processed per-batch below
            seg_patch_tokens_list = [p[:, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens_list = [p[:, 1:, :] for p in det_patch_tokens]

        # Process each image in the batch
        for i in range(image.shape[0]):
            seg_patch_tokens = [p[i] for p in seg_patch_tokens_list]
            det_patch_tokens = [p[i] for p in det_patch_tokens_list]
            
            if CLASS_INDEX[args.obj] > 0:
                # few-shot, seg head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    cos = cos_sim(seg_mem_features[idx], p.unsqueeze(0))
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(anomaly_map_few_shot,
                                                         size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                seg_score_map_few.append(score_map_few)

                # zero-shot, seg head
                anomaly_maps = []
                for layer in range(len(seg_patch_tokens)):
                    p_norm = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * p_norm @ text_features).unsqueeze(0)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                    anomaly_maps.append(anomaly_map.cpu().numpy())
                score_map_zero = np.sum(anomaly_maps, axis=0)
                seg_score_map_zero.append(score_map_zero)

            else:
                # few-shot, det head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(det_patch_tokens):
                    cos = cos_sim(det_mem_features[idx], p.unsqueeze(0))
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(anomaly_map_few_shot,
                                                         size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                score_few_det = anomaly_map_few_shot.mean()
                det_image_scores_few.append(score_few_det)
                
                # zero-shot, det head
                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    p_norm = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * p_norm @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                det_image_scores_zero.append(anomaly_score.cpu().item())

    gt_list = np.array(gt_list)

    if CLASS_INDEX[args.obj] > 0:
        gt_mask_list = np.array(gt_mask_list)
        gt_mask_list = (gt_mask_list > 0).astype(np.int_)
        
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)
        
        # Normalize scores
        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min())
        
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        
        # Calculate metrics
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')
        
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores.reshape(len(segment_scores), -1), axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')
        
        return seg_roc_auc + roc_auc_im
    
    else:
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        # Normalize scores
        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min())

        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')
        
        return img_roc_auc_det


if __name__ == '__main__':
    main()
