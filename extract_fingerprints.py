import os


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import glob
import json
import torch
import random
import numpy as np
from PIL import Image
import open_clip
from torchvision import transforms
from sparc.model import get_sae_model_class


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

CONFIG = {
    "dataset_root": "dataset",
    "checkpoint_dir": "model",
    "clip_weights": "clip_weights/open_clip_pytorch_model.bin",
    "db_output_path": "semantic_database.pt",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 32,
    "sae_output_key": "logits_clip_img"
}


def load_dino(device):


    hub_dir = torch.hub.get_dir()
    repo_dir = os.path.join(hub_dir, "facebookresearch_dinov2_main")
    checkpoint_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(repo_dir):
        raise RuntimeError(
            "DINOv2 hub repo is not cached and this machine has no network.\n"
            f"  expected: {repo_dir}\n"
            "  fix: copy the cache from a machine that has it, or set TORCH_HOME "
            "to a directory that contains it."
        )
    cached = [
        name
        for name in os.listdir(checkpoint_dir)
        if name.startswith("dinov2_vitl14_reg") and name.endswith(".pth")
    ] if os.path.isdir(checkpoint_dir) else []
    if not cached:
        raise RuntimeError(
            "DINOv2 weights are not cached and this machine has no network.\n"
            f"  expected something like: {checkpoint_dir}/dinov2_vitl14_reg4_pretrain.pth\n"
            "  fix: copy it from a machine that has it, or set TORCH_HOME."
        )
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14_reg", force_reload=False
    ).to(device)
    model.eval()
    return model


def setup_image_models(config):
    print("loading vision encoders")
    device = config["device"]
    dino_model = load_dino(device)
    dino_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    clip_model, _, clip_transform = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained=config["clip_weights"], device=device
    )
    clip_model.eval()
    run_config_path = os.path.join(config["checkpoint_dir"], "run_config.json")
    weights_path = os.path.join(config["checkpoint_dir"], "msae_checkpoint.pth")
    with open(run_config_path, 'r') as f:
        run_args = json.load(f)['args']
    SaeModelClass = get_sae_model_class(run_args.get('topk_type', 'global'))
    n_latents = run_args['n_latents']
    sparc_model = SaeModelClass(
        d_streams={"dino": 1024, "clip_img": 768, "clip_txt": 768},
        n_latents=n_latents, k=run_args['k'], auxk=run_args['auxk']
    ).to(device)
    sparc_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    sparc_model.eval()
    return dino_model, dino_transform, clip_model, clip_transform, sparc_model, n_latents


@torch.no_grad()
def build_database(output_path=None, sae_key=None, min_occurrences=None,
                   db_images_per_sample=9):


    if output_path is not None:
        CONFIG["db_output_path"] = output_path
    if sae_key is not None:
        CONFIG["sae_output_key"] = sae_key

    models = setup_image_models(CONFIG)
    dino_m, dino_t, clip_m, clip_t, sparc_m, n_latents = models
    device = CONFIG["device"]
    target_key = CONFIG["sae_output_key"]

    sample_dirs = sorted([
        d for d in glob.glob(os.path.join(CONFIG["dataset_root"], "sample_*"))
        if os.path.isdir(d)
    ])
    print(f"{len(sample_dirs)} sample directories")

    min_occurrences = 2 if min_occurrences is None else min_occurrences
    db_vectors, db_labels = [], []

    for i, sample_dir in enumerate(sample_dirs):
        label = os.path.basename(sample_dir)

        txt_path = os.path.join(CONFIG["dataset_root"], f"{label}.txt")
        if not os.path.exists(txt_path):
            alt_txt_path = os.path.join(sample_dir, f"{label}.txt")
            if os.path.exists(alt_txt_path):
                txt_path = alt_txt_path
            else:
                print(f"  {txt_path} missing, skipping")
                continue

        with open(txt_path, 'r', encoding='utf-8') as f:
            raw_text = f.read().strip()

        if not raw_text:
            print(f"  {label}: empty text, skipping")
            continue

        tokenized = open_clip.tokenize([raw_text]).to(device)
        txt_feat = clip_m.encode_text(tokenized)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        all_img_paths = sorted(
            glob.glob(os.path.join(sample_dir, "*.png"))
            + glob.glob(os.path.join(sample_dir, "*.jpg"))
        )
        img_paths = all_img_paths[:db_images_per_sample]

        if not img_paths:
            print(f"  {sample_dir} has no images, skipping")
            continue

        if len(all_img_paths) <= db_images_per_sample:
            print(
                f"Warning: {label} has no held-out query image "
                f"(images={len(all_img_paths)}, db_images={db_images_per_sample})"
            )

        dense_list = []
        for batch_start in range(0, len(img_paths), CONFIG["batch_size"]):
            batch_paths = img_paths[batch_start: batch_start + CONFIG["batch_size"]]
            batch_size = len(batch_paths)
            d_imgs = torch.stack([dino_t(Image.open(p).convert('RGB')) for p in batch_paths]).to(device)
            c_imgs = torch.stack([clip_t(Image.open(p).convert('RGB')) for p in batch_paths]).to(device)

            d_feat = dino_m(d_imgs)
            c_feat = clip_m.encode_image(c_imgs)
            c_feat /= c_feat.norm(dim=-1, keepdim=True)

            expanded_txt_feat = txt_feat.expand(batch_size, -1)

            img_output = sparc_m({
                "dino": d_feat,
                "clip_img": c_feat,
                "clip_txt": expanded_txt_feat
            })

            if target_key not in img_output:
                raise KeyError(f"key '{target_key}' not in the model output; available: {list(img_output.keys())}")
            dense_list.append(img_output[target_key].cpu())

        if not dense_list:
            continue

        stacked_dense = torch.cat(dense_list, dim=0)

        flat_vals = stacked_dense[stacked_dense > 0]
        if flat_vals.numel() == 0:
            continue
        quantile_99 = torch.quantile(flat_vals, 0.99)
        threshold = quantile_99 * 0.01
        stacked_dense_filtered = torch.where(stacked_dense > threshold, stacked_dense, torch.zeros_like(stacked_dense))

        is_active = stacked_dense_filtered > 0
        occurrence_count = is_active.sum(dim=0)

        stable_mask = occurrence_count >= min_occurrences

        mean_vec = stacked_dense_filtered.mean(dim=0)
        prototype_vec = torch.zeros_like(mean_vec)
        prototype_vec[stable_mask] = mean_vec[stable_mask]

        db_vectors.append(prototype_vec)
        db_labels.append(label)
        print(
            f"[{i + 1:>3}/{len(sample_dirs)}] {label} | stable latents (>= {min_occurrences}x): {stable_mask.sum().item():>9}")

    if not db_vectors:
        print("database construction failed")
        return

    all_vecs = torch.stack(db_vectors)
    df = (all_vecs > 0).sum(dim=0).float()
    weights = torch.log((len(db_labels) + 1.0) / (df + 1.0)) + 1.0

    torch.save(
        {
            "vectors": all_vecs,
            "labels": db_labels,
            "weights": weights,
            "n_latents": n_latents,
            "db_images_per_sample": db_images_per_sample,
        },
        CONFIG["db_output_path"],
    )
    print(f"\ndatabase written to {CONFIG['db_output_path']} (first nine images per group, threshold {min_occurrences})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the SAE semantic database.")
    parser.add_argument("--dataset-root", default=None, help="root of the known-sample dataset")
    parser.add_argument("--output", default=None, help="database output path")
    parser.add_argument("--sae-key", default=None, help="SAE output key")
    parser.add_argument("--min-occurrences", type=int, default=None, help="minimum occurrences for a stable latent")

    parser.add_argument(
        "--db-images-per-sample",
        type=int,
        default=9,
        help="Number of images per sample used to build the database.",
    )
    args = parser.parse_args()
    if args.dataset_root:
        CONFIG["dataset_root"] = args.dataset_root

    build_database(
        output_path=args.output,
        sae_key=args.sae_key,
        min_occurrences=args.min_occurrences,
        db_images_per_sample=args.db_images_per_sample,
    )
