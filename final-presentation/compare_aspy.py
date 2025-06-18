# %%
import cv2
import pandas as pd
from ignite.metrics import SSIM
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# %%
def compute_ssim(baseline_image, images):
    """
    Compute SSIM scores between a baseline image and a batch of images.

    Args:
        baseline_image (ndarray): A single baseline image of shape (H, W, 3).
        images (ndarray): A batch of images of shape (B, H, W, 3).

    Returns:
        ndarray: An array of shape (B,) containing SSIM scores.
    """

    # Ensure inputs are numpy arrays
    baseline_image = np.asarray(baseline_image)
    images = np.asarray(images)

    try:
        # Convert to torch tensors and move to CUDA
        baseline_tensor = torch.from_numpy(baseline_image).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().cuda() / 255.0

        # Initialize SSIM metric
        ssim_metric = SSIM(data_range=1.0)

        # Compute SSIM scores
        ssim_scores = []
        for img in images_tensor:
            img = img.unsqueeze(0)  # Add batch dimension
            ssim_metric.update((img, baseline_tensor))
            ssim_score = ssim_metric.compute()
            ssim_scores.append(ssim_score)  # Convert to Python float
            ssim_metric.reset()

        return np.array(ssim_scores)
    finally:
        # Explicitly delete tensors and clear GPU memory
        del baseline_tensor, images_tensor, ssim_metric
        torch.cuda.empty_cache()

# %%
def make_ssim_table(base_img_path: Path, compressed_imgs: list, compression_levels: list, compressed_img_sizes: list):
    category = base_img_path.parent.name
    name = base_img_path.name
    base_img = cv2.imread(str(base_img_path), cv2.IMREAD_COLOR)

    data = dict()
    data["category"] = [category for _ in range(len(compressed_imgs))]
    data["name"] = [name for _ in range(len(compressed_imgs))]
    data["level"] = compression_levels
    data["size"] = compressed_img_sizes
    data["ssim"] = list(compute_ssim(base_img, np.array(compressed_imgs)))
    return pd.DataFrame(data)

# %%
def find_uncompressed(p: Path):
    category = p.parent.name
    level = int(p.parent.parent.name)
    bits = int(p.stem.split("###")[0])
    key = p.name.split("###")[1]
    loc = Path("../data/splitimages/test") / category / key
    assert loc.exists()
    return bits, level, loc

# %%
from collections import defaultdict

def load_multicut_images():
    data = defaultdict(lambda: [[], [], []]) # key -> 
    ROOT = Path("samples/multicut")

    for category in ROOT.glob("*"):
        if not category.is_dir(): continue
        for img_p in category.glob("**/*.png"):
            bits, level, location = find_uncompressed(img_p)

            data[str(location)][0].append(bits)
            data[str(location)][1].append(level)
            data[str(location)][2].append(img_p)
    
    return data

mc_img_data = load_multicut_images()

# %%
# dfs = []
# for key_path, (bits, levels, compressed_img_paths) in tqdm(mc_img_data.items()):
#     uncompressed_img = cv2.imread(key_path, cv2.IMREAD_COLOR)
#     dfs.append(make_ssim_table(Path(key_path), [cv2.imread(str(i), cv2.IMREAD_COLOR) for i in compressed_img_paths], levels, bits))

# df = pd.concat(dfs)
# df.to_csv("data-multicut.csv", index=False)

# %%
# def jpeg_compress(img, q):
#     res, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
#     assert res
#     decoded_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
#     return len(data) * 8, decoded_img

# ROOT = Path("../data/splitimages/test")
# JPEG_QUALITIES = [1] + list(range(5, 101, 5))

# dfs = []

# for category in ROOT.glob("*"):
#     if not category.is_dir(): continue
#     for img_p in tqdm(list(category.glob("*.png"))):
#         img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)


#         sizes = []
#         imgs = []


#         for q in JPEG_QUALITIES:
#             compressed_size, compressed = jpeg_compress(img, q)
#             sizes.append(compressed_size)
#             imgs.append(compressed)

#         dfs.append(make_ssim_table(img_p, imgs, JPEG_QUALITIES, sizes))

# df = pd.concat(dfs)
# df.to_csv("data-jpeg.csv", index=False)

# %%
# def jpeg2000_compress(img, q):
#     res, data = cv2.imencode(".jp2", img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, q])
#     assert res
#     decoded_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
#     return len(data) * 8, decoded_img

# ROOT = Path("../data/splitimages/test")
# JPEG_2000_QUALITIES = [1] + list(range(50, 1001, 50))

# dfs = []

# for category in ROOT.glob("*"):
#     if not category.is_dir(): continue

#     for img_p in tqdm(list(category.glob("*.png"))):
#         img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)

#         sizes = []
#         imgs = []

#         try:
#             for q in JPEG_2000_QUALITIES:
#                 compressed_size, compressed = jpeg2000_compress(img, q)
#                 sizes.append(compressed_size)
#                 imgs.append(compressed)

#             dfs.append(make_ssim_table(img_p, imgs, JPEG_2000_QUALITIES, sizes))
#         except:
#             print("Failure for file: ", str(img_p))

# df = pd.concat(dfs)
# df.to_csv("data-jpeg2000.csv", index=False)

# %%
# def webp_compress(img, q):
#     res, data = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, q])
#     assert res
#     decoded_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
#     return len(data) * 8, decoded_img


# ROOT = Path("../data/splitimages/test")
# WEBP_QUALS = [1] + list(range(5, 101, 5))

# dfs = []

# for category in ROOT.glob("*"):
#     if not category.is_dir(): continue

#     for img_p in tqdm(list(category.glob("*.png"))):
#         img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)

#         sizes = []
#         imgs = []

#         try:
#             for q in WEBP_QUALS:
#                 compressed_size, compressed = webp_compress(img, q)
#                 sizes.append(compressed_size)
#                 imgs.append(compressed)

#             dfs.append(make_ssim_table(img_p, imgs, WEBP_QUALS, sizes))
#         except:
#             print("Failure for file: ", str(img_p))

# df = pd.concat(dfs)
# df.to_csv("data-webp.csv", index=False)

# %%
import cv2
import numpy as np
from PIL import Image
import io
import numpy as np
from pillow_heif import register_heif_opener
from PIL import Image
register_heif_opener()

def heif_compress(img, q):
    # Convert OpenCV image (numpy array) to PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Save the image as HEIC in memory
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format="HEIF", quality=q)
    
    # Get the size of the encoded image in bits
    bit_size = len(img_bytes.getvalue()) * 8
    
    # Decode the HEIC image back to OpenCV format
    img_bytes.seek(0)
    decoded_pil = Image.open(img_bytes)
    decoded_img = cv2.cvtColor(np.array(decoded_pil), cv2.COLOR_RGB2BGR)
    
    return bit_size, decoded_img

ROOT = Path("../data/splitimages/test")
HEIC_QUALS = [1] + list(range(5, 101, 5))
dfs = []

for category in ROOT.glob("*"):
    if not category.is_dir(): continue

    for img_p in tqdm(list(category.glob("*.png"))):
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)

        sizes = []
        imgs = []
        try:
            for q in HEIC_QUALS:
                compressed_size, compressed = heif_compress(img, q)
                sizes.append(compressed_size)
                imgs.append(compressed)

            dfs.append(make_ssim_table(img_p, imgs, HEIC_QUALS, sizes))
        except:
            print("Failure for file: ", str(img_p))

df = pd.concat(dfs)
df.to_csv("data-heif.csv", index=False)


