import cv2
from PIL import Image, ImageDraw
import math
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from tqdm import tqdm
import time


# reading configuration
charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
BOS = '[B]'
EOS = '[E]'
PAD = '[P]'
specials_first = (EOS,)
specials_last = (BOS, PAD)
itos = specials_first + tuple(charset) + specials_last
stoi = {s: i for i, s in enumerate(itos)}
eos_id, bos_id, pad_id = [stoi[s] for s in specials_first + specials_last]
itos = specials_first + tuple(charset) + specials_last

# image transform
preprocess_parseq = T.Compose([
        T.Resize((32, 128), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])


# decode the model output
def tokenizer_filter(probs, ids):
    ids = ids.tolist()
    try:
        eos_idx = ids.index(eos_id)
    except ValueError:
        eos_idx = len(ids)  # Nothing to truncate.
    # Truncate after EOS
    ids = ids[:eos_idx]
    probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
    return probs, ids

def ids2tok(token_ids):
    tokens = [itos[i] for i in token_ids]
    return ''.join(tokens)

def decode(token_dists):
    """Decode a batch of token distributions.
    Args:
        token_dists: softmax probabilities over the token distribution. Shape: N, L, C
        raw: return unprocessed labels (will return list of list of strings)

    Returns:
        list of string labels (arbitrary length) and
        their corresponding sequence probabilities as a list of Tensors
    """
    batch_tokens = []
    batch_probs = []
    for dist in token_dists:
        probs, ids = dist.max(-1)  # greedy selection
        probs, ids = tokenizer_filter(probs, ids)
        tokens = ids2tok(ids)
        batch_tokens.append(tokens)
        batch_probs.append(probs)
    return batch_tokens, batch_probs


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main(keyframe, parseq_path, parseq_eps=0.7, batch_size=8, device="gpu_device", img_type=".jpg"):

    parseq_model = torch.jit.load(parseq_path)

    image_folders = os.listdir(keyframe)[:1000]
    num_labels = len(image_folders)
    label_ids = [image_folder for image_folder in image_folders]
    
    print(num_labels)
    
    T_parseq = 0
    N_parseq = 0
    for i in tqdm(range(num_labels)):
        label_id = label_ids[i]
        label_folder = os.path.join(keyframe, str(label_id))
        
        ocr_list = []
        T_tmp = 0        
        if os.path.exists(label_folder):
            text_img_paths = [os.path.join(label_folder, x) for x in os.listdir(label_folder) if x.endswith("jpg")]
            N = len(text_img_paths)
            n_batch = N//batch_size+1 # number of batches
            for ii in range(n_batch):
                if batch_size*(ii+1) <= N:
                    input_holder = torch.zeros(batch_size, 3, 32, 128)
                    img_paths_tmp = text_img_paths[(ii)*batch_size:batch_size*(ii+1)]
                else:
                    input_holder = torch.zeros(N-(n_batch-1)*batch_size, 3, 32, 128)
                    img_paths_tmp = text_img_paths[(ii)*batch_size:N]

                readings = []
                #load the input holder
                for jj in range(len(img_paths_tmp)):
                            
                    img_input = Image.open(img_paths_tmp[jj]).convert('RGB')
                    img_input = preprocess_parseq(img_input.convert('RGB')).unsqueeze(0)
                    # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
                    
                    input_holder[jj, :, :, :] = img_input
                    
                # print(input_holder.size())
                start = time.time()
                with torch.no_grad():  
                    logits = parseq_model(input_holder.to(device))
                end = time.time()
                T_tmp += (end-start)
                
                pred = logits.softmax(-1)
                labels, confidences = decode(pred)    
                # print(len(labels))
                # print(len(file_names_tmp))
                for k in range(len(img_paths_tmp)):
                    readings.append("".join([labels[k][i] for i in range(len(labels[k])) if confidences[k][i] > parseq_eps]))
                                    
                for i_tmp in range(len(readings)):
                    reading = readings[i_tmp]
                    if len(reading) > 0:
                        outfilename = os.path.join(img_paths_tmp[i_tmp].replace("jpg", "txt"))
                        outfile = open(outfilename, "w")
                        outfile.write(reading)
                        outfile.close()

                    ocr_list.append(reading)
        N_parseq += N
        T_parseq += T_tmp
        
        # print("[INFO] reading took {:.6f} seconds".format(T_tmp/n))
        # print("[INFO] reading took {:.6f} seconds".format(T_tmp))
        
    print("Time of Parseq")
    print(T_parseq)
    print("Number of text images")  
    print(N_parseq)


if __name__ == "__main__":
    keyframe = "time_test"
    parseq_path = "parseq_cpu_dynamic_torchscript.pt"
    main(keyframe, parseq_path, parseq_eps=0.2, batch_size=16, device=torch.device("cpu"), img_type=".jpg")