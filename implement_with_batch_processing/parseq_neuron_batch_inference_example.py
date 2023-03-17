import cv2
import torch
import os
import time
import torch_neuron

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


def img_preprocess(img_orig, nH=32, nW=128):
    # img_orig is a numpy array from cv2.imread(img_path)
    original_image = img_orig[:, :, ::-1]

    resized_image = cv2.resize(original_image, [nW, nH], interpolation=cv2.INTER_CUBIC)
    resized_image = (resized_image - 0.5 * 255) / (0.5 * 255)

    image = torch.as_tensor(resized_image.astype("float32").transpose(2, 0, 1))

    return image

def batch_inference(images, model, batch_size):
    model_parallel = torch.neuron.DataParallel(model)
    # pre process those images to resize them to the tensor input
    
    
    N = len(images)
    n_batch = N//batch_size + 1
    labels = []
    for i in range(n_batch):
        # print(list(range((i-1)*batch_size,batch_size*i)))
        if batch_size*(i+1) < N:
            input_holder = torch.zeros(batch_size, 3, 32, 128)
            imgs_tmp = images[(i)*batch_size:batch_size*(i+1)]
        else:
            input_holder = torch.zeros(N-(n_batch-1)*batch_size, 3, 32, 128)
            imgs_tmp = images[(i)*batch_size:N]
        
        # print(input_holder.size())
        for j in range(len(imgs_tmp)):
            image = cv2.cvtColor(imgs_tmp[j], cv2.COLOR_BGR2RGB)        
            img_tensor = img_preprocess(image)
            input_holder[j, :, :, :] = img_tensor[:,:,:]
            
        start = time.time()
        logits_neuron = model_parallel(input_holder)
        end = time.time()
        
    
        pred = logits_neuron.softmax(-1)
        label, confidence = decode(pred)
        labels.append(label)

    return labels



if __name__ == "__main__":
    img_folder = "digits_demo"

    img_paths = [os.path.join(img_folder, x) for x in os.listdir(img_folder) if x.endswith("jpg")]

    images = [cv2.imread(x) for x in img_paths[:5]]

    neuron_model = torch.jit.load('parseq_model_traced_neuron_new.pt')
    
    labels = batch_inference(images, neuron_model, 4)

    print(labels)