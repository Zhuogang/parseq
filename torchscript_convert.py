import torch
import torchvision
import onnx
import onnxruntime as ort
from torchvision import transforms as T
from PIL import Image

preprocess_parseq = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])

def main(batch_size, img_folder = "/home/ubuntu/parseq/demo_images"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().cuda()
    test_img_path = "demo_images/art-01107.jpg"
    img = Image.open(test_img_path).convert('RGB')
    # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
    img = preprocess_parseq(img.convert('RGB')).unsqueeze(0).to(device)

    logits = parseq(img)
    logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

    # Greedy decoding
    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)
    print(confidence)
    print('Decoded label = {}'.format(label[0]))

    dummy_input = torch.rand(batch_size, 3, 32, 128) 

    output_path = "parseq_model_traced_cuda.pt"
    parseq.to_torchscript(output_path)

    ts_model = torch.jit.load(output_path)

    logits = ts_model(img)
    logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

    # Greedy decoding
    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)
    print('Decoded label = {}'.format(label[0]))


if __name__ == '__main__':
    batch_size = 1
    main(batch_size)