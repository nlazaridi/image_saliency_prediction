import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import transforms as trans
from torchvision import transforms
from PIL import Image
import argparse

def test_net(args):

    cudnn.benchmark = True

    net = torch.jit.load('vst_torchscript.pt')
    if args.device == 'gpu':
        net.cuda()
    if args.device == 'cpu':
        net.cpu()
    net.eval()

    transform = trans.Compose([
            trans.Scale((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
        ])

    image = Image.open(args.img_path).convert('RGB')
    image_w = image.width
    image_h = image.height
    if args.device == 'gpu':
        image = transform(image).unsqueeze(0).cuda()

    if args.device == 'cpu':
        image = transform(image).unsqueeze(0).cpu()


    outputs_saliency, _ = net(image)
    _, _, _, mask_1_1 = outputs_saliency
    output_s = F.sigmoid(mask_1_1)
    if args.device == 'gpu':
        output_s = output_s.data.cuda().squeeze(0)

    if args.device == 'cpu':
        output_s = output_s.data.cpu().squeeze(0)

    transform = trans.Compose([
        transforms.ToPILImage(),
        trans.Scale((image_w, image_h))
    ])
    output_s = transform(output_s)

    filename = args.img_path.split('/')[-1].split('.')[0]

    # save saliency maps
    
    if not os.path.exists(args.save_test_path_root):
        os.makedirs(args.save_test_path_root)
    output_s.save(os.path.join(args.save_test_path_root, filename + '_mask.png'))
    return(output_s)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--Testing', default=True, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--img_path', type=str, default='/data2/nlazaridis/sal_dataset/DHF1K/test_set/0731/images/0245.png')
    parser.add_argument('--test_paths', type=str, default='/data2/nlazaridis/sal_dataset/DHF1K/test_set/0731')
    parser.add_argument('--device', default='cpu', type=str, help='gpu or cpu')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if args.Testing:
        saliency_mask = test_net(args)
    

