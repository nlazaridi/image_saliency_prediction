import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse

class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        #ssert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

def test_net(args):

    cudnn.benchmark = True

    net = torch.jit.load('vst_torchscript_2.pt')
    if args.device == 'gpu':
        net.cuda()
    if args.device == 'cpu':
        net.cpu()
    net.eval()

    transform = transforms.Compose([
            Scale((args.img_size, args.img_size)),
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


    saliency_mask = net(image)
    output_s = F.sigmoid(saliency_mask)
    if args.device == 'gpu':
        output_s = output_s.data.cuda().squeeze(0)

    if args.device == 'cpu':
        output_s = output_s.data.cpu().squeeze(0)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        Scale((image_w, image_h))
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
    parser.add_argument('--device', default='cpu', type=str, help='gpu or cpu')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    if args.Testing:
        saliency_mask = test_net(args)
    

