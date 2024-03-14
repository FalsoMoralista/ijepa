
import PIL
from datasets.PlantCLEF2022 import make_PlantCLEF2022
import torchvision.transforms as transforms
import torchvision
import torch

#import PIL.ImageStat

transform_list = [] 

#transform_list += [transforms.Resize(224, interpolation=PIL.Image.BICUBIC)] # to maintain same ratio w.r.t. 224 images    
transform_list += [transforms.ToTensor()]
transform = transforms.Compose(transform_list)

dataset, loader, sampler = make_PlantCLEF2022(
        transform=transform,
        batch_size=1,
        training=True,
        num_workers=8,
        root_path='/home/rtcalumby/adam/luciano/LifeCLEFPlant2022/',
        image_folder='',
        drop_last=False)
ipe = len(loader)
classes = dataset.class_to_idx

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data in loader:
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels
    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


print('Dataset Lenght: ', ipe)
means = []
stds = []
for image in enumerate(loader):
    image = torch.squeeze(image[1][0])

    img = torchvision.transforms.functional.to_pil_image(image)
    means.append( PIL.ImageStat.Stat(img).mean)
    stds.append( PIL.ImageStat.Stat(img).stddev)

mean_r = std_r = mean_g = std_g = mean_b = std_b = 0
for i in range(len(means)):
    mean_r += means[i][0]    
    mean_g += means[i][1]    
    mean_b += means[i][2]

    std_r += stds[i][0]    
    std_g += stds[i][1]    
    std_b += stds[i][2]

mean_r /= ipe
mean_g /= ipe        
mean_b /= ipe        

std_r /= ipe
std_g /= ipe
std_b /= ipe

print([mean_r, mean_g, mean_b])
print([std_r, std_g, std_b])
