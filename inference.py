from gradio.helpers import log_message
from tqdm import tqdm
import time
import datetime
from argparse import ArgumentParser
import numpy as np
from config import *
from data import CelebA
import torch.utils.data as data
from networks.Ensemble_models import Ensemble
from networks.Generalize_Model import Models as Black_models
from networks.vgg import Vgg16
from tools.color_space import rgb2ycbcr_np, ycbcr_to_tensor, ycbcr_to_rgb
from tools.metrics_compute import compute_metrics, compute_psnr, prepare_lpips, get_perceptual_loss
from tools.tool import *
from torchvision import transforms
import cv2
from logger import setup_logger
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from networks.bisenet import BiSeNet
# from networks.PG_face_dualmask import define_G as face_G
# from networks.PG_hair_dualmask import define_G as hair_G
from networks.PG_face_model import define_G as face_G
from networks.PG_hair_model import define_G as hair_G


def load_face_pretrained_weights(model, weight_path):
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path)
        if "face_protection_net" in state_dict:
            model.load_state_dict(state_dict["face_protection_net"], strict=False)
            print(f"Loaded pretrained weights from {weight_path}")
            return True
        else:
            print(f"Warning: 'face_protection_net' key not found in {weight_path}")
    else:
        print(f"Warning: Pretrained weight file not found at {weight_path}")
    return False

def load_hair_pretrained_weights(model, weight_path):
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path)
        if "hair_protection_net" in state_dict:
            model.load_state_dict(state_dict["hair_protection_net"], strict=False)
            print(f"Loaded pretrained weights from {weight_path}")
            return True 
        else:
            print(f"Warning: 'hair_protection_net' key not found in {weight_path}")
    else:
        print(f"Warning: Pretrained weight file not found at {weight_path}")
    return False 

def load_model(model_name: str, num_classes: int, weight_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load and initialize the BiSeNet model.

    Args:
        model_name: Name of the backbone model (e.g., "resnet18")
        num_classes: Number of segmentation classes
        weight_path: Path to the model weights file
        device: Device to load the model onto

    Returns:
        torch.nn.Module: Initialized and loaded model
    """
    model = BiSeNet(num_classes, backbone_name=model_name)
    model.to(device)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        raise ValueError(f"Weights not found from given path ({weight_path})")

    model.eval()
    return model

def create_single_labels_attgan(att_a, selected_attrs):
    num_attrs = len(selected_attrs)

    b_trg_list = []

    for i in range(num_attrs):
        b_trg = att_a.clone()
        b_trg_list.append(b_trg)

        b_trg = att_a.clone()
        b_trg[:, i] = 1 - b_trg[:, i]
        b_trg_list.append(b_trg)

    return b_trg_list

def preprocess_single_image_attgan(image_path, img_size=256, device='cuda', num_attrs=13):
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    all_attrs = [
        "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair",
        "Bushy_Eyebrows", "Eyeglasses", "Male", "Mouth_Slightly_Open",
        "Mustache", "No_Beard", "Pale_Skin", "Young"
    ]

    c_org = torch.zeros((1, 13)).to(device)
    att_a = torch.zeros((1, 13)).to(device)

    selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
    for i, attr in enumerate(selected_attrs):
        idx = all_attrs.index(attr)
        c_org[0, idx] = 0
        att_a[0, idx] = 0
    filename = os.path.basename(image_path)
    return image_tensor, att_a, c_org, filename

def preprocess_single_image_fgan(image_path, img_size=256, device='cuda'):
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    c_org = torch.tensor([[0, 0, 0, 0, 0]]).to(device)
    att_a = torch.tensor([[0, 0, 0, 0, 0]]).to(device)

    filename = os.path.basename(image_path)

    return image_tensor, att_a, c_org, filename


def save_image_tensor(tensor, save_path, filename):
    with torch.no_grad():
        if tensor.ndim == 4:
            if tensor.shape[0] > 1:
                print(f"Warning: The tensor contains multiple images：({tensor.shape[0]}), save only the first one")
            tensor = tensor[0]

        tensor = tensor.cpu().detach()
        tensor = torch.clamp(tensor * 0.5 + 0.5, 0, 1)

        pil_image = transforms.ToPILImage()(tensor)
        pil_image.save(os.path.join(save_path, filename))

    print("All results have been saved.")

def save_single_results(results, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    original_results = results[0]
    defense_results = results[1]

    print(f"Number of original results: {len(original_results)}")
    # save_image_tensor(original_result, save_path, f"original_{filename}")
    for i, original_result in enumerate(original_results):
        print(f"Save the {i + 1} original results, shape: {original_result.shape}")
        save_image_tensor(original_result, save_path, f"original_{i + 1}_{filename}")
    print(f"Number of defense results: {len(defense_results)}")
    for i, defense_result in enumerate(defense_results):
        print(f"Save the {i + 1} Defense results, shape: {defense_result.shape}")
        save_image_tensor(defense_result, save_path, f"defense_{i + 1}_{filename}")



if __name__ == '__main__':
    config_parser = ArgumentParser()
    config_parser.add_argument('--flag', default=False, type=bool, help='is save results')
    config_parser.add_argument('--model_choice', default='hisd', type=str, help='compute metrics choice')
    config_parser.add_argument('--StarG_path', default="./checkpoints/stargan/200000-G.ckpt", type=str,
                               help='Stargan Weight Path')
    config_parser.add_argument('--AttentionG_path', default="./checkpoints/attentiongan/200000-G.ckpt", type=str,
                               help='AttentionGAN Weight Path')
    config_parser.add_argument('--test_path', default='./demo_results', type=str, help='Test Result Path')
    config_parser.add_argument('--test_img', default='./test_samples/test_face_2.png', type=str, help='Test Image Name')
    config_parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    config_parser.add_argument('--eposilon', default=0.01, type=float, help='Perturbation Scale')
    config_parser.add_argument('--img_size', default=256, type=float, help='Image Size')
    config_parser.add_argument('--dataset_path', default='../../dataset/CelebAMask-HQ/CelebA-HQ-img/',
                               type=str, help='Dataset Path')
    config_parser.add_argument('--attribute_txt_path',
                               default='../../dataset/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt',
                               type=str, help='Attribute Txt Path')
    config_parser.add_argument('--selected_attrs', default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"], type=list, help='Attribute Selection')
    config_parser.add_argument('--face_pretrained_weights', default='./checkpoints/single_model_adv/hisd_adv/hisd_face_perturb.pth',
                               type=str,
                               help='Path to pretrained weights for continuing training')
    config_parser.add_argument('--hair_pretrained_weights',
                               default='./checkpoints/single_model_adv/hisd_adv/hisd_hair_perturb.pth',
                               type=str,
                               help='Path to pretrained weights for continuing training')

    opts = config_parser.parse_args()
    print(opts)

    save_path = opts.test_path + '/' + opts.model_choice +'-inference-result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # prepare
    y_mask = Y_mask(opts)

    # compute metrics prepare
    lpips_model, lpips_model2 = prepare_lpips()
    vgg = Vgg16().to(device)
    vgg.eval()
    criterion = torch.nn.MSELoss()

    # net prepare
    Models = Ensemble(opts)
    GAN = Black_models(opts)
    Bise_net = load_model(model_name="resnet34", num_classes=19, weight_path="./checkpoints/resnet34.pt", device=device)
    Bise_net.eval()

    # PG load
    features = 384
    n_heads = 6
    n_blocks = 6
    ffn_features = 1536
    embed_features = 384
    activ = 'gelu'
    norm = 'layer'
    image_shape = (3, 256, 256)
    unet_features_list = [48, 96, 192, 384]
    unet_activ = 'leakyrelu'
    unet_norm = 'instance'
    unet_downsample = 'conv'
    unet_upsample = 'upsample-conv'
    rezero = True
    activ_output = 'sigmoid'


    face_PG = face_G(input_nc, output_nc, ngf, 'unet_64', 'instance', not no_dropout, init_type, init_gain).to(
        device)
    hair_PG = hair_G(input_nc, output_nc, ngf, 'unet_64', 'instance', not no_dropout, init_type, init_gain).to(device)

    '''load PG pretrained weights'''
    if opts.face_pretrained_weights:
        face_success = load_face_pretrained_weights(face_PG, opts.face_pretrained_weights)
        print("Face Training from pretrained weights")
        hair_success = load_hair_pretrained_weights(hair_PG, opts.hair_pretrained_weights)
        print("hair Training from pretrained weights")
        if not face_success:
            print("Face Training from scratch")
        if not hair_success:
            print("Hair Training from scratch")
    else:
        print("No pretrained weights provided, training from scratch")

    face_PG.eval()
    hair_PG.eval()

    test_dataset = CelebA(opts.dataset_path, opts.attribute_txt_path,
                         opts.img_size, 'test', attrs,
                         opts.selected_attrs)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1, num_workers=4,
        shuffle=True
    )
    print('The number of val Iterations = %d' % len(test_dataloader))

    psnr_value, ssim_value, lpips_alex, lpips_vgg = 0.0, 0.0, 0.0, 0.0
    succ_num, total_num, n_dist = 0.0, 0.0, 0.0
    l1_error, l2_error = 0.0, 0.0
    vgg_sum = 0.0
    psnr_adv, ssim_adv, lpips_alexs_adv, lpips_vggs_adv = 0.0, 0.0, 0.0, 0.0

    logger_save_path = opts.test_path + '/' + opts.model_choice +'-2'
    # path_isexists()

    logger = setup_logger(logger_save_path, 'test_result.log', 'test_logger')
    logger.info(f'Start testing...')
    start_time = time.time()

    if opts.model_choice == 'fgan':
        x_real, att_a, c_org, filename = preprocess_single_image_fgan(opts.test_img, opts.img_size, device)
        ref_img, att_a, c_org, filename = preprocess_single_image_fgan('./test_samples/res_1.jpg', opts.img_size, device)
    elif opts.model_choice == 'attgan':
        x_real, att_a, c_org, filename = preprocess_single_image_attgan(opts.test_img, opts.img_size, device, len(opts.selected_attrs))
        ref_img, att_a, c_org, filename = preprocess_single_image_attgan(
            './test_samples/res_1.jpg', opts.img_size, device, len(opts.selected_attrs))
    else:
        x_real, att_a, c_org, filename = preprocess_single_image_fgan(opts.test_img, opts.img_size, device)
        ref_img, att_a, c_org, filename = preprocess_single_image_fgan(
            './test_samples/res_1.jpg', opts.img_size, device)

    with torch.no_grad():
        c_trg_list = Models.create_labels(c_org)

        # b_trg_list = create_single_labels_attgan(att_a, opts.selected_attrs)
        x_ori = tensor2numpy(x_real)
        x_ref = tensor2numpy(ref_img)

        if opts.model_choice == 'fgan':
            ori_outs = Models.fgan_outs(x_real, c_trg_list)
        elif opts.model_choice == 'attgan':
            b_trg_list = Models.create_labels_attgan(att_a)
            ori_outs = Models.attgan_outs(x_real, b_trg_list)
        elif opts.model_choice == 'hisd':
            ori_outs = Models.hisd_outs(x_real)
        else:
            ori_outs = GAN.model_out(x_real, c_trg_list)

        # compute saliency mask using BiSeNet
        bise_outputs = Bise_net(x_real)
        if isinstance(bise_outputs, tuple) or isinstance(bise_outputs, list):
            main_output = bise_outputs[0] 
        else:
            main_output = bise_outputs

        face_predicted_mask = main_output.detach().squeeze(0)
        face_predicted_mask = torch.argmax(face_predicted_mask, dim=0)
        face_predicted_mask = face_predicted_mask.cpu().numpy()

        hair_predicted_mask = main_output.detach().squeeze(0)
        hair_predicted_mask = torch.argmax(hair_predicted_mask, dim=0)
        hair_predicted_mask = hair_predicted_mask.cpu().numpy()

        face_mask_pil = Image.fromarray(face_predicted_mask.astype(np.uint8))
        face_restored_mask = face_mask_pil.resize((256, 256), resample=Image.NEAREST)
        face_predicted_mask = np.array(face_restored_mask)

        hair_mask_pil = Image.fromarray(hair_predicted_mask.astype(np.uint8))
        hair_restored_mask = hair_mask_pil.resize((256, 256), resample=Image.NEAREST)
        hair_predicted_mask = np.array(hair_restored_mask)

        face_predicted_mask_binary = np.zeros_like(face_predicted_mask, dtype=np.uint8)
        hair_predicted_mask_binary = np.zeros_like(hair_predicted_mask, dtype=np.uint8)

        # 0:背景, 1:皮肤, 2:左眉, 3:右眉, 4:左眼, 5:右眼, 6:眼镜, 7:左耳, 8:右耳,
        # 9:耳环, 10:鼻子, 11:嘴, 12:上唇, 13:下唇, 14:脖子, 15:领子, 16:衣服, 17:头发, 18:帽子
        face_classes = [1]
        hair_classes = [17, 18]

        for class_id in face_classes:
            face_predicted_mask_binary[face_predicted_mask == class_id] = 1

        for class_id in hair_classes:
            hair_predicted_mask_binary[hair_predicted_mask == class_id] = 1

        face_predicted_mask_binary = ndimage.binary_closing(face_predicted_mask_binary,
                                                            structure=np.ones((3, 3))).astype(
            np.uint8)
        face_predicted_mask_binary = ndimage.binary_fill_holes(face_predicted_mask_binary).astype(np.uint8)

        face_predicted_mask = torch.from_numpy(face_predicted_mask_binary).unsqueeze(0).to(device)

        hair_predicted_mask_binary = ndimage.binary_closing(hair_predicted_mask_binary,
                                                            structure=np.ones((3, 3))).astype(
            np.uint8)
        hair_predicted_mask_binary = ndimage.binary_fill_holes(hair_predicted_mask_binary).astype(np.uint8)
        hair_predicted_mask = torch.from_numpy(hair_predicted_mask_binary).unsqueeze(0).to(device)

        # convert RGB to YCbCr
        x_ycbcr = rgb2ycbcr_np(x_ori)
        x_y = ycbcr_to_tensor(x_ycbcr).cuda()
        x_r_ycbcr = rgb2ycbcr_np(x_ref)
        x_r = ycbcr_to_tensor(x_r_ycbcr).cuda()

        face_adv_A = face_PG(x_y, x_r, face_predicted_mask)
        hair_adv_A = hair_PG(x_y, x_r, hair_predicted_mask)

   
        face_adv_noise = face_adv_A
        face_adv_noise = face_adv_noise * y_mask
        face_adv_noise = torch.clamp(face_adv_noise, -opts.eposilon, opts.eposilon)

        hair_adv_noise = hair_adv_A
        hair_adv_noise = hair_adv_noise * y_mask
        hair_adv_noise = torch.clamp(hair_adv_noise, -opts.eposilon, opts.eposilon)

        x_L_adv = x_y + face_adv_noise + hair_adv_noise
        x_adv = ycbcr_to_rgb(x_L_adv)
  
        adv_A = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_adv.contiguous())

        if opts.model_choice == 'fgan':
            adv_outs = Models.fgan_outs(adv_A, c_trg_list)
        elif opts.model_choice == 'attgan':
            adv_outs = Models.attgan_outs(adv_A, b_trg_list)
        elif opts.model_choice == 'hisd':
            adv_outs = Models.hisd_outs(adv_A)
        else:
            adv_outs = GAN.model_out(adv_A, c_trg_list)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"代码运行时间: {execution_time:.5f} 秒")
        results = []
        # results.append(torch.cat(x_ori, dim=0))
        results.append(torch.cat(ori_outs[1:], dim=0))
        results.append(torch.cat(adv_outs[1:], dim=0))
        save_single_results(results, save_path, filename)

        ##### compute metrics #####
        # between clean image and defensed image
        psnr_adv, ssim_adv, lpips_alexs_adv, lpips_vggs_adv = compute_metrics(x_real, adv_A, lpips_model, lpips_model2)

        psnr_value, ssim_value, lpips_alex, lpips_vgg = 0.0, 0.0, 0.0, 0.0
        l1_error, l2_error, vgg_sum = 0.0, 0.0, 0.0
        succ_num, total_num, n_dist = 0.0, 0.0, 0.0

        for i in range(len(adv_outs) - 1):
            psnr_temp, ssim_temp, lpips_alex_temp, lpips_vgg_temp = compute_metrics(ori_outs[i + 1], adv_outs[i + 1],
                                                                                    lpips_model, lpips_model2)
            l1_error += torch.nn.functional.l1_loss(ori_outs[i + 1], adv_outs[i + 1]).item()
            l2_error += torch.nn.functional.mse_loss(ori_outs[i + 1], adv_outs[i + 1]).item()
            loss_style, loss_content = get_perceptual_loss(vgg, ori_outs[i + 1], adv_outs[i + 1])
            vgg_per = (STYLE_WEIGHT * loss_style + CONTENT_WEIGHT * loss_content).item()
            vgg_sum += vgg_per
            psnr_value += psnr_temp
            ssim_value += ssim_temp
            lpips_alex += lpips_alex_temp
            lpips_vgg += lpips_vgg_temp

            # ASR
            mask_d = abs(ori_outs[i + 1] - x_real)
            mask_d = mask_d[0, 0, :, :] + mask_d[0, 1, :, :] + mask_d[0, 2, :, :]
            mask_d[mask_d > 0.5] = 1
            mask_d[mask_d < 0.5] = 0
            if (((ori_outs[i + 1] * mask_d - adv_outs[i + 1] * mask_d) ** 2).sum() / (mask_d.sum() * 3)) >= 0.05:
                n_dist += 1

            dis = criterion(ori_outs[i + 1], adv_outs[i + 1])
            if dis >= 0.05:
                succ_num = succ_num + 1
            total_num = total_num + 1

    len_crg = len(adv_outs) - 1 if len(adv_outs) > 1 else 1

    # MASR
    if total_num > 0:
        asr = succ_num / total_num
        mask_asr = n_dist / total_num
    else:
        asr = 0.0
        mask_asr = 0.0

    if len_crg > 0:
        psnr_value /= len_crg
        ssim_value /= len_crg
        lpips_alex /= len_crg
        lpips_vgg /= len_crg
        l1_error /= len_crg
        l2_error /= len_crg
        vgg_sum /= len_crg

    log_message = "\nThe Average Metrics between clean and defensed images:\n"
    log_message += opts.face_pretrained_weights
    log_message += opts.hair_pretrained_weights
    log_message += '\n'
    log_message += f'psnr_adv: {psnr_adv:.3f}'
    log_message += f', ssim_adv: {ssim_adv:.4f}'
    log_message += f', lpips(alex)_adv: {lpips_alexs_adv:.5f}'
    log_message += f', lpips(vgg)_adv: {lpips_vggs_adv:.5f}'
    log_message += "\nThe Average Metrics between clean outputs and defensed outputs:\n"
    log_message += f'psnr: {psnr_value:.3f}'
    log_message += f', ssim: {ssim_value:.4f}'
    log_message += f', lpips(alex): {lpips_alex:.5f}'
    log_message += f', lpips(vgg): {lpips_vgg:.5f}'
    log_message += f', l1_error: {l1_error:.5f}'
    log_message += f', l2_error: {l2_error:.5f}'
    log_message += f', vgg: {vgg_sum:.5f}'
    log_message += f', asr: {asr:.5f}'
    log_message += f', mask_asr: {mask_asr:.5f}'

    print(log_message)
    et = str(datetime.timedelta(seconds=execution_time))[:-1]
    log_file_path = os.path.join(save_path, "test_results.txt")

    with open(log_file_path, 'a') as f:
        f.write("\n\n======= Test at: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " =======\n")
        f.write(log_message)
        f.write("\nTime use for five attributes: " + et + "\n")

    print(f"The inference result has been saved to: {log_file_path}")


