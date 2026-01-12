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
from tools.phase_mix import PhaseMix
import cv2
from logger import setup_logger
import torch.nn.functional as F
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


if __name__ == '__main__':
    config_parser = ArgumentParser()
    config_parser.add_argument('--flag', default=False, type=bool, help='is save results')
    config_parser.add_argument('--model_choice', default='hisd', type=str, help='compute metrics choice')
    config_parser.add_argument('--StarG_path', default="./checkpoints/stargan/200000-G.ckpt", type=str,
                               help='Stargan Weight Path')
    config_parser.add_argument('--AttentionG_path', default="./checkpoints/attentiongan/200000-G.ckpt", type=str,
                               help='AttentionGAN Weight Path')
    config_parser.add_argument('--test_path', default='./test_results', type=str, help='Test Result Path')
    config_parser.add_argument('--test_img',
                               default='./test_samples/test_face.png',
                               type=str, help='Test Image Name')
    config_parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    config_parser.add_argument('--eposilon', default=0.02, type=float, help='Perturbation Scale')
    config_parser.add_argument('--img_size', default=256, type=float, help='Image Size')
    config_parser.add_argument('--dataset_path', default='../../dataset/CelebAMask-HQ/CelebA-HQ-img/',
                               type=str, help='Dataset Path')
    config_parser.add_argument('--attribute_txt_path',
                               default='../../dataset/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt',
                               type=str, help='Attribute Txt Path')
    config_parser.add_argument('--selected_attrs', default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
                               type=list, help='Attribute Selection')
    config_parser.add_argument('--face_pretrained_weights',
                               default='./checkpoints/cross_model_adv/face_perturb_mask_out.pth',
                               type=str,
                               help='Path to pretrained weights for continuing training')
    config_parser.add_argument('--hair_pretrained_weights',
                               default='./checkpoints/cross_model_adv/hair_perturb_mask_out.pth',
                               type=str,
                               help='Path to pretrained weights for continuing training')
    opts = config_parser.parse_args()
    print(opts)
    phase_swap = PhaseMix(swap_strength=1.0).to(device)

    save_path = opts.test_path + '/' + opts.model_choice +'-dual-test-result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    y_mask = Y_mask(opts)

    lpips_model, lpips_model2 = prepare_lpips()
    vgg = Vgg16().to(device)
    vgg.eval()
    criterion = torch.nn.MSELoss()

    Models = Ensemble(opts)
    GAN = Black_models(opts)
    Bise_net = load_model(model_name="resnet34", num_classes=19, weight_path="./checkpoints/resnet34.pt", device=device)
    Bise_net.eval()

    # PG load
    face_PG = face_G(input_nc, output_nc, ngf, 'unet_64', 'instance', not no_dropout, init_type, init_gain).to(
        device)

    hair_PG = hair_G(input_nc, output_nc, ngf, 'unet_64', 'instance', not no_dropout, init_type, init_gain).to(device)
    if opts.face_pretrained_weights:
        face_success = load_face_pretrained_weights(face_PG, opts.face_pretrained_weights)
        print("Face Training from pretrained weights")
        hair_success = load_hair_pretrained_weights(hair_PG, opts.hair_pretrained_weights)
        # hair_success = False
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
        shuffle=False
    )
    print('The number of val Iterations = %d' % len(test_dataloader))
    
    ref_img, att_a, c_org, filename = preprocess_single_image_fgan(
        './test_samples/res_1.jpg', opts.img_size, device)
    psnr_value, ssim_value, lpips_alexs, lpips_vggs = 0.0, 0.0, 0.0, 0.0
    succ_num, total_num, n_dist = 0.0, 0.0, 0.0
    l1_error, l2_error = 0.0, 0.0
    vgg_sum = 0.0
    psnr_adv, ssim_adv, lpips_alexs_adv, lpips_vggs_adv = 0.0, 0.0, 0.0, 0.0
    print("Start testing...")

    for idx, (img_a, att_a, c_org, filename) in enumerate(tqdm(test_dataloader, desc='')):
        with torch.no_grad():
            x_real = img_a.to(device).clone().detach()
            c_trg_list = Models.create_labels(c_org)
            b_trg_list = Models.create_labels_attgan(att_a) # attgan

            x_ori = tensor2numpy(x_real)
            if opts.model_choice == 'fgan':
                ori_outs = Models.fgan_outs(x_real, c_trg_list)
            elif opts.model_choice == 'attgan':
                ori_outs = Models.attgan_outs(x_real, b_trg_list)
            elif opts.model_choice == 'hisd':
                ori_outs = Models.hisd_outs(x_real)
            else:
                ori_outs = GAN.model_out(x_real, c_trg_list)

            x_ori = tensor2numpy(x_real)
            x_ref = tensor2numpy(ref_img)

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

            start_time = time.time()
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

            results = []
            results.append(torch.cat(ori_outs, dim=0))
            results.append(torch.cat(adv_outs, dim=0))
            save_grid_img(results, save_path, idx)

            psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics(x_real, adv_A, lpips_model, lpips_model2)
            psnr_adv += psnr_temp
            ssim_adv += ssim_temp
            lpips_alexs_adv += lpips_alex
            lpips_vggs_adv += lpips_vgg

            for i in range(len(adv_outs)-1):
                psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics(ori_outs[i+1], adv_outs[i+1], lpips_model, lpips_model2)
                l1_error += torch.nn.functional.l1_loss(ori_outs[i+1], adv_outs[i+1]).item()
                l2_error += torch.nn.functional.mse_loss(ori_outs[i+1], adv_outs[i+1]).item()
                loss_style, loss_content = get_perceptual_loss(vgg, ori_outs[i+1], adv_outs[i+1])
                vgg_per = (STYLE_WEIGHT * loss_style + CONTENT_WEIGHT * loss_content).item()
                vgg_sum += vgg_per
                psnr_value += psnr_temp
                ssim_value += ssim_temp
                lpips_alexs += lpips_alex
                lpips_vggs += lpips_vgg

                # ASR
                mask_d = abs(ori_outs[i+1] - x_real)
                mask_d = mask_d[0, 0, :, :] + mask_d[0, 1, :, :] + mask_d[0, 2, :, :]
                mask_d[mask_d > 0.5] = 1
                mask_d[mask_d < 0.5] = 0
                if (((ori_outs[i+1] * mask_d - adv_outs[i+1] * mask_d) ** 2).sum() / (mask_d.sum() * 3)) >= 0.05:
                    n_dist += 1

                dis = criterion(ori_outs[i+1], adv_outs[i+1])
                if dis >= 0.05:
                    succ_num = succ_num + 1
                total_num = total_num + 1

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"代码运行时间: {execution_time:.2f} 秒")
    len_crg = 5
    psnr_adv /= (idx + 1)
    ssim_adv /= (idx + 1)
    lpips_alexs_adv /= (idx + 1)
    lpips_vggs_adv /= (idx + 1)

    asr = succ_num / total_num
    mask_asr = n_dist / total_num
    psnr_value /= (idx + 1) * len_crg
    ssim_value /= (idx + 1) * len_crg
    lpips_alexs /= (idx + 1) * len_crg
    lpips_vggs /= (idx + 1) * len_crg
    l1_error /= (idx + 1) * len_crg
    l2_error /= (idx + 1) * len_crg
    vgg_sum /= (idx + 1) * len_crg
    log_message = "\nThe Average Metrics between clean and defensed images:\n"
    log_message += f'psnr_adv: {psnr_adv:.3f}'
    log_message += f', ssim_adv: {ssim_adv:.4f}'
    log_message += f', lpips(alex)_adv: {lpips_alexs_adv:.5f}'
    log_message += f', lpips(vgg)_adv: {lpips_vggs_adv:.5f}'
    log_message += "\nThe Average Metrics between clean outputs and defensed outputs:\n"
    log_message += f'psnr: {psnr_value:.3f}'
    log_message += f', ssim: {ssim_value:.4f}'
    log_message += f', lpips(alex): {lpips_alexs:.5f}'
    log_message += f', lpips(vgg): {lpips_vggs:.5f}'
    log_message += f', l1_error: {l1_error:.5f}'
    log_message += f', l2_error: {l2_error:.5f}'
    log_message += f', vgg: {vgg_sum:.5f}'
    log_message += f', asr: {asr:.5f}'
    log_message += f', mask_asr: {mask_asr:.5f}'

    print(log_message)
    et = str(datetime.timedelta(seconds=execution_time))[:-1]
    # print("time use:" + str(et))

    log_file_path = os.path.join(save_path, "test_results.txt")
    with open(log_file_path, 'a') as f:
        f.write("\n\n======= Test at: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " =======\n")
        f.write(log_message)
        f.write("\nTime use for five attributes: " + et + "\n")

    print(f"The test results have been saved to: {log_file_path}")

