from tqdm import tqdm
from torch.optim import SGD, Adam, AdamW
from argparse import ArgumentParser
from config import *
from data import CelebA
import torch.utils.data as data
from networks.Ensemble_models import Ensemble
from networks.FGAN import Discriminator as D
from logger import setup_logger
from tools.color_space import rgb2ycbcr_np, ycbcr_to_tensor, ycbcr_to_rgb
from tools.metrics_compute import compute_metrics, compute_psnr, prepare_lpips
from tools.tool import *
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from networks.bisenet import BiSeNet
from scipy import ndimage
from networks.PG_face_dualmask import define_G as face_G
from networks.PG_hair_dualmask import define_G as hair_G
# from networks.PG_face_model import define_G as face_G
# from networks.PG_hair_model import define_G as hair_G


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


if __name__ == '__main__':
    config_parser = ArgumentParser()
    config_parser.add_argument('--iter_num', default=100, type=float, help='Training Iterations Numer')
    config_parser.add_argument('--perturb_wt', default=10, type=float, help='Perturbation Weight')
    config_parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    config_parser.add_argument('--loss_type', default='l2', type=str, help='Loss Type')
    config_parser.add_argument('--lr', default=0.0001, type=float, help='Learning Rate')
    config_parser.add_argument('--eposilon', default=0.02, type=float, help='Perturbation Scale')
    config_parser.add_argument('--img_size', default=256, type=float, help='Image Size')
    config_parser.add_argument('--show_iter', default=1, type=int, help='Show Results After Every Iters')

    config_parser.add_argument('--dataset_path', default='D:/dataset/CelebAMask-HQ/CelebA-HQ-img/', type=str, help='Dataset Path')
    config_parser.add_argument('--attribute_txt_path', default='D:/dataset/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt', type=str, help='Attribute Txt Path')
    config_parser.add_argument('--selected_attrs', default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"], type=list, help='Attribute Selection')
    config_parser.add_argument('--face_pretrained_weights', default='./train_results/weight/face_perturb.pth',
                               type=str,
                               help='Path to pretrained weights for continuing training')
    config_parser.add_argument('--hair_pretrained_weights',
                               default='./train_results/weight/hair_perturb.pth',
                               type=str,
                               help='Path to pretrained weights for continuing training')

    opts = config_parser.parse_args()

    print(opts)
    perturb_wt = opts.perturb_wt
    batch_size = opts.batch_size
    loss_type = opts.loss_type
    lr = opts.lr
    eposilon = opts.eposilon
    img_size = opts.img_size
    show_iter = opts.show_iter

    logger_save_path = r"D:\python_project\AdvDeepfake"
    # path_isexists()

    logger = setup_logger(logger_save_path, 'result.log', 'train_logger')
    logger.info(f'Loading model.')

    ######  data set prepare  ######
    train_dataset = CelebA(opts.dataset_path, opts.attribute_txt_path,
                           img_size, 'train', attrs,
                           opts.selected_attrs)
    val_dataset = CelebA(opts.dataset_path, opts.attribute_txt_path,
                         img_size, 'val', attrs,
                         opts.selected_attrs)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=1, num_workers=4,
        shuffle=True
    )
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=1, num_workers=4,
        shuffle=False
    )

    # net prepare
    lpips_model, lpips_model2 = prepare_lpips()
    Models = Ensemble(opts)
    Bise_net = load_model(model_name="resnet34", num_classes=19, weight_path="./checkpoints/resnet34.pt", device=device)
    Bise_net.eval()

    print('The number of train Iterations = %d' % len(train_dataloader))
    print('The number of val Iterations = %d' % len(val_dataloader))

    optim_list = [(SGD, {'lr': lr}), (AdamW, {'lr': lr*4})]
    my_optim = optim_list[1]
    optimizer = my_optim[0]
    optim_args = my_optim[1]
    y_mask = Y_mask(opts)

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


    netDisc = D().to(device)

    face_model_optim = optimizer(params=list(face_PG.parameters()), **optim_args)
    hair_model_optim = optimizer(params=list(hair_PG.parameters()), **optim_args)
    disc_optim = optimizer(netDisc.parameters(), lr=lr*2)

    best_loss = float("inf")

    delta = 0.05
    balance_factor = 0.1

    max_psnr_fgan, max_psnr_attgan, max_psnr_hisd = 50.0, 50.0, 50.0

    min_weight = 0.15  
    reset_interval = 20  
    boost_interval = 15 

    weights_softmax = torch.tensor([0.3, 0.4, 0.3], requires_grad=True)

    fgan_history = []
    attgan_history = []
    hisd_history = []

    performance_m1 = torch.tensor(15.0)
    performance_m2 = torch.tensor(18.0)
    performance_m3 = torch.tensor(15.0)
    threshold_m1 = torch.tensor(16.0)
    threshold_m2 = torch.tensor(20.0)
    threshold_m3 = torch.tensor(16.0)

    for epoch in range(1, opts.iter_num):
        train_imgs_fgan, val_imgs_fgan = [], []
        train_imgs_attgan, val_imgs_attgan = [], []
        train_imgs_hisd, val_imgs_hisd = [], []
        train_current_loss = {'D/loss_real': 0., 'D/loss_fake': 0., 'D/loss_gp': 0.,'G/loss_fake': 0., 'G/loss_attack': 0.}
        psnr_value, ssim_value, lpips_alexs, lpips_vggs = 0.0, 0.0, 0.0, 0.0
        psnr_fgan, psnr_attgan, psnr_hisd = 0.0, 0.0, 0.0
        for idx, (img_a, att_a, c_org, filename) in enumerate(tqdm(train_dataloader, desc='')):
            c_trg_list = Models.create_labels(c_org)
            b_trg_list = Models.create_labels_attgan(att_a) 

            ref_img = Image.open('./test_samples/res_1.jpg').convert(
                'RGB').resize((256, 256))

            ref_img = np.array(ref_img) / 255.0
            x_real = img_a.to(device).clone().detach()
            x_ori = tensor2numpy(x_real)
            x_r_ycbcr = rgb2ycbcr_np(ref_img)
            x_r = ycbcr_to_tensor(x_r_ycbcr).cuda()

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
            # face_classes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
            face_classes = [1]
            hair_classes = [17, 18]
            # face_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

            for class_id in face_classes:
                face_predicted_mask_binary[face_predicted_mask == class_id] = 1

            for class_id in hair_classes:
                hair_predicted_mask_binary[hair_predicted_mask == class_id] = 1

            face_predicted_mask_binary = ndimage.binary_closing(face_predicted_mask_binary, structure=np.ones((3, 3))).astype(
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
            x_y = ycbcr_to_tensor(x_ycbcr).to(device)
            x_r_ycbcr = rgb2ycbcr_np(ref_img)
            x_r = ycbcr_to_tensor(x_r_ycbcr).cuda()

            if epoch % show_iter == 0 and idx < 10:
                ori_outs_fgan, ori_outs_attgan, ori_outs_hisd = Models.ensemble_models_out(x_real, c_trg_list, b_trg_list)

            face_adv_A = face_PG(x_y, x_r, face_predicted_mask)
            hair_adv_A = hair_PG(x_y, x_r, hair_predicted_mask)

            # face_adv_A = face_PG(x_y, x_r, face_predicted_mask)
            # hair_adv_A = hair_PG(x_y, hair_predicted_mask)

            face_adv_noise = face_adv_A
            face_adv_noise = face_adv_noise * y_mask
            face_adv_noise = torch.clamp(face_adv_noise, -eposilon, eposilon)

            hair_adv_noise = hair_adv_A
            hair_adv_noise = hair_adv_noise * y_mask
            hair_adv_noise = torch.clamp(hair_adv_noise, -eposilon, eposilon)

            x_L_adv = x_y + face_adv_noise + hair_adv_noise
            x_adv = ycbcr_to_rgb(x_L_adv)
            adv_A = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_adv.contiguous())

            # optimize D
            # compute loss with real images.
            out_src, out_cls = netDisc(x_real)
            d_loss_real = - torch.mean(out_src)

            # compute loss with fake images.
            out_src, _ = netDisc(adv_A.detach_())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
            x_hat = (alpha * x_real.data + (1 - alpha) * adv_A.data).requires_grad_(True)
            out_src, _ = netDisc(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat)

            d_loss = d_loss_real + d_loss_fake + 10 * d_loss_gp

            disc_optim.zero_grad()
            face_model_optim.zero_grad()
            hair_model_optim.zero_grad()

            d_loss.backward(retain_graph=True)
            disc_optim.step()

            train_current_loss['D/loss_real'] += d_loss_real.item()
            train_current_loss['D/loss_fake'] += d_loss_fake.item()
            train_current_loss['D/loss_gp'] += d_loss_gp.item()

            # optimize G

            face_adv_A = face_PG(x_y, x_r, face_predicted_mask)
            hair_adv_A = hair_PG(x_y, x_r, hair_predicted_mask)

            # face_adv_A = face_PG(x_y, x_r, face_predicted_mask)
            # hair_adv_A = hair_PG(x_y, hair_predicted_mask)

            face_adv_noise = face_adv_A
            face_adv_noise = face_adv_noise * y_mask
            face_adv_noise = torch.clamp(face_adv_noise, -eposilon, eposilon)

            hair_adv_noise = hair_adv_A
            hair_adv_noise = hair_adv_noise * y_mask
            hair_adv_noise = torch.clamp(hair_adv_noise, -eposilon, eposilon)


            x_L_adv = x_y + face_adv_noise + hair_adv_noise
            x_adv = ycbcr_to_rgb(x_L_adv)
 
            adv_A = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_adv.contiguous())

            out_src, _ = netDisc(adv_A)
            g_loss_fake = - torch.mean(out_src)

            loss_adv = Models.ensemble_compute_loss(adv_A, x_real, c_trg_list, b_trg_list, weights_softmax)

            g_loss = g_loss_fake + 10 * loss_adv
            disc_optim.zero_grad()
            face_model_optim.zero_grad()
            hair_model_optim.zero_grad()
            g_loss.backward(retain_graph=True)
            face_model_optim.step()
            hair_model_optim.step()

            if idx < 10 and epoch % show_iter == 0:
                adv_outs_fgan, adv_outs_attgan, adv_outs_hisd = Models.ensemble_models_out(adv_A, c_trg_list, b_trg_list)

            train_current_loss['G/loss_attack'] += loss_adv.item()
            train_current_loss['G/loss_fake'] += g_loss_fake.item()

            psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics(x_real, adv_A, lpips_model, lpips_model2)
            psnr_value += psnr_temp
            ssim_value += ssim_temp
            lpips_alexs += lpips_alex
            lpips_vggs += lpips_vgg

            for i in range(len(opts.selected_attrs)):
                psnr_fgan += compute_psnr(ori_outs_fgan[i+1], adv_outs_fgan[i+1])
                psnr_attgan += compute_psnr(ori_outs_attgan[i + 1], adv_outs_attgan[i + 1])
                psnr_hisd += compute_psnr(ori_outs_hisd[i + 1], adv_outs_hisd[i + 1])

            if epoch % show_iter == 0 and idx < 10:
                train_imgs_fgan.append(torch.cat(ori_outs_fgan, dim=0))
                train_imgs_fgan.append(torch.cat(adv_outs_fgan, dim=0))
                train_imgs_attgan.append(torch.cat(ori_outs_attgan, dim=0))
                train_imgs_attgan.append(torch.cat(adv_outs_attgan, dim=0))
                train_imgs_hisd.append(torch.cat(ori_outs_hisd, dim=0))
                train_imgs_hisd.append(torch.cat(adv_outs_hisd, dim=0))

        if train_imgs_fgan and train_imgs_attgan and train_imgs_hisd:
            save_grid_img(train_imgs_fgan, save_train_path_fgan, epoch)
            save_grid_img(train_imgs_attgan, save_train_path_attgan, epoch)
            save_grid_img(train_imgs_hisd, save_train_path_hisd, epoch)


        psnr_value /= len(train_dataloader)
        ssim_value /= len(train_dataloader)
        lpips_alexs /= len(train_dataloader)
        lpips_vggs /= len(train_dataloader)

        psnr_fgan /= (len(train_dataloader) * 5)
        psnr_attgan /= (len(train_dataloader) * 5)
        psnr_hisd /= (len(train_dataloader) * 5)

        if epoch % 1 == 0:
            if psnr_fgan < performance_m1:
                performance_m1 = psnr_fgan
                weights_softmax.data[0] += delta  
                weights_softmax.data[1] -= delta * balance_factor
                weights_softmax.data[2] -= delta * balance_factor
                print("performance_m1: ", performance_m1)
                print("weights_softmax_update: ", weights_softmax)
            if psnr_attgan < performance_m2:
                performance_m2 = psnr_attgan
                weights_softmax.data[1] += delta
                weights_softmax.data[0] -= delta * balance_factor
                weights_softmax.data[2] -= delta * balance_factor
                print("performance_m2: ", performance_m2)
                print("weights_softmax_update: ", weights_softmax)
            if psnr_hisd < performance_m3:
                performance_m3 = psnr_hisd
                weights_softmax.data[2] += delta
                weights_softmax.data[0] -= delta * balance_factor
                weights_softmax.data[1] -= delta * balance_factor
                print("performance_m3: ", performance_m3)
                print("weights_softmax_update: ", weights_softmax)
            with torch.no_grad():
                total_weight = torch.sum(weights_softmax)
                weights_softmax.data = weights_softmax.data / total_weight
        if epoch % 5 == 0:
            if psnr_fgan > threshold_m1:
                weights_softmax.data[0] += delta
                weights_softmax.data[1] -= delta * balance_factor
                weights_softmax.data[2] -= delta * balance_factor
                print("performance_m1: ", performance_m1)
                print("weights_softmax_update: ", weights_softmax)
            if psnr_attgan > threshold_m2:
                weights_softmax.data[1] += delta
                weights_softmax.data[0] -= delta * balance_factor
                weights_softmax.data[2] -= delta * balance_factor
                print("performance_m2: ", performance_m2)
                print("weights_softmax_update: ", weights_softmax)
            if psnr_hisd > threshold_m3:
                weights_softmax.data[2] += delta
                weights_softmax.data[0] -= delta * balance_factor
                weights_softmax.data[1] -= delta * balance_factor
                print("performance_m3: ", performance_m3)
                print("weights_softmax_update: ", weights_softmax)
            with torch.no_grad():
                total_weight = torch.sum(weights_softmax)
                weights_softmax.data = weights_softmax.data / total_weight

            with torch.no_grad():
                total_weight = torch.sum(weights_softmax)
                weights_softmax.data = weights_softmax.data / total_weight

            print(f"Epoch {epoch}: Weights updated - FGAN: {weights_softmax[0]:.3f}, "
                  f"AttGAN: {weights_softmax[1]:.3f}, HISD: {weights_softmax[2]:.3f}")
            print(f"Current PSNR - FGAN: {psnr_fgan:.2f} (target: {performance_m1:.2f}), "
                  f"AttGAN: {psnr_attgan:.2f} (target: {performance_m2:.2f}), "
                  f"HISD: {psnr_hisd:.2f} (target: {performance_m3:.2f})")


        if epoch > 0 and epoch % reset_interval == 0:
            mean_weight = torch.mean(weights_softmax)
            weights_softmax = (weights_softmax + mean_weight) / 2
            with torch.no_grad():
                total_weight = torch.sum(weights_softmax)
                weights_softmax.data = weights_softmax.data / total_weight

            print(f"Epoch {epoch}: Weights RESET - FGAN: {weights_softmax[0]:.3f}, "
                  f"AttGAN: {weights_softmax[1]:.3f}, HISD: {weights_softmax[2]:.3f}")


        train_current_loss['D/loss_real'] /= len(train_dataloader)
        train_current_loss['D/loss_fake'] /= len(train_dataloader)
        train_current_loss['D/loss_gp'] /= len(train_dataloader)
        train_current_loss['G/loss_fake'] /= len(train_dataloader)
        train_current_loss['G/loss_attack'] /= len(train_dataloader)


        val_current_loss = {'G/loss_attack': 0.}
        val_psnr_value, val_ssim_value, val_lpips_alexs, val_lpips_vggs = 0.0, 0.0, 0.0, 0.0
        for idx, (img_a, att_a, c_org, filename) in enumerate(tqdm(val_dataloader, desc='')):
            with torch.no_grad():
                x_real = img_a.to(device).clone().detach()
                c_trg_list = Models.create_labels(c_org)
                b_trg_list = Models.create_labels_attgan(att_a)

                ref_img = Image.open('./test_samples/res_1.jpg').convert(
                    'RGB').resize((256, 256))
                ref_img = np.array(ref_img) / 255.0
                x_ori = tensor2numpy(x_real)

                x_ycbcr = rgb2ycbcr_np(x_ori)
                x_y = ycbcr_to_tensor(x_ycbcr).cuda()
                x_r_ycbcr = rgb2ycbcr_np(ref_img)
                x_r = ycbcr_to_tensor(x_r_ycbcr).cuda()

                if epoch % show_iter == 0 and idx < 10:
                    ori_outs_fgan, ori_outs_attgan, ori_outs_hisd = Models.ensemble_models_out(x_real, c_trg_list, b_trg_list)

                face_adv_A = face_PG(x_y, x_r, face_predicted_mask)
                hair_adv_A = hair_PG(x_y, x_r, hair_predicted_mask)

                # face_adv_A = face_PG(x_y, x_r, face_predicted_mask)
                # hair_adv_A = hair_PG(x_y, hair_predicted_mask)

                face_adv_noise = face_adv_A
                face_adv_noise = face_adv_noise * y_mask
                face_adv_noise = torch.clamp(face_adv_noise, -eposilon, eposilon)

                hair_adv_noise = hair_adv_A
                hair_adv_noise = hair_adv_noise * y_mask
                hair_adv_noise = torch.clamp(hair_adv_noise, -eposilon, eposilon)

                x_L_adv = x_y + face_adv_noise + hair_adv_noise
                x_adv = ycbcr_to_rgb(x_L_adv)
                adv_A = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_adv.contiguous())

                #### compute loss ####
                loss = Models.ensemble_compute_loss(adv_A, x_real, c_trg_list, b_trg_list, weights_softmax)

                if idx < 10 and epoch % show_iter == 0:
                    adv_outs_fgan, adv_outs_attgan, adv_outs_hisd = Models.ensemble_models_out(adv_A, c_trg_list, b_trg_list)

                if epoch % show_iter == 0 and idx < 10:
                    val_imgs_fgan.append(torch.cat(ori_outs_fgan, dim=0))
                    val_imgs_fgan.append(torch.cat(adv_outs_fgan, dim=0))
                    val_imgs_attgan.append(torch.cat(ori_outs_attgan, dim=0))
                    val_imgs_attgan.append(torch.cat(adv_outs_attgan, dim=0))
                    val_imgs_hisd.append(torch.cat(ori_outs_hisd, dim=0))
                    val_imgs_hisd.append(torch.cat(adv_outs_hisd, dim=0))

                val_current_loss['G/loss_attack'] += loss.item()

                psnr_temp, ssim_temp, lpips_alex, lpips_vgg = compute_metrics(x_real, adv_A, lpips_model, lpips_model2)
                val_psnr_value += psnr_temp
                val_ssim_value += ssim_temp
                val_lpips_alexs += lpips_alex
                val_lpips_vggs += lpips_vgg

        if val_imgs_fgan and val_imgs_attgan and val_imgs_hisd:
            save_grid_img(val_imgs_fgan, save_val_path_fgan, epoch)
            save_grid_img(val_imgs_attgan, save_val_path_attgan, epoch)
            save_grid_img(val_imgs_hisd, save_val_path_hisd, epoch)

        val_current_loss['G/loss_attack'] /= len(val_dataloader)

        val_psnr_value /= len(val_dataloader)
        val_ssim_value /= len(val_dataloader)
        val_lpips_alexs /= len(val_dataloader)
        val_lpips_vggs /= len(val_dataloader)

        log_message = ''
        for tag, value in train_current_loss.items():
            log_message += ", {}: {:.3f}".format(tag, value)

        log_message += f', psnr: {psnr_value:.3f}'
        log_message += f', ssim: {ssim_value:.4f}'
        log_message += f', lpips: {(lpips_alexs + lpips_vggs) / 2:.5f}'
        log_message += f", val_loss: {val_current_loss['G/loss_attack']:.3f}"
        log_message += f', val_psnr: {val_psnr_value:.3f}'
        log_message += f', val_ssim: {val_ssim_value:.4f}'
        log_message += f', val_lpips: {(val_lpips_alexs + val_lpips_vggs) / 2:.5f}'
        log_message += f', psnr_fgan: {psnr_fgan:.4f}'
        log_message += f', psnr_attgan: {psnr_attgan:.4f}'
        log_message += f', psnr_hisd: {psnr_hisd:.4f}'

        print(log_message)
        if logger:
            logger.debug(f'Step: {epoch:05d}, '
                          f'lr: {lr:.2e}, '
                         f'e: {eposilon:.2e},'
                          f'{log_message}')

        if psnr_fgan < (max_psnr_fgan + 0.5) or psnr_attgan < (max_psnr_attgan + 0.5) or psnr_hisd < (max_psnr_hisd + 0.5):
            max_psnr_fgan = psnr_fgan
            max_psnr_attgan = psnr_attgan
            max_psnr_hisd = psnr_hisd

            if not os.path.exists(weight_save_path):
                os.makedirs(weight_save_path, exist_ok=True)
            face_save_filename_model = 'face_perturb_%s.pth' % (epoch)
            face_save_path = os.path.join(weight_save_path, face_save_filename_model)
            hair_save_filename_model = 'hair_perturb_%s.pth' % (epoch)
            hair_save_path = os.path.join(weight_save_path, hair_save_filename_model)
            print('Updating the noise model')
            torch.save({"face_protection_net": face_PG.state_dict()}, face_save_path)
            torch.save({"hair_protection_net": hair_PG.state_dict()}, hair_save_path)
            best_loss = val_current_loss['G/loss_attack']

        print(
            'Epoch {} / {} \t Train Loss: {:.3f} \t Val Loss: {:.3f}'.format(epoch, opts.iter_num,
                                                                            train_current_loss['G/loss_attack'],
                                                                            val_current_loss['G/loss_attack']))
        face_save_filename_model = 'face_perturb_latest.pth'
        face_save_path = os.path.join(weight_save_path, face_save_filename_model)
        hair_save_filename_model = 'hair_perturb_latest.pth'
        hair_save_path = os.path.join(weight_save_path, hair_save_filename_model)
        torch.save({"face_protection_net": face_PG.state_dict()}, face_save_path)
        torch.save({"hair_protection_net": hair_PG.state_dict()}, hair_save_path)

        if epoch > 1 and (psnr_fgan > 40 or psnr_hisd > 50 or psnr_attgan > 40):
            print('trash_training...')
            break
