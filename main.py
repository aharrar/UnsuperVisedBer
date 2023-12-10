import copy
import os
import random
from statistics import median, stdev, mean

import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from expirments.load import make_adata_from_batches
from pre_procesing.utils import load_and_pre_process_data
from unsupervised_dataset import UnsupervisedDataset
from unsupervised.autoencoder import Encoder
from unsupervised.ind_discrimnator import IndDiscriminator
from unsupervised.ber_network import DecoderBer, Net
from unsupervised.utils import indep_loss, eval_mmd, gradient_penalty, lr_scheduler, get_cdca_term


def get_data_calibrated(src_data, target_data, encoder, decoder):
    encoder.eval()
    decoder.eval()
    y_src = torch.zeros(src_data.shape[0])
    y_target = torch.ones(target_data.shape[0])
    code_src = encoder(src_data)
    code_target = encoder(target_data)
    recon_src, _ = decoder(code_src, y_src)
    _, recon_target = decoder(code_target, y_target)
    _, calibrated_src = decoder(code_src, 1 - y_src)
    calibrated_target, _ = decoder(code_target, 1 - y_target)

    return code_src, code_target, recon_src, recon_target, calibrated_src, calibrated_target


def validate(src, target, encoder, decoder):
    encoder.eval()
    decoder.eval()
    code_src, code_target, recon_src, recon_target, calibrated_src, calibrated_target = get_data_calibrated(src,
                                                                                                            target,
                                                                                                            encoder,
                                                                                                            decoder)
    mmd = min(eval_mmd(calibrated_target, src), eval_mmd(calibrated_src, target))
    return mmd


def train2(src, target, data_loader, net, ind_discriminator, ae_optim, ind_disc_optim,
           config
           ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    recon_criterion = nn.MSELoss()
    recon_losses = []
    independence_losses = []
    coef = [1, config['coef_1'], 1]
    mmd_list = []

    l1_list = []
    l2_list = []
    best_mmd = torch.tensor(10000)
    smoothed_disc_loss = 0

    for epoch in tqdm(range(config["epochs"])):

        net.encoder.train(True)
        net.decoder.train(True)
        ind_discriminator.train(True)
        average_discriminator_loss = 0
        average_ae_loss = 0
        counter = 0
        for step, (batch_x, batch_y, batch_id) in enumerate(data_loader):
            counter += 1
            batch_x = batch_x.float().to(device=device).detach().clone()
            batch_id = batch_id.float().to(device=device).detach().clone()
            mask0 = batch_id == 0
            mask1 = batch_id == 1

            ind_discriminator.zero_grad()
            code_real = net.encoder(batch_x)
            logist = ind_discriminator(code_real.detach())
            independence = coef[0] * indep_loss(logist, batch_id, should_be_dependent=True)
            independence_loss_value = independence.item()
            independence.backward()
            ind_disc_optim.step()
            average_discriminator_loss += abs(independence_loss_value)
            ############################# train autoencoder ###############
            if epoch % 1 == 0:
                net.encoder.zero_grad()
                net.decoder.zero_grad()

                code_real = net.encoder(batch_x).float()

                recon_batch_a, _ = net.decoder(code_real[mask0], batch_id[mask0])
                _, recon_batch_b = net.decoder(code_real[mask1], batch_id[mask1])

                recon_loss_a = recon_criterion(recon_batch_a, batch_x[mask0])
                recon_loss_b = recon_criterion(recon_batch_b, batch_x[mask1])

                logist = ind_discriminator(code_real.detach())
                independence = indep_loss(logist, batch_id, should_be_dependent=False)
                ae_loss = coef[1] * (recon_loss_a / recon_loss_a.item() + recon_loss_b / recon_loss_b.item()) + coef[
                    2] * independence / independence.item()
                ae_loss_value = recon_loss_a.item() + recon_loss_b.item()
                l1_list.append(recon_loss_a + recon_loss_b)
                l2_list.append(independence)

                ae_loss.backward()
                ae_optim.step()
                average_ae_loss += ae_loss_value

        if epoch % 3 == 0 and epoch > 0:
            mmd = validate(src, target, net.encoder, net.decoder)
            net.encoder.eval()
            mmd_code = eval_mmd(net.encoder(src), net.encoder(target))
            net.encoder.train()
            mmd_list.append(mmd_code.detach().numpy())
            if mmd < best_mmd:
                best_mmd = mmd
                print(best_mmd)
                net.save(config["save_weights"])

            # Save the best model

        if len(mmd_list) > 2:
            coef[0] = 1
            coef[1] = config["coef_1"] / np.std(np.array(mmd_list)) / np.mean(np.array(mmd_list))
            coef[2] = 1  # np.std(np.array(mmd_list)) / np.mean(np.array(mmd_list))
            # else:
            #     coef[0] = 1
            #     coef[1] = 30  # / np.std(np.array(mmd_list)) / np.mean(np.array(mmd_list))
            #     coef[2] = 1 / np.std(np.array(mmd_list)) / np.mean(np.array(mmd_list)) + random.randint(0, 30)

        smoothed_disc_loss = 0.95 * smoothed_disc_loss + 0.05 * independence.item()
        current_lr = ind_disc_optim.param_groups[0]['lr']

        for param_group in ind_disc_optim.param_groups:
            param_group['lr'] = current_lr * lr_scheduler(smoothed_disc_loss, 0.63)

        recon_losses.append(average_ae_loss / counter)
        independence_losses.append(average_discriminator_loss / counter)

        recon_losses.append(average_ae_loss / counter)
        independence_losses.append(average_discriminator_loss / counter)

    print(f"----------{best_mmd}----------")

    return net, recon_losses, independence_losses


def ber_for_notebook(config, adata1, adata2, model_shrinking, embed='', load_pre_weights=''):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    min_max_scaler_src = MinMaxScaler((-0.7, 0.7))
    min_max_scaler_target = MinMaxScaler((-0.7, 0.7))
    if embed == '':
        src_data_without_labels = adata1.X
        target_data_without_labels = adata2.X
    else:
        src_data_without_labels = adata1.obsm[embed]
        target_data_without_labels = adata2.obsm[embed]

    source_labels = torch.tensor(adata1.obs['celltype'])
    target_labels = torch.tensor(adata2.obs['celltype'])

    src_data_without_labels = min_max_scaler_src.fit_transform(src_data_without_labels)
    target_data_without_labels = min_max_scaler_target.fit_transform(target_data_without_labels)

    dataset = UnsupervisedDataset(src_data_without_labels, target_data_without_labels, source_labels, target_labels)
    src = torch.tensor(src_data_without_labels).float()
    target = torch.tensor(target_data_without_labels).float()
    train_loader = DataLoader(dataset, shuffle=True, batch_size=config["batch_size"], drop_last=True)
    code_dim = 25
    input_dim = src_data_without_labels.shape[1]

    encoder = Encoder(input_dim,
                      hidden_dim=20,
                      drop_prob=0.1,
                      code_snape=code_dim)

    decoder = DecoderBer(code_dim=code_dim,
                         hidden_dim=100,
                         output_dim=input_dim,
                         drop_prob=0.1,
                         )
    net = Net(encoder, decoder)
    ind_discriminator = IndDiscriminator(
        input_dim=code_dim,
        hidden_dim=20,
        drop_prob=0.1)

    ae_optim = torch.optim.Adam(list(net.parameters()),
                                lr=config["lr"],
                                weight_decay=config["weight_decay"])

    ind_disc_optim = torch.optim.SGD(ind_discriminator.parameters(),
                                     lr=config["lr"],
                                     weight_decay=config["weight_decay"])
    if load_pre_weights == '':
        net, recon_losses, independence_losses = train2(src, target, train_loader, net,
                                                        ind_discriminator,
                                                        ae_optim, ind_disc_optim,
                                                        config)

        net.from_pretrain(os.path.join(config["save_weights"]))
    else:
        net.from_pretrain(load_pre_weights)

    net.eval()
    code_src, code_target, recon_src, recon_target, calibrated_src, calibrated_target = get_data_calibrated(src, target,
                                                                                                            net.encoder,
                                                                                                            net.decoder)

    normalized_calibrated_target = torch.tensor(
        min_max_scaler_src.inverse_transform(calibrated_target.detach().numpy()))
    normalized_target = torch.tensor(min_max_scaler_target.inverse_transform(target.detach().numpy()))
    normalized_src = torch.tensor(min_max_scaler_src.inverse_transform(src.detach().numpy()))
    normalized_calibrated_src = torch.tensor(min_max_scaler_target.inverse_transform(calibrated_src.detach().numpy()))

    normalized_target = model_shrinking.decoder(normalized_target)
    normalized_calibrated_src = model_shrinking.decoder(normalized_calibrated_src)

    normalized_src = model_shrinking.decoder(normalized_src)
    normalized_calibrated_target = model_shrinking.decoder(normalized_calibrated_target)

    adata_target_calibrated_src = make_adata_from_batches(normalized_target.detach().numpy(),
                                                          normalized_calibrated_src.detach().numpy(), target_labels,
                                                          source_labels)

    adata_src_calibrated_target = make_adata_from_batches(normalized_src.detach().numpy(),
                                                          normalized_calibrated_target.detach().numpy(), source_labels,
                                                          target_labels)

    return adata_src_calibrated_target, adata_target_calibrated_src
