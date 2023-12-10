
def train(src, target, data_loader, encoder, decoder, ind_discriminator, ae_optim, ind_disc_optim, ae_scheduler,
          disc_scheduler,
          config
          ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    recon_criterion = nn.MSELoss()
    encoder_best_model_wts = copy.deepcopy(encoder.state_dict())
    decoder_best_model_wts = copy.deepcopy(decoder.state_dict())
    recon_losses = []
    independence_losses = []
    coef = [0.1, 100, 1]
    mmd_list = []
    best_mmd = 10000
    smoothed_disc_loss = 0
    l1_list = []
    l2_list = []
    for epoch in tqdm(range(config["epochs"])):

        average_discriminator_loss = 0
        average_ae_loss = 0

        counter = 0
        for step, (batch_x, batch_y) in enumerate(data_loader):
            encoder.train(True)
            decoder.train(True)
            ind_discriminator.train(True)
            counter += 1
            batch_x = batch_x.float().to(device=device).detach()
            batch_y = batch_y.int().to(device=device).detach()
            mask0 = batch_y == 0
            mask1 = batch_y == 1
            ######## train discrimnator
            ind_discriminator.zero_grad()
            code_real = encoder(batch_x)
            independence = torch.mean(ind_discriminator(code_real[mask1])) - torch.mean(
                ind_discriminator(code_real[mask0]))
            gradient_penlty = gradient_penalty(code_real[mask0], code_real[mask1], ind_discriminator)
            independence_loss = (independence + coef[0] * gradient_penlty)

            independence_loss.backward()
            ind_disc_optim.step()
            average_discriminator_loss += abs(independence.item())
            #### ae train
            if epoch % 2 == 0:
                encoder.zero_grad()
                decoder.zero_grad()
                code_real = encoder(batch_x)

                recon_batch_a, _ = decoder(code_real[mask0], batch_y[mask0])
                _, recon_batch_b = decoder(code_real[mask1], batch_y[mask1])

                adverserial_loss = nn.MSELoss()(torch.mean(ind_discriminator(code_real[mask0])),
                                                torch.mean(ind_discriminator(code_real[mask1])))
                recon_loss_a = recon_criterion(recon_batch_a, batch_x[mask0])
                recon_loss_b = recon_criterion(recon_batch_b, batch_x[mask1])
                ae_loss = coef[1] * (recon_loss_a / recon_loss_a.item() + recon_loss_b / recon_loss_b.item()) + coef[
                    2] * adverserial_loss / adverserial_loss.item()
                ae_loss_value = recon_loss_a.item() + recon_loss_b.item() + adverserial_loss.item()
                ae_loss.backward()
                ae_optim.step()
                average_ae_loss += ae_loss_value
        if epoch % 5 == 0 and epoch > 0:
            print(f"best mmd {best_mmd}")
            mmd = validate(src.clone(), target.clone(), encoder, decoder)

            # Save the best model
            if mmd < best_mmd:
                best_mmd = mmd
                encoder_best_model_wts = copy.deepcopy(encoder.state_dict())
                decoder_best_model_wts = copy.deepcopy(decoder.state_dict())

        # if epoch % 10 == 0 and epoch > 0:
        #
        #     code_src, code_target, recon_src, recon_target, calibrated_src, calibrated_trget = get_data_calibrated(
        #         src.clone(), target.clone(),
        #         encoder,
        #         decoder)
        #     mmd_code = float(eval_mmd(code_src, code_target))
        #     mmd_list.append(mmd_code)
        #     if len(mmd_list) > 1:# and mmd_code <= median(mmd_list):
        #         coef[0] = 0.1
        #         coef[1] = mean(mmd_list)/stdev(mmd_list)
        #         coef[2] = 1/mean(mmd_list)/stdev(mmd_list)
        #     # else:
        #     #     coef[0] = 0.1
        #     #     coef[1] = 1  # 17 * (1 / mmd_code)
        #     #     coef[2] = 100

        if epoch % 100 == 0 and epoch > 0:
            encoder.eval()
            decoder.eval()
            code_src, code_target, recon_src, recon_target, calibrated_src, calibrated_target = get_data_calibrated(
                src.clone(),
                target.clone(),
                encoder,
                decoder)
            print(f"best asw = {best_mmd}")
            mmd_code = float(eval_mmd(code_src, code_target))
            recon_loss_src = recon_criterion(recon_src, src.clone())
            recon_loss_target = recon_criterion(recon_target, target.clone())
            print(f"-----------------{epoch}-----------------")
            print(
                f"recon loss src- {recon_loss_src} ,recon loss target- {recon_loss_target}")
            print(f"mmd between code batch1 to batch2 {mmd_code}")
            print(f"mmd betweeen src_to_src to target_to_src: {eval_mmd(calibrated_target, src)}")
            print(f"mmd betweeen target_to_target to target_to_src: {eval_mmd(calibrated_src, target)}")
            src_pca = get_pca_data(src.clone())
            target_pca = get_pca_data(target.clone())
            recon_target_pca = get_pca_data(recon_target.detach().numpy())
            calibrated_src_pca = get_pca_data(calibrated_src.detach().numpy())
            calibrated_target_pca = get_pca_data(calibrated_target.detach().numpy())
            scatterHist(target_pca[:, 0],
                        target_pca[:, 1],
                        recon_target_pca[:, 0],
                        recon_target_pca[:, 1],
                        "pc1", "pc2", title="train data after calibration",
                        name1='target', name2='recon target', plots_dir='')

            scatterHist(target_pca[:, 0],
                        target_pca[:, 1],
                        calibrated_src_pca[:, 0],
                        calibrated_src_pca[:, 1],
                        "pc1", "pc2", title="train data after calibration",
                        name1='target', name2='calibrated src', plots_dir='')
            scatterHist(src_pca[:, 0],
                        src_pca[:, 1],
                        calibrated_target_pca[:, 0],
                        calibrated_target_pca[:, 1],
                        "pc1", "pc2", title="train data after calibration",
                        name1='src', name2='calibrated target', plots_dir='')

        smoothed_disc_loss = 0.95 * smoothed_disc_loss + 0.05 * independence.item()
        current_lr = ind_disc_optim.param_groups[0]['lr']

        for param_group in ind_disc_optim.param_groups:
            param_group['lr'] = current_lr * lr_scheduler(smoothed_disc_loss, 0)

        recon_losses.append(average_ae_loss / counter)
        independence_losses.append(average_discriminator_loss / counter)

    encoder.load_state_dict(encoder_best_model_wts)
    decoder.load_state_dict(decoder_best_model_wts)
    print(f"----------{best_mmd}----------")
    return encoder, decoder, recon_losses, independence_losses