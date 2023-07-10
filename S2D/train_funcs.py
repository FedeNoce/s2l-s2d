import os
import torch
import time


def train_autoencoder_dataloader(dataloader_train,
                                 device, model, optim, loss_fn,
                                 start_epoch, n_epochs, result_dir):

    for epoch in range(start_epoch, n_epochs):
        model.train()
        tloss = 0
        start_time = time.time()
        for b, sample_dict in enumerate(dataloader_train):
            optim.zero_grad()
            tx = sample_dict['points'].to(device)
            t_target = sample_dict['points_target'].to(device)
            landmarks = sample_dict['landmarks'].to(device)
            landmarks = torch.squeeze(landmarks)
            tx_hat, displacement = model(tx, landmarks)
            loss = loss_fn(t_target, tx_hat, tx, displacement, device)

            loss.backward()
            optim.step()
            tloss += loss


        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t train_loss={} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, tloss / b, elapsed_time), flush=True)

        torch.save({'epoch': epoch,
                    'autoencoder_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, os.path.join(result_dir, 's2d_coma_florence.pth.tar'))

    print('~FIN~')


