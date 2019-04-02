__author__ = 'yawli'


import os
import math
from decimal import Decimal
import torch.optim as optim
import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm
from trainer import Trainer


class TrainerFT(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(TrainerFT, self).__init__(args, loader, my_model, my_loss, ckp)
        # self.args = args
        # self.scale = args.scale
        #
        # self.ckp = ckp
        # self.loader_train = loader.loader_train
        # self.loader_test = loader.loader_test
        # self.model = my_model
        # self.loss = my_loss
        self.optimizer = self.make_optimizer(args, self.model)
        # self.scheduler = utility.make_scheduler(args, self.optimizer)
        #
        # if self.args.load != '.':
        #     self.optimizer.load_state_dict(
        #         torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
        #     )
        #     for _ in range(len(ckp.log)): self.scheduler.step()
        #
        # self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        # from IPython import embed; embed(); exit()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, nl, mk, hr, _, idx_scale) in enumerate(self.loader_train):
            # from IPython import embed; embed(); exit()
            lr, nl, mk, hr = self.prepare([lr, nl, mk, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(idx_scale, lr, nl, mk)
            # from IPython import embed; embed(); exit()
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.3f}+{:.3f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
            # from IPython import embed; embed(); exit()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, nl, mk, hr, filename, _) in enumerate(tqdm_test):
                    print('FLAG')
                    print(filename)
                    filename = filename[0]
                    print(filename)
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, nl, mk, hr = self.prepare([lr, nl, mk, hr])
                    else:
                        lr, nl, mk, = self.prepare([lr, nl, mk])

                    sr = self.model(idx_scale, lr, nl, mk)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    print(sr.shape)
                    b, c, h, w = sr.shape
                    hr = hr[:, :, :h, :w]
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def make_optimizer(self, args, model):
        trainable = filter(lambda x: x.requires_grad, model.model.parameters())
        # from IPython import embed; embed(); exit()
        finetune_id = list(map(id, model.model.body_ft.parameters())) \
                      + list(map(id, model.model.tail_ft.parameters()))#\
                      #+ list(map(id, model.model.tail_ft2.parameters()))
        base_params = filter(lambda x: id(x) not in finetune_id, trainable)
        trainable = filter(lambda x: x.requires_grad, model.model.parameters())
        finetune_params = filter(lambda x: id(x) in finetune_id, trainable)
        if args.optimizer == 'SGD':
            optimizer_function = optim.SGD
            kwargs = {'momentum': args.momentum}
        elif args.optimizer == 'ADAM':
            optimizer_function = optim.Adam
            kwargs = {
                'betas': (args.beta1, args.beta2),
                'eps': args.epsilon
            }
        elif args.optimizer == 'RMSprop':
            optimizer_function = optim.RMSprop
            kwargs = {'eps': args.epsilon}

        kwargs['lr'] = args.lr * 0.1
        kwargs['weight_decay'] = args.weight_decay
        # from IPython import embed; embed(); exit()
        return optimizer_function([
                                      {'params': base_params},
                                      {'params': finetune_params, 'lr': args.lr}
                                  ], **kwargs)
