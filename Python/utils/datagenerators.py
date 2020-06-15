import progressbar
from torch.autograd import Variable
import torch

class Model_Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler=None, save_path=None, norm_clipping=None, cuda=True):
        self.cuda = cuda
        self.epoch_text = progressbar.FormatCustomText('Epoch %(epoch).i/%(n_epoch).i', dict(epoch=0, n_epoch=0))
        self.loss_text = progressbar.FormatCustomText('Loss: %(loss).4f', dict(loss=0))
        self.val_loss_text = progressbar.FormatCustomText('Val. Loss: %(loss).4f', dict(loss=0))
        self.pqr_text = progressbar.FormatCustomText('PQR10: %(pqr).2f', dict(pqr=0))
        self.val_pqr_text = progressbar.FormatCustomText('Val. PQR10: %(pqr).2f', dict(pqr=0))
        self.norm_clipping = norm_clipping
        self.save_path = save_path

        self.widgets = [self.epoch_text, ': ', progressbar.Percentage(), ', ', progressbar.ETA(), '; ',
                        self.loss_text, ', ', self.pqr_text, ', ',
                        self.val_loss_text, ', ', self.val_pqr_text]
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.bar = None

        self.hist_loss = []
        self.hist_pqr = []
        self.hist_loss_val = []
        self.hist_pqr_val = []

    def _execute(self, data_loader, loss_func, training=True, text_updater=None, pqr_updater=None):
        self.model.train(training)
        losses = []
        pqrs = []

        for batch_i in range(len(data_loader)):
            batch = next(data_loader)

            if len(batch) == 4:
                b_x, b_feat, b_y, b_pqr = batch
            elif len(batch) == 3:
                b_x, b_y, b_pqr = batch
                b_feat = None
            else:
                b_x, b_y = batch
                b_pqr, b_feat = None
            b_y = b_y.long()

            if self.cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                b_pqr = b_pqr.cuda() if b_pqr is not None else None
                b_feat = b_feat.cuda() if b_feat is not None else None

            if b_feat is not None:
                output = self.model(b_x, b_feat)
            else:
                output = self.model(b_x)

            loss = loss_func(output, b_y)
            losses.append(loss.item())

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                if self.norm_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.norm_clipping)
                self.optimizer.step()

            if text_updater is not None:
                l = sum(losses) / len(losses)
                text_updater.update_mapping(loss=l)
            if pqr_updater is not None and b_pqr is not None:
                pqr = self.calc_pqr(b_y, output, b_pqr)
                pqrs.append(pqr.item())
                p = sum(pqrs) / len(pqrs)
                pqr_updater.update_mapping(pqr=p)
            self.bar += 1
            del b_x, b_y, output, loss, b_pqr, b_feat

        return (sum(losses) / len(losses)), (sum(pqrs) / len(pqrs))

    def _reset_bar(self, max_value=1):
        self.loss_text.update_mapping(loss=0)
        self.val_loss_text.update_mapping(loss=0)
        self.pqr_text.update_mapping(pqr=0)
        self.val_pqr_text.update_mapping(pqr=0)
        self.bar = progressbar.ProgressBar(widgets=self.widgets, max_value=max_value).start()

    def train(self, loss_function, epochs):
        for epoch in range(epochs):
            self.epoch_text.update_mapping(epoch=(epoch+1), n_epoch=epochs)
            training_ = iter(self.train_loader)
            validating_ = iter(self.val_loader)
            total_iter = len(training_) + len(validating_)
            self._reset_bar(total_iter)
            loss, pqr = self._execute(training_, loss_function, True, self.loss_text, self.pqr_text)
            loss_val, pqr_val = self._execute(validating_, loss_function, False, self.val_loss_text, self.val_pqr_text)
            self.bar.finish()
            if self.scheduler is not None:
                self.scheduler.step()

            self.hist_loss.append(loss)
            self.hist_pqr.append(pqr)
            self.hist_loss_val.append(loss_val)
            self.hist_pqr_val.append(pqr_val)

            if self.save_path is not None and loss_val <= min(self.hist_loss_val):
                torch.save(self.model, self.save_path)


    def calc_pqr(self, y_true, y_pred, b_pqr):
        y_cls = torch.argmax(y_pred, dim=-1) if len(y_true.shape) <= 1 else torch.argmin(y_pred, dim=-1)
        pqrs = b_pqr[torch.arange(b_pqr.size(0)), y_cls]
        return torch.mean(pqrs)
