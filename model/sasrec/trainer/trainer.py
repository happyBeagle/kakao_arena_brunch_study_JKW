from tqdm import tqdm
import torch
import numpy as np
import os 

class Trainer:
    def __init__(self, model, loss, optimizer, train_loader, val_loader, config):
        self.model = model
        self.loss = loss 
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
    
    def _train_epoch(self, epoch):
        print(f'TRAINING EPOCH {epoch}')
        self.model.train() 
        train_loss = 0
        for seq, pos, neg, src_mask in tqdm(self.train_loader): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            pos_logits, neg_logits = self.model(seq, pos, neg, src_mask)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.config.SYSTEM.DEVICE), torch.zeros(neg_logits.shape, device=self.config.SYSTEM.DEVICE)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            self.optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = self.loss(pos_logits[indices], pos_labels[indices])
            loss += self.loss(neg_logits[indices], neg_labels[indices])
            for param in self.model.item_emb.parameters(): loss += self.config.TRAIN.L2_EMD * torch.norm(param)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        print(f"[TRAIN]loss in epoch {epoch}: {train_loss}") # expected 0.4~0.6 after init few epochs
    
    def _val_epoch(self, epoch):
        self.model.eval()

        val_loss = 0
        
        with torch.no_grad():
            for batch_idx, (seq, pos, neg, src_mask) in enumerate(self.val_loader):
                   logits = self.model.predict_(seq)
                   loss = self.loss(logits, pos)
                   val_loss += loss.item()
        print(f"[VALID]loss in epoch {epoch}: {val_loss}")

        # if epoch % 10 == 0:
        #     self.model.eval()
        #     t1 = time.time() - t0
        #     T += t1
        #     print('Evaluating', end='')
        #     t_test = evaluate(self.model, dataset, args)
        #     t_valid = evaluate_valid(self.model, dataset, args)
        #     print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
        #             % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

        #     f.write(str(t_valid) + ' ' + str(t_test) + '\n')
        #     f.flush()
        #     t0 = time.time()
        #     self.model.train()

        # if epoch % 10 == 0:
        #     folder = args.dataset + '_' + args.train_dir
        #     fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        #     fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
        #     torch.save(self.model.state_dict(), os.path.join(folder, fname))
    def _save_checkpoint(self, epoch):
        save_path = './tmp'
        print(f'Saveing checkpoint {save_path}..')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        file_name = f'Sasrec_{epoch}'
        file_path = os.path.join(save_path, file_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, file_path)

    def train(self, epochs):
        
        for epoch in range(1, epochs + 1):
            self._train_epoch(epoch)
            self._val_epoch(epoch)
            self._save_checkpoint(epoch)
            
