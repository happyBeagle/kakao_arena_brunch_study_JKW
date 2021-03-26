# %%
import numpy as np 
from model.ConvMF.convmodel import CNN as CNN
# %%
class ConvMF:
    def __init__(self, rate_table, vocab_size, if_cuda, cnn_epoch = 20, cnn_lr = 1e-4,
                  lambda_u=1, lambda_v=100, dimension=50,
                 dropout_rate=0.2, emb_dim=200, max_len=200, num_kernel_per_ws=100):
        self.__num_user = rate_table.shape[0]
        self.__num_item = rate_table.shape[1] 
        self.__rate_table = rate_table
        self.__vocab_size = vocab_size
        self.__lambda_u = lambda_u
        self.__lambda_v = lambda_v
        self.__dimension = dimension
        self.__item_w = np.ones(num_item, dtype=float)
     
        self.__train_cnn = TrainCNN(cnn_epoch, cnn_lr, dimension, vocab_size, emb_dimension, 
                        dropout_rate, max_len, num_kernel_per_ws, if_cuda)
             
        self.__theta = train_cnn.get_projection_layer(cnn_input)
        self.__U = np.random.uniform(size=(self.__num_user, dimension))
        self.__V = self.__theta

    def train(self, cnn_input, max_epochs, train_user, train_item, valid_user, test_user):
        a, b = 1, 0

        PREV_LOSS =1e-50
        
        Train_R_I = train_user[1]
        Train_R_J = train_item[1] 
        Test_R = test_user[1]
        Valid_R = valid_user[1]

        pre_val_eval = 1e10
        best_tr_eval, best_val_eval, best_te_eval = 1e10, 1e10, 1e10

   
        endure_count = 5
        count = 0
        
        for e in range(max_epochs):
            loss = 0
        
            VV = b * (self.__V.T.dot(self.__V)) + self.__lambda_u * np.eye(self.__dimension)
            sub_loss = np.zeros(self.__num_user)

            for i in range(self.__num_user):
                idx_item = train_user[0][i]
                V_i = self.__V[idx_item]
                R_i = Train_R_I[i]
                A = VV + (a - b) * (V_i.T.dot(V_i))
                B = (a * V_i * np.tile(R_i, (self.__dimension, 1)).T).sum(0)

                self.__U[i] = np.linalg.solve(A, B)

                sub_loss[i] = -0.5 * self.__lambda_u * np.dot(self.__U[i], self.__U[i])

            loss = loss + np.sum(sub_loss)

            sub_loss = np.zeros(self.__num_item)
            UU = b * (self.__U.T.dot(self.__U))
            for j in range(self.__num_item):
                idx_user = train_item[0][j]
                U_j = self.__U[idx_user]
                R_j = Train_R_J[j]

                tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
                A = tmp_A + self.__lambda_v * self.__item_weight[j] * np.eye(self.__dimension)
                B = (a * U_j * np.tile(R_j, (self.__dimension, 1)).T).sum(0) + self.__lambda_v * self.__item_weight[j] * self.__theta[j]
                self.__V[j] = np.linalg.solve(A, B)

                sub_loss[j] = -0.5 * np.square(R_j * a).sum()
                sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(self.__V[j])) * R_j)
                sub_loss[j] = sub_loss[j] - 0.5 * np.dot(self.__V[j].dot(tmp_A), self.__V[j])

            loss = loss + np.sum(sub_loss)

            self.__train_cnn.train(cnn_input, self.__V)
            self.__theta = self.__train_cnn.get_projection_layer(cnn_input)

            tr_eval = self.eval_RMSE(Train_R_I, train_user[0])
            val_eval = self.eval_RMSE(Valid_R, valid_user[0])
            te_eval = self.eval_RMSE(Test_R, test_user[0])

            converge = abs((loss - PREV_LOSS) / PREV_LOSS)

            if val_eval < pre_val_eval:
                best_tr_eval, best_val_eval, best_te_eval = tr_eval, val_eval, te_eval
            else:
                count += 1

            pre_val_eval = val_eval

            if count == endure_count:
                break

            PREV_LOSS = loss
        print("\n\nBest Model: Train: %.5f Valid: %.5f Test: %.5f" % (best_tr_eval, best_val_eval, best_te_eval))
  

    def eval_RMSE(self, R, TS):
        sub_rmse = np.zeros(self.__num_user)
        TS_count = 0
        for i in range(self.__num_user):
            idx_item = TS[i]
            if len(idx_item) == 0:
                continue
            TS_count += len(idx_item)
            approx_R_i = self.__U[i].dot(self.__V[idx_item].T)
            R_i = R[i]

            sub_rmse[i] = np.square(approx_R_i - R_i).sum()

        rmse = np.sqrt(sub_rmse.sum() / TS_count)

        return rmse
    
    @property
    def U(self):
        return self.__U
    
    @property
    def V(self):
        return self.__V
# %%
