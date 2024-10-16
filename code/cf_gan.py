from os import PRIO_USER, killpg, posix_fadvise
from typing import ItemsView
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from dis_model import DIS
from gen_model import GEN
from sup_gen_model import SUP_GEN
import pickle as cPickle
import numpy as np
import utils as ut
import multiprocessing
from scipy.spatial.distance import pdist, squareform
import data_ml_seq as user_seq
import math
import heapq
import linecache
from parser import parse_args
args = parse_args()
cores = multiprocessing.cpu_count()

#########################################################################################
# Hyper-parameters
#########################################################################################
TOPN = args.topn
EMB_DIM = args.emb_dim
USER_NUM = args.user_num  #anime  #3654                  #ml    #938
ITEM_NUM = args.item_num          #2033/1609(for Anime)         #1008
BATCH_SIZE = args.batch_size
INIT_DELTA = 0.05
SEQ_LEN = args.dpp_set_len
CATE_NUM = args.cate_num

temp_gen = args.temp_gen
g_lr = args.gen_lr
d_lr = args.dis_lr
g_lamda = args.g_lamda
d_lamda = args.d_lamda
g_dpp_lamda = args.dpp_lamda
epoch_size = args.num_epoch
d_epoch_size = args.d_num_epoch    #d_epoch % 5 == 0   
g_epoch_size = args.g_num_epoch
sample_lambda = args.sample_lambda
gen_for_d_epoch = args.gen_for_d_epoch
is_sup_gen = args.is_sup_gen

pre_emb = args.pre_emb   #embeddings.pkl for anime-seq no bias 9-25

Ks = [3, 5, 10, 20, 50]

sigma = 1. #for dpp gaussian kernel

strn = str(TOPN)
all_items = set(range(ITEM_NUM))
workdir = args.data_path + args.dataset +'/'

DIS_TRAIN_FILE = workdir + 'dpp_dis_train'+strn+'.txt'

#########################################################################################
# Load data
#########################################################################################
iidcate_map = {}  #iid:cates
## movie_id:cate_ids, cate_ids is not only one
with open(workdir + 'cate_id.txt') as f_cate:
    for l in f_cate.readlines():
        if len(l) == 0: break
        l = l.strip('\n')
        items = [int(i) for i in l.split(' ')]
        iid, cate_ids = items[0], items[1:]
        iidcate_map[iid] = cate_ids

user_pos_train = {}
item_set = set()
with open(workdir + 'train.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iids = line[1:]
        for iid in iids:
            iid = int(iid)
            item_set.add(iid)
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]

user_pos_test = {}
with open(workdir + 'test.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iids = line[1:]
        for iid in iids:
            iid = int(iid)
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]

all_users = user_pos_train.keys()
sorted(all_users)

#################### cond seq user train ######################
#user_cond_seq = user_seq.get_user_seq()

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def cc_at_k(cc, k):
    cates = set()
    for i in range(k):
        for c in cc[i]:
           cates.add(c)
    return len(cates) / CATE_NUM

def simple_test_one_user(x):
    rating = x[0]
    u = x[1]

    test_items = list(all_items - set(user_pos_train[u]))
    
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    
    r = []
    cc = []
    for i in K_max_item_score:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)
        cc.append(iidcate_map[i])

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    p_20 = np.mean(r[:20])
    p_50 = np.mean(r[:50])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)
    ndcg_20 = ndcg_at_k(r, 20)
    ndcg_50 = ndcg_at_k(r, 50)
    cc_3 = cc_at_k(cc, 3)
    cc_5 = cc_at_k(cc, 5)
    cc_10 = cc_at_k(cc, 10)
    cc_20 = cc_at_k(cc, 20)
    cc_50 = cc_at_k(cc, 50)

    return np.array([p_3, p_5, p_10, p_20, p_50, ndcg_3, ndcg_5, ndcg_10, ndcg_20, ndcg_50, cc_3, cc_5, cc_10, cc_20, cc_50])

def simple_test(sess, model):
    result = np.array([0.] * 15)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    test_users = list(user_pos_test.keys())
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret

def seq_test_one_user(x):
    pred = x[0]
    u = x[1]

    r = []
    cc = []
    for i in pred:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)
        cc.append(iidcate_map[i])

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    p_20 = np.mean(r[:20])
    p_50 = np.mean(r[:50])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)
    ndcg_20 = ndcg_at_k(r, 20)
    ndcg_50 = ndcg_at_k(r, 50)
    cc_3 = cc_at_k(cc, 3)
    cc_5 = cc_at_k(cc, 5)
    cc_10 = cc_at_k(cc, 10)
    cc_20 = cc_at_k(cc, 20)
    cc_50 = cc_at_k(cc, 50)

    return np.array([p_3, p_5, p_10, p_20, p_50, ndcg_3, ndcg_5, ndcg_10, ndcg_20, ndcg_50, cc_3, cc_5, cc_10, cc_20, cc_50])

def seq_test(pred_items):
    result = np.array([0.]*15)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    test_users = list(user_pos_test.keys())  #list(range(len(test_users)))
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        user_batch_pred = pred_items[index:index + batch_size]
        index += batch_size

        user_batch_rating_uid = zip(user_batch_pred, user_batch)
        batch_result = pool.map(seq_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def diversity_kernel(emb):
    pairwise_sq_dists = squareform(pdist(emb, 'sqeuclidean'))
    gamma = 1.0 / (2 * sigma ** 2)
    K = np.exp(-pairwise_sq_dists*gamma)
    return K

##################### Fast Greedy MAP Inference for DPP ###############
def greedy_dpp(kernel_matrix, max_length, epsilon=1E-12):
    
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items

def cond_greedy_dpp(kernel_matrix, cond_set, topn, epsilon=1E-10):
    item_size = kernel_matrix.shape[0]
    max_length = len(cond_set) + topn
    cis = np.zeros((max_length, item_size))  
    di2s = np.copy(np.diag(kernel_matrix))
    k = 0
    for iid in cond_set:
        ci_optimal = cis[:k, iid]
        di_optimal = math.sqrt(di2s[iid])
        elements = kernel_matrix[iid, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        k += 1
    selected_items = cond_set
    selected_item = iid
    di2s[cond_set[:-1]] = -np.inf
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items[-topn:]

def cond_dpp(kernel_matrix, cond_set, max_length=SEQ_LEN):

    cur_det = np.linalg.slogdet(kernel_matrix[cond_set][:, cond_set])

    selected_items = cond_set
    det_list = [-np.inf]*ITEM_NUM
    while len(selected_items) < max_length:
        for i in np.arange(ITEM_NUM):
            if i not in selected_items:
                add_items = np.append(selected_items,[i])
                s, changed_det = np.linalg.slogdet(kernel_matrix[add_items][:, add_items])
                print(changed_det)
                det_list[i] = changed_det
        next_item = np.argmax(np.array(det_list))
        selected_items = np.append(selected_items, next_item)
        det_list[next_item] = -np.inf
    return selected_items[int(max_length/2):]

def generate_for_d(sess, model, filename):
    data = []
    ################ greedy sample ##############
    item_embeddings = sess.run(model.item_embeddings)
    #print("item_embeddings:", item_embeddings)
    item_diversity = diversity_kernel(item_embeddings)
    pos_sum = 0
    for u in user_pos_train:
        pos = np.array(user_pos_train[u])
        rating = sess.run(model.all_rating, {model.u: [u]}) 
        pos_sum += len(user_pos_train[u])
        rating = np.array(rating[0]) / temp_gen  # Temperature
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)
        
        user_kernel = np.dot(np.dot(np.diag(exp_rating),item_diversity), np.diag(exp_rating))

        ######## neg_samples = dpp topn
        if len(pos) < TOPN:
            neg_sample = greedy_dpp(user_kernel, len(pos)) 
        else:
            neg_sample = greedy_dpp(user_kernel, TOPN)
            
        for i in range(len(pos)): #0 1 2 3 4 5 6 7 8 9 10 11 
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg_sample[i%TOPN]))

    print(pos_sum)
    with open(filename, 'w') as fout:
        fout.write('\n'.join(data))

def main():
    print("load model...")
    param = cPickle.load(open(workdir + pre_emb, 'rb'), encoding="latin1")
    if is_sup_gen:
        generator = SUP_GEN(ITEM_NUM, USER_NUM, EMB_DIM, lamda = g_lamda,  dpp_lamda = g_dpp_lamda, param=param, initdelta=INIT_DELTA,
                        learning_rate=g_lr)
    else:
        generator = GEN(ITEM_NUM, USER_NUM, EMB_DIM, lamda = g_lamda,  dpp_lamda = g_dpp_lamda, param=param, initdelta=INIT_DELTA,
                        learning_rate=g_lr)
    discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda = d_lamda, param=param, initdelta=INIT_DELTA,
                        learning_rate=d_lr)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print("gen ", simple_test(sess, generator))
    print("dis ", simple_test(sess, discriminator))

    dis_log = open(workdir + 'dpp_dis_log'+strn+'.txt', 'w')
    gen_log = open(workdir + 'dpp_gen_log'+strn+'.txt', 'w')

    # minimax training
    best = 0.
    for epoch in range(epoch_size):
        if epoch >= 0:
            # Train D
            print("start D Training:")
            for d_epoch in range(d_epoch_size): #100
                d_loss = 0
                if d_epoch % gen_for_d_epoch == 0:
                    generate_for_d(sess, generator, DIS_TRAIN_FILE)
                    train_size = ut.file_len(DIS_TRAIN_FILE)
                index = 1
                #linecache.updatecache(DIS_TRAIN_FILE)
                while True:
                    if index > train_size:
                        break
                    if index + BATCH_SIZE <= train_size + 1:
                        input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index, BATCH_SIZE)
                    else:
                        print("train_size1", train_size)
                        input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index,
                                                                                train_size - index + 1)
                    index += BATCH_SIZE

                    _, bd_loss = sess.run([discriminator.d_updates, discriminator.pre_loss],
                                 feed_dict={discriminator.u: input_user, discriminator.i: input_item,
                                           discriminator.label: input_label})
                    d_loss += np.mean(bd_loss)

                result = simple_test(sess, discriminator)   # item recommendation

                print("epoch ", epoch, "d_epoch", d_epoch, "dis: ", result)
                buf = '\t'.join([str(x) for x in result])

                p_5 = result[1]
                if p_5 > best:
                    print('best: ', result)
                    best = p_5
                    discriminator.save_model(sess, workdir+'dpp_discriminator'+strn+'.pkl')
            # Train G
            print("start G Training:")
            for g_epoch in range(g_epoch_size):  # 50

                item_embeddings = sess.run(generator.item_embeddings)
                item_diversity = diversity_kernel(item_embeddings)

                loss, g_loss, d_loss, norm_loss, dpp_norm_loss = 0, 0, 0, 0, 0
                pred_items = []

                for u in user_pos_train:
                    pos = user_pos_train[u]
                    rating = sess.run(generator.all_logits, {generator.u: u})

                    exp_rating = np.exp(rating)
                    prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta
                    #user_kernel = np.dot(np.dot(np.diag(exp_rating), item_diversity), np.diag(exp_rating))
                    pn = (1 - sample_lambda) * prob
                    pn[pos] += sample_lambda * 1.0 / len(pos)
                    user_kernel = np.dot(np.dot(np.diag(pn), item_diversity), np.diag(pn))
                    
                    if len(pos) < TOPN:
                        sample = greedy_dpp(user_kernel, len(pos)) 
                    else:
                        sample = greedy_dpp(user_kernel, TOPN)
                    
                    ##### sample = np.random.choice(np.arange(ITEM_NUM), size=len(pos), p=pn)  #importance sampling
                    
                    ############# Get reward and adapt it with importance sampling
                    reward = sess.run(discriminator.reward, {discriminator.u: u, discriminator.i: sample})
                    reward = reward * prob[sample] / pn[sample]  
                    ############# randomly selecting pos and neg items of u for supervised generator
                    if is_sup_gen:
                        test_items = np.array(list(all_items - set(pos)))
                        sup_neg = np.random.choice(test_items, len(pos), replace=False)
                        sup_ids = np.concatenate((np.array(pos), sup_neg))
                        sup_labels = np.array([1]*len(pos) + [0]*len(sup_neg))
                        _, uloss, ug_loss, unorm_loss, ud_loss, udpp_norm_loss = sess.run(
                                    [generator.gan_updates, generator.gen_loss, generator.gan_loss, 
                                    generator.norm_loss, generator.det, generator.dpp_norm], 
                                    {generator.u: u, generator.i: sample, generator.reward: reward, 
                                    generator.sup_labels: sup_labels, generator.sup_ids: sup_ids,                          
                                    generator.len: len(sample)})
                    else:
                        _, uloss, ug_loss, unorm_loss, ud_loss, udpp_norm_loss = sess.run(
                                    [generator.gan_updates, generator.gen_loss, generator.gan_loss, 
                                    generator.norm_loss, generator.det, generator.dpp_norm], 
                                    {generator.u: u, generator.i: sample, generator.reward: reward, 
                                    #generator.dpp_i: np.array(dpp_sample), 
                                    generator.len: len(sample)})  #如果仅关注topn的diversity，这里len为sample长度，并且不需要dpp_i, 否则为topn
                    #print("loss:", uloss, "=ug_loss+", ug_loss, "unorm_loss+", unorm_loss, "ud_loss", ud_loss, "udpp_norm_loss+", udpp_norm_loss)
                    loss += uloss
                    g_loss += ug_loss
                    d_loss += ud_loss
                    norm_loss += unorm_loss
                    dpp_norm_loss += udpp_norm_loss
                    '''
                    ##############################################################
                    ## conditional/sequential recommendation  
                    ##############################################################
                    if u in user_pos_test:
                        rating = sess.run(generator.all_rating, {generator.u: [u]}) 
                        rating = np.array(rating[0])  # Temperature
                        exp_rating = np.exp(rating)

                        user_kernel = np.dot(np.dot(np.diag(exp_rating), item_diversity), np.diag(exp_rating))
                        
                        u_pred_items = cond_greedy_dpp(user_kernel, pos.copy(), 50)
                        pred_items.append(u_pred_items)
                result = seq_test(pred_items)  #sequential/conditional recommendation
                '''
               
                result = simple_test(sess, generator)  #item recommendation

                print("########### epoch ", epoch, "g_epoch:", g_epoch, "gen: ", result)
                buf = '\t'.join([str(x) for x in result])
                gen_log.write(str(epoch) + '\t' + buf + '\n')
                gen_log.flush()

                p_5 = result[1]
                if p_5 > best:
                    print('best: ', result)
                    best = p_5
                    generator.save_model(sess, workdir+'dpp_generator_test.pkl')

    gen_log.close()
    dis_log.close()

if __name__ == '__main__':
    main()
