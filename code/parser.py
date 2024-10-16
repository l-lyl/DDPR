import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="GAN+DPP.")
    
    parser.add_argument('--data_path', nargs='?', default='../item_recommendation/',
                        help='Input data path.')

    parser.add_argument('--dataset', nargs='?', default='processed_ml-100k-seq',
                        help='Input data path.')

    parser.add_argument('--pre_emb', nargs='?', default= 'ngcf_emb_64_bias.pkl', 
                        help='pretrained embeddings.')
    
    parser.add_argument('--num_epoch', type=int, default=20,
                        help='Number of epoch.')

    parser.add_argument('--g_num_epoch', type=int, default=5,
                        help='Number of gen epoch.')

    parser.add_argument('--d_num_epoch', type=int, default=1,
                        help='Number of edis poch.')
    
    parser.add_argument('--gen_for_d_epoch', type=int, default=1,
                        help='when to generate samples for dis mdel .')

    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Embedding size.')

    parser.add_argument('--item_num', type=int, default=1008,
                        help='item size.')
    
    parser.add_argument('--user_num', type=int, default=938,
                        help='user size.')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size.')

    parser.add_argument('--dpp_set_len', type=int, default=5,
                        help='dpp set len.')

    parser.add_argument('--cate_num', type=int, default=43,
                        help='17 for ml, 43 for Anime')

    parser.add_argument('--topn', type=int, default=10,
                        help='dpp sample num')

    parser.add_argument('--g_has_bias', type=int, default=1,
                        help='use item bias or not')

    parser.add_argument('--d_has_bias', type=int, default=1,
                        help='use item bias or not')

    parser.add_argument('--temp_gen', type=float, default=1.,
                        help='for gen for d.')

    parser.add_argument('--sample_lambda', type=float, default=1.,
                        help='for gen for d.')

    parser.add_argument('--gen_lr', type=float, default=1e-3,
                        help='gen model lr.')

    parser.add_argument('--dis_lr', type=float, default=1e-5,
                        help='dis model lr.')

    parser.add_argument('--g_lamda', type=float, default=0.,
                        help='gen model l2 loss weight.')

    parser.add_argument('--d_lamda', type=float, default=0.001,
                        help='dis model l2 loss weight.')

    parser.add_argument('--dpp_lamda', type=float, default=0.,
                        help='i embeddings of dpp norm weight.')

    parser.add_argument('--d_param', type=float, default=1.,
                        help='whether dis use the pretrained param.')

    parser.add_argument('--dpp_loss_weight', type=float, default=1.,
                        help='weight of dpp loss.')

    parser.add_argument('--is_sup_gen', type=float, default=0.,
                        help='using supervised loss in gen or not.')

    return parser.parse_args()
