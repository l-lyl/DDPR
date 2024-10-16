import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle as cPickle
from parser import parse_args
args = parse_args()

class GEN():
    def __init__(self, itemNum, userNum, emb_dim, lamda, dpp_lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.dpp_lamda = dpp_lamda
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.dpp_loss_weight = args.dpp_loss_weight
        self.g_params = []

        self.set_len = 5    #used for dpp determinant
        self.dpp_sigma = tf.constant(1.)  #for gausian kernel

        with tf.variable_scope('generator'):
            if self.param == None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                if args.g_has_bias:
                    self.item_bias = tf.Variable(self.param[2])
                else:
                    #self.item_bias = tf.Variable(tf.zeros([self.itemNum]), trainable = False)
                    self.item_bias = tf.Variable(tf.zeros([self.itemNum])) #need trainable when anime-seq-3, ml-seq
    
            self.g_params = [self.user_embeddings, self.item_embeddings, self.item_bias]

        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.reward = tf.placeholder(tf.float32)
        self.len = tf.placeholder(tf.int32)
        self.dpp_i = tf.placeholder(tf.int32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.dpp_i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.dpp_i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.i_prob = tf.gather(
            tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]),
            self.i)
        
        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) 
        self.norm_loss = self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias))
        
        self.pair_distance = tf.reduce_sum((tf.expand_dims(self.i_embedding, 1) - tf.expand_dims(self.i_embedding, 0))**2,2)
        self.item_kernel = tf.exp(-self.pair_distance*(1.0 / (2 * tf.square(self.dpp_sigma))))

        self.det = -self.loop_split(self.len, self.set_len, self.item_kernel)  #set items with fixed size
        self.dpp_norm = self.dpp_lamda * tf.nn.l2_loss(self.i_embedding)
        
        self.gen_loss = self.gan_loss + self.norm_loss + self.dpp_loss_weight*(self.det + self.dpp_norm)

        g_opt = tf.train.GradientDescentOptimizer(self.learning_rate)  #tf.train.AdamOptimizer  ?
        self.gan_updates = g_opt.minimize(self.gen_loss, var_list=self.g_params)
        
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias

    def save_model(self, sess, filename):
        param = sess.run(self.g_params)
        cPickle.dump(param, open(filename, 'wb'))

    def cond(self, a, time, len, size, kernel):
        return tf.less((time+1)*size, len)

    def body(self, a, time, len, size, kernel):
        k = tf.slice(kernel, [time*size, time*size], [size, size]) + tf.linalg.diag(tf.ones((size)))
        #k = tf.Print(k, ['self.k=', k, '  '])
        det = tf.log(1-tf.exp(-0.01*tf.linalg.det(k)))
        #det = tf.Print(det, ['self.det=', det, '  '])
        a = a.write(time, det)
        return a, time+1, len, size, kernel

    def loop_split(self, len, size, kernel):
        time = tf.constant(0)
        a = tf.TensorArray(dtype=tf.float32, size = 1, dynamic_size=True, clear_after_read=False)
        result, _, _, _, _ = tf.while_loop(self.cond, self.body, [a, time, len, size, kernel])
        re = tf.reduce_mean(result.stack(),axis = 0)
        return re

    

