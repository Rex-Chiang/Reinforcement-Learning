import numpy as np
import tensorflow as tf

np.random.seed(1) #each run have the same random numbers
tf.set_random_seed(1) #each run have the same random numbers

global Num #numbers of antennas
Num= 20

class DeepQNetwork:
    def __init__(
            self,
            n_actions,# 40, numbers of actions, left: 20, right: 20
            n_features,# 20, numbers of antennas
            learning_rate=0.01,
            reward_decay=0.9,# 獎勵衰減度
            e_greedy=0.9,# 貪婪度 (探索度=0.1)
            replace_target_iter=300, #target, eval參數置換循環門檻
            memory_size=500,# 記憶庫大小
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))# "n_features*2": s, s_ ; "+2": a, r

        # consist of [target_net, evaluate_net]
        self._build_net()# DeepQNetwork一被呼叫變自動執行_build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')# 從指定集合中取出全部變量並回傳list
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]# tf.assign() change t's value to e's value

        self.sess = tf.Session()

        #if output_graph:
            # $ tensorboard --logdir=logs
        #    tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []


    def _build_net(self):
            # ------------------ all inputs ------------------------
            #如果想要從外部傳入data
            #placeholder用於定義過程在執行的时候再賦與具體的值
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
            self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward, shape= (batch-size, )
            self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action, shape= (batch-size, )
    
            w_initializer= tf.random_normal_initializer(0., 0.3)
            b_initializer= tf.constant_initializer(0.1)
    
            # ------------------ build evaluate_net ------------------
            #如果想要達到重複利用變量
            #使用tf.variable_scope(), 並搭配 tf.get_variable() 產生和提取變量
            #在重複使用的時要强調 scope.reuse_variables()
            with tf.variable_scope('eval_net'):
                e1 = tf.layers.dense(self.s,
                                     40,# 該層的神經單元節點數 shape= (len(self.s), 節點數)
                                     tf.nn.relu,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer,
                                     name='e1')
                
                self.q_eval = tf.layers.dense(e1,
                                              self.n_actions,# output的神經單元節點數 shape= (len(self.s), n_actions)
                                              kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer,
                                              name='q')
    
            # ------------------ build target_net ------------------
            with tf.variable_scope('target_net'):
                t1 = tf.layers.dense(self.s_,
                                     40,# 該層的神經單元節點數 shape= (batch-size, 節點數)
                                     tf.nn.relu,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer,
                                     name='t1')
                
                self.q_next = tf.layers.dense(t1,
                                              self.n_actions,# output的神經單元節點數 shape= (batch-size, n_actions)
                                              kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer,
                                              name='t2')
    
            with tf.variable_scope('q_target'):
                q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')# axis=1, 找出各行中最大值组成一個tensor
                self.q_target = tf.stop_gradient(q_target)# Fixed Q-targets, 阻擋節點BP的梯度 shape= (batch-size, )
                
            with tf.variable_scope('q_eval'):
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1) # shape= (batch-size, 2)
                
                self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)# 允許在多维上進行索引 shape= (batch-size, )
                
            # 取平均的 loss 是為了輸出時可以查看,  tf 内部還是每個 loss 單獨傳遞
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))# tf.reduce_mean()回傳各個元素的平均值
            
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
    
    def store_transition(self, s, a, r, s_): #Experience replay
            if not hasattr(self, 'memory_counter'): #hasattr() 用於判斷對象（object）是否包含對應的属性
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_)) #水平(按列順序)把數組给堆疊起来
            # replace the old memory with new memory
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1
            
            
    def Choose_Actions(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :] ##轉變一維矩陣observation的形狀成 (1* len(observation))
        
        #a1= np.arange(1, Num+1)
        #a2= np.arange(-1, -Num-1, -1)     
        #actions= np.concatenate((a1,a2))
        actions= np.arange(0,Num*2)
        
        if np.random.uniform() < self.epsilon: #貪婪度=0.9(探索度=0.1)
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation}) #輸出為 shape= (len(actions), )
            action = np.argmax(actions_value) #數組中最大元素的索引
        else:
            action= np.random.choice(actions)
        print(action)
        return action
       
    def learn(self):
            # check to replace target parameters
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.sess.run(self.target_replace_op)
                print('\ntarget_params_replaced\n')
    
            # sample batch memory from all memory
            # 從memory中隨機取出batch_size個當作batch_memory
            if self.memory_counter > self.memory_size: #如果記憶庫資料足夠就使用全部來選, 不夠就目前資料量來選
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]
            #print(batch_memory.shape)
            #print(batch_memory[:, self.n_features])
            #print(4444)
            # 訓練 eval_net
            _, cost = self.sess.run(
                [self._train_op, self.loss],
                feed_dict={
                    self.s: batch_memory[:, :self.n_features],
                    self.a: batch_memory[:, self.n_features],# shape= (batch-size, )
                    self.r: batch_memory[:, self.n_features + 1],# shape= (batch-size, )
                    self.s_: batch_memory[:, -self.n_features:],
                })
            
            self.cost_his.append(cost)# 紀錄 cost 誤差
    
            # 逐漸增加 epsilon, 降低行為的随機性
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
            self.learn_step_counter += 1
            print("Learn_step_counter:", self.learn_step_counter)
            
    def plot_cost(self):
            import matplotlib.pyplot as plt
            plt.plot(np.arange(len(self.cost_his)), self.cost_his)
            plt.ylabel('Cost')
            plt.xlabel('training steps')
            plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(40,20, output_graph=False)        