
# coding: utf-8

# In[40]:


import numpy as np
import gym
import cv2
import random
from collections import deque
import tensorflow as tf


# In[41]:


env = gym.make('Breakout-v0')


# In[42]:


action_size = env.action_space.n
obs_space = env.observation_space
print ( "Action Size : ", action_size )
print ( "Observation Space : ", obs_space )
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())


# In[43]:


def preprocess_frame(frame):
    gray = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
    normalised_frame = gray/255.0
    final_frame = cv2.resize( normalised_frame, (84,110) )
    return final_frame


# In[44]:


stack_size = 4

stacked_frames = deque( [ np.zeros( (110, 84, 1), dtype=np.int ) for i in range(stack_size) ] , maxlen = 4 )

def stack_frames( stacked_frames, state, is_new_episode ):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque( [ np.zeros( (110, 84, 1), dtype=np.int ) for i in range(stack_size) ] , maxlen = 4 )
        for _ in range(stack_size):
            stacked_frames.append( frame )
        stack_state = np.stack( stacked_frames, axis=2 )
    else:
        stacked_frames.append(frame)
        stack_state = np.stack( stacked_frames, axis=2 )
        
    return stack_state, stacked_frames


# In[45]:


### MODEL HYPERPARAMETERS
state_size = [110, 84, 4]      
learning_rate =  0.0005

total_episodes = 50            
max_steps = 50000              
batch_size = 64                

explore_start = 1.0            
explore_stop = 0.01            
decay_rate = 0.00001           

gamma = 0.9

pretrain_length = batch_size   
memory_size = 1000000 
training = True


# In[46]:


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            
            self.inputs_ = tf.placeholder(tf.float32, [None, 110, 84, 4], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")
            
            
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")

            self.conv2_out = tf.nn.relu(self.conv2, name="conv2_out")            
            
            
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 64,
                                 kernel_size = [3,3],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")

            self.conv3_out = tf.nn.relu(self.conv3, name="conv3_out")
            
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            
            self.fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.relu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")
            
            self.output = tf.layers.dense(inputs = self.fc, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)
            

            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))
            
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


# In[47]:


tf.reset_default_graph()
DQNetwork = DQNetwork(state_size=state_size, action_size=action_size, learning_rate=learning_rate)


# In[48]:


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add( self, experience ):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice( np.arange(buffer_size), size=batch_size, replace=False )
        return [ self.buffer[i] for i in index ]
    


# In[49]:


## Initialise Memory
memory = Memory(max_size = memory_size)
for i in range(pretrain_length):
    
    if i == 0:
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True  )
        
    choice = random.randint(1, len(possible_actions)) - 1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(choice)
    
    next_state, stacked_frames = stack_frames( stacked_frames, next_state, False )
    
    if done:
        
        next_state = np.zeros( state.shape )
        memory.add( (state, reward, action, next_state, done) )
        state = env.reset()
        state, stacked_frames = stack_frames( stacked_frames, state, True )
        
    else:
        memory.add( (state, action, reward, next_state, done) )
        state = next_state
        


# In[50]:


writer = tf.summary.FileWriter('./tb/dqn')
tf.summary.scalar("Loss", DQNetwork.loss)
write_op = tf.summary.merge_all()


# In[51]:


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if ( explore_probability > exp_exp_tradeoff ):
        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice] 
    else:
        Qs = sess.run( DQNetwork.output, feed_dict = { DQNetwork.inputs_ : state.reshape( (1, 110, 84, 4) ) } )
        choice = np.argmax(Qs)
        action = possible_actions[choice]
    
    return action, explore_probability


# In[52]:
rewards_list = []

# Saver will help us to save our model
saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0
        
        for episode in range(total_episodes):
            # Set step to 0
            step = 0
            
            # Initialize the rewards of the episode
            episode_rewards = []
            
            # Make a new episode and observe the first state
            state = env.reset()
            
            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            
            while step < max_steps:
                step += 1
                
                #Increase decay_step
                decay_step +=1
                
                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
                
                #Perform the action and get the next_state, reward, and done information
                next_state, reward, done, _ = env.step( np.argmax(action) )
                
                # Add the reward to total reward
                episode_rewards.append(reward)
                
                # If the game is finished
                if done:
                    # The episode ends so no next state
                    next_state = np.zeros((110,84,3), dtype=np.uint8)
                    
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                                  'Total reward: {}'.format(total_reward),
                                  'Explore P: {:.4f}'.format(explore_probability),
                                'Training Loss {:.4f}'.format(loss))

                    rewards_list.append((episode, total_reward))

                    # Store transition <st,at,rt+1,st+1> in memory D
                    memory.add((state, action, reward, next_state, done))

                else:
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                
                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state
                    

                ### LEARNING PART            
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch]) 
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state 
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                        feed_dict={DQNetwork.inputs_: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb})

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                       DQNetwork.target_Q: targets_mb,
                                                       DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")

