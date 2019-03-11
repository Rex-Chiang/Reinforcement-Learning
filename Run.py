import ENV
import numpy as np
import matplotlib.pyplot as plt
from RFL import DeepQNetwork
from matplotlib import animation
#import collections

#def init(): #to plot the GIF
#    line.set_ydata(Func.BeamFunc(xxx))
#    return line,

def update(i): #to plot the GIF
    Wave.set_ydata(ENV.BeamFunc(Plot_data[i]))
    return Wave,

def run():
    step = 0
    #compare = lambda x, y: collections.Counter(x) == collections.Counter(y)    
    for episode in range(100):
        # initial observation
        observation = ENV.reset() #get the initial observaion (position)
        rewards= [] #record rewards in the episode        
        Plot_data= [] #record the observation (position) to plot
        Init_X= observation #the initial observation (position)
        
        while True:
            # RL choose action based on observation
            action = RL.Choose_Actions(observation)

            # RL take action and get next observation, reward, next Maximal side-lobe
            observation_, reward, done, MSL_ = ENV.step(action, observation)
            
            rewards.append(reward)
            print("Episode:", episode)
            print("Step", step)
            print("Max Side Lobe:", MSL_)
            print("Position:", observation)
            print("reward:", sum(rewards))           
            print('*'*40)
            #if compare(observation, observation_):
            #    print("SAME")
            #else:
            #    print("Different")
            
            RL.store_transition(observation, action, reward, observation_) #store the data
            
            if (step > 200) and (step % 5 == 0): #當步數大於200後, 每5步學習一次
                RL.learn()
                        
            observation = observation_# swap observation
            
            Plot_data.append(observation)
            
            # break while loop when end of this episode
            if done:
                break
            step += 1

    print('Complete')
    
    return Plot_data, Init_X
    
    
    
if __name__ == "__main__":
    RL = DeepQNetwork(40, 20,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    
    Plot_data, Init_X= run()
    
    fig, ax = plt.subplots()
    theta= np.arange(0, np.pi, 0.001)
    
    Wave0= ax.plot(theta*180/np.pi, ENV.BeamFunc(Init_X), 'r--') #plot the initial wave (red line)
    Wave,= ax.plot(theta*180/np.pi, ENV.BeamFunc(Init_X), 'b--') #update the wave (blue line), ","為了更新值時的類型匹配
        
    ax.set_xlabel(r'$theta$')
    ax.set_ylabel(r'$dB$')

    ax.axis([0,180,-30,0])
    ax.grid(color= 'g', linestyle= '--', linewidth= 1, alpha= 0.4)
    
    
    ani= animation.FuncAnimation(fig, update, frames=len(Plot_data), interval=100, blit=False) #get dymanic Waves
    ani.save('4.gif', writer='imagemagick') #save the GIF
    
    plt.show()
        
    RL.plot_cost()#!/usr/bin/env python3
