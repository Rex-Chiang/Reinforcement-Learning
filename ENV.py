import numpy as np
from scipy.signal import argrelmax
import matplotlib.pyplot as plt

global Num #numbers of antennas
Num= 20

#get initial position    
def Random():
    X= np.arange(1.5, 11.5, .5) #initial position    
    for i in range(Num):
        X[i]+= np.random.rand() #let the initial position have little different
    return np.round(X, 3)
# move antennas position based on action
def Move(observation_X, action, limit):
    # move 0.01 each time
    if action < Num:
        observation_X[action]+= 0.01
        
    elif action > Num-1:
        observation_X[action-Num]-= 0.01
    
    observation_X= intervalX(observation_X, limit) #let position interval "< limit", "> 0.5"
    
    return observation_X
#let position interval "< limit", "> 0.5"
def intervalX(position,limit):
    Max_position= 15
    Min_position= 1
    new_position= []
    for i in position:
        i= round(i,3)
        new_position.append(i)
    for i in range(0,len(new_position)- 1):
        if (new_position[i+1]- new_position[i] <= 0.5):
            new_position[i+1]= new_position[i]+ 0.5
        elif (new_position[i+1]- new_position[i] > limit):
            new_position[i+1]= new_position[i]+ limit
        if new_position[i] < Min_position:
            new_position[i]= Min_position
        elif new_position[i] > Max_position:
            new_position[i]= Max_position
    return np.array(new_position)
# object function
def BeamFunc(X):
    final_list1= []
       
    theta= np.arange(0, np.pi, 0.001) #Angle for Num elements, from 0-180
    
    for i in range(0, len(theta)):
        final_function1= 0
        BeamFunc= trans(2*np.pi*X*np.cos(theta[i]), 2*np.pi*X*np.cos(theta[i]), Num)
        
        for j in range(Num):
            final_function1+= BeamFunc[j]
            
        final_list1.append(abs(final_function1)/ Num)

    final_list1= 20*np.log10(final_list1) #Nomalize the pattern   
    
    return final_list1
#To transmit the input into complex form
def trans(a, b, element_nums):
    Out=[]
    for i in range(element_nums):
        out= complex(np.cos(a[i]), np.sin(b[i]))
        Out.append(out)
    Out= np.array(Out)
    return Out
# get the maximal and minimal index of the wave value
def Get_limit_point(List,theta):
    Max= max(List)
    Min= min(List)
    Max_index= list(List).index(Max)
    Min_index= list(List).index(Min)
    Max_angle= theta[Max_index]*180/np.pi
    Min_angle= theta[Min_index]*180/np.pi
    
    print("Max point:", round(Max,4), "Min point:", round(Min,4))
    print("Max point index:", Max_index, "Min point index:", Min_index)
    print("Max point theta:", Max_angle, "Min point theta:", Min_angle)
# get each peak of the waves (red dot)
def Get_Maxdot(List):
    peak= argrelmax(List, order=1) #(peaks index)
    for i in range(len(List)):
        if i not in peak[0]:
            List[i]= None
    return List
# get maximal side-lobe (instead of the max peak)
def Get_Max_Side_lobe(List):
    SideLobe_Value=[]
    peak= argrelmax(List, order=1) #(peaks index)
    peak= list(peak[0])
    peak.remove(1571) #remove the max peak
    for i in peak:
        SideLobe_Value.append(List[i])
    Max_Side_lobe=max(SideLobe_Value) 
    return Max_Side_lobe
# judge the reward value
def Get_reward(s, s_):
    # if next_MSL <= now_MSL, get reward
    if s_ <= s:
        reward= 1
    else:
        reward= -1
    
    return reward
# plot the instant waves
def Plot(List):
    theta= np.arange(0, np.pi, 0.001) #Angle for Num elements, from 0-180
    plt.plot(theta*180/np.pi, List, 'b--') #plot waves (blue line)
    plt.plot(theta*180/np.pi, Get_Maxdot(List), 'ro') #plot peaks (red dot)    
    plt.xlabel('theta')
    plt.ylabel('dB')
    plt.axis([0,180,-30,0])
    plt.grid(color= 'g', linestyle= '--', linewidth= 1, alpha= 0.4)
    plt.show()
# reset the observation, get initial position
def reset():
    Pos= Random()
    Pos= intervalX(Pos, 1) #let position interval "< limit", "> 0.5"
    return Pos
#check if the the goal reached
def Done(MSL):
    if MSL < -15:
        return True
# main RL activity
def step(action, observation):

    #Plot(BeamFunc(Pos))
    MSL= Get_Max_Side_lobe(BeamFunc(observation)) #observation= antennas position
    
    New_Observation= Move(observation, action, 1) #get next observation
    
    MSL_= Get_Max_Side_lobe(BeamFunc(New_Observation)) #get next maximal side-lobe
   
    reward= Get_reward(MSL, MSL_) #get reward
    
    done= Done(MSL_) #check if the the goal reached
    
    Plot(BeamFunc(New_Observation)) #plot the instant waves
    
    return New_Observation, reward, done, MSL_