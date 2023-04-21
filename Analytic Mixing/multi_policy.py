#**********************************

#------------Imports--------------

#**********************************
import policy_lib               #Custom policies
import numpy as np 
np.set_printoptions(suppress = True) 
import seaborn as sns           #For plotting style
import graph_plotter            #Custom visualizer
import vector_grid_goal         #Custom environment
import gym                      #Support for RL environment.
import matplotlib.pyplot as plt
import time
import os
from matplotlib import cm
from matplotlib import interactive
from matplotlib.ticker import LinearLocator
import warnings
warnings.filterwarnings("ignore")

#********************************

#----------Settings--------------

#***********************************
run = {}

#---------Env Settings------------
run['env_map'] = np.array([[0,0,0,0,0],
                        [0,1,0,1,0],
                        [0,0,0,1,0],
                        [0,1,0,0,1],
                        [0,0,1,0,0]])
#run['env_name'] = 'FrozenLake-v1'
run['env_name'] = 'vector_grid_goal'                        
run['grid_dims'] = (len(run['env_map'][0]),len(run['env_map'][1]))
run['player_location'] = (0,0)
run['goal_location'] = (run['grid_dims'][0] - 1, run['grid_dims'][1] - 1)  
print(run['grid_dims'] ,  run['goal_location'])   

#----------Stochastic Controls--------------
run['seed'] = 6000

#------------Training Loop Controls-----------
run['n_episodes'] = 2000
run['max_steps'] = 30

#-----------Q_Policy Control---------
run['gamma'] = 0.83          #discount factor 0.9
run['lr'] = 0.3                 #learning rate 0.1

#-----------Random Policy Control---------------
run['max_epsilon'] = 1              #initialize the exploration probability to 1
run['epsilon_decay'] = 0.01        #exploration decreasing decay for exponential decreasing
run['min_epsilon'] = 0.001

#-------------Analytic Policy Control---------------
run["mip_flag"] = True
run["cycle_flag"] = True
run['analytic_policy_update_frequency'] = 500
run['random_endpoints'] = False
run['reward_endpoints'] = True
run['analytic_policy_active'] = True
run['analytic_policy_chance'] = 0.3

run['q_visuals'] = False

#-------------Visualization Controls------------------
sns.set(style='darkgrid')
run['visualizer'] = True
run['vis_steps'] = False          #Visualize every step
run['vis_frequency'] = 100           #Visualize every x episodes


#********************************

#-----Init the environment-------

#**********************************
env = vector_grid_goal.CustomEnv(grid_dims=run['grid_dims'], player_location=run['player_location'], goal_location=run['goal_location'], map=run['env_map'])
run['env'] = env


#*******************************

#-----Init Policies------------

#********************************

#Init Q Policy
Q_policy = policy_lib.Q_Policy(run)

#Init Random Policy
Random_policy = policy_lib.Random_Policy(env, run['seed'])

#Init Analytic Policy
Analytic_policy = policy_lib.Analytic_Policy(run)

#-------------Establish Policy Structure----------------
Q_policy_status = {
'name': 'Q',
'episode_dominant': False,
'action_dominant': False,
'episode_dominants_remaining': 0,
'action_dominants_remaining': 0,
'policy_object': Q_policy,
}


R_policy_status = {
'name': 'R',
'episode_dominant': False,
'action_dominant': False,
'episode_dominants_remaining': 0,
'action_dominants_remaining': 0,
'policy_object': Random_policy,
}


A_policy_status = {
'name': 'A',
'episode_dominant': False,
'action_dominant': False,
'episode_dominants_remaining': 0,
'action_dominants_remaining': 0,
'policy_object': Analytic_policy,
}

policy_list = [Q_policy_status, R_policy_status, A_policy_status]


#********************************

#---------Init Graphics----------

#********************************
#plotter = graph_plotter.Graph_Visualizer(run['grid_dims'][0], run['grid_dims'][1])


#********************************

#----------Train---------------

#********************************
#Take variables out of run dict for clarity
for m in [True,False]:
    run["analytic_policy_active"]=m

    n_episodes = run['n_episodes']
    max_epsilon = run['max_epsilon']             #initialize the exploration probability to 1
    epsilon = max_epsilon
    epsilon_decay = run['epsilon_decay']       #exploration decreasing decay for exponential decreasing
    min_epsilon = run['min_epsilon']
    max_steps = run['max_steps']
    env = run['env']
    analytic_policy_active = run['analytic_policy_active']
    analytic_policy_chance = run['analytic_policy_chance']
    default_policy = Q_policy

    reward_per_episode = []

    an_policy = {}
    q_policy = {}


    for e in range(n_episodes):
        #------------Reset Environment--------------
        state = env.reset()
        done = False
        episode_reward = 0
        
        
        prev_run = []
        prev_first = 0
        prev_last = 0
        

        #------------------------------------------------
        #----------DETERMINE DOMINANT POLICY-------------
        #------------------------------------------------
        #---------------Reset Random Policies------------
        for entry in policy_list:
            entry['action_dominant'] = False
            
        
        #----------------Chance of Analytic Policy--------------
        if analytic_policy_active and np.random.uniform(0,1) < analytic_policy_chance:
            #If analytic policy applies, apply it
            #print("Using analytic policy")
            
            A_policy_status['action_dominant'] = True
            analytic_policy_chance=analytic_policy_chance-0.001
                 
            # print(e,"episode number")
            # print(analytic_policy_chance)
            # input("enter")
        else:
            #-----------Chance of Random Policy--------------
            if np.random.uniform(0,1) < epsilon:
                
                R_policy_status['action_dominant'] = True           #Random policy is dominant
            
            #---------------Chance of Q_Policy-----------------
            else:
                Q_policy_status['action_dominant'] = True           #Q_policy is dominant
                

        for i in range(max_steps):
            
            
            #--------Determine dominant policy-----------
            active_policy = default_policy
            #print(policy_list)
            for entry in policy_list:
                if entry['action_dominant']:
                    active_policy = entry['policy_object']
            
            #----------GET ACTION FROM DOMINANT POLICY------------------
            action = active_policy.get_action(state)
            #print(active_policy)
            
            if action == 'NA':
                #print("Defaulting")
                action = default_policy.get_action(state)
                
            
            
            #------------RUN ACTION--------------------
            next_state, reward, done, _ = env.step(action)
            
            #For now, translate states to ints
            state = int(state)
            action = int(action)
            next_state = int(next_state)
            
            
            
            #--------------------------------------------
            #-------------Update Policies and Memory-----
            #--------------------------------------------
            Q_policy.update_policy(state, action, reward, next_state)                   
            Analytic_policy.update_known_info(state, action, reward, next_state)
            
            #print("Known info")
            #print(Analytic_policy.state_reward)
            #print(Analytic_policy.known_states)
            #print(Analytic_policy.known_edges)
            
            #print("Alg form")
            #Analytic_policy.convert_to_alg_form()
            #print(Analytic_policy.learned_info)
            
            #Step Visualizer
            if run['visualizer'] and run['vis_steps']:
                print('s:', state, 'a:', action, 'ns:',next_state, 'r:', reward,  'd:', done)
                #plotter.draw_plot(Analytic_policy.state_reward, Analytic_policy.known_states, Analytic_policy.known_edges)
                input("Press Enter")
            
            episode_reward += reward    #Increment reward
            state = next_state      #update state to next_state
            
            #If done, end episode
            if done: break
            
            '''
            #Print Q Table
            print("Q table")
            np.set_printoptions(precision=4)
            np.set_printoptions(floatmode = 'fixed')
            np.set_printoptions(sign = ' ')
            for i, row in enumerate(Q_policy.Q_table):
                row_print = "{0:<6}{1:>8}".format(i, str(row))
                #print(i, ':', row)
                print(row_print)
            '''
            #input("Stop")
            
            
        #------------------------------------------------------
        #--------------End of Episode Updates and Displays-----
        #------------------------------------------------------
        #Update reward record
        reward_per_episode.append(episode_reward) 
        if A_policy_status['action_dominant']==True:
            an_policy[e]=episode_reward
        if Q_policy_status['action_dominant']==True:
            q_policy[e]=episode_reward            
        #Decrease epsilon
        epsilon = max(min_epsilon, np.exp(-epsilon_decay*e))
        #print("epsilon", epsilon)
        
        #Decrease analytic solver chance
        #run['analytic_policy_chance'] = max(min_epsilon, np.exp(-epsilon_decay*e))
        
        if e % 10 and e != 0:
            #print("--------------Episode:", str(e), "-----------------")
            pass
        if run['visualizer'] and e% run['vis_frequency'] == 0 and e != 0:
            #Get greedy path
            path, action_list, Q_path_reward = Q_policy.greedy_run()
            print(path, action_list, Q_path_reward)
            
            #Get current analytic path
            recommended_path = Analytic_policy.get_path()
        
            #plotter.draw_plot(Analytic_policy.state_reward, Analytic_policy.known_states, Analytic_policy.known_edges, path, recommended_path)
            #time.sleep(0.5)
            #input("Pause")
        
        if run['analytic_policy_active'] and e % run['analytic_policy_update_frequency'] == 0 and e != 0:
            #print('ap')
            #Make the conversion to the proper form
            Analytic_policy.convert_to_alg_form()
            
            #Run the solver
            if len(Analytic_policy.learned_info) > 0:
                Analytic_policy.update_policy(run)
            
    
        

    #----------------------------------------
    #-----------End of Training Actions------
    #----------------------------------------
    running_avg = []
    window = 1
    for point in range(window, len(reward_per_episode)):
        running_avg.append(np.mean(reward_per_episode[point-window: point]))
            
    #plt.plot(running_avg[1000:1250])
    #plt.scatter(list(an_policy),list(an_policy.values()))
    plt.scatter(list(q_policy),list(q_policy.values()))
    plt.legend(["Analytical Crossover","Regular Q"])
    plt.show()
    
plt.title("Rewards vs Episode")
plt.savefig(os.path.join('output_images','rewards' +"croosover"+'.png'))
plt.show()        
        
