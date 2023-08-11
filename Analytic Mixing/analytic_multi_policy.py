#**********************************

#------------Imports--------------

#**********************************
import analytic_policy_lib               #Custom policies
import numpy as np 
rng = np.random.default_rng(12345)
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
import json
import copy
import random
import analytic_grapher

import warnings
warnings.filterwarnings("ignore")


#--------------------------------------------------
#---------------General Functions------------------
#--------------------------------------------------
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def get_discrete_state(self, state, bins, obsSpaceSize):
    #https://github.com/JackFurby/CartPole-v0
    stateIndex = []
    for i in range(obsSpaceSize):
        stateIndex.append(np.digitize(state[i], bins[i]) - 1) # -1 will turn bin into index
    return tuple(stateIndex)
   
def init_environment(env_config):
    env_name = env_config['env_name']
    
    if env_name == 'FrozenLake-v1':
        env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=env_config['is_slippery'])
        #env = gym.make("FrozenLake-v1", is_slippery=False)
    
    if env_name == 'vector_grid_goal':
        grid_dims = (7,7)
        player_location = (0,0)
        goal_location = (6,6)
        custom_map = np.array([[0,1,1,1,0,0,0],
                                [0,0,1,0,0,1,0],
                                [0,0,0,0,0,0,1],
                                [0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,1],
                                [0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0]])
    
        #env = vector_grid_goal.CustomEnv(grid_dims=grid_dims, player_location=player_location, goal_location=goal_location, map=custom_map)
        env = vector_grid_goal.CustomEnv(grid_dims=env_config['grid_dims'], player_location=env_config['player_location'], goal_location=env_config['goal_location'], map=env_config['env_map'])

    return env                
   

#*******************************************

#----------Default Settings-----------------

#********************************************
default_run = {}

 


#---------Env Settings------------
#Env Config
default_run['env_config'] = {}
default_run['env_config']['env_map'] = np.array([[0,0,0,0,0],
                                                [0,0,0,0,0],
                                                [0,1,0,1,0],
                                                [0,1,0,0,1],
                                                [1,0,1,0,0]])
#default_run['env_config']['env_name'] = 'FrozenLake-v1'
default_run['env_config']['env_name'] = 'vector_grid_goal'                        
default_run['env_config']['grid_dims'] = (len(default_run['env_config']['env_map'][0]),len(default_run['env_config']['env_map'][1]))
default_run['env_config']['player_location'] = (0,0)
default_run['env_config']['goal_location'] = (default_run['env_config']['grid_dims'][0] - 1, default_run['env_config']['grid_dims'][1] - 1)  
print(default_run['env_config']['grid_dims'] ,  default_run['env_config']['goal_location'])   


#----------Stochastic Controls--------------
default_run['seed'] = 6000

#------------Training Loop Controls-----------
default_run['n_episodes'] = 2000
default_run['max_steps'] = 30

#-----------Q_Policy Control---------
default_run['gamma'] = 0.95         #discount factor 0.9
default_run['lr'] = 0.1               #learning rate 0.1

#-----------Random Policy Control---------------
default_run['max_epsilon'] = 1              #initialize the exploration probability to 1
default_run['epsilon_decay'] = 0.01 #0.01        #exploration decreasing decay for exponential decreasing
default_run['min_epsilon'] = 0.001

#-------------Analytic Policy Control---------------
default_run["mip_flag"] = True
default_run["cycle_flag"] = True
default_run['analytic_policy_update_frequency'] = 500
default_run['random_endpoints'] = False
default_run['reward_endpoints'] = True
default_run['analytic_policy_active'] = True
default_run['analytic_policy_chance'] = 0.3 #0.3

default_run['q_visuals'] = False

#-------------Visualization Controls------------------
sns.set(style='darkgrid')
default_run['visualizer'] = True
default_run['vis_steps'] = False          #Visualize every step
default_run['vis_frequency'] = 100           #Visualize every x episodes

default_run['output_path'] = 'analytic_policy_output'
if not os.path.exists(default_run['output_path']):
    os.makedirs(default_run['output_path'], exist_ok = True) 

default_run['output_dict'] = {} 

#------------------------------------
#------------Generate Experiment-----
#------------------------------------
def gen_analytic_policy_active(default_run):
    #Description: Test epsilon setting 0 vs 50%
    #Try with two epsilon settings and 10 different random seeds.
    #epsilon_maxes = [0, 0.05, 0.1, 0.25, 0.5]
    
    a_policy_active_list = [True, False]
    a_policy_chances = [0.3]
    random_seeds = rng.integers(low=0, high=9999, size=10)
    
    new_experiment = {'runs':[]}
    new_experiment['generation_time']= time.time()
    new_experiment['variables'] = ['analytic_policy_active', 'analytic_policy_chance', 'np_seed']
    new_experiment['varied_quantities'] = ['analytic_policy_active', 'analytic_policy_chance']
    
    #color_list = ['green', 'blue', 'red', 'yellow', 'orange', 'brown']
    #color_counter = 0
    for policy_active in a_policy_active_list:
        for policy_chance in a_policy_chances:
            for seed in random_seeds:
                new_run = copy.deepcopy(default_run)
                
                #Adjusted Settings
                new_run['analytic_policy_active'] = policy_active
                new_run['analytic_policy_chance'] = policy_chance
                
                #Standard Settings
                new_run['np_seed'] = seed
                new_run['env_seed'] = seed
                new_run['python_seed'] = seed
                #new_run['color'] = color_list[color_counter]
                new_run['label'] = "Analytic Policy_Active: " + str(policy_active)
                
                print("Settings: ", new_run['analytic_policy_active'], ": ", seed)
                
                #Add run to experiment
                new_experiment['runs'].append(copy.deepcopy(new_run))
            
            #color_counter += 1   
    print("Returning new experiment")
    return new_experiment  
    
#----------------------------------------------
#----------------Generate The Experiment-------
#----------------------------------------------
#Generate or Load Experiment
folder_mode = False
generate_mode = True

experiment = {'runs': []}

if generate_mode:
    #Generate experiment
    experiment = copy.deepcopy(gen_analytic_policy_active(default_run))

    #Save experiment to folder
    experiment_name = str(experiment['generation_time']) + '.json'
    
    if not os.path.exists('saved_experiments'):
        os.makedirs('saved_experiments', exist_ok = True) 
    
    with open(os.path.join('saved_experiments', experiment_name), 'w') as f:
        json.dump(experiment, f, cls=MyEncoder)




#----------------------------------------------
#---------------Run The Experiment-------------
#----------------------------------------------
for run in experiment['runs']:
    #Start Timer
    #print("Run")
    #print(run)
    run['run_start_time'] = time.time()
    
    #Init the environment
    env = init_environment(run['env_config'])

    #*******************************

    #-----Init Policies------------

    #********************************

    #Init Q Policy
    Q_policy = analytic_policy_lib.Q_Policy(run, env)

    #Init Random Policy
    Random_policy = analytic_policy_lib.Random_Policy(env, run['seed'])

    #Init Analytic Policy
    Analytic_policy = analytic_policy_lib.Analytic_Policy(run, env)

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
    #run["analytic_policy_active"]=m
    n_episodes = run['n_episodes']
    max_epsilon = run['max_epsilon']             #initialize the exploration probability to 1
    epsilon = max_epsilon
    epsilon_decay = run['epsilon_decay']       #exploration decreasing decay for exponential decreasing
    min_epsilon = run['min_epsilon']
    max_steps = run['max_steps']
    #env = run['env']
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
        
        if e % 200 and e != 0:
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
    #-----------End of Run Actions------
    #----------------------------------------
    '''
    running_avg = []
    window = 100
    for point in range(window, len(reward_per_episode)):
        running_avg.append(np.mean(reward_per_episode[point-window: point]))
    '''
    
    #plt.plot(running_avg[1000:1250])
    #plt.scatter(list(an_policy),list(an_policy.values()))
    #plt.scatter(list(q_policy),list(q_policy.values()))
    #plt.plot(list(q_policy),list(q_policy.values()))
    #plt.legend(["Analytical Crossover","Regular Q"])
    #plt.show()
        
    #plt.title("Rewards vs Episode")
    #plt.savefig(os.path.join('output_images','rewards' +"croosover"+'.png'))
    #plt.show()
    
    #Save to output dict.
    run['output_dict']['reward_per_episode'] = reward_per_episode

        
#--------------------------------------------------------
#--------------End of Experiment Processing--------------
#--------------------------------------------------------

#Save
#save_name = str(round(time.time())) + "_json_output.json"

out_folder_name = str(experiment['generation_time']) + '_experiment'
os.makedirs(os.path.join('results', out_folder_name), exist_ok=True)

save_name = "json_output.json"
with open(os.path.join('results', out_folder_name, save_name ), 'w') as f:
    json.dump(experiment, f, cls=MyEncoder)    

#Save a text file with the changed variables
save_name = '+'.join(experiment['variables'])
file_path = os.path.join('results', out_folder_name, save_name)
with open(file_path, 'w') as f:
    pass

print("Processing Visuals")
analytic_grapher.Analyze_Experiment(experiment)
    
        
'''        
plt.legend(["With Analytical Crossover","Regular Q"])
plt.title("Rewards vs Episode")
plt.savefig(os.path.join('output_images','rewards' +"croosover"+'.png'))
plt.show()        
'''                
