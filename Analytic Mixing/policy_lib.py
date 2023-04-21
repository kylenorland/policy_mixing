#Imports
import numpy as np
import gym                      #Support for RL environment.
import smart_crossover          #Solver: By Sam
import graph_plotter            #Visualization helper   
from datetime import datetime
import random             



class Q_Policy:
    def __init__(self, init_dict):
        print("Initializing Q_policy")
        #Init settings
        for k, v in init_dict.items():
            setattr(self, k, v)
        
        #Init the Q_table
        n_observations = self.env.observation_space.n
        n_actions = self.env.action_space.n
        self.Q_table = np.zeros((n_observations,n_actions))
    
    def get_action(self, state):
        return np.argmax(self.Q_table[state,:])
    
    def update_policy(self, state, action, reward, next_state):
        #Update Q_table
        self.Q_table[state, action] = self.Q_table[state, action] + self.lr*(reward + self.gamma*max(self.Q_table[next_state,:]) - self.Q_table[state, action])
    
        #Update parameters
        
    
    def greedy_run(self):
        #Finds the greedy path through a q_table by running env
    
        #Restart environment
        state = self.env.reset()
        done = False
        path = []
        actions = []
        total_reward = 0
        
        for i in range(20):
            action = np.argmax(self.Q_table[state,:])
            next_state, reward, done, _ = self.env.step(action)
        
            #Keep track of actions and states and rewards
            actions.append(action)
            path.append((int(state), int(next_state)))
            total_reward += reward
            state = next_state
            
            #Break if done
            if done:
                break
            
        return path, actions, total_reward
        
        #Returns the current greedy path


class Random_Policy:
        def __init__(self, env, seed):
            self.env = env
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            # env.action_space.np_random.seed(seed) 
            print("initializing random policy")
        
        def get_action(self, state):
            return self.env.action_space.sample()
        
class Analytic_Policy:
        def __init__(self, init_dict):
            #Init settings
            for k, v in init_dict.items():
                setattr(self, k, v)
            
            #Known info memory
            self.recommended_path = []
            self.policy = {}
            self.state_reward = {}
            self.known_states = []
            self.known_edges = {}
            self.san_dict = {}      #State action next dictionary.
            
            self.learned_info = {}     #Compatible format with solver.
            self.num_rewards = []

        def update_known_info(self, state, action, reward, next_state):
            #-----------Update state knowledge-------------
            if state not in self.known_states:
                self.known_states.append(state)
            if next_state not in self.known_states:
                self.known_states.append(next_state)
            
            #------------Update edge knowledge---------------
            if state in self.known_edges:
                if next_state not in self.known_edges[state]:
                    self.known_edges[state].append(next_state)
            else:
                self.known_edges[state] = [next_state]
                
            #---------Update reward knowledge---------------
            if next_state in self.state_reward:
                if reward > self.state_reward[next_state]:
                    self.state_reward[next_state] = reward
            else:
                self.state_reward[next_state] = reward
                
            #-----------Update state action next dictionary-------------
            if state in self.san_dict:
                if next_state not in self.san_dict[state]:
                    self.san_dict[state][next_state] = action
            else:
                self.san_dict[state] = {next_state: action}
            
            if len(self.num_rewards)==0 or self.num_rewards[-1]!=sum(list(self.state_reward.values())):
                
                self.num_rewards.append(sum(list(self.state_reward.values())))            
        
        def update_policy(self, run):
            #Calculates the analytically suggested path
            #Start timer to time solver
            start = datetime.now()
            #Run the crossover
            #print("Starting crossover")
            
            if len(self.num_rewards)>1 and self.num_rewards[-1]!=self.num_rewards[-2]: 
                self.num_rewards[-2]=self.num_rewards[-1]
                crossover=smart_crossover.Smart_Crossover(self.learned_info, run)
                
                print("Solution time: ", datetime.now() - start)

                print("Objective Value:",crossover.value)
                
                solution_path = crossover.solution
                #Update recommended path
                self.recommended_path = solution_path
                print("Solution path", solution_path)
            
            else:
                print("not re-computed")
            #Find path and also make policy dictionary
            sa_path = []
            
            for edge in self.recommended_path:
                if edge[0] in self.known_edges and edge[1] in self.known_edges[edge[0]]:
                    #Find the correct action in san_dict
                    action = self.san_dict[edge[0]][edge[1]]  
                else:
                    #Choose a random action from action space
                    print("taking random action")
                    action = self.env.action_space.sample()
                    
                sa_path.append((edge[0], action))
                self.policy[edge[0]] = action
                
            print("SA_Path")
            print(sa_path)
            print("New Policy")
            print(self.policy)
            
        def get_action(self, state):
            if state in self.policy:
                return self.policy[state]
            else:
                return 'NA'
            
        def get_path(self):
            return self.recommended_path
        
        def convert_to_alg_form(self):
            #Create an entry within self.learned info for compatibility
            new_info = {}
            self.learned_info = {0:new_info}

            #Set up structure
            new_info['state-reward'] = self.state_reward
            new_info['known_states'] = self.known_states
            new_info['not_known'] = []
            new_info['edges_exist'] = []
            
            #Convert to that structure
            #Edges
            for key, edge_group in self.known_edges.items():
                for end_point in edge_group:
                    new_info['edges_exist'].append((key, end_point))
            
            #Not known states: Needs fix, also slow(Talk with Sam)
            new_info['not_known'] = [(from_state, to_state) for from_state in new_info['known_states'] for to_state in new_info['known_states'] if (from_state, to_state) not in new_info['edges_exist']]
            
            #By here, the new_info should be updated
            
            
            
            
            