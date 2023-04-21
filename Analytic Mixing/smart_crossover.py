#Sam, modified by Kyle Norland
#Modified 10/10/22

from xmlrpc.client import boolean
import numpy as np
import cvxpy as cvx
import random
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import gurobipy as gp
from datetime import datetime
import time



class Smart_Crossover:

    def __init__(self,learners_info, run):
        print("-"*20)
        print("In smart crossover code")
        print("-"*20)
        self.learned_info=learners_info
        self.run = run
        
        self.mip_flag = run['mip_flag']
        self.cycle_flag = run['cycle_flag']
        
        #Generate the ensemble.
        self.ensemble = self.ensemble_learned_info()
        
        #print("Ensembled learned info")
        #NOTE: Ensemble unlearned will blow up.
        #print(self.ensemble)
        #Graph it
        self.graph = nx.DiGraph()
        
        #Prepare
        rew_matrix,source_state,sink_state = self.prep()
        self.source_state = source_state
        self.sink_state = sink_state
        #Check that everything is there:
        print("Index of source and sink", source_state, "", sink_state)
        
        self.solution=[]
        condition=True
        cycle = []
        self.cycle=[]
        self.n_iter = 0
        self.value=0
       
        
        while condition:
        
            self.solution,cycle = self.optimize_mip(rew_matrix,source_state,sink_state,cycle)
            self.n_iter+=1
            if len(cycle)==0:
                condition=False
        
        
    def ensemble_learned_info(self):
        #Collect globally known data.
        total_info = {}
        known_states = []
        state_reward = {}
        edges = []
        not_known = []
        known_not_exist = []
        
        
        for learner in self.learned_info.values():
            #print("learner")
            #print(learner)
            #Add known states
            known_states.extend(learner['known_states'])
            
            #Add known rewards (deterministic)
            for state, reward in learner['state-reward'].items():
                state_reward[state] = reward
            
            #Add known edges
            edges.extend(learner['edges_exist'])
        
        #Only unique states
        known_states = list(set(known_states))
        known_states.sort()
        
        #Figure out known not edges.
        for from_state in known_states:
            temp_not = []
            temp_exists = []
            trans_counter = 0
            for to_state in known_states:
                if (from_state, to_state) in edges:
                    trans_counter += 1
                    #temp_exists.append((from_state, to_state))
                else:
                    temp_not.append((from_state, to_state))
            #If transitions greater or equal to actions, can eliminate others. (Wrong, change to reflect having tried all actions)
            if trans_counter >= 4: #Hardcoded, CHANGE
                known_not_exist.extend(temp_not)
                
            
            
            
        #Figure out new not_known edges
        not_known = [(from_state, to_state) for from_state in known_states for to_state in known_states if (from_state, to_state) not in edges]
        not_known = [transition for transition in not_known if transition not in known_not_exist]
        not_known = list(set(not_known))        
        not_known.sort()



        edges = list(set(edges))        
        edges.sort()         
        
        
        total_info={"state-reward":state_reward,"edges":edges,"not_known":not_known, "known_states": known_states, "known_not_exist": known_not_exist}
        
        
        print("State rewards")
        print(total_info['state-reward'])
        '''
        print("Known states")
        print(known_states)
        print("Edges")
        print(edges)
        
        print("Not known")
        
        #total_info['not_known'].sort(key=lambda x:x[0])
        print(not_known)
        print()
        '''

          
        '''
        total_info={}
        edges={}
        unknown_exist=[]

        for learner in self.learned_info:
            #Edges has visited states?
            edges[learner]=list(self.learned_info[learner]["state-reward"])
            print("Edges for learner")
            print(edges[learner])
            
            for state in self.learned_info[learner]["state-reward"]:
                if state not in list(total_info):
                    total_info[state]=self.learned_info[learner]["state-reward"][state]
                    
            #Is there a check for if another state has learned connections that are not known to another?        
            for unknowns in self.learned_info[learner]["not_known"]:
                if unknowns not in unknown_exist:
                    unknown_exist.append(unknowns)
                    
        total_info={"state-reward":total_info,"edges":edges,"not_known":unknown_exist}
        print("Not known")
        total_info['not_known'] = list(set(total_info['not_known']))        
        total_info['not_known'].sort()
        #total_info['not_known'].sort(key=lambda x:x[0])
        print(total_info['not_known'])
        print()
        #print("State rewards")
        #print(total_info['state-reward'])
        #print("Edges")
        #print(total_info['edges'])
        '''
        
        return total_info
        
        
    def color_nodes(self,g):
        nodes_with_reward=[]
        nodes_without_reward=[]
        color = []
        for node in g.nodes:
            if self.ensemble["state-reward"][node]>0:
                nodes_with_reward.append(node)
                color.append("red")
            else:
                nodes_without_reward.append(node)
                color.append("blue")
        
        
        return color
    def color_edges(self,g):
        
        edges_color = []
        for edge in g.edges:
            if edge in self.ensemble["not_known"]:
                edges_color.append('gray')
            else:
                edges_color.append('black')
        return edges_color
        
        
    def prep(self):

        sns.set(style='darkgrid')
        '''
        dict_states = list(self.ensemble["edges"].values())
        #print("Dict states")
        #print(dict_states)
        new=[]
        for i in dict_states:
            for j in i:
                new.append(j)
        
        new.sort()
        array_states=[*set(new)]
        print("Array states")
        print(array_states)
        
        #Randomly generated end of edges?
        array_edges=[(i[j],i[j+1]) for i in dict_states for j in range(len(i)-1)]
        '''
        
        #print("Array edges")
        #print(array_edges)
        #array_edges.sort(key=lambda x:x[0])
        
        
        
        #-------------------------------------------
        #---------Add known edges and nodes---------
        #-------------------------------------------
        

        array_states = self.ensemble['state-reward']
        array_edges = self.ensemble['edges']
        
        '''
        array_edges = list(set(array_edges))        
        array_edges.sort()
        
        print("Known states")
        print(array_states)
        print("Known edges")
        print(array_edges)
        '''
        self.graph.add_nodes_from(self.ensemble['known_states'])
        #self.graph.add_edges_from(self.ensemble['edges'])
        
        #self.graph.add_nodes_from(array_states)
        self.graph.add_edges_from(array_edges)
        
        '''
        #add rewards to nodes in graph (Removed, no known use?)
        for node in self.graph.nodes:
            self.graph.nodes[node]["reward"]=self.ensemble["state-reward"][node]
        '''

        #plot learned graph        
        #plt.figure(figsize=(20,10))
        color = self.color_nodes(self.graph)
        #nx.draw(self.graph,node_color=color,with_labels=True)
        #plt.show()
        
        
        #------------------------------------------------------------
        #-----------------Declare Start and Finish Nodes-------------
        #------------------------------------------------------------
        #randomize source
        choose_learner = random.choice(list(self.learned_info))
        #source_state = list(self.learned_info[choose_learner]["state-reward"])[0]
        #print("source learner")
        #print(list(self.learned_info[choose_learner]["state-reward"]))
        #print(list(self.learned_info[choose_learner]["state-reward"])[0])
        #randomize sink
        #!!! Will sometimes cause infinite loop
        
        #------------------------------
        #Now using first and lasts from recorded runs.
        #source_state = self.learned_info[random.choice(list(self.learned_info))]['prev_run'][0][0]
        #sink_state = self.learned_info[random.choice(list(self.learned_info))]['prev_run'][-1][0]
        
        #Use zero as a start
        source_state = 0 #self.learned_info[random.choice(list(self.learned_info))]['prev_first']
        
        if self.run['random_endpoints']:
            sink_state = random.choice(self.ensemble['known_states'])
            #self.learned_info[random.choice(list(self.learned_info))]['prev_last']
            #print("OG Source", source_state, "OG sink", sink_state)
            
            while sink_state == source_state:
                sink_state = random.choice(self.ensemble['known_states'])
        

        if self.run['reward_endpoints']:
            print("Chose reward endpoints")
            #Grab all reward spots
            reward_squares = []
            #print('Ensemble', self.ensemble['state-reward'])
            for key, value in self.ensemble['state-reward'].items():
                #print(key, value)
                if value > 0:
                    reward_squares.append(key)
            #print("Rewards Squares", reward_squares)
            if len(reward_squares) > 0:
                #sink_state = random.choice(reward_squares)
                sink_state = max(reward_squares)
            else:
                sink_state = random.choice(self.ensemble['known_states'])
                
                while sink_state == source_state:
                    sink_state = random.choice(self.ensemble['known_states'])
                
        #Print what source and sink are chosen
        print("Source", source_state, "sink", sink_state)
        
        
        

        
        '''
        sink_state = source_state
        while sink_state == source_state:
            choose_learner = random.choice(list(self.learned_info))
            sink_state = list(self.learned_info[choose_learner]["state-reward"])[-1]
        '''

            
        #print("sink learner")
        #print(list(self.learned_info[choose_learner]["state-reward"]))
        #print(list(self.learned_info[choose_learner]["state-reward"])[-1])
        
        color[list(self.graph.nodes).index(source_state)]="green"
        color[list(self.graph.nodes).index(sink_state)]="brown"
        
        #Make the adjacency matrix out of the previously added edges.
        adj_matrix = np.array(nx.attr_matrix(self.graph,rc_order=self.graph.nodes))
        #print("adjacency matrix")
        #print(adj_matrix)
        
        #Convert the reward dictionary to a useful form
        #print("state reward")
        #print(self.ensemble['state-reward'])
        
        #Put in order
        ordered_sr = list(zip(self.ensemble['state-reward'].keys(), self.ensemble['state-reward'].values()))
        ordered_sr.sort()
        #print("Rewards")
        #print(ordered_sr)
        
        #-------------------------------------------
        #-----------Create Reward Matrix-----------
        #-------------------------------------------
     
        #First part of reward matrix (add major negatives to disincentivize known non-existent edges) TODO
        rew_matrix = np.zeros(adj_matrix.shape)
        
        #Add known reward categories.
        for row_index in range(len(adj_matrix)):
            for col_index in range(len(adj_matrix)):
                #Add known rewards (between nodes)
                if adj_matrix[row_index][col_index]==1:
                    rew_matrix[row_index][col_index]=ordered_sr[col_index][1]
                
                #Known edge reward: Known edge, no reward.
                elif rew_matrix[row_index][col_index]==0 and adj_matrix[row_index][col_index]==1:
                    rew_matrix[row_index][col_index]=-5
                
                #Potential edge rewards (-2)
                elif rew_matrix[row_index][col_index]==0 and adj_matrix[row_index][col_index]==0:
                    rew_matrix[row_index][col_index]=-10
                    
                #Add known non-existent reward penalty
                else: 
                    rew_matrix[row_index][col_index]=-100
                    
              
        #Where the not known edges are added.
        #self.graph.add_edges_from(self.ensemble['not_known'])
        
        #If altered, make it -2 reward for nonexistent edges
        #altered = np.array(nx.attr_matrix(self.graph,rc_order=self.graph.nodes))
        
        #print("altered matrix")
        #print(altered)
        #print(len(altered[0]))
        '''
        print("base reward matrix")
        print(rew_matrix)
        print(len(rew_matrix[0]))
        '''       
        
        #print("reward matrix")
        #print(rew_matrix)

        #plt.figure()
        
        #edges_color = self.color_edges(self.graph)
        #nx.draw(self.graph,node_color=color,with_labels=True,edge_color=edges_color)
        #plt.savefig("new_graph.png")
        
        #print("Index check")
        #print(self.ensemble['known_states'])
        self.source_state = self.ensemble['known_states'][self.ensemble['known_states'].index(source_state)]
        self.sink_state = self.ensemble['known_states'][self.ensemble['known_states'].index(sink_state)]
        #self.source_state = array_states[array_states.index(source_state)]
        #self.sink_state = array_states[array_states.index(sink_state)]
        print("Start and end states: ", self.source_state, self.sink_state)
        
        return rew_matrix, self.ensemble['known_states'].index(source_state), self.ensemble['known_states'].index(sink_state)

    def optimize_mip(self,rew_matrix,source_state,sink_state,cycle):
        
        #Extra computational time from making multiple adjacency matrices?
        self.graph.remove_edges_from(list(nx.selfloop_edges(self.graph)))
        start = time.time()
        rew_matrix=rew_matrix
        adj_matrix = np.array(nx.attr_matrix(self.graph,rc_order=self.graph.nodes))
        #create variables
        if self.mip_flag:
            x=cvx.Variable(adj_matrix.shape,boolean=True)
        else:
            x=cvx.Variable(adj_matrix.shape)
        
        #create constraints, x is binary
        constraints = [x<=1,x>=0]
        #constraints += [theta<=1,x>=0]
        
        #Could this be set to <= 0 for better performance?
        for row_index in range(len(adj_matrix)):
            for col_index in range(len(adj_matrix)): 
                if adj_matrix[row_index][col_index]==0:
                    #known to not exist edges are set to 0s
                    constraints += [x[row_index][col_index]==0]
                
                #constraint of only allowing directed edges
                #!!!!!!!!!!!!!
                #TODO: Fix, this one doesn't make sense.
                if self.mip_flag:
                    constraints += [x[row_index][col_index]+x[col_index][row_index]<=1]

        #Sum of inflow is the sum of outflow
         
        vector = np.zeros(len(adj_matrix))
        vector[sink_state]=-1
        vector[source_state]=1
        constraints += [cvx.sum(x,axis=1)-cvx.sum(x,axis=0)==vector]
        
        #force only one  entry and only one exit 
        if self.mip_flag:
            constraints += [cvx.sum(x,axis=0)<=1]
            constraints += [cvx.sum(x,axis=1)<=1]
            
        #if edge is activated it has to have another edge which is incident.
        # constraints += [cvx.sum(x,axis=1)+cvx.sum(x,axis=0)>=2-m*(1-theta)]
        # constraints += [cvx.sum(x,axis=1)+cvx.sum(x,axis=0)<=1+m*(theta)]        

        #Add constraints for found cycles:
        #Q: Can cycles be avoided with some constraints?
        if self.cycle_flag:
            for cycle_i in self.cycle: 
                index_nodes_cycle = [list(self.graph.nodes).index(node) for node in cycle_i]
                edges_cycle = [(index_nodes_cycle[i],index_nodes_cycle[i+1]) for i in range(len(index_nodes_cycle)-1)]
                edges_cycle.append((index_nodes_cycle[-1],index_nodes_cycle[0]))
                exp = 0
                for pair in edges_cycle:
                    exp+=x[pair[0]][pair[1]]
                num=len(cycle_i)
                constraints += [exp<= num-1]
                #print("cleaned cycle")

        #create obj_function 
        problem=cvx.Problem(cvx.Maximize(cvx.sum(cvx.multiply(rew_matrix,x))),constraints)
        if not self.mip_flag:
            prob_data,_,_ = problem.get_problem_data(cvx.SCS)
            print(prob_data["c"])
            print(prob_data["A"])
            print(prob_data["b"])
            input("here it is")    
        problem.solve("SCIPY")
        
        
       
        #solve
        #for writing model//works when unimodular
        
        #for solving MIP
        
        #problem.solve("CPLEX","model.lp",verbose=True)
        
        if problem.status == "infeasible":
            for cons in constraints:
                print("Infeasible")
                print("Dual value")
                print(cons.dual_value)
                return
        else:
            self.value = problem.value
            print(time.time()-start)
            new_pairs=[]
            for row_index in range(len(x.value)):
                for col_index in range(len(x.value)):
                    if x.value[row_index][col_index]==1:
                        new_pairs.append((list(self.graph.nodes)[row_index],list(self.graph.nodes)[col_index]))
            #print(x.value)
            g=nx.DiGraph()
            g.add_edges_from(new_pairs)
            
            #------------------Run to solve cycles---------------------
            if self.cycle_flag:
                #find cycles:
                cycles=sorted(nx.simple_cycles(g))
                #print("found cycles",cycles)
                
                if len(cycles)>0:
                    for cycle in cycles:
                    #    print(cycle)
                        
                        if cycle not in self.cycle:
                            self.cycle.append(cycle)
                            #print("added cycle")
                    #print("found cycles:",self.cycle)
                    return list(g.edges),cycles
                
                else: 
                    
                    # nodes = g.nodes
                    # new_colors = []
                    # color = self.color_nodes(g)
                    
                    # #color[list(g.nodes).index(list(self.graph.nodes)[source_state])]="green"
                    # #color[list(g.nodes).index(list(self.graph.nodes)[sink_state])]="brown"
                    # edges_color = self.color_edges(g)
                    # print(nodes)
                    # print(list(self.graph.nodes))
                    # for node in nodes:
                    #     new_list=(list(self.graph.nodes))
                    #     new_colors.append(color[new_list.index(node)])
                    
                    # plt.figure(figsize=(20,10))
                    
                    # nx.draw(g,with_labels=True,node_color=new_colors)
                    # plt.show()
                    # print(g.edges)
                    
                    #ordered_path =nx.dijkstra_path(g,list(self.graph.nodes)[source_state],list(self.graph.nodes)[sink_state])
                    #ordered_path = [(ordered_path[i],ordered_path[i+1]) for i in range(len(ordered_path)-1)]
                    return list(g.edges),cycles
                    
            else:
                cycles = []
                ordered_path =nx.dijkstra_path(g,list(self.graph.nodes)[source_state],list(self.graph.nodes)[sink_state])
                ordered_path = [(ordered_path[i],ordered_path[i+1]) for i in range(len(ordered_path)-1)]
                
                # nodes = g.nodes
                # new_colors = []
                # color = self.color_nodes(g)
                
                # #color[list(g.nodes).index(list(self.graph.nodes)[source_state])]="green"
                # #color[list(g.nodes).index(list(self.graph.nodes)[sink_state])]="brown"
                # edges_color = self.color_edges(g)
                # print(nodes)
                # print(list(self.graph.nodes))
                # # for node in nodes:
                # #     new_list=(list(self.graph.nodes))
                # #     new_colors.append(color[new_list.index(node)])
                
                '''
                #To show graph
                plt.figure(figsize=(20,10))
                
                nx.draw(g,with_labels=True)
                plt.show()
                print(g.edges)
                '''
                
                
                return ordered_path,cycles      

    
    
