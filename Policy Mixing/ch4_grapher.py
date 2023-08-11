#Author: Kyle Norland
#Date: 9-17-22
#Purpose: Secondary code to analyze agents.


#-------------------Imports-----------------------
#General
import numpy as np
import os
import time
import random
import time
from datetime import datetime
import json
import re
import matplotlib.pyplot as plt
import copy

def Analyze_Experiment(experiment):
    #Set up the figure   
    fig = plt.figure()
    ax = plt.subplot(111)

    #For output in folder, if meets conditions, graph all.
    total_run_rewards = {}
    num_runs = 1    #Special adjustment, averaging actually not necessary in this case.
    plot_names = []

    average_mode = False

    if average_mode:
        for iterator, entry in enumerate(experiment['runs']):    
            print("Run: ", iterator)
            
            #plt.plot(entry['output_dict']['reward_per_episode'])
            #plt.show()
            
            run = entry['output_dict']
            #avg_reward = [x['avg'] for x in run['stats']]
            reward_per_episode = run['reward_per_episode']

            
            running_avg = []
            window = 100
            for point in range(window, len(reward_per_episode)):
                running_avg.append(np.mean(reward_per_episode[point-window: point]))
                    
            
            
            #Make label
            set_name = ""
            set_name += "set_name"
            #set_name += "eps_max: " + str(entry['epsilon_max']) + " seed: " + str(entry['np_seed'])
            plot_names.append(set_name)
            
            #Generate equivalent number of timesteps for x axis
            #timesteps = [(x * pop_size * episode_len) for x in range(0,len(avg_reward))]
            
            
            ax.plot(running_avg, label=entry['label'], c=entry['color'])
            '''
            if iterator in total_run_rewards:
                temp_list = []
                temp_list = [a + b for a,b in zip(total_run_rewards[iterator], avg_reward)]  
                total_run_rewards[iterator] = temp_list[:]
                
            else:
                total_run_rewards[iterator] = avg_reward
            '''
            
        '''    
        #Plot each of the iterators
        for key, value in total_run_rewards.items():
            avg_value = [x / num_runs for x in value]
            #print("avg_value: ", avg_value)
            ax.plot(avg_value, label= plot_names[key], c=)
        '''
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.4,
                     box.width, box.height * 0.6])
            
        plt.title("Tester")
        plt.ylabel("Reward")
        plt.xlabel("# Generations")
        #plt.ylim(-0.1, 1.1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True, shadow=True)
        save_name = os.path.join('results', str(experiment['generation_time']) + '_experiment', 'avg_reward_vs_generation.pdf')
        plt.savefig(save_name, bbox_inches='tight')
        plt.show()   
        
    else:
        #Not average mode
        
        #Count unique pairs of variables.
        #Find first unique one, then get all of them.
        known_run_types = []
        
        for run in experiment['runs']:
            if run['run_type'] not in known_run_types:
                #Add it
                known_run_types.append(run['run_type'])
                
                #Add to list and sum others of that type.
                print("Known run types")
                print(known_run_types)
                
                #Sum with others
                sum_rpe = np.array([])
                #np.array(run['output_dict']['reward_per_episode'])
                num_same = 0
                label = run['label']
                print(label)
                
                #Add the
                for other_run in experiment['runs']:
                    #Compare with variables from other run. If same
                    if other_run['run_type'] == run['run_type']:
                        print("Found a match")
                        if sum_rpe.size > 0:
                            sum_rpe += np.array(other_run['output_dict']['reward_per_episode'])
                        else:
                            sum_rpe = np.array(other_run['output_dict']['reward_per_episode'])
                        
                        #Increment the number
                        num_same += 1
            
                #With all summed, plot average
                print(num_same, " of the same type")
                sum_rpe = sum_rpe.astype('float64')
                avg_rpe = sum_rpe / num_same
                
                #ax.plot(running_avg, label=entry['label'])
                #Running average window.
                running_avg = []
                window = 100
                for point in range(window, avg_rpe.size):
                    running_avg.append(np.mean(avg_rpe[point-window: point]))
                
                #Note: Running avg starts at window (Adjust plot)
                ax.plot(running_avg, label=label)
                
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.4,
                     box.width, box.height * 0.6])
            
        #plt.title("Tester")
        plt.title(experiment['plot_title'])
        plt.ylabel("Average Reward")
        plt.xlabel("# Generations")
        #plt.ylim(-0.1, 1.1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True, shadow=True)
        save_name = os.path.join('results', str(experiment['generation_time']) + '_experiment', 'avg_reward_vs_generation.pdf')
        plt.savefig(save_name, bbox_inches='tight')
        plt.show()           


if __name__ == "__main__":    
    #Set up the figure   
    fig = plt.figure()
    ax = plt.subplot(111)

    #For output in folder, if meets conditions, graph all.
    main_folder = "macro_GA_output"
    total_run_rewards = {}
    num_runs = 0
    plot_names = []

    experiment_mode = True
    if experiment_mode == True:
        for file_name in os.listdir(main_folder):
            if re.search(".*\.json", file_name):
                print(file_name)
                num_runs += 1
                
                with open(os.path.join(main_folder, file_name), 'r') as f:
                    experiment_list = json.load(f)
                    #print(experiment_list['run_date'])
                    
                for iterator, entry in enumerate(experiment_list):    
                    run = entry['output_dict']
                    avg_reward = [x['avg'] for x in run['stats']]
                    
                    #Make label
                    set_name = ""
                    set_name += "eps_max: " + str(entry['epsilon_max']) + " seed: " + str(entry['np_seed'])
                    plot_names.append(set_name)
                    
                    #Generate equivalent number of timesteps for x axis
                    #timesteps = [(x * pop_size * episode_len) for x in range(0,len(avg_reward))]
                    
                    #ax.plot(avg_reward, label= str(iterator))
                    if iterator in total_run_rewards:
                        temp_list = []
                        temp_list = [a + b for a,b in zip(total_run_rewards[iterator], avg_reward)]  
                        total_run_rewards[iterator] = temp_list[:]
                        
                    else:
                        total_run_rewards[iterator] = avg_reward


        #Plot each of the iterators
        for key, value in total_run_rewards.items():
            avg_value = [x / num_runs for x in value]
            ax.plot(avg_value, label= plot_names[key])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.4,
                     box.width, box.height * 0.6])
            
        plt.title("Tester")
        plt.ylabel("Reward")
        plt.xlabel("# Generations")
        plt.ylim(-0.1, 1.1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fancybox=True, shadow=True)
        save_name = str(round(time.time())) + "_avg_reward.jpg"
        plt.savefig("output.jpg")
        plt.show()      
                    

    else:
        for file_name in os.listdir(main_folder):
            if re.search(".*\.json", file_name):
                print(file_name)
                num_runs += 1
                
                with open(os.path.join(main_folder, file_name), 'r') as f:
                    output_dict = json.load(f)
                    print(output_dict['run_date'])
            
                #Prepare data set
                for iterator, run in enumerate(output_dict['runs']):
                    avg_reward = [x['avg'] for x in run['stats']]
                    
                    #Make label
                    set_name = ""
                    for entry in run['action_table']:
                        set_name += str(entry) + "-"
                    
                    #Generate equivalent number of timesteps for x axis
                    #timesteps = [(x * pop_size * episode_len) for x in range(0,len(avg_reward))]
                    
                    #ax.plot(avg_reward, label= str(iterator))
                    if iterator in total_run_rewards:
                        temp_list = []
                        temp_list = [a + b for a,b in zip(total_run_rewards[iterator], avg_reward)]  
                        total_run_rewards[iterator] = temp_list[:]
                        
                    else:
                        total_run_rewards[iterator] = avg_reward


        #Plot each of the iterators
        for key, value in total_run_rewards.items():
            avg_value = [x / num_runs for x in value]
            ax.plot(avg_value, label= str(key))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.4,
                     box.width, box.height * 0.6])
            
        plt.title("Tester")
        plt.ylabel("Reward")
        plt.xlabel("# Generations")
        #plt.ylim(-0.1, 1.1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fancybox=True, shadow=True)
        save_name = str(round(time.time())) + "_avg_reward.jpg"
        plt.savefig("output.jpg")
        plt.show()      
                    
        