#Kyle Norland
#1/24/23
#Custom visualizer for grid world Markov Chains
#Also for visualizing Q-Learning values.

#-------------
#--Imports----
#-------------
import pygame
import sys
from pygame.locals import *
import time
import math

#Global Colors
WHITE=(255,255,255)
BLACK=(0,0,0)
BLUE=(0,0,255)
GREEN=(0,255,0)
RED=(255,0,0)
YELLOW=(255,255,0)

class Graph_Visualizer:
    def __init__(self, n_rows, n_columns):
        #Initialize pygame
        print("Initializing plot")
        
        #---Setttings-----
        self.screen_width = 800
        self.screen_height = 600
        self.aspect_ratio = self.screen_width/self.screen_height
        self.x = 50
        self.y = 50
        self.box_width = 75
        self.box_height = 75
        self.circle_radius = 15
        self.border_width = 2
        
        #----Arguments-------
        self.n_rows = n_rows
        self.n_columns = n_columns
        
        #------------Pygame init actions-------------
        pygame.init()# Setting your screen size with a tuple of the screen width and screen height
        self.screen = pygame.display.set_mode((800,600))# Setting a random caption title for your pygame graphical window.
        pygame.display.set_caption("pygame test")
        
        #Determine midpoints
        self.draw_grid(set_midpoints=True)


    def draw_grid(self, set_midpoints=False):
        #x,y is upper left corner of overall grid
        #middle points stores where the middle of each square is.
        if set_midpoints: self.midpoints = []
        
        
        for i in range(self.n_rows):
            print(i)
            points_row =[] 
            for j in range(self.n_columns):
                pygame.draw.rect(self.screen, BLACK, (self.x + j*self.box_width, self.y + i*self.box_height, self.box_width, self.box_height), width=self.border_width)
                #Save the middle point
                if set_midpoints:
                    points_row.append((self.x + j*self.box_width + (0.5*self.box_width), self.y + i*self.box_height + (0.5*self.box_height)))
            if set_midpoints:
                self.midpoints.append(points_row)
            
    def angle(self, A, B, aspectRatio):
        x = B[0] - A[0]
        y = B[1] - A[1]
        angle = math.atan2(-y, x / aspectRatio)
        return angle
        
    def angle_two(self, A,B):
        x = B[0] - A[0]
        y = B[1] - A[1]
        angle = math.atan2(-y, x)
        return angle
        
    def distance(self, A, B):
            return math.hypot(B[0] - A[0], B[1] - A[1])    
    
    def plot_edges(self, edges, color=(0,0,0)):
        #screen, nrows, ncolumns, midpoints, edges, color=(0,0,0)):
        #Plots a slightly curved path from start to finish
        
        for entry in edges:
            #print(entry)
            #Get first point
            first_index = self.to_2d_index(entry[0])
            first_center = self.midpoints[first_index[0]][first_index[1]]
            
            #Get second point
            second_index = self.to_2d_index(entry[1])
            second_center = self.midpoints[second_index[0]][second_index[1]]
            
            #Calculate distance
            interpoint_distance = self.distance(first_center, second_center)
            
            #Figure out which is left and right (should also be top if on same plane. As well as a flag for which way to put the center.
            center_flag = 'left'
            if (first_center[0] < second_center[0]) or (first_center[0] == second_center[0] and first_center[1] > second_center[1]):
                left_point = first_center
                right_point = second_center
                center_flag = 'left'
                #print('a')
                
            elif (first_center[0] == second_center[0] and first_center[1] > second_center[1]):
                left_point = first_center
                right_point = second_center
                center_flag = 'left'
                #print('b')
     
            else:
                left_point = second_center
                right_point = first_center
                center_flag = 'left'
                #print('c')
                
            #print("Left_point", left_point)
            #print("Right_point", right_point)
            #print("Center_flag", center_flag)
                
            #------Choose radius---------------
            r = self.distance(left_point, right_point)
            
               
            #Calculate the midpoint and angle
            #straight_angle = angle(left_point, right_point, aspect_ratio)
            
            straight_angle = self.angle_two(left_point, right_point)
            
            points_center = (left_point[0] + (0.5*(right_point[0]-left_point[0])), (left_point[1] + (0.5*(right_point[1]-left_point[1]))))
            #print("points_center", points_center)
            
            #perpendicular angle (to left)
            if center_flag == 'left':
                perp_angle = straight_angle + (math.pi/2)
            if center_flag == 'right':
                perp_angle = straight_angle - (math.pi/2)
            
            #print("straight_angle", straight_angle)
            #print("perpendicular_angle", perp_angle)
            #Calculate center point
            a = r
            b = self.distance(left_point, right_point) 
            h = math.sqrt(a**2 - (0.5*b)**2)
            #print("h", h)
            
            center_point = (points_center[0] + h*math.cos(perp_angle), points_center[1] - h*math.sin(perp_angle))
            
            start_angle = self.angle_two(center_point, left_point) + (math.pi/2)
            end_angle = self.angle_two(center_point, right_point) + (math.pi/2)
            
            if center_flag == 'right':    
                end_angle = end_angle + math.pi
                start_angle = start_angle + math.pi
            
            x = int(center_point[0] - r)
            y = int(center_point[1] - r)
            width = 2*r
            height = 2*r
            
            #print("Points", right_point, left_point)
            #print(start_angle, end_angle, x,y,width, height)
            start_degrees = math.degrees(start_angle)
            end_degrees = math.degrees(end_angle)
            if start_degrees < 0: start_degrees += 360
            if end_degrees < 0: end_degrees += 360
            #print(start_degrees, end_degrees)
            
            #pygame.draw.arc(screen, (0, 0, 0), (x, y, width, height), *angles)
            
            start_angle = start_angle - math.pi/2
            end_angle = end_angle - math.pi/2
            
            #Draw the arc
            pygame.draw.arc(self.screen, color, (x, y, width, height), start_angle, end_angle, width=1)
            
            #Draw the triangle at the end of the arc
            triangle_point = (0,0)
            triangle_angle = 0
            if second_center == right_point:
                triangle_point = second_center
                triangle_angle = end_angle + ((1/2)*math.pi)
                #print("Case a")
            elif second_center == left_point:
                triangle_point = second_center
                triangle_angle = start_angle + ((1/2)*math.pi) + math.pi
                #print("Case b")
            
            #print("triangle_point", triangle_point)
            #print("triangle_angle", triangle_angle)
            
            triangle_points = []
            #3 Times find the points at 120 degrees further clockwise. 2/3 radians.
            side_length = 10
            cp = triangle_point
            ca = triangle_angle
            #angle_series = [(1/6)*math.pi, (2/3)*math.pi, (1/6)*math.pi
            #Draw a check line for direction
            pygame.draw.line(self.screen,(0,0,0), cp, (cp[0] + side_length*math.cos(ca), cp[1] - side_length*math.sin(ca)))
            
            #First one
            next_angle = ca + (math.pi - ((1/6) * math.pi)) #Plus 150 degrees for 30 degree angle from straight up.
            next_x = cp[0] + side_length*math.cos(next_angle)
            next_y = cp[1] - side_length*math.sin(next_angle)
            triangle_points.append((next_x, next_y))
            cp = (next_x, next_y)
            ca = (next_angle)
            
            #Second
            next_angle = ca + ((2/3) * math.pi) #Plus 120 degrees for 60 degree angle from straight up.
            next_x = cp[0] + side_length*math.cos(next_angle)
            next_y = cp[1] - side_length*math.sin(next_angle)
            triangle_points.append((next_x, next_y))
            cp = (next_x, next_y)
            ca = (next_angle)        
            
            #Third
            next_angle = ca + ((2/3) * math.pi) #Plus 120 degrees for 60 degree angle from straight up.
            next_x = cp[0] + side_length*math.cos(next_angle)
            next_y = cp[1] - side_length*math.sin(next_angle)
            triangle_points.append((next_x, next_y))
            cp = (next_x, next_y)
            ca = (next_angle)  
            
            #Draw polygon of triangle and fill with same color as line.
            #print("triangle_points: ", triangle_points)
            pygame.draw.polygon(self.screen, color, triangle_points)
        
    def plot_known_rewards(self, known_rewards):
        print("Known rewards")
        for key, entry in known_rewards.items():
            if entry > 0:
                #Convert to 2d index
                index = self.to_2d_index(key)
                center = self.midpoints[index[0]][index[1]]
                
                #Plot rectangle at midpoint
      
                pygame.draw.rect(self.screen, YELLOW, (int((center[0] - (0.25* self.box_width))), int((center[1] - (0.25 * self.box_height))), int(0.5*self.box_width), int(0.5*self.box_height)), width=0)       
                #input("Check")
            
    def plot_known_states(self, known_states):
        #screen, nrows, ncolumns, midpoints, explored):
        for entry in known_states:
            #Convert to 2d index
            index = self.to_2d_index(entry)
            center = self.midpoints[index[0]][index[1]]
            
            #Plot circle at midpoint
            pygame.draw.circle(self.screen, GREEN, center, radius=self.circle_radius, width=0)

    def to_2d_index(self, index):
        row = index // self.n_columns
        column = index % self.n_columns
        return (row, column)

   
    def draw_plot(self, state_reward, known_states, known_edges, greedy_path=[], optim_path=[], q_edges=[(0,2)]):
        #screen, nrows, ncolumns, explored, known_rewards, known_edges, greedy_path=[], optim_path=[], q_edges=[(0,2)]):
        #Blank out screen
        self.screen.fill(WHITE)
        
        #Initial Draw
        
        #print("Middle points", midpoints)
        #print("plotting")
        self.draw_grid()
        self.plot_known_rewards(state_reward)
        self.plot_known_states(known_states)
        
        #Plot known edges
        #Preprocess known edges
        processed_edges = []
        for key, edge_group in known_edges.items():
            for end_point in edge_group:
                processed_edges.append((key, end_point))
                
        self.plot_edges(processed_edges, color=(119,255,51))
        
        #Plot optim path
        self.plot_edges(optim_path, color=(255,0,0))
        
        #Plot greedy path
        self.plot_edges(greedy_path, color=(255,0,255)) 
        
        #Plot q_edges
        #self.plot_edges(q_edges, color=(255,200,0))
        
        pygame.display.update()

    def test_plotter(self):
        pygame.init()# Setting your screen size with a tuple of the screen width and screen height
        screen_width = 800
        screen_height = 600
        aspect_ratio = screen_width/screen_height
        screen = pygame.display.set_mode((800,600))# Setting a random caption title for your pygame graphical window.
        pygame.display.set_caption("pygame test")
        
        #Blank out screen
        screen.fill(WHITE)
        
        #Graph information
        nrows = 4
        ncolumns = 4
        
        #Initial Draw
        midpoints = draw_grid(50, 50, ncolumns, nrows, 76, 76, 2)
        print("Middle points", midpoints)
        
        
        #Array of explored states (1-16, etc)
        #position = current_row*nrows + current_col

        explored = [0,1,5,8,10]
        known_rewards = [0,5,9,12]
        known_edges = [(8,9),(10,5), (9,6),  (5,9), (2,12)]
        
        #Plot known data
        
        plot_known_rewards(nrows, ncolumns, midpoints, known_rewards)
        plot_explored(nrows, ncolumns, midpoints, explored)
        plot_known_edges(nrows, ncolumns, midpoints, known_edges)
        
        #Plot edges
        
        #Plot solved edges
        
        
        
        
        
        #Run Loop
        while True:
            #Check for exit
            for event in pygame.event.get():
                if event.type==QUIT:
                    pygame.quit()
                    sys.exit()
            #Update screen        
            pygame.display.update()  
            time.sleep(0.25)


if __name__ == "__main__":
    print("hi")
    my_vis = Graph_Visualizer()




