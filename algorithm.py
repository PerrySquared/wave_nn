import numpy as np

# np.set_printoptions(precision=1e-2, edgeitems=sys.maxsize)

# 0 - free space
# int > 0 - path
# -1 - obstacle
# -2 - contact source
# -3 - contact end

class Wave_Route:
    
    def __init__(self, a, start_y, start_x, end_y, end_x):
        self.a = a
        self.start_y = start_y
        self.start_x = start_x
        self.end_y = end_y
        self.end_x = end_x
        a[self.start_y][self.start_x] = -2
        a[self.end_y][self.end_x] = -3
        
    def validate_current_state(self, array, y, x, index):
        
        row_len, column_len = array.shape
        new_positions_found = 0
        
        if y > 0:
            if array[y - 1, x] == 0:
                array[y - 1, x] = index
                new_positions_found += 1           
            if array[y - 1, x] == -3: # exit condition check
                return array, True, new_positions_found
            
        if y < row_len - 1:
            if array[y + 1, x] == 0:
                array[y + 1, x] = index
                new_positions_found += 1                        
            if array[y + 1, x] == -3: # exit condition check
                    return array, True, new_positions_found
                
        if x > 0:
            if array[y, x - 1] == 0:
                array[y, x - 1] = index   
                new_positions_found += 1                  
            if array[y, x - 1] == -3: # exit condition check
                    return array, True, new_positions_found        
       
        if x < column_len - 1:
            if array[y, x + 1] == 0:
                array[y, x + 1] = index 
                new_positions_found += 1           
            if array[y, x + 1] == -3: # exit condition check
                    return array, True, new_positions_found
        
        
        return array, False, new_positions_found

        
    def wave_generator(self, array, y, x):
        
        row_len, column_len = array.shape
        prev_array = np.copy(array)
        total_positions = 0
        wave_number = 1
        
        array, exit, new_positions_found = self.validate_current_state(array, y, x, wave_number)
        
        if new_positions_found == 0:
            return array, wave_number - 1, False
        
        wave_number += 1
        
        exit_while = False
        while exit_while == False:
            
            prev_array = np.copy(array)
            
            for i in range(row_len):
                for j in range(column_len):
                    if array[i, j] == wave_number - 1:

                        array, exit, new_positions_found = self.validate_current_state(array, i, j, wave_number)
                        total_positions += new_positions_found

                        if exit == True:
                            exit_while = True
                            array = prev_array
                            break
                         
            if total_positions == 0:
                # print("NO ROUTE FOUND")
                return  array, wave_number - 1, False
                        
            total_positions = 0                         
            wave_number += 1         
            
        return array, wave_number - 1, True


    def path_backtracking(self, array, y, x, wave_number):
        
        path = np.zeros((0,2), dtype=int) 
        path = np.append(path, [[self.end_y, self.end_x]], axis=0)
        
        row_len, column_len = array.shape

        for i in range(wave_number, -1, -1):

            if y > 0:
                if array[y - 1, x] == i:         
                    path = np.append(path, [[y - 1, x]], axis=0)
                    y = y - 1  
            if y < row_len - 1:
                if array[y + 1, x] == i:
                    path = np.append(path, [[y + 1, x]], axis=0)
                    y = y + 1          
                    
            if x > 0:
                if array[y, x - 1] == i:
                    path = np.append(path, [[y, x - 1]], axis=0)
                    x = x - 1          
            if x < column_len - 1:
                if array[y, x + 1] == i:
                    path = np.append(path, [[y, x + 1]], axis=0)
                    x = x + 1
        
        path = np.append(path, [[self.start_y, self.start_x]], axis=0)

        return(path)


    def final_route(self, array, path):
        for i in path :
            array[i[0], i[1]] = 1
            
        return array

    def output(self):
        row_len, column_len = self.a.shape
        a_zeros = np.zeros((row_len, column_len), dtype=float)
        
        #print("MATRIX WITH START AND END\n", self.a, "\n")
        
        array, wave_number, success = self.wave_generator(self.a, self.start_y, self.start_x)
        
        if success == False:
            return a_zeros
        
        path = self.path_backtracking(array, self.end_y, self.end_x, wave_number)
        array_output = self.final_route(a_zeros, path)
        
        return array_output

      
# a = np.array([
#     [0, 0, 0, 0, 0, 0, 0],
#     [0, 0, -1, 0, 0, 0, 0],
#     [-1, 0, -1, -1, -1, 0, 0], 
#     [0, -1, -1, -1, 0, 0, 0],
#     [0, 0, 0, -1, 0, 0, 0]
# ])

# # a_copy = np.copy(a)

# print("enter start Y and X values")
# s_y, s_x = map(int, input().split())
# print("enter end Y and X values")
# e_y, e_x = map(int, input().split())

# print("\n\n INITIAL MATRIX\n", a, "\n")

# array, wave_number = wave_generator(a, start_y, start_x)
# path = path_backtracking(array, end_y, end_x, wave_number)
# output = final_route(a_copy, path)
# wave1 = Wave_Route(a,s_y,s_x,e_y,e_x)
# print("\n OUTPUT \n", wave1.output(), "\n")