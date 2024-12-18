# %% [markdown]
# # Test

# %%

import numpy as np
import os
var = input()
#var = var.encode('unicode_escape').decode('utf-8')
var = var.replace("\\", "\\\\")
#normalized_path = os.path.abspath(var)
print(var)
# path = var.replace(r'\', '/)
# print(path)
arr = np.loadtxt( var, delimiter=",")
print(arr)

class Matrix:
    
    def __init__(self, row = None, columns = None):
        pass

# %%
class StateMachine:
    
    def __init__(self):
        self.__CurrentState = 1  
        self.__run()
              

    # This is the function for state 1, the home screen
    # Home marks the default screen, and is the only screen to allow a exit from the program
    # Home will return the state to move to.
    def __home(self):

        #intro text and options to move into further states
        print("Welcome to the Matrix Operation Manager!\n\n"
              "To use this program type the number of the option and hit enter.\n"
              "Please select from the following options:\n"
              "1.) Create a Matrix\n"
              "2.) List current Matrices available\n"
              "3.) Perform a Matrix Operation\n"
              "4.) Delete a Matrix\n"
              "5.) Exit"
        )

        #input options and validation call
        options = [ 1, 2, 3, 4, 5]
        userInput = None
        userInput = self.__inputValidation(options, userInput)
        
        #local state dictionary to transform user input into the class wide defined states
        stateDict = { 1: 2, 
                      2: 0,
                      3: 0,
                      4: 0,
                      5: 0}

        return stateDict[userInput]
  

    def __createMatrix(self):
        print("Please select from the following options:\n"
              "1.) Create a random matrix\n"
              "2.) Import a matrix\n"
              "3.) Manually create a matrix\n"
              "4.) Return to home\n"
        )

        options = [ 1, 2, 3, 4]
        userInput = None
        userInput = self.__inputValidation(options, userInput)

        #local state dictionary to transform user input into the class wide defined states
        stateDict = { 1: 3, 
                      2: 4,
                      3: 5,
                      4: 1}

        return stateDict[userInput]

    def __createRandom(self):
        print(1)

    def __inputValidation(self, options, userInput):
        
        try:
            userInput = int(input())
            if userInput not in options:
                raise Exception
        except:
            while userInput not in options:
                print("Please enter the number of one of the options above:\n ")
                try:
                    userInput = int(input())
                except:
                    print("That is not a whole number.\n")
            
        return userInput

    
    __STATES = { 1: __home, 
                 2: __createMatrix
                }
    
    def __run(self):
        while self.__CurrentState != 0:
            self.__CurrentState = self.__STATES[self.__CurrentState](self)
        
        print("Have a nice day!")
            
        


obj = StateMachine()

# %%



