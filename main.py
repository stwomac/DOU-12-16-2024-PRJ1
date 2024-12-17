# %%


# %% [markdown]
# # Test

# %%


# %%
class StateMachine:
    
    __CurrentState = 1
    

    def __init__(self):
        self.__run()        


    def __home(self):
        print("Welcome to the Matrix Operation Manager!\n\n"
              "To use this program type the number of the option and hit enter.\n"
              "Please select from the following options:\n"
              "1.) Create a Matrix\n"
              "2.) List current Matrices available\n"
              "3.) Perform a Matrix Operation\n"
              "4.) Delete a Matrix\n"
              "5.) Exit"
        )

        options = [1,2,3,4, 5]
        userInput = int(input())
        
        userInput = self.__inputValidation(options, userInput)
        
        if userInput == 5:
            return 0
        else:
            return userInput
  
    #def __createMatrix()
    def __inputValidation(self, options, userInput):
        while userInput not in options:
            print("Please enter the number of one of the options above: ")
            userInput = int(input())
        
        return userInput

    STATES = { "1": __home}

    def __run(self):
        while self.__CurrentState != 0:
            self.__CurrentState = self.STATES[self.__CurrentState]()



obj = StateMachine()




