# %% [markdown]
# # Imports

# %%
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# # Matrix Class

# %%
# Class utilized to represent a 2-D Matrix, and the operations one can perform on one.
class Matrix:
    
    # Default Construcion method, used for quick conversions of Numpy Arrays to Matrix Objects
    # If it recieves something besides a Numpy Array, will instead provide None Values
    #
    # npMatrix - a Numpy Matrix
    def __init__( self, npMatrix ):
        
        if isinstance(npMatrix, np.ndarray):

            self.__npMatrix = npMatrix

            # Used to determine if a 1D Array was recieved and then converts it to a 2D
            try:
                self.__rows, self.__cols = self.__npMatrix.shape
            except:
                self.__rows = 1
                self.__cols = self.__npMatrix.shape[0]
                self.__npMatrix = np.reshape(self.__npMatrix, (self.__rows, self.__cols))
        else:
            self.__npMatrix = None
            self.__rows, self.__cols = (None, None)
        
    # Constructor Method used to arrays of random sizes and values, expects paramters to limit the randomness.
    #
    # rows - int - The amount of rows of the Array
    # cols - int - The amount of columns of the array
    # wholeNum - bool - wether the values inside should be whole numbers or floats
    #
    # Will return either a Matrix object or None
    @classmethod
    def createRandom( cls, 
                        rows, 
                        cols,
                        wholeNum ):
        
        # parameter validation
        if type(rows) is not int or type(cols) is not int:
            print("You have attempted to make a matrix with non int columns or rows, this does not work.")
            return
        elif rows < 1 or cols < 1:
            print("You have attempted to make a matrix with columns or rows less than 1")
            return
        elif type(wholeNum) is not bool:
            print("wholeNum must have a bool passed in")
            return
        
        # Generates the np array and then sends it to the default constructor
        if wholeNum:
            return cls(np.random.randint(low = 0, high = 100, size=(rows, cols)))
        else:
            return cls(np.random.uniform( size=(rows, cols)) * 100)
        
    # Constructor used to create a Matrix based on a CSV file
    # path - str - the path to the csv file
    #
    # Will return either a Matrix object or None
    @classmethod      
    def createImport(cls, path):
        
        # Imports file, will return nothing if not provided a proper path, a CSV file, or if the file contents are incompatible with transforming into a numpy array
        try:
            return cls(np.loadtxt( path, delimiter=",", ndmin=2))
        except:
            pass

    # Constructor used to create a Matrix based off of a list of lists.
    #
    # Will return either a Matrix object or None
    @classmethod
    def createManual(cls, listMatrix):
        
        # Returns none if given a empty list, a list does not have a list inside of it, or a non-list
        if type(listMatrix) is not list:
            return

        if len(listMatrix) == 0:
            return

        if type(listMatrix[0]) is not list:
            return 
        try:
            return cls(np.array(listMatrix))
        except:
            pass
        
    # Adds two matrices together
    #
    # Will return either a Matrix object or None
    def add(self, matrixB):
        if self.__npMatrix.shape != matrixB.getMatrix().shape:
            print("To add Matrices they must have the same amount of rows and columns.")
        else:
            return Matrix(npMatrix = np.add(self.__npMatrix,matrixB.getMatrix()))

    # Subtracts two matrices
    #
    # Will return either a Matrix object or None
    def subtract(self, matrixB):
        if self.__npMatrix.shape != matrixB.getMatrix().shape:
            print("To subtract Matrices they must have the same amount of rows and columns.")
        else:
            return Matrix(npMatrix = np.subtract(self.__npMatrix,matrixB.getMatrix()))

    # Multiplies two matrices together
    #
    # Will return either a Matrix object or None
    def multiply(self, matrixB):
        if self.__cols != matrixB.getRows():
            print("To multiply Matrices the amount of columns of the first Martix must match the amount of rows of the second.")
        else:
            return Matrix(npMatrix = np.multiply(self.__npMatrix,matrixB.getMatrix()))

    # Transposes the current matrix
    #
    # Will return either a Matrix object or None
    def transpose(self):
        if self.__npMatrix is None:
            pass
        else:
            return Matrix(npMatrix =  np.transpose(self.__npMatrix))

    # Finds the determinate of the current matrix
    #
    # Will return either float64, complex 128, or None
    def determinate(self):
        if self.__rows != self.__cols:
            print("A matrix must be square (same number of rows and columns) to have a determinant.")
        else:
            return np.linalg.det(self.__npMatrix)

    # Finds the inverse of the current Matrix
    #
    # Will return either a Matrix object or None
    def inverse(self):
        if self.__cols != self.__rows:
            print("A matrix must be square (same number of rows and columns) to have a inverse.")
        elif self.determinate() == 0:
            print("A matrix must not have a determinate of 0 to have a inverse.")
        else:
            return Matrix(npMatrix = np.linalg.inv(self.__npMatrix))
        
    # Finds the identity matrix of the current Matrix
    #
    # Will return either a Matrix object or None
    def identity(self):
        if self.__cols != self.__rows:
            print("A matrix must be square (same number of rows and columns) to have a identity matrix.")
        elif self.__rows is None:
            pass
        elif self.__rows < 1:
            print("Somehow you made a Matrix with less than one row.")
        else:
            return Matrix(npMatrix = np.identity(self.__rows))

    # Finds the zero matrix of the current matrix
    #
    # Will return either a Matrix object or None
    def zero(self):
        if type(self.__rows) is not int or type(self.__cols) is not int:
            print("Their is a problem, the Matrix's shape is not made of ints.")
        elif self.__rows < 1:
            print("Somehow you made a Matrix with less than one row.")
        else:
            return Matrix(npMatrix = np.zeros(self.__npMatrix.shape))

    # Function needed for Eigen Decomposition
    # values - expects the eigenvalues from a np.linalg.eig or np.linalg.eigh call
    # Returns a Matrix
    def __diagonal(self, values):
        return Matrix(npMatrix = np.diag(values))

    # Finds the eigen values, vectors, and decomposition of the current matrix.
    #
    # Will return either a list with a Matrix object for each of the eigen value,
    # vector, and decomposition or will return None
    def eigenDecomp(self):

        if self.__cols != self.__rows:
            print("A matrix must be square (same number of rows and columns) to perform eigen decomposition.\n")
        else:
            eigValues = None
            eigVectors = None

            #check for symmetric array
            if ((self.__npMatrix==self.transpose().getMatrix()).all()):
                eigValues, eigVectors = np.linalg.eigh(self.__npMatrix)
                eigVectors = Matrix(eigVectors)
                return [Matrix(npMatrix = np.dot(eigVectors.getMatrix(), 
                                                np.dot(self.__diagonal(eigValues).getMatrix(), eigVectors.transpose().getMatrix()))),
                        eigVectors,
                        eigValues]
            else:
                eigValues, eigVectors = np.linalg.eig(self.__npMatrix)
                eigVectors = Matrix(eigVectors)
                
                # Required for testing, transforms the np [] array to a np [[]] array
                eigHold = []
                eigHold.append(eigValues.tolist())
                eigValuesMatrix = Matrix.createManual(eigHold)

                return [Matrix(npMatrix = np.dot(eigVectors.getMatrix(), 
                                                np.dot(self.__diagonal(eigValues).getMatrix(), eigVectors.inverse().getMatrix()))),
                        eigVectors,
                        eigValuesMatrix]

    # Getters

    # returns the numpy matrix
    def getMatrix(self):
        return self.__npMatrix
    
    # returns the rows of the matrix
    def getRows(self):
        return self.__rows
    
    # returns the columns of the matrix
    def getCols(self):
        return self.__cols
    
    # creates a pop-up image of the heatmap of the current matrix
    def showVisualization(self):
        sns.heatmap(self.__npMatrix, annot = True, cmap ='plasma', 
            linecolor ='black', linewidths = 1)
        plt.show()

    # prints the current matrix to the terminal
    def printMatrix(self):
        if self.__npMatrix is None:
            print('None')
        else:
            print(str(self.__npMatrix))



# %% [markdown]
# # State Machine Class

# %%

# This class provides the UI/UX for the program, and a object of 
# which should be instantiated at the start of the program.
#
# Works by moving the user through different states based on user inputs, 
# with the only memory preserved being the CSV files stored in the matrices folder.
#
# Will infintely run until the user exits through the home screen.
class StateMachine:
    
    def __init__(self):

        # Checks to see if a folder exists to store matrices that can be used by the program
        # if it does not, it creates it instead
        folder1 = "matrices"
        os.chdir(".")

        if os.path.isdir(folder1):
            pass
        else:
            os.mkdir(folder1)

        # sets the current state to the Home Screen
        self.__CurrentState = 1  

        # Begins the program loop
        self.__run()

        

    # State Functions:          

    # This is the function for state 1, the home screen
    # Home marks the default screen, and is the only screen to allow a exit from the program
    # 
    # Will return the state to move to.
    def __home(self):

        # intro text and options to move into further states
        print("Welcome to the Matrix Operation Manager!\n\n"
              "To use this program type the number of the option and hit enter.\n"
              "Please select from the following options:\n"
              "1.) Create a Matrix\n"
              "2.) List current Matrices available\n"
              "3.) Perform a Matrix Operation\n"
              "4.) Delete a Matrix\n"
              "5.) Exit"
        )

        # input options and validation call
        options = [ 1, 2, 3, 4, 5]
        userInput = None
        userInput = self.__inputValidation(options, userInput)
        
        # local state dictionary to transform user input into the class wide defined states
        stateDict = { 1: 2, 
                      2: 6,
                      3: 8,
                      4: 7,
                      5: 0 }

        return stateDict[userInput]
    
    # State 2 the create Matrix screen, will let users choose between 3 options to create
    # matrices for use in the program
    # 
    # Will return the state to move to.
    def __createMatrix(self):
        # UI options
        print("Please select from the following options:\n"
              "1.) Create a random matrix\n"
              "2.) Import a matrix\n"
              "3.) Manually create a matrix\n"
              "4.) Return to home"
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

    # State 3 - Create Random Matrix, allows the user to create a save a randomly generated matrix
    # 
    # Will return the state to move to.
    def __createRandom(self):
        randomMatrix = None
        
        # UI Options
        print("Random Matrix Creation:\n"
              "1.) Completely random\n"
              "2.) Input parameters\n"
              "3.) Return to home"
        )

        # Input validation
        options = [ 1, 2, 3]
        userInput = None
        userInput = self.__inputValidation(options, userInput)


        match userInput:
            # Case 1 - random matrix based on program defined parameters, 
            # Matrices will have 1 - 10 rows and columns, and be made up of whole numbers
            case 1:
                randomMatrix = Matrix.createRandom(np.random.randint(1,11),np.random.randint(1,11),True)
                self.__postCreate(randomMatrix)
            
            # Case 2 - random matrix based on user defined amount of rows, columns, and wether it will have whole numbers or not
            case 2:
                print("Please type in the number of rows for the matrix:")
                
                # input validation rows
                rows = input()
                while(True):
                    
                    try:
                        if rows.isdigit():
                            rows = int(rows)
                            if rows > 0:
                                break
                        else:
                            print("Please enter a integer greater than 0.")
                            rows = input()
                    except:
                        print("Please enter a integer greater than 0.")
                        rows = input()
                
                print("Please type in the number of columns for the matrix:")
                
                # input validation columns
                cols = input()
                while(True):
                    try:
                        if cols.isdigit():
                            cols = int(cols)
                            if cols > 0:
                                break
                        else:
                            print("Please enter a integer greater than 0.")
                            cols = input()
                    except:
                        print("Please enter a integer greater than 0.")
                        cols = input()
                
                print("Would you like the Matrix to have only whole numbers?:\n"
                      "1.) Yes\n"
                      "2.) No"
                )

                # Whole Number input validation
                options2 = [ 1, 2 ]
                userInput2 = None
                userInput2 = self.__inputValidation(options2, userInput2)

                wholeNums = None
                match userInput2:
                    case 1:
                        wholeNums = True
                    case 2:
                        wholeNums = False

                # matrix is created
                randomMatrix = Matrix.createRandom(rows, cols, wholeNums)
                self.__postCreate(randomMatrix)
        
        # local state dictionary to transform user input into the class wide defined states
        stateDict = { 1: 3, 
                      2: 3,
                      3: 1 }

        return stateDict[userInput]

    # State 4 - Imports a Matrix, allows the user to import a matrix into the program from a user defined path
    # 
    # Will return the state to move to.
    def __createImport(self):

        userInput = None

        
        print("Please input the path to the file you wish to import.\n"
              "The Matrix Operations Manager only allows imports of CSV files and numeric arrays.")
        
        importMatrix = None

        # Loop to allow for continuous imports until a user selects a option to stop
        while importMatrix is None:

            importMatrix = Matrix.createImport(input())
            # input validation
            if importMatrix is None:
                print("Either the path you provided was incorrect, or you did not provide a CSV file,\n"
                      " or if the file contents are incompatible with transforming into a numpy array"
                      )
                print("Would you like to try again?\n"
                      "1.) Yes\n"
                      "2.) No")
                
                options = [ 1, 2 ]
                userInput = self.__inputValidation(options, userInput)

                if userInput == 2:
                    break
            
            # check to ensure imported matrix is usable by the program,
            # the program will not save non numeric matrices
            elif not np.issubdtype(importMatrix.getMatrix().dtype, np.number):
                print("The matrix provided is non numeric")
                print("Would you like to try again?\n"
                      "1.) Yes\n"
                      "2.) No")
                
                options = [ 1, 2 ]
                userInput = self.__inputValidation(options, userInput)

                if userInput == 2:
                    break

                importMatrix = None
            
            if importMatrix is not None:
                self.__postCreate(importMatrix)

                print("Would you like to import another Matrix?\n"
                          "1.) Yes\n"
                          "2.) No")
                
                options = [ 1, 2 ]
                userInput = self.__inputValidation(options, userInput)

                match userInput:
                    case 1:
                        importMatrix = None
                        print("Please input the path to the file you wish to import.\n"
                              "The Matrix Operations Manager only allows imports of CSV files and numeric arrays."
                              )
                    case 2:
                        break
        # local state dictionary to transform user input into the class wide defined states
        stateDict = { 1: 4, 
                      2: 1 }

        return stateDict[userInput]
    
    # State 5 - Create a Matrix Manually, allows the user to create a matrix by terminal inputs.
    # User must finish the inputs before being let back into the rest of the program.
    # 
    # Will return the state to move to.    
    def __createManual(self):

        print("Manual Matrix Creation:\n"
              "Please type in the number of rows for the matrix:"
              )
        
        # Row Input Validation
        rows = input()
        while(True):
            
            try:
                if rows.isdigit():
                    rows = int(rows)
                    if rows > 0:
                        break
                else:
                    print("Please enter a integer greater than 0.")
                    rows = input()
            except:
                print("Please enter a integer greater than 0.")
                rows = input()
        
        print("Please type in the number of columns for the matrix:")
        
        # Column input validation
        cols = input()
        while(True):
            try:
                if cols.isdigit():
                    cols = int(cols)
                    if cols > 0:
                        break
                else:
                    print("Please enter a integer greater than 0.")
                    cols = input()
            except:
                print("Please enter a integer greater than 0.")
                cols = input()
        
        matrixList = []

        # Fill the matrixList with user input
        print("Enter the elements row by row:")
        for i in range(rows):
            # Create an empty list for the current row
            row = []
            for j in range(cols):
                # Append user input to the current row
                print(f"Enter element for row {i+1}, column {j+1}: ")
                element =  self.__numberValidation(input())
                row.append(element)
            # Append the completed row to the matrix
            matrixList.append(row)

        # Create the Matrix Object from the List of Lists
        listMatrix = Matrix.createManual(matrixList)

        self.__postCreate(listMatrix)

        # Note that this function does not loop, instead just lets the StateMachine return to this state.
        print("Would you like to create another Matrix?\n"
              "1.) Yes\n"
              "2.) No")

        # Input Validation        
        options = [ 1, 2 ]
        userInput = None
        userInput = self.__inputValidation(options, userInput)

        print(userInput)
        
        # local state dictionary to transform user input into the class wide defined states
        stateDict = { 1: 5, 
                      2: 1 }

        return stateDict[userInput]

    # State 6 - List of saved matrices, lists in the terminal the matrices available to perform operations on
    # 
    # Will return the state to move to.
    def __listSaved(self):

        print("The current Matrices available are:")

        counter = 0
        for file_name in os.listdir("matrices/"):
        # Check if the file has a .csv extension
            if file_name.endswith(".csv"):
                # Remove the .csv extension and print the name
                print(file_name[:-4], end="\t")
                counter += 1

                if counter % 5 == 0:
                    print()
        print()

        # local state dictionary to transform user input into the class wide defined states
        stateDict = { 1: 1 }

        return stateDict[1]
    
    # State 7 - Delete Screen, lets users delete the csv files saved in the matrices folder
    # 
    # Will return the state to move to.
    def __deleteScreen(self):
        
        print("Please enter the name of the Matrix you wish to delete:")

        self.__deleteMatrix(input())
        
        print("Would you like to delete another Matrix?\n"
              "1.) Yes\n"
              "2.) No")
                
        # input validation
        options = [ 1, 2 ]
        userInput = None
        userInput = self.__inputValidation(options, userInput)

        print(userInput)

        # local state dictionary to transform user input into the class wide defined states
        stateDict = { 1: 7, 
                      2: 1 }

        return stateDict[userInput]
    
    # State 8 - Matrix Operations, lets users perform various Matrix Operations from the Matrix Class
    # 
    # Will return the state to move to.
    def __matrixOperations(self):
        
        # matrixA is a tuple with 2 values inside
        # 0 - a str with the name of the matrix
        # 1 - the Matrix Object imported from the csv with its name
        matrixA = None
        
        print("Matrix Operations:")
        #This loop continues the state untill they choose option 12 (Exit)
        while(True):
            
            # matrixB is a tuple with 2 values inside
            # 0 - a str with the name of the matrix
            # 1 - the Matrix Object imported from the csv with its name
            matrixB = None
            
            # Details the current Matrix operations will be performed on, this does not change the
            # state of the csv file the matrix is saved in
            matrixStr = "Current Matrix: "

            if matrixA is None:
                matrixStr += "None"
            else:
                matrixStr += matrixA[0]

            print(matrixStr)
            
            print("Please select from the following operations:\n"
              "1.) Select a matrix\n"
              "2.) Add (Current + Other)\n"
              "3.) Subtract (Current - Other)\n"
              "4.) Multiply (Current * Other)\n"
              "5.) Transpose \n"
              "6.) Determinate\n"
              "7.) Inverse\n"
              "8.) Identity\n"
              "9.) Eigen Decomposition\n"
              "10.) Print to console\n"
              "11.) Show Heatmap\n"
              "12.) Return to Home\n"
            )

            # input validation
            options = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            userInput = None
            userInput = self.__inputValidation(options, userInput)

            match userInput:
                # Sets the matrix that the user would like to perform operations on.
                case 1:
                    print("Please enter the name of the matrix you want to perform operations on:")
                    matrixA = self.__loadMatrix(input())

                    if matrixA[1] is None:
                        print("That matrix does not currently exist in the Matrix Operation Manager")
                        matrixA = None
                        continue
                
                # Addition
                case 2:
                    if matrixA is None:
                        print("Please select a matrix before trying to perform a operation.")
                        continue
                    else:
                        cancel = False
                        matrixB = None
                        while matrixB is None:
                            print(f"Please enter the name of the matrix you want to add to {matrixA[0]}:")
                            matrixB = self.__loadMatrix(input())

                            if matrixB[1] is None:
                                matrixB = None
                                print("That matrix does not currently exist in the Matrix Operation Manager\n"
                                      "Would you like to try again?:\n"
                                      "1.) Yes\n"
                                      "2.) No"
                                )
                                
                                
                                options2 = [ 1, 2 ]
                                userInput2 = None
                                userInput2 = self.__inputValidation(options2, userInput2)

                                if userInput2 == 2:
                                    cancel = True
                                    break
                        
                        if cancel:
                            continue

                        addMatrix = matrixA[1].add(matrixB[1])

                        if addMatrix is not None:
                            self.__postOp(addMatrix)
                
                # Subtraction
                case 3:
                    if matrixA is None:
                        print("Please select a matrix before trying to perform a operation.")
                        continue
                    else:
                        cancel = False
                        matrixB = None
                        while matrixB is None:
                            print(f"Please enter the name of the matrix you want to subtract from {matrixA[0]}:")
                            matrixB = self.__loadMatrix(input())

                            if matrixB[1] is None:
                                matrixB = None
                                print("That matrix does not currently exist in the Matrix Operation Manager\n"
                                      "Would you like to try again?:\n"
                                      "1.) Yes\n"
                                      "2.) No"
                                )
                                
                                
                                options2 = [ 1, 2 ]
                                userInput2 = None
                                userInput2 = self.__inputValidation(options2, userInput2)

                                if userInput2 == 2:
                                    cancel = True
                                    break
                        
                        if cancel:
                            continue

                        subMatrix = matrixA[1].subtract(matrixB[1])

                        if subMatrix is not None:
                            self.__postOp(subMatrix)
                
                # Multiplication
                case 4:
                    if matrixA is None:
                        print("Please select a matrix before trying to perform a operation.")
                        continue
                    else:
                        cancel = False
                        while matrixB is None:
                            print(f"Please enter the name of the matrix you want to multiply with {matrixA[0]}:")
                            matrixB = self.__loadMatrix(input())

                            if matrixB[1] is None:
                                matrixB = None
                                print("That matrix does not currently exist in the Matrix Operation Manager\n"
                                      "Would you like to try again?:\n"
                                      "1.) Yes\n"
                                      "2.) No"
                                )
                                
                                
                                options2 = [ 1, 2 ]
                                userInput2 = None
                                userInput2 = self.__inputValidation(options2, userInput2)

                                if userInput2 == 2:
                                    cancel = True
                                    break
                        
                        if cancel:
                            continue

                        multMatrix = matrixA[1].multiply(matrixB[1])

                        if multMatrix is not None:
                            self.__postOp(multMatrix)
                
                # Transposition
                case 5:
                    if matrixA is None:
                        print("Please select a matrix before trying to perform a operation.")
                        continue
                    else:
                        tranMatrix = matrixA[1].transpose()

                        if tranMatrix is not None:
                            self.__postOp(tranMatrix)
                
                # Determinate
                case 6:
                    if matrixA is None:
                        print("Please select a matrix before trying to perform a operation.")
                        continue
                    else:
                        det = matrixA[1].determinate()

                        if det is not None:
                            print(f"The determinate of the Matrix is: {det}")
                
                # Inverse
                case 7:
                    if matrixA is None:
                        print("Please select a matrix before trying to perform a operation.")
                        continue
                    else:
                        invMatrix = matrixA[1].inverse()

                        if invMatrix is not None:
                            self.__postOp(invMatrix)
                
                # Identity
                case 8:
                    if matrixA is None:
                        print("Please select a matrix before trying to perform a operation.")
                        continue
                    else:
                        idMatrix = matrixA[1].identity()

                        if idMatrix is not None:
                            self.__postOp(idMatrix)
                
                # Eigen Decomposition
                case 9:
                    if matrixA is None:
                        print("Please select a matrix before trying to perform a operation.")
                        continue
                    else:
                        eigList = matrixA[1].eigenDecomp()

                        if eigList is not None:
                            print("The eigen values are:")
                            print(eigList[2].printMatrix())
                            print("The eigen vectors are:")
                            eigList[1].printMatrix()
                            print("The eigen decomposition is:")
                            eigList[0].printMatrix()

                            # Options to let users save the matrices from eigen decomposition
                            print("Would you like to save the eigen values?\n"
                                  "1.) Yes\n"
                                  "2.) No"
                                  )

                            options = [ 1, 2 ]
                            userInput = None
                            userInput = self.__inputValidation(options, userInput)

                            if userInput == 1:
                                print("Please enter the name of the matrix:")
                                name = input()
                                self.__saveMatrix(name, eigList[2])
                                print("Your matrix has been saved!")

                            print("Would you like to save the eigen vectors?\n"
                                  "1.) Yes\n"
                                  "2.) No"
                                  )

                            options = [ 1, 2 ]
                            userInput = None
                            userInput = self.__inputValidation(options, userInput)

                            if userInput == 1:
                                print("Please enter the name of the matrix:")
                                name = input()
                                self.__saveMatrix(name, eigList[1])
                                print("Your matrix has been saved!")

                            print("Would you like to save the eigen decomposition?\n"
                                  "1.) Yes\n"
                                  "2.) No"
                                  )

                            options = [ 1, 2 ]
                            userInput = None
                            userInput = self.__inputValidation(options, userInput)

                            if userInput == 1:
                                print("Please enter the name of the matrix:")
                                name = input()
                                self.__saveMatrix(name, eigList[0])
                                print("Your matrix has been saved!")

                # Print matrix to terminal
                case 10:
                    if matrixA is None:
                        print("Please select a matrix before trying to perform a operation.")
                        continue
                    else:
                        self.__viewConsoleMatrix(matrixA[1])

                # Creates a pop-up of the heatmap representation of the matrix
                case 11:
                    if matrixA is None:
                        print("Please select a matrix before trying to perform a operation.")
                        continue
                    else:
                        self.__viewImage(matrixA[1])
                # Exit - Ends the loop
                case 12:
                    break

        # When loop is ended returns the user to home
        stateDict = { 1: 1}

        return stateDict[1]      

    # Helper Functions:

    # Returns a tuple with the stripped name of the matrix, and the Matrix Object
    #
    # name - str - The name of a matrix in the matrices folder without the file extension
    def __loadMatrix(self, name):
        
        path = "matrices\\" + name + ".csv"
        return (name, Matrix.createImport(path))
    
    # Regularly used code that is ran after nearly every Matrix Operation
    # Gives the user the ability to see the resulting matrix and lets them save it
    #
    # matrix - a Matrix Object
    def __postOp(self, matrix):
        print("The resulting matrix from the operation is:")
        self.__viewConsoleMatrix(matrix)

        print("Would you like to save this matrix?\n"
              "1.) Yes\n"
              "2.) No"
              )
        
        options = [ 1, 2 ]
        userInput = None
        userInput = self.__inputValidation(options, userInput)

        if userInput == 1:
            print("Please enter the name of the matrix:")
            name = input()
            self.__saveMatrix(name, matrix)
            print("Your matrix has been saved!")

    # Regularly used code that is ran on the Create States,
    # Lets a user see the matrix and lets them choose to save it
    #
    # matrix - a Matrix Object
    def __postCreate(self, matrix):

        print("Would you like to view the created matrix?\n"
              "1.) Yes\n"
              "2.) No"
              )
        
        options = [ 1, 2 ]
        userInput = None
        userInput = self.__inputValidation(options, userInput)

        if userInput == 1:
            self.__viewConsoleMatrix(matrix)

        
        print("Would you like to save the created matrix?\n"
              "1.) Yes\n"
              "2.) No"
              )
        
        options = [ 1, 2 ]
        userInput = None
        userInput = self.__inputValidation(options, userInput)

        if userInput == 1:
            print("Please enter the name of the matrix:")
            name = input()
            self.__saveMatrix(name, matrix)
            print("Your matrix has been saved!")

    # Used to verify that the user correctly selected one of the options provided
    # will infintly loop untill the user puts in a valid option
    #
    # options - a list of ints
    # userInput - a empty variable
    #
    # returns the user's validated input transformed into a int
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
    
    # used to validate that the user provided a number in Manual Matrix Creation
    #
    # returns a float
    def __numberValidation(self, number):
        while True:
            try:
                value = float(number)
                return value
            except:
                print("Invalid input. Please enter a number.")
                number = input()
        
    # prints a numpy matrix to the terminal     
    # 
    # matrix - a matrix object   
    def __viewConsoleMatrix(self, matrix):
        matrix.printMatrix()

    # creates a pop up image of a heatmap of the matrix
    #
    # matrix - a matrix object
    def __viewImage(self, matrix):
        matrix.showVisualization()

    # Attempts to delete a matrix in the matrices folder based on the stripped filename
    #
    # filename - the name of a file in the matrices folder without the file extension
    def __deleteMatrix(self, filename):

        try:
            path = "matrices\\" + filename + ".csv"
            os.remove(path)
        except:
            print("That matrix does not exist in the Matrix Operation Manager.")

    # Attempts to save a matrix in the matrices folder based on the stripped filename
    #
    # filename - the name you want to save a file in the matrices folder without the file extension
    # matrix - a Matrix Object
    def __saveMatrix(self, filename, matrix):

        if matrix is None:
            print("You cannot save a empty or null matrix.")
        else:
            path = "matrices\\" + filename + ".csv"
            np.savetxt(path, matrix.getMatrix(), delimiter=",")

    # Process Functions:
    
    # A dictionary of class functions, with the keys representing the State Number
    __STATES = { 1: __home, 
                 2: __createMatrix,
                 3: __createRandom,
                 4: __createImport,
                 5: __createManual,
                 6: __listSaved,
                 7: __deleteScreen,
                 8: __matrixOperations
                }
    
    # This function runs the State Machine, and ends it when the user reaches state 0
    def __run(self):
        while self.__CurrentState != 0:
            self.__CurrentState = self.__STATES[self.__CurrentState](self)
        
        print("Have a nice day!")
            
        




# %% [markdown]
# # Start Command

# %%
# On start up instantiate the Machine
def main():
    begin = StateMachine()

if __name__ == "__main__":
    main()


