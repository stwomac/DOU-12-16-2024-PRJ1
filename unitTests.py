# %%

#imports needed for testing
import unittest
from unittest.mock import patch
import numpy as np
import os
from script import Matrix, StateMachine
import csv


# %%
class TestMatrixClass(unittest.TestCase):
    
    def setUp(self):
        #setup variables
        self.matrix1 = Matrix.createManual([[1,2,3]])
        self.matrix2 = Matrix.createManual([[1,2,3],[4,5,6],[7,8,9]])
        
        # used for import testing, creates a csv file
        self.path = "matrices\\" + "unitTest" + ".csv"
        np.savetxt(self.path, self.matrix2.getMatrix(), delimiter=",")

    def tearDown(self):
        # delete test csv
        os.remove(self.path)

    def testCreate(self):
        #test for when a np array is passed in
        matrix = Matrix(np.array([[1,2,3]]))
        np.testing.assert_allclose(matrix.getMatrix(), self.matrix1.getMatrix())
        
        # test for non numpy array
        matrix = Matrix("huh")
        self.assertEqual(matrix.getMatrix(), None)

    def testCreateRandom(self):
        # input validation tests 
        matrix = Matrix.createRandom(0,0,False)
        self.assertEqual(matrix, None)

        matrix = Matrix.createRandom("hi", 2, True)
        self.assertEqual(matrix, None)

        matrix = Matrix.createRandom( -1, 2, True)
        self.assertEqual(matrix, None)

        matrix = Matrix.createRandom( 1, "hi", True)
        self.assertEqual(matrix, None)

        matrix = Matrix.createRandom(1, -1, True)
        self.assertEqual(matrix, None)

        matrix = Matrix.createRandom(1, 2, "hi")
        self.assertEqual(matrix, None)

        # check to see that random creates a matrix with parameters properly
        matrix = Matrix.createRandom(1, 2, True)
        self.assertEqual(matrix.getRows(), 1)
        self.assertEqual(matrix.getCols(), 2)

        numCheck = matrix.getMatrix() % 1
        numCheck = np.all(numCheck == 0)
        self.assertEqual(numCheck, True)

        matrix = Matrix.createRandom(1, 2, False)
        numCheck = matrix.getMatrix() % 1
        numCheck = np.all(numCheck == 0)
        self.assertEqual(numCheck, False)
        
    def testCreateImport(self):
        # test for import with a path that exists
        matrix = Matrix.createImport(self.path)
        np.testing.assert_allclose(matrix.getMatrix(), self.matrix2.getMatrix())

        # test for import with incorrect path/file type
        matrix = Matrix.createImport(1)
        self.assertEqual(matrix, None)

    def testCreateManual(self):
        # check for invalid inputs on create call
        matrix = Matrix.createManual(1)
        self.assertEqual(matrix, None)

        matrix = Matrix.createManual([])
        self.assertEqual(matrix, None)

        matrix = Matrix.createManual([1,2])
        self.assertEqual(matrix, None)

        # test for creation by passing a python list
        matrix = Matrix.createManual([[1,2,3],[4,5,6],[7,8,9]])
        np.testing.assert_allclose(matrix.getMatrix(), self.matrix2.getMatrix())

    def testAdd(self):
        # test for Matrix Addition
        sum = np.add(np.array([[1,2,3]]), np.array([[1,2,3]]))
        np.testing.assert_allclose(self.matrix1.add(self.matrix1).getMatrix(), sum)

        # test for invalid addition
        matrix = Matrix.createManual([[1,2]])
        self.assertEqual(matrix.add(self.matrix1), None)

    def testSubtract(self):
        # test for Matrix subtraction
        sum = np.subtract(np.array([[1,2,3]]), np.array([[1,2,3]]))
        np.testing.assert_allclose(self.matrix1.subtract(self.matrix1).getMatrix(), sum)

        # test for invalid subtraction
        matrix = Matrix.createManual([[1,2]])
        self.assertEqual(matrix.subtract(self.matrix1), None)

    def testMultiply(self):
        # test for Matrix multiplication
        mult = np.multiply(np.array([[1,2,3]]), np.array([[1],[2],[3]]))
        matrix = self.matrix1.multiply(self.matrix1.transpose())
        np.testing.assert_allclose(matrix.getMatrix(), mult)

        # test for invalid multiplication
        self.assertEqual(self.matrix1.multiply(self.matrix1), None)

    def testTranspose(self):
        # test for Matrix Transposition
        sum = np.subtract(np.array([[1,2,3]]), np.array([[1,2,3]]))
        np.testing.assert_allclose(self.matrix1.subtract(self.matrix1).getMatrix(), sum)

    def testDeterminate(self):
        # test for getting a Determinate
        det = self.matrix2.determinate()
        self.assertEqual(det, np.linalg.det(np.array([[1,2,3],[4,5,6],[7,8,9]])))

        # test for trying to get a determinate from a invalid matrix
        self.assertEqual(self.matrix1.determinate(), None)

    def testInverse(self):
        # test for determinate of 0
        self.assertEqual(self.matrix2.inverse(), None)

        # test for non square matrix
        self.assertEqual(self.matrix1.inverse(), None)

        # test for Matrix inverse
        inv = np.linalg.inv(np.array([[1,2,3],[4,5,6],[7,8,11]]))
        np.testing.assert_allclose(Matrix.createManual([[1,2,3],[4,5,6],[7,8,11]]).inverse().getMatrix(), inv)

    def testIdentity(self):
        # test for invalid Matrix
        matrix = Matrix("huh")
        self.assertEqual(matrix.identity(), None)

        # test for non square Matrix
        self.assertEqual(self.matrix1.identity(), None)

        # test for Matrix Identity
        np.testing.assert_allclose(self.matrix2.identity().getMatrix(), np.identity(3))

    def testZero(self):
        # test for invalid Matrix
        matrix = Matrix("huh")
        self.assertEqual(matrix.zero(), None)

        # test to recieve a zero Matrix
        np.testing.assert_allclose(self.matrix1.zero().getMatrix(), np.zeros((1,3)))

    def testEigenDecomp(self):
        
        # Create the Eigen Values, Vectors, and Decomposition
        nMat = np.array([[1,2,3],[4,5,6],[7,8,9]])
        neVal, neVec = np.linalg.eig(nMat)
        neDec = np.dot(neVec, np.dot(np.diag(neVal), np.linalg.inv(neVec)))

        # Turn the Eigen values into a proper [[]] array
        neValMat = []
        neValMat.append(neVal.tolist())
        neValMat = np.array(neValMat)
        
        # test for eigen decomposition
        matEig = self.matrix2.eigenDecomp()
        np.testing.assert_allclose(matEig[0].getMatrix(), neDec)
        np.testing.assert_allclose(matEig[1].getMatrix(), neVec)
        np.testing.assert_allclose(matEig[2].getMatrix(), neValMat)

    def testGetMatrix(self):
        # test for invalid matrix providing None of through all Getters
        matrix = Matrix("huh")
        self.assertEqual(matrix.getMatrix(), None)

        # test for ability to get the numpy matrix
        np.testing.assert_allclose(self.matrix1.getMatrix(), np.array([[1,2,3]]))

    def testGetRows(self):
        # test for invalid matrix providing None of through all Getters
        matrix = Matrix("huh")
        self.assertEqual(matrix.getRows(), None)

        # test for ability to get the rows
        nMat = np.array([[1,2,3]])
        np.testing.assert_allclose(self.matrix1.getRows(), nMat.shape[0] )

    def testGetCols(self):
        # test for invalid matrix providing None of through all Getters
        matrix = Matrix("huh")
        self.assertEqual(matrix.getCols(), None)

        # test for ability to get the columns
        nMat = np.array([[1,2,3]])
        np.testing.assert_allclose(self.matrix1.getCols(), nMat.shape[1] )

    @patch("matplotlib.pyplot.show")
    def testShowVisualization(self, mock_show):
        # test to see that Heatmap popup goes off
        self.matrix1.showVisualization()
        mock_show.assert_called_once()

    @patch('builtins.print')
    def testPrintMatrix(self, mock_print):
        # test to see that program does not crash when you try to print a null Matrix
        matrix = Matrix("huh")
        matrix.printMatrix()
        mock_print.assert_called_with('None')

        # test to see that matrix prints to console correctly
        self.matrix1.printMatrix()
        mock_print.assert_called_with('[[1 2 3]]')

# %%

# plan of attack here is to mock the inputs into creating a matrix, 
# verifying that it was created correctly, then deleting it
# the state machine will be booted up at the start of each test, then ended
# then we verify matrix, then delete it
class TestStateMachineClass(unittest.TestCase):

    def setUp(self):
        
        # Matrices used for testing functionality        
        
        # used to create a empty Matrix file
        path = "matrices\\" + "testNone" + ".csv"
        try:
            np.savetxt(path, Matrix("a").getMatrix(), delimiter=",")
        except:
            pass

        path = "matrices\\" + "test1x3" + ".csv"
        np.savetxt(path, Matrix.createManual([[1,2,3]]).getMatrix(), delimiter=",")
        self.test1x3 = ("test1x3", Matrix.createImport(path))

        path = "matrices\\" + "testInvertible" + ".csv"
        np.savetxt(path, Matrix.createManual([[1,2,3],[4,5,6],[7,8,11]]).getMatrix(), delimiter=",")
        self.testInvertible = ("testInvertible", Matrix.createImport(path))

        path = "matrices\\" + "test3x3" + ".csv"
        np.savetxt(path, Matrix.createManual([[1,2,3],[4,5,6],[7,8,9]]).getMatrix(), delimiter=",")
        self.test3x3 = ( "test3x3", Matrix.createImport(path))

        # used to create a csv with mixed values to show improper imported files
        path = "matrices\\" + "testNonNumeric" + ".csv"
        mixedArray = np.array([[1,"lol",3]],dtype=object)
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in mixedArray:
                writer.writerow(row)

        self.testNonNumeric = ( "testNonNumeric", Matrix.createImport(path))

    def tearDown(self):
        # tear down created test files

        path = "matrices\\" + "testNone" + ".csv"
        os.remove(path)
        path = "matrices\\" + "test1x3" + ".csv"
        os.remove(path)
        path = "matrices\\" + "testInvertible" + ".csv"
        os.remove(path)
        path = "matrices\\" + "test3x3" + ".csv"
        os.remove(path)
        path = "matrices\\" + "testNonNumeric" + ".csv"
        os.remove(path)

    # Tests below utilize mock input to test for UI fuctionality, they list of strings
    # details the path a user would need to take to perform actions with the script

    @patch('script.input', create=True)
    def testFolderCreate(self, mock_input):
        # test to see that saved matrices folder exists on when program is started
        mock_input.side_effect = ["5"]
        StateMachine()
        folder1 = "matrices"
        os.chdir(".")
        self.assertEqual(os.path.isdir(folder1), True)
        
    @patch('script.input', create=True)
    def testInputValidation(self, mock_input):
        # test to see private inputValidation method works
        mock_input.side_effect = ["12", "5"]
        StateMachine()
    
    @patch('script.input', create=True)
    def testPostCreate(self, mock_input):
        # test to see private postCreate method works (show matrix + save matrix)
        mock_input.side_effect = ["1", "1", "1", "1", "1", "testPostCreate","3", "5"]
        StateMachine()

        path = "matrices\\" + "testPostCreate" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        os.remove(path)

        # test to see private postCreate method works (do not show matrix + save matrix)
        mock_input.side_effect = ["1", "1", "1", "2", "1", "testPostCreate","3", "5"]
        StateMachine()

        path = "matrices\\" + "testPostCreate" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        os.remove(path)

        # test to see private postCreate method works (show matrix + do not save matrix)
        mock_input.side_effect = ["1", "1", "1", "1", "2", "3", "5"]
        StateMachine()

        # test to see private postCreate method works (do not show matrix + do not save matrix)
        mock_input.side_effect = ["1", "1", "1", "2", "2", "3", "5"]
        StateMachine()
    
    @patch('script.input', create=True)
    def testHome(self, mock_input):
        
        # test to see that you can exit program from the home screen
        mock_input.side_effect = ["5"]
        StateMachine()
    
    @patch('script.input', create=True)
    def testCreateMatrix(self, mock_input):
        # test to see that you can reach the create matrix screen and then exit
        mock_input.side_effect = ["1", "4", "5"]
        StateMachine()

    @patch('script.input', create=True)
    def testCreateRandom(self, mock_input):

        #completely random test
        mock_input.side_effect = ["1", "1", "1", # Create completely random
                                  "1", "1", "test","3", "5"] # Save Matrix and Exit
        StateMachine()
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        os.remove(path)

        #random with correct parameters + whole nums
        mock_input.side_effect = ["1", "1", "2", # Create with Parameters
                                  "1", "2", "1",            # Parameters Chosen
                                  "1", "1", "test","3", "5"] # Save Matrix and Exit
        StateMachine()
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        os.remove(path)

        #random with correct parameters without whole nums
        mock_input.side_effect = ["1", "1", "2", # Create with Parameters
                                  "1", "2", "2",            # Parameters Chosen
                                  "1", "1", "test","3", "5"] # Save Matrix and Exit
        StateMachine()
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        os.remove(path)

        #random with negative rows
        mock_input.side_effect = ["1", "1", "2", # Create with Parameters
                                  "-1", "1", "2", "1",            # Parameters Chosen
                                  "1", "1", "test","3", "5"] # Save Matrix and Exit
        StateMachine()
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        os.remove(path)

        #random with a string for a row
        mock_input.side_effect = ["1", "1", "2", # Create with Parameters
                                  "lol", "1", "2", "1",            # Parameters Chosen
                                  "1", "1", "test","3", "5"] # Save Matrix and Exit
        StateMachine()
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        os.remove(path)

        #random with zero rows
        mock_input.side_effect = ["1", "1", "2", # Create with Parameters
                                  "0", "1", "2", "1",            # Parameters Chosen
                                  "1", "1", "test","3", "5"] # Save Matrix and Exit
        StateMachine()
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        os.remove(path)
        
        # random with negative columns
        mock_input.side_effect = ["1", "1", "2", # Create with Parameters
                                  "1", "-2", "2", "1",            # Parameters Chosen
                                  "1", "1", "test","3", "5"] # Save Matrix and Exit
        StateMachine()
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        os.remove(path)
        
        #random with a string for a column
        mock_input.side_effect = ["1", "1", "2", # Create with Parameters
                                  "1", "wat", "2", "1",            # Parameters Chosen
                                  "1", "1", "test","3", "5"] # Save Matrix and Exit
        StateMachine()
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        os.remove(path)

        #random with zero columns
        mock_input.side_effect = ["1", "1", "2", # Create with Parameters
                                  "1", "0", "2", "1",            # Parameters Chosen
                                  "1", "1", "test","3", "5"] # Save Matrix and Exit
        StateMachine()
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        os.remove(path)

    @patch('script.input', create=True)
    def testCreateImport(self, mock_input):

        # test normal import
        path = "matrices/" + self.test1x3[0] + ".csv"
        mock_input.side_effect = ["1", "2", path, # Create with import
                                  "1", "1", "test", "2", "5"] # Save Matrix and Exit
        StateMachine()
        
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.test1x3[1].getMatrix())
        os.remove(path)

        # test no/wrong path import
        path = "matrices/" + "doesNotExist" + ".csv"
        mock_input.side_effect = ["1", "2", path, # Create with import
                                  "2", "5"] # Save Matrix and Exit
        StateMachine()
        
        path = "matrices\\" + "doesNotExist" + ".csv"
        self.assertEqual(os.path.exists(path), False)
        
        # test try again yes
        path1 = "matrices/" + "doesNotExist" + ".csv"
        path2 = "matrices/" + self.test1x3[0] + ".csv"
        mock_input.side_effect = ["1", "2", path1,"1", path2, # Create with import twice
                                  "1", "1", "test", "2", "5"] # Save Matrix and Exit
        StateMachine()
        
        path = "matrices\\" + "doesNotExist" + ".csv"
        self.assertEqual(os.path.exists(path), False)

        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.test1x3[1].getMatrix())
        os.remove(path)

        # # test try again no
        path = "matrices/" + "test" + ".csv"
        mock_input.side_effect = ["1", "2", path, # Create with import
                                  "2", "5"] # Save Matrix and Exit
        StateMachine()
        
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), False)

        # # test import of non numeric import
        path = "matrices/" + self.testNonNumeric[0] + ".csv"
        mock_input.side_effect = ["1", "2", path, # Create with import
                                  "2", "5"] # Save Matrix and Exit
        StateMachine()
        
        # # test seconday import yes
        path = "matrices/" + self.test1x3[0] + ".csv"
        mock_input.side_effect = ["1", "2", path, # Create with import
                                  "1", "1", "test", # first matrix saved
                                  "1", path, # Create with import
                                  "1", "1", "test2", # second matrix saved
                                  "2", "5"] # Exit
        StateMachine()
        
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.test1x3[1].getMatrix())
        os.remove(path)

        path = "matrices\\" + "test2" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.test1x3[1].getMatrix())
        os.remove(path)

        # # test seconday import no
        path = "matrices/" + self.test1x3[0] + ".csv"
        mock_input.side_effect = ["1", "2", path, # Create with import
                                  "1", "1", "test", "2", "5"] # Save Matrix and Exit
        StateMachine()
        
        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.test1x3[1].getMatrix())
        os.remove(path)
        
    @patch('script.input', create=True)
    def testCreateManual(self, mock_input):
        # np.savetxt(path, Matrix.createManual([[1,2,3]]).getMatrix(), delimiter=",")
        
        mock_input.side_effect = ["1", "3",  # Create with manual
                                  "huh", "-3", "2.2", # wrong input row test
                                  "1", #row input
                                  "huh", "-3", "2.2", # wrong input column test
                                  "3", #column input
                                  "1", "2", "3", #matrix input
                                  "1", "1", "testa", "2", "5"] # Save Matrix and Exit

        StateMachine()

        path = "matrices\\" + "testa" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.test1x3[1].getMatrix())
        os.remove(path)
        
        #test the create loop
        mock_input.side_effect = ["1", "3",  # Create with manual
                                  "1", #row input
                                  "3", #column input
                                  "1", "2", "3", #matrix input
                                  "2", "2", "1", 
                                  "1", #row input
                                  "3", #column input
                                  "1", "2", "3", #matrix input
                                  "1", "1", "test2",
                                  "2","5"] # Save Matrix and Exit
        
        StateMachine()

        path = "matrices\\" + "test2" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.test1x3[1].getMatrix())
        os.remove(path)

    @patch('script.input', create=True)
    def testListSaved(self, mock_input):
        mock_input.side_effect = ["2", "5"] # List Matrices and Exit

        StateMachine()

    @patch('script.input', create=True)
    def testDeleteScreen(self, mock_input):
        
        # test Single Delete
        path = "matrices\\" + "testDelete" + ".csv"
        np.savetxt(path, Matrix.createManual([[1,2,3]]).getMatrix(), delimiter=",")
        self.assertEqual(os.path.exists(path), True)

        mock_input.side_effect = ["4", "testDelete", "2" ,"5"] # Delete Matrix and Exit
        StateMachine()

        path = "matrices\\" + "testDelete" + ".csv"
        self.assertEqual(os.path.exists(path), False)

        # test Multi-Delete
        path = "matrices\\" + "testDelete1" + ".csv"
        np.savetxt(path, Matrix.createManual([[1,2,3]]).getMatrix(), delimiter=",")
        self.assertEqual(os.path.exists(path), True)

        path = "matrices\\" + "testDelete2" + ".csv"
        np.savetxt(path, Matrix.createManual([[1,2,3]]).getMatrix(), delimiter=",")
        self.assertEqual(os.path.exists(path), True)

        mock_input.side_effect = ["4", "testDelete1", "1", "testDelete2", "2" ,"5"] # Delete Matrix and Exit
        StateMachine()

        path = "matrices\\" + "testDelete1" + ".csv"
        self.assertEqual(os.path.exists(path), False)
        path = "matrices\\" + "testDelete2" + ".csv"
        self.assertEqual(os.path.exists(path), False)

        # test trying to delete a matrix that doesnt exist
        mock_input.side_effect = ["4", "testDelete3", "2" ,"5"] # Delete Matrix and Exit
        StateMachine()

    @patch('script.input', create=True)
    def testMatrixOpLoadRequired(self, mock_input):
        # test that to use matrix operations you most go through option 1 first
        mock_input.side_effect = ["3", # Enter Matrix Operations
                                  "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", # try each operation without loading a matrix
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

    @patch('script.input', create=True)
    def testMatrixOp1(self, mock_input):
        #test load matrix and exit
        mock_input.side_effect = ["3", "1", # Matrix Operation
                                  self.test1x3[0], 
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        # test trying to load a non-existant matrix
        mock_input.side_effect = ["3", "1", # Matrix Operation
                                  "testNonExist", 
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()
    
    @patch('script.input', create=True)
    def testMatrixOp2(self, mock_input):
        # test add non existent matrix
        mock_input.side_effect = ["3", "1", self.test3x3[0], "2", # Load Matrix  and perform Operation
                                  "testNonExist", "2", # try to add non-existent matrix
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        # test add
        mock_input.side_effect = ["3", "1", self.test3x3[0], "2", # Load Matrix  and perform Operation
                                  "testNonExist", # Try to add a nonexistent Matrix
                                  "1", self.test3x3[0], # Retry adding one that does exist
                                  "1", "test", # Save Matrix
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.test3x3[1].add(self.test3x3[1]).getMatrix())
        os.remove(path)

        # test incompatible addition
        mock_input.side_effect = ["3", "1", self.test1x3[0], "2", # Load Matrix  and perform Operation
                                  "testNonExist", "1", self.test3x3[0], # Incompatible Addition
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

    @patch('script.input', create=True)
    def testMatrixOp3(self, mock_input):
        # test subtract non existent matrix
        mock_input.side_effect = ["3", "1", self.test3x3[0], "3", # Load Matrix  and perform Operation
                                  "testNonExist", "2", # try to subtract non-existent matrix
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        # test subtract
        mock_input.side_effect = ["3", "1", self.test3x3[0], "3", # Load Matrix  and perform Operation
                                  "testNonExist", # Try to subtract a nonexistent Matrix
                                  "1", self.test3x3[0], # Retry subtracting one that does exist
                                  "1", "test", # Save Matrix
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.test3x3[1].subtract(self.test3x3[1]).getMatrix())
        os.remove(path)

        # test incompatible subtraction
        mock_input.side_effect = ["3", "1", self.test1x3[0], "3", # Load Matrix  and perform Operation
                                  "testNonExist", "1", self.test3x3[0], # Incompatible Subtraction
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

    @patch('script.input', create=True)
    def testMatrixOp4(self, mock_input):
        # test multiply non existent matrix
        mock_input.side_effect = ["3", "1", self.test3x3[0], "4", # Load Matrix  and perform Operation
                                  "testNonExist", "2", 
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        # test multiply
        mock_input.side_effect = ["3", "1", self.test3x3[0], "4", # Load Matrix  and perform Operation
                                  "testNonExist", # Try to multiply a nonexistent Matrix
                                  "1", self.test3x3[0], # Retry multiply one that does exist
                                  "1", "test", # Save Matrix
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.test3x3[1].multiply(self.test3x3[1]).getMatrix())
        os.remove(path)

        # test incompatible multiply
        mock_input.side_effect = ["3", "1", self.test1x3[0], "4", # Load Matrix  and perform Operation
                                  "testNonExist", "1", self.test1x3[0], # Incompatible Multiplication
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), False)

    @patch('script.input', create=True)
    def testMatrixOp5(self, mock_input):
        # test transpose
        mock_input.side_effect = ["3", "1", self.test1x3[0], "5", # Load Matrix  and perform Operation
                                  "1", "testT", # Save Matrix
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        path = "matrices\\" + "testT" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.test1x3[1].transpose().getMatrix())
        os.remove(path)

    @patch('script.input', create=True)
    def testMatrixOp6(self, mock_input):
        # test incompatible determinate
        mock_input.side_effect = ["3", "1", self.test1x3[0], "6", # Load Matrix  and perform Operation
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        # test determinate
        mock_input.side_effect = ["3", "1", self.test3x3[0], "6", # Load Matrix  and perform Operation
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()


    @patch('script.input', create=True)
    def testMatrixOp7(self, mock_input):
        # test inverse
        mock_input.side_effect = ["3", "1", self.testInvertible[0], "7", # Load Matrix  and perform Operation
                                  "1", "test", # Save Matrix
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.testInvertible[1].inverse().getMatrix())
        os.remove(path)

        # test incompatible inverse
        mock_input.side_effect = ["3", "1", self.test1x3[0], "7", # Load Matrix  and perform Operation
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), False)

    @patch('script.input', create=True)
    def testMatrixOp8(self, mock_input):
        # test identity
        mock_input.side_effect = ["3", "1", self.test3x3[0], "8", # Load Matrix  and perform Operation
                                  "1", "test", # Save Matrix
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), self.test3x3[1].identity().getMatrix())
        os.remove(path)

        # test incompatible identity
        mock_input.side_effect = ["3", "1", self.test1x3[0], "8", # Load Matrix  and perform Operation
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        path = "matrices\\" + "test" + ".csv"
        self.assertEqual(os.path.exists(path), False)

    @patch('script.input', create=True)
    def testMatrixOp9(self, mock_input):
        # Test eigen decomposition 
        mock_input.side_effect = ["3", "1", self.test3x3[0], "9", # Load Matrix  and perform Operation
                                  "1", "testA", "1", "testB", "1", "testC", # Save Matrices
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

        testArray = self.test3x3[1].eigenDecomp()
        path = "matrices\\" + "testa" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), testArray[2].getMatrix())
        os.remove(path)

        path = "matrices\\" + "testb" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), testArray[1].getMatrix())
        os.remove(path)

        path = "matrices\\" + "testc" + ".csv"
        self.assertEqual(os.path.exists(path), True)
        np.testing.assert_allclose(Matrix.createImport(path).getMatrix(), testArray[0].getMatrix())
        os.remove(path)

        # test eigen decomposition without saving
        mock_input.side_effect = ["3", "1", self.test3x3[0], "9", # Load Matrix  and perform Operation
                                  "2", "2", "2", # Do Not Save Matrices
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        StateMachine()

    @patch('script.input', create=True)
    def testMatrixOp10(self, mock_input):
        # Test print to console
        mock_input.side_effect = ["3", "1", self.test1x3[0], "10", # Load Matrix  and perform Operation
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        
        StateMachine()


    @patch("script.plt.show")
    @patch('script.input', create=True)
    def testMatrixOp11(self, mock_input, mock_show):
        # test for Matrix Visualization pop-up
        mock_input.side_effect = ["3", "1", self.test1x3[0], "11", # Load Matrix  and perform Operation
                                  "12" ,"5"] # Exit Matrix Operations and Exit Program
        
        StateMachine()
        mock_show.assert_called_once()
        pass

   
    @patch('script.input', create=True)
    def testMatrixOp12(self, mock_input):
        # test for exiting matrix operations
        mock_input.side_effect = ["3", "12" ,"5"] 
        
        StateMachine()


if __name__ == "__main__":
    unittest.main()


