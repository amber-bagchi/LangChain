from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = """# Python Program to solve the n-queens problem

# Function to check if it is safe to place
# the queen at board[row][col]
def isSafe(mat, row, col):
    n = len(mat)

    # Check this col on upper side
    for i in range(row):
        if mat[i][col]:
            return False

    # Check upper diagonal on left side
    for i, j in zip(range(row - 1, -1, -1), 
                    range(col - 1, -1, -1)):
        if mat[i][j]:
            return False

    # Check upper diagonal on right side
    for i, j in zip(range(row - 1, -1, -1), 
                    	range(col + 1, n)):
        if mat[i][j]:
            return False

    return True

def placeQueens(row, mat):
    n = len(mat)

    # If all queens are placed
    # then return true
    if row == n:
        return True

    # Consider the row and try placing
    # queen in all columns one by one
    for i in range(n):

        # Check if the queen can be placed
        if isSafe(mat, row, i):
            mat[row][i] = 1
            if placeQueens(row + 1, mat):
                return True
            mat[row][i] = 0

    return False

# Function to find the solution
# to the N-Queens problem
def nQueen(n):

    # Initialize the board
    mat = [[0 for _ in range(n)] for _ in range(n)]

    # If the solution exists
    if placeQueens(0, mat):

        # to store the columns of the queens
        ans = []
        for i in range(n):
            for j in range(n):
                if mat[i][j]:
                    ans.append(j + 1)
        return ans
    else:
        return [-1]

if __name__ == "__main__":
    n = 4
    ans = nQueen(n)
    print(" ".join(map(str, ans)))

"""


spliter = RecursiveCharacterTextSplitter.from_language(language='python', chunk_size=800)

result = spliter.split_text(text)
print(result[0])
print(len(result))


