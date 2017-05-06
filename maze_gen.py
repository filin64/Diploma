#This module generates maze depending on size, wall number etc

import numpy as np
n = 32
m = 32
wall_num = 60
maze = [[0 for i in range(n)] for j in range(m)]
def print_maze():
    for i in range(n):
        for j in range(m):
            print (maze[i][j], end='')
        print ('\n')
def write_maze():
    f = open('env/Map_v08', 'w')
    for i in range(n):
        for j in range(m):
            f.writelines(str(maze[i][j]))
        f.write('\n')
    f.close()
def set_bounds():
    for i in range(n):
        for j in range(m):
            if i == 0 or i == n - 1 or j == 0 or j == m - 1:
                maze[i][j] = '#'
def set_hWall(i, j, l):
    try:
        for k in range(l):
            maze[i][j+k] = '#'
    except Exception:
        print ("hWall")
def set_vWall(i, j, l):
    try:
        for k in range(l):
            maze[i+k][j] = '#'
    except Exception:
        print ("vWall")
set_bounds()
for k in range(wall_num):
    i = np.random.randint(2, n)
    j = np.random.randint(2, m)
    l = np.random.randint(5, 10)
    p = np.random.rand()
    if p < 0.5:
        set_hWall(i, j, l)
    else:
        set_vWall(i, j, l)
print_maze()
write_maze()