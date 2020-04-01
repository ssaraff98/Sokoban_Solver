import util
import os, sys
import datetime, time
import argparse
import copy
import math
import signal, gc
from queue import Queue

def __print_map_info__(problem):
    map = problem.map
    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j].wall is True:
                print(f'i = {i} j = {j} WALL')
            elif map[i][j].target is True:
                print(f'i = {i} j = {j} TARGET')
            else:
                print(f'i = {i} j = {j}')

def __compute_deads__(problem):
    #corner checking
    deads = set()
    map = problem.map
    for i in range(1, len(map)): #starts at 1 because first row is outside bounds
        for j in range(1, len(map[i]) - 1): #ends before the last y because it's outside bounds
            if map[i][j].wall is True:
                continue
            try:
                if map[i][j].target is False and (map[i - 1][j].wall is True or map[i + 1][j].wall is True): #block to left or right is wall and current block isn't target
                    try:
                        if map[i][j - 1].wall is True or map[i][j + 1].wall is True: #block above or below is wall
                            deads.add((i, j)) #corner
                            continue
                    except: pass
            except: pass

    #uses corners to find dead spaces along walls
    deadCopy = copy.deepcopy(deads)
    for p1 in deadCopy:
        for p2 in deadCopy:
            addPoints = True
            if abs(p1[0] - p2[0]) < 2 and abs(p1[1] - p2[1]) < 2:
                continue
            if p1[0] == p2[0] or p1[1] == p2[1]:
                if not __check_for_wall_between__(map, p1, p2):
                    points = __wall_continues__(map, p1, p2)
                    for point in points:
                        if map[point[0]][point[1]].target is True:
                            addPoints = False
                    if addPoints:
                        for point in points:
                            deads.add(point)
    return deads

def __wall_continues__(map, p1, p2): #follows a wall from corner to corner, returns unordered list of points between p1 and p2 inclusive
    wallsAbove = True
    wallsBelow = True
    wallsRight = True
    wallsLeft = True
    points = []
    if p1[1] == p2[1]:
        if p1[0] > p2[0]:
            start = p2[0]
            end = p1[0]
        else:
            start = p1[0]
            end = p2[0]
        for i in range(start, end):
            points.append((i, p1[1]))
            if map[i][p1[1] + 1].wall is False:
                wallsAbove = False
            if map[i][p1[1] - 1].wall is False:
                wallsBelow = False
        if wallsAbove or wallsBelow:
            if p1 not in points:
                points.append(p1)
            if p2 not in points:
                points.append(p2)
            return points
        else:
            return []
    elif p1[0] == p2[0]:
        if p1[1] > p2[1]:
            start = p2[1]
            end = p1[1]
        else:
            start = p1[1]
            end = p2[1]
        for i in range(start, end):
            points.append((p1[0], i))
            if map[p1[0] + 1][i].wall is False:
                wallsRight = False
            if map[p1[0] - 1][i].wall is False:
                wallsLeft = False
        if wallsRight or wallsLeft:
            if p1 not in points:
                points.append(p1)
            if p2 not in points:
                points.append(p2)
            return points
        else:
            return []

def __compute_far_walls__(problem): #returns [left wall x val, up wall y val, right wall x val, down wall y val]
    map = problem.map
    walls = []
    Lwall = -1
    Rwall = -1
    Uwall = -1
    Dwall = -1
    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j].wall is True:
                walls.append((i, j))
                if Lwall < 0:
                    Lwall = i
                    Rwall = i
                    Uwall = i
                    Dwall = i
                if i < Lwall:
                    Lwall = i
                if i > Rwall:
                    Rwall = i
                if j < Dwall:
                    Dwall = j
                if j > Uwall:
                    Uwall = j

    return [Lwall, Uwall, Rwall, Dwall]

#checks for wall between two points that are either on the same x or y axis
def __check_for_wall_between__(map, point1, point2):
    wallBetween = False
    if point1[1] == point2[1]:
        if point1[0] > point2[0]:
            start = point2[0]
            end = point1[0]
        else:
            start = point1[0]
            end = point2[0]
        for i in range(start, end):
            if map[i][point1[1]].wall is True:
                wallBetween = True
                break
    elif point1[0] == point2[0]:
        if point1[1] > point2[1]:
            start = point2[1]
            end = point1[1]
        else:
            start = point1[1]
            end = point2[1]
        for i in range(start, end):
            if map[point1[0]][i].wall is True:
                wallBetween = True
                break
    return wallBetween

# Checks if a box is stuck between a wall and a box on adjacent sides
# Returns true if check is true
def __stuck_between_obstacles__(s, problem, x, y):
    map = problem.map

    if map[x - 1][y].wall == True or map[x + 1][y].wall == True or map[x][y - 1].wall == True or map[x][y + 1].wall == True:
        for i, j in s.boxes():
            if i == x and j == y:
                continue
            if ((x - 1) == i and y == j) or ((x + 1) == i and y == j) or (x == i and (y - 1) == j) or (x == i and (y + 1) == j):
                return True

# Checks if a box is stuck between two boxes on adjacent sides
# Returns true if check is true
# !!!! Not implemented during run - FIX !!!!
def __stuck_between_boxes__(s, problem, x, y):
    for i, j in s.boxes():
        if i == x and j == y:
            continue
        if ((x - 1) == i and y == j) or ((x + 1) == i and y == j):
            if (x == i and (y - 1) == j) or (x == i and (y + 1) == j):
                return True

class SokobanState:
    # player: 2-tuple representing player location (coordinates)
    # boxes: list of 2-tuples indicating box locations
    def __init__(self, player, boxes):
        # self.data stores the state
        self.data = tuple([player] + sorted(boxes))
        # below are cache variables to avoid duplicated computation
        self.all_adj_cache = None
        self.adj = {}
        self.dead = None
        self.solved = None
    def __str__(self):
        return 'player: ' + str(self.player()) + ' boxes: ' + str(self.boxes())
    def __eq__(self, other):
        return type(self) == type(other) and self.data == other.data
    def __lt__(self, other):
        return self.data < other.data
    def __hash__(self):
        return hash(self.data)

    # return player location
    def player(self):
        return self.data[0]

    # return boxes locations
    def boxes(self):
        return self.data[1:]

    def is_goal(self, problem):
        if self.solved is None:
            self.solved = all(problem.map[b[0]][b[1]].target for b in self.boxes())
        return self.solved

    def act(self, problem, act):
        if act in self.adj: return self.adj[act]
        else:
            val = problem.valid_move(self, act)
            self.adj[act] = val
            return val

    def deadp(self, problem):
        self.dead = False
        for box in self.data[1:]:
            if box in problem.dead_ends:
                self.dead = True
                break
        return self.dead

    def all_adj(self, problem):
        if self.all_adj_cache is None:
            succ = []
            for move in 'udlr':
                valid, box_moved, nextS = self.act(problem, move)
                if valid:
                    succ.append((move, nextS, 1))
            self.all_adj_cache = succ
        return self.all_adj_cache

class MapTile:
    def __init__(self, wall=False, floor=False, target=False):
        self.wall = wall
        self.floor = floor
        self.target = target

def parse_move(move):
    if move == 'u': return (-1,0)
    elif move == 'd': return (1,0)
    elif move == 'l': return (0,-1)
    elif move == 'r': return (0,1)
    raise Exception('Invalid move character.')

class DrawObj:
    WALL = '\033[37;47m \033[0m'
    PLAYER = '\033[97;40m@\033[0m'
    BOX_OFF = '\033[30;101mX\033[0m'
    BOX_ON = '\033[30;102mX\033[0m'
    TARGET = '\033[97;40m*\033[0m'
    FLOOR = '\033[30;40m \033[0m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class SokobanProblem(util.SearchProblem):
    # valid sokoban characters
    valid_chars = '#@+$*. '

    def __init__(self, map, dead_detection=False):
        self.map = [[]]
        self.dead_detection = dead_detection
        self.init_player = (0,0)
        self.init_boxes = []
        self.numboxes = 0
        self.targets = []
        self.parse_map(map)
        self.dead_ends = __compute_deads__(self)

    # parse the input string into game map
    # Wall              #
    # Player            @
    # Player on target  +
    # Box               $
    # Box on target     *
    # Target            .
    # Floor             (space)
    def parse_map(self, input_str):
        coordinates = lambda: (len(self.map)-1, len(self.map[-1])-1)
        for c in input_str:
            if c == '#':
                self.map[-1].append(MapTile(wall=True))
            elif c == ' ':
                self.map[-1].append(MapTile(floor=True))
            elif c == '@':
                self.map[-1].append(MapTile(floor=True))
                self.init_player = coordinates()
            elif c == '+':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_player = coordinates()
                self.targets.append(coordinates())
            elif c == '$':
                self.map[-1].append(MapTile(floor=True))
                self.init_boxes.append(coordinates())
            elif c == '*':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_boxes.append(coordinates())
                self.targets.append(coordinates())
            elif c == '.':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.targets.append(coordinates())
            elif c == '\n':
                self.map.append([])
        assert len(self.init_boxes) == len(self.targets), 'Number of boxes must match number of targets.'
        self.numboxes = len(self.init_boxes)

    def print_state(self, s):
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                target = self.map[row][col].target
                box = (row,col) in s.boxes()
                player = (row,col) == s.player()
                if box and target:
                    print(DrawObj.BOX_ON, end='')
                elif player and target: print(DrawObj.PLAYER, end='')
                elif target: print(DrawObj.TARGET, end='')
                elif box: print(DrawObj.BOX_OFF, end='')
                elif player: print(DrawObj.PLAYER, end='')
                elif self.map[row][col].wall: print(DrawObj.WALL, end='')
                else: print(DrawObj.FLOOR, end='')
            print()

    # decide if a move is valid
    # return: (whether a move is valid, whether a box is moved, the next state)
    def valid_move(self, s, move, p=None, f=False):
        if p is None:
            p = s.player()
        dx,dy = parse_move(move)
        x1 = p[0] + dx
        y1 = p[1] + dy
        x2 = x1 + dx
        y2 = y1 + dy
        if self.map[x1][y1].wall:
            return False, False, None
        elif (x1,y1) in s.boxes():
            if self.map[x2][y2].floor and (x2,y2) not in s.boxes():
                if f:
                    return True, (x1, y1), SokobanState((x1, y1), [b if b != (x1, y1) else (x2, y2) for b in s.boxes()])
                return True, True, SokobanState((x1,y1),
                    [b if b != (x1,y1) else (x2,y2) for b in s.boxes()])
            else:
                return False, False, None
        else:
            return True, False, SokobanState((x1,y1), s.boxes())

    ##############################################################################
    # Problem 1: Dead end detection                                              #
    # Modify the function below. We are calling the deadp function for the state #
    # so the result can be cached in that state. Feel free to modify any part of #
    # the code or do something different from us.                                #
    # Our solution to this problem affects or adds approximately 50 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    # detect dead end
    def dead_end(self, s):
        if not self.dead_detection:
            return False
        return s.deadp(self)

    def start(self):
        return SokobanState(self.init_player, self.init_boxes)

    def goalp(self, s):
        return s.is_goal(self)

    def expand(self, s):
        if self.dead_end(s):
            return []
        return s.all_adj(self)

def __find_box_path__(s, problem):
    start_state = s.player()
    succ = []
    explored = []

    frontier = Queue()
    frontier.put(start_state)

    while not frontier.empty():
        state = frontier.get()
        if state in explored:
            continue
        explored.append(state)

        for move in 'udlr':
            valid, box_moved, next_state = problem.valid_move(s, move, state, f=True)
            if box_moved:
                succ.append(((box_moved, move), next_state, 1))
            elif valid:
                next_position = next_state.player()
                if next_position not in explored:
                    frontier.put(next_position)
    return succ

class SokobanProblemFaster(SokobanProblem):
    ##############################################################################
    # Problem 2: Action compression                                              #
    # Redefine the expand function in the derived class so that it overrides the #
    # previous one. You may need to modify the solve_sokoban function as well to #
    # account for the change in the action sequence returned by the search       #
    # algorithm. Feel free to make any changes anywhere in the code.             #
    # Our solution to this problem affects or adds approximately 80 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################

    def expand(self, s):
        if self.dead_end(s):
            return []

        box_actions = __find_box_path__(s, self)
        return box_actions

def __manhattan_distance__(coordinate1, coordinate2):
    return abs(coordinate1[0] - coordinate2[0]) + abs(coordinate1[1] - coordinate2[1])

class Heuristic:
    def __init__(self, problem):
        self.problem = problem

    ##############################################################################
    # Problem 3: Simple admissible heuristic                                     #
    # Implement a simple admissible heuristic function that can be computed      #
    # quickly based on Manhattan distance. Feel free to make any changes         #
    # anywhere in the code.                                                      #
    # Our solution to this problem affects or adds approximately 10 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################

    # Heuristic is the sum of minimum distances from every box to a target
    def heuristic(self, s):
        h = 0
        for box in s.boxes():
            if box in self.problem.targets:
                continue

            minimum_distance = math.inf
            for target in self.problem.targets:
                distance = __manhattan_distance__(box, target)

                if distance < minimum_distance:
                    minimum_distance = distance

            h += minimum_distance
        return h

    ##############################################################################
    # Problem 4: Better heuristic.                                               #
    # Implement a better and possibly more complicated heuristic that need not   #
    # always be admissible, but improves the search on more complicated Sokoban  #
    # levels most of the time. Feel free to make any changes anywhere in the     #
    # code. Our heuristic does some significant work at problem initialization   #
    # and caches it.                                                             #
    # Our solution to this problem affects or adds approximately 40 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################

    # Heuristic is the sum of minimum distance from every box to a unique target
    # and distance between each box and the player
    # If box is in a dead state then heuristic is inifinity
    def heuristic2(self, s):
        unused_targets = self.problem.targets.copy()
        h = 0

        for box in s.boxes():
            if box in self.problem.targets:
                continue

            if box in self.problem.dead_ends:
                return math.inf

            h += __manhattan_distance__(box, s.player())

            minimum_distance = math.inf
            best_target = None
            for target in unused_targets:
                distance = __manhattan_distance__(box, target)

                if distance < minimum_distance:
                    minimum_distance = distance
                    best_target = target

            h += minimum_distance
            unused_targets.remove(best_target)
        return h

# solve sokoban map using specified algorithm
def solve_sokoban(map, algorithm='ucs', dead_detection=False):
    # problem algorithm
    if 'f' in algorithm:
        problem = SokobanProblemFaster(map, dead_detection)
    else:
        problem = SokobanProblem(map, dead_detection)

    # search algorithm
    h = Heuristic(problem).heuristic2 if ('2' in algorithm) else Heuristic(problem).heuristic
    if 'a' in algorithm:
        search = util.AStarSearch(heuristic=h)
    else:
        search = util.UniformCostSearch()

    # solve problem
    search.solve(problem)
    if search.actions is not None:
        print('length {} soln is {}'.format(len(search.actions), search.actions))
    if 'f' in algorithm:
        return search.totalCost, search.actions, search.numStatesExplored
    else:
        return search.totalCost, search.actions, search.numStatesExplored

# animate the sequence of actions in sokoban map
def animate_sokoban_solution(map, seq, dt=0.2):
    problem = SokobanProblem(map)
    state = problem.start()
    clear = 'cls' if os.name == 'nt' else 'clear'
    for i in range(len(seq)):
        os.system(clear)
        print(seq[:i] + DrawObj.UNDERLINE + seq[i] + DrawObj.END + seq[i+1:])
        problem.print_state(state)
        time.sleep(dt)
        valid, _, state = problem.valid_move(state, seq[i])
        if not valid:
            raise Exception('Cannot move ' + seq[i] + ' in state ' + str(state))
    os.system(clear)
    print(seq)
    problem.print_state(state)

# read level map from file, returns map represented as string
def read_map_from_file(file, level):
    map = ''
    start = False
    found = False
    with open(file, 'r') as f:
        for line in f:
            if line[0] == "'": continue
            if line.strip().lower()[:5] == 'level':
                if start: break
                if line.strip().lower() == 'level ' + level:
                    found = True
                    start = True
                    continue
            if start:
                if line[0] in SokobanProblem.valid_chars:
                    map += line
                else: break
    if not found:
        raise Exception('Level ' + level + ' not found')
    return map.strip('\n')

# extract all levels from file
def extract_levels(file):
    levels = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip().lower()[:5] == 'level':
                levels += [line.strip().lower()[6:]]
    return levels

def solve_map(file, level, algorithm, dead, simulate):
    map = read_map_from_file(file, level)
    print(map)
    tic = datetime.datetime.now()
    cost, sol, nstates = solve_sokoban(map, algorithm, dead)
    toc = datetime.datetime.now()
    print('Time consumed: {:.3f} seconds using {} and exploring {} states'.format(
        (toc - tic).seconds + (toc - tic).microseconds/1e6, algorithm, nstates))
    seq = ''.join(sol)
    print(len(seq), 'moves')
    print(' '.join(seq[i:i+5] for i in range(0, len(seq), 5)))
    if simulate:
        animate_sokoban_solution(map, seq)

def main():
    parser = argparse.ArgumentParser(description="Solve Sokoban map")
    parser.add_argument("level", help="Level name or 'all'")
    parser.add_argument("algorithm", help="ucs | [f][a[2]] | all")
    parser.add_argument("-d", "--dead", help="Turn on dead state detection (default off)", action="store_true")
    parser.add_argument("-s", "--simulate", help="Simulate the solution (default off)", action="store_true")
    parser.add_argument("-f", "--file", help="File name storing the levels (levels.txt default)", default='levels.txt')
    parser.add_argument("-t", "--timeout", help="Seconds to allow (default 300)", type=int, default=300)

    args = parser.parse_args()
    level = args.level
    algorithm = args.algorithm
    dead = args.dead
    simulate = args.simulate
    file = args.file
    maxSeconds = args.timeout

    if (algorithm == 'all' and level == 'all'):
        raise Exception('Cannot do all levels with all algorithms')

    def solve_now(): solve_map(file, level, algorithm, dead, simulate)

    def solve_with_timeout(maxSeconds):
        try:
            util.TimeoutFunction(solve_now, maxSeconds)()
        except KeyboardInterrupt:
            raise
        except MemoryError as e:
            signal.alarm(0)
            gc.collect()
            print('Memory limit exceeded.')
        except util.TimeoutFunctionException as e:
            signal.alarm(0)
            print('Time limit (%s seconds) exceeded.' % maxSeconds)

    if level == 'all':
        levels = extract_levels(file)
        for level in levels:
            print('Starting level {}'.format(level), file=sys.stderr)
            sys.stdout.flush()
            solve_with_timeout(maxSeconds)
    elif algorithm == 'all':
        for algorithm in ['ucs', 'a', 'a2', 'f', 'fa', 'fa2']:
            print('Starting algorithm {}'.format(algorithm), file=sys.stderr)
            sys.stdout.flush()
            solve_with_timeout(maxSeconds)
    else:
        solve_with_timeout(maxSeconds)

if __name__ == '__main__':
    main()
