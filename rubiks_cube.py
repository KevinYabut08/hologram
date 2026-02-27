import numpy as np
from enum import Enum

class Face(Enum):
    FRONT = 0
    BACK = 1
    UP = 2
    DOWN = 3
    LEFT = 4
    RIGHT = 5

class Color(Enum):
    WHITE = 0
    YELLOW = 1
    RED = 2
    ORANGE = 3
    GREEN = 4
    BLUE = 5

class RubiksCube:
    def __init__(self):
        # Initialize solved cube with standard Rubik's Cube colors
        self.cube = {
            Face.FRONT: np.full((3, 3), Color.GREEN),
            Face.BACK: np.full((3, 3), Color.BLUE),
            Face.UP: np.full((3, 3), Color.WHITE),
            Face.DOWN: np.full((3, 3), Color.YELLOW),
            Face.LEFT: np.full((3, 3), Color.ORANGE),
            Face.RIGHT: np.full((3, 3), Color.RED)
        }
        
        # Scramble moves
        self.move_history = []
        self.is_scrambled = False
        
        # Game state
        self.move_count = 0
        self.start_time = None
        
    def scramble(self, moves=20):
        """Scramble the cube with random moves"""
        moves_list = ['U', 'U\'', 'D', 'D\'', 'L', 'L\'', 'R', 'R\'', 'F', 'F\'', 'B', 'B\'']
        scramble_moves = []
        
        for _ in range(moves):
            move = np.random.choice(moves_list)
            self.perform_move(move)
            scramble_moves.append(move)
            
        self.is_scrambled = True
        self.move_history = scramble_moves
        self.move_count = 0
        return " ".join(scramble_moves)
    
    def perform_move(self, move):
        """Perform a Rubik's cube move notation"""
        if move == 'U':
            self._rotate_U(clockwise=True)
        elif move == 'U\'':
            self._rotate_U(clockwise=False)
        elif move == 'D':
            self._rotate_D(clockwise=True)
        elif move == 'D\'':
            self._rotate_D(clockwise=False)
        elif move == 'L':
            self._rotate_L(clockwise=True)
        elif move == 'L\'':
            self._rotate_L(clockwise=False)
        elif move == 'R':
            self._rotate_R(clockwise=True)
        elif move == 'R\'':
            self._rotate_R(clockwise=False)
        elif move == 'F':
            self._rotate_F(clockwise=True)
        elif move == 'F\'':
            self._rotate_F(clockwise=False)
        elif move == 'B':
            self._rotate_B(clockwise=True)
        elif move == 'B\'':
            self._rotate_B(clockwise=False)
    
    def _rotate_U(self, clockwise=True):
        """Rotate UP face"""
        # Rotate the face itself
        self.cube[Face.UP] = np.rot90(self.cube[Face.UP], -1 if clockwise else 1)
        
        # Rotate adjacent edges
        if clockwise:
            # Save front top row
            temp = self.cube[Face.FRONT][0, :].copy()
            # Front top row <- right top row
            self.cube[Face.FRONT][0, :] = self.cube[Face.RIGHT][0, :]
            # Right top row <- back top row
            self.cube[Face.RIGHT][0, :] = self.cube[Face.BACK][0, :]
            # Back top row <- left top row
            self.cube[Face.BACK][0, :] = self.cube[Face.LEFT][0, :]
            # Left top row <- saved front top row
            self.cube[Face.LEFT][0, :] = temp
        else:
            # Counter-clockwise
            temp = self.cube[Face.FRONT][0, :].copy()
            self.cube[Face.FRONT][0, :] = self.cube[Face.LEFT][0, :]
            self.cube[Face.LEFT][0, :] = self.cube[Face.BACK][0, :]
            self.cube[Face.BACK][0, :] = self.cube[Face.RIGHT][0, :]
            self.cube[Face.RIGHT][0, :] = temp
    
    def _rotate_D(self, clockwise=True):
        """Rotate DOWN face"""
        self.cube[Face.DOWN] = np.rot90(self.cube[Face.DOWN], -1 if clockwise else 1)
        
        if clockwise:
            temp = self.cube[Face.FRONT][2, :].copy()
            self.cube[Face.FRONT][2, :] = self.cube[Face.LEFT][2, :]
            self.cube[Face.LEFT][2, :] = self.cube[Face.BACK][2, :]
            self.cube[Face.BACK][2, :] = self.cube[Face.RIGHT][2, :]
            self.cube[Face.RIGHT][2, :] = temp
        else:
            temp = self.cube[Face.FRONT][2, :].copy()
            self.cube[Face.FRONT][2, :] = self.cube[Face.RIGHT][2, :]
            self.cube[Face.RIGHT][2, :] = self.cube[Face.BACK][2, :]
            self.cube[Face.BACK][2, :] = self.cube[Face.LEFT][2, :]
            self.cube[Face.LEFT][2, :] = temp
    
    def _rotate_L(self, clockwise=True):
        """Rotate LEFT face"""
        self.cube[Face.LEFT] = np.rot90(self.cube[Face.LEFT], -1 if clockwise else 1)
        
        if clockwise:
            # Save front left column
            temp = self.cube[Face.FRONT][:, 0].copy()
            # Front left column <- up left column
            self.cube[Face.FRONT][:, 0] = self.cube[Face.UP][:, 0]
            # Up left column <- back right column (reversed)
            self.cube[Face.UP][:, 0] = self.cube[Face.BACK][::-1, 2]
            # Back right column <- down left column
            self.cube[Face.BACK][:, 2] = self.cube[Face.DOWN][::-1, 0]
            # Down left column <- saved temp
            self.cube[Face.DOWN][:, 0] = temp
        else:
            temp = self.cube[Face.FRONT][:, 0].copy()
            self.cube[Face.FRONT][:, 0] = self.cube[Face.DOWN][:, 0]
            self.cube[Face.DOWN][:, 0] = self.cube[Face.BACK][::-1, 2]
            self.cube[Face.BACK][:, 2] = self.cube[Face.UP][::-1, 0]
            self.cube[Face.UP][:, 0] = temp
    
    def _rotate_R(self, clockwise=True):
        """Rotate RIGHT face"""
        self.cube[Face.RIGHT] = np.rot90(self.cube[Face.RIGHT], -1 if clockwise else 1)
        
        if clockwise:
            temp = self.cube[Face.FRONT][:, 2].copy()
            self.cube[Face.FRONT][:, 2] = self.cube[Face.DOWN][:, 2]
            self.cube[Face.DOWN][:, 2] = self.cube[Face.BACK][::-1, 0]
            self.cube[Face.BACK][:, 0] = self.cube[Face.UP][::-1, 2]
            self.cube[Face.UP][:, 2] = temp
        else:
            temp = self.cube[Face.FRONT][:, 2].copy()
            self.cube[Face.FRONT][:, 2] = self.cube[Face.UP][:, 2]
            self.cube[Face.UP][:, 2] = self.cube[Face.BACK][::-1, 0]
            self.cube[Face.BACK][:, 0] = self.cube[Face.DOWN][::-1, 2]
            self.cube[Face.DOWN][:, 2] = temp
    
    def _rotate_F(self, clockwise=True):
        """Rotate FRONT face"""
        self.cube[Face.FRONT] = np.rot90(self.cube[Face.FRONT], -1 if clockwise else 1)
        
        if clockwise:
            temp = self.cube[Face.UP][2, :].copy()
            # Up bottom row <- left right column (reversed)
            self.cube[Face.UP][2, :] = self.cube[Face.LEFT][::-1, 2]
            # Left right column <- down top row
            self.cube[Face.LEFT][:, 2] = self.cube[Face.DOWN][0, :]
            # Down top row <- right left column (reversed)
            self.cube[Face.DOWN][0, :] = self.cube[Face.RIGHT][::-1, 0]
            # Right left column <- saved temp
            self.cube[Face.RIGHT][:, 0] = temp
        else:
            temp = self.cube[Face.UP][2, :].copy()
            self.cube[Face.UP][2, :] = self.cube[Face.RIGHT][:, 0]
            self.cube[Face.RIGHT][:, 0] = self.cube[Face.DOWN][0, ::-1]
            self.cube[Face.DOWN][0, :] = self.cube[Face.LEFT][:, 2]
            self.cube[Face.LEFT][:, 2] = temp[::-1]
    
    def _rotate_B(self, clockwise=True):
        """Rotate BACK face"""
        self.cube[Face.BACK] = np.rot90(self.cube[Face.BACK], -1 if clockwise else 1)
        
        if clockwise:
            temp = self.cube[Face.UP][0, :].copy()
            self.cube[Face.UP][0, :] = self.cube[Face.RIGHT][:, 2]
            self.cube[Face.RIGHT][:, 2] = self.cube[Face.DOWN][2, ::-1]
            self.cube[Face.DOWN][2, :] = self.cube[Face.LEFT][:, 0]
            self.cube[Face.LEFT][:, 0] = temp[::-1]
        else:
            temp = self.cube[Face.UP][0, :].copy()
            self.cube[Face.UP][0, :] = self.cube[Face.LEFT][::-1, 0]
            self.cube[Face.LEFT][:, 0] = self.cube[Face.DOWN][2, :]
            self.cube[Face.DOWN][2, :] = self.cube[Face.RIGHT][::-1, 2]
            self.cube[Face.RIGHT][:, 2] = temp
    
    def check_solved(self):
        """Check if cube is solved (all faces uniform)"""
        for face in Face:
            # Check if all stickers on a face are the same color
            if not np.all(self.cube[face] == self.cube[face][0,0]):
                return False
        return True
    
    def get_face_colors(self, face):
        """Get colors for a specific face as RGB values"""
        color_map = {
            Color.WHITE: (1.0, 1.0, 1.0),    # White
            Color.YELLOW: (1.0, 1.0, 0.0),   # Yellow
            Color.RED: (1.0, 0.0, 0.0),      # Red
            Color.ORANGE: (1.0, 0.5, 0.0),   # Orange
            Color.GREEN: (0.0, 1.0, 0.0),    # Green
            Color.BLUE: (0.0, 0.0, 1.0)      # Blue
        }
        
        # Return 3x3 array of RGB tuples
        face_colors = []
        for i in range(3):
            row = []
            for j in range(3):
                color = self.cube[face][i, j]
                row.append(color_map[color])
            face_colors.append(row)
        return face_colors
    
    def get_solution_moves(self):
        """Get solution moves from current state"""
        # Simplified: Just return reversed scramble
        solution = []
        for move in reversed(self.move_history):
            if "'" in move:
                solution.append(move.replace("'", ""))
            else:
                solution.append(move + "'")
        return solution
    
    def get_cube_state(self):
        """Get current cube state as a string"""
        state = ""
        for face in [Face.UP, Face.LEFT, Face.FRONT, Face.RIGHT, Face.BACK, Face.DOWN]:
            for i in range(3):
                for j in range(3):
                    state += str(self.cube[face][i, j].value)
        return state