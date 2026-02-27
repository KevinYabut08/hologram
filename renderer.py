import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import time

class ImmersiveCubeRenderer:
    def __init__(self):
        pygame.init()
        self.display = (1600, 1000)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("🎮 PHYSICAL RUBIK'S CUBE - Grab & Twist!")
        
        # Initialize OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glTranslatef(0.0, 0.0, -10)
        
        # Multiple lights for better 3D feel
        glLightfv(GL_LIGHT0, GL_POSITION, [5, 5, 5, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        
        glLightfv(GL_LIGHT1, GL_POSITION, [-5, -5, 5, 1])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.5, 0.5, 0.5, 1])
        
        # Cube visual properties
        self.cube_size = 2.5  # Larger cube
        self.cubie_size = self.cube_size / 3.0
        self.cubie_gap = 0.05  # More gap for tactile feel
        self.bevel_size = 0.08  # Rounded edges
        
        # Materials for each color
        self.setup_materials()
        
        # Animation states
        self.animating = False
        self.animation_angle = 0
        self.animating_face = None
        self.animating_direction = 1
        
        # Physical feedback
        self.grab_highlight = 0
        self.twist_feedback = 0
        self.snap_time = 0
        
        # Cube logic (you'll need to integrate your existing cube logic)
        # self.rubiks_cube = RubiksCube()
        
        # Font
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
    
    def setup_materials(self):
        """Setup materials with physical properties"""
        # Shiny plastic materials
        self.materials = {
            'white': {'diffuse': [1.0, 1.0, 1.0], 'specular': [0.7, 0.7, 0.7], 'shininess': 100},
            'yellow': {'diffuse': [1.0, 1.0, 0.0], 'specular': [0.7, 0.7, 0.0], 'shininess': 100},
            'red': {'diffuse': [1.0, 0.0, 0.0], 'specular': [0.7, 0.0, 0.0], 'shininess': 100},
            'orange': {'diffuse': [1.0, 0.5, 0.0], 'specular': [0.7, 0.35, 0.0], 'shininess': 100},
            'blue': {'diffuse': [0.0, 0.0, 1.0], 'specular': [0.0, 0.0, 0.7], 'shininess': 100},
            'green': {'diffuse': [0.0, 1.0, 0.0], 'specular': [0.0, 0.7, 0.0], 'shininess': 100},
            'black': {'diffuse': [0.1, 0.1, 0.1], 'specular': [0.3, 0.3, 0.3], 'shininess': 50},
            'highlight': {'diffuse': [1.0, 1.0, 0.5], 'specular': [1.0, 1.0, 0.5], 'shininess': 120}
        }
    
    def draw_beveled_cubie(self, x, y, z, colors, highlight_face=None):
        """Draw a cubie with beveled edges for tactile feel"""
        half = self.cubie_size / 2 - self.cubie_gap
        bevel = self.bevel_size
        
        # Draw each face with bevel
        faces = [
            # Front face
            {'normal': (0, 0, 1), 'center': (0, 0, half), 
             'vertices': [
                 (-half+bevel, -half+bevel, half), (half-bevel, -half+bevel, half),
                 (half-bevel, half-bevel, half), (-half+bevel, half-bevel, half)
             ], 'color': colors[0]},
            
            # Back face
            {'normal': (0, 0, -1), 'center': (0, 0, -half),
             'vertices': [
                 (-half+bevel, -half+bevel, -half), (-half+bevel, half-bevel, -half),
                 (half-bevel, half-bevel, -half), (half-bevel, -half+bevel, -half)
             ], 'color': colors[1]},
            
            # Top face
            {'normal': (0, 1, 0), 'center': (0, half, 0),
             'vertices': [
                 (-half+bevel, half, -half+bevel), (-half+bevel, half, half-bevel),
                 (half-bevel, half, half-bevel), (half-bevel, half, -half+bevel)
             ], 'color': colors[2]},
            
            # Bottom face
            {'normal': (0, -1, 0), 'center': (0, -half, 0),
             'vertices': [
                 (-half+bevel, -half, -half+bevel), (half-bevel, -half, -half+bevel),
                 (half-bevel, -half, half-bevel), (-half+bevel, -half, half-bevel)
             ], 'color': colors[3]},
            
            # Left face
            {'normal': (-1, 0, 0), 'center': (-half, 0, 0),
             'vertices': [
                 (-half, -half+bevel, -half+bevel), (-half, -half+bevel, half-bevel),
                 (-half, half-bevel, half-bevel), (-half, half-bevel, -half+bevel)
             ], 'color': colors[4]},
            
            # Right face
            {'normal': (1, 0, 0), 'center': (half, 0, 0),
             'vertices': [
                 (half, -half+bevel, -half+bevel), (half, half-bevel, -half+bevel),
                 (half, half-bevel, half-bevel), (half, -half+bevel, half-bevel)
             ], 'color': colors[5]}
        ]
        
        # Draw faces
        for i, face in enumerate(faces):
            if face['color']:
                material = self.materials.get(face['color'], self.materials['black'])
                
                # Apply highlight if this is the grabbed face
                if highlight_face == i:
                    highlight = self.grab_highlight
                    glMaterialfv(GL_FRONT, GL_DIFFUSE, [
                        material['diffuse'][0] * (1 - highlight) + self.materials['highlight']['diffuse'][0] * highlight,
                        material['diffuse'][1] * (1 - highlight) + self.materials['highlight']['diffuse'][1] * highlight,
                        material['diffuse'][2] * (1 - highlight) + self.materials['highlight']['diffuse'][2] * highlight
                    ])
                else:
                    glMaterialfv(GL_FRONT, GL_DIFFUSE, material['diffuse'])
                
                glMaterialfv(GL_FRONT, GL_SPECULAR, material['specular'])
                glMaterialf(GL_FRONT, GL_SHININESS, material['shininess'])
                
                glBegin(GL_QUADS)
                glNormal3fv(face['normal'])
                for vertex in face['vertices']:
                    glVertex3f(x + vertex[0], y + vertex[1], z + vertex[2])
                glEnd()
        
        # Draw bevel edges
        glDisable(GL_LIGHTING)
        glColor3f(0.05, 0.05, 0.05)
        glLineWidth(1.5)
        
        # Draw edge lines
        edges = [
            # Front face edges
            (x-half+bevel, y-half+bevel, z+half), (x+half-bevel, y-half+bevel, z+half),
            (x+half-bevel, y-half+bevel, z+half), (x+half-bevel, y+half-bevel, z+half),
            (x+half-bevel, y+half-bevel, z+half), (x-half+bevel, y+half-bevel, z+half),
            (x-half+bevel, y+half-bevel, z+half), (x-half+bevel, y-half+bevel, z+half),
        ]
        
        glBegin(GL_LINES)
        for edge in edges:
            glVertex3f(edge[0], edge[1], edge[2])
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_grabbed_face_highlight(self, face, twist_angle):
        """Draw visual feedback for grabbed face"""
        if not face:
            return
        
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        # Pulsing highlight based on grab strength
        pulse = (math.sin(time.time() * 8) + 1) / 2
        alpha = 0.3 + 0.2 * pulse
        
        # Color based on twist direction
        if abs(twist_angle) > 10:
            if twist_angle > 0:
                glColor4f(0.0, 1.0, 0.0, alpha)  # Green for clockwise
            else:
                glColor4f(1.0, 0.0, 0.0, alpha)  # Red for counter-clockwise
        else:
            glColor4f(1.0, 1.0, 0.0, alpha)  # Yellow for neutral
        
        # Draw a transparent plane on the grabbed face
        glPushMatrix()
        
        # Position based on which face
        face_positions = {
            'U': (0, self.cube_size/2 + 0.1, 0),
            'D': (0, -self.cube_size/2 - 0.1, 0),
            'L': (-self.cube_size/2 - 0.1, 0, 0),
            'R': (self.cube_size/2 + 0.1, 0, 0),
            'F': (0, 0, self.cube_size/2 + 0.1),
            'B': (0, 0, -self.cube_size/2 - 0.1)
        }
        
        if face in face_positions:
            pos = face_positions[face]
            glTranslatef(pos[0], pos[1], pos[2])
            
            # Rotate to face camera
            if face in ['U', 'D']:
                glRotatef(90, 1, 0, 0) if face == 'U' else glRotatef(-90, 1, 0, 0)
            elif face in ['L', 'R']:
                glRotatef(90, 0, 1, 0) if face == 'L' else glRotatef(-90, 0, 1, 0)
            
            # Draw highlight quad
            size = self.cube_size/2
            glBegin(GL_QUADS)
            glVertex3f(-size, -size, 0)
            glVertex3f(size, -size, 0)
            glVertex3f(size, size, 0)
            glVertex3f(-size, size, 0)
            glEnd()
        
        glPopMatrix()
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
    
    def draw_twist_indicator(self, face, angle):
        """Draw arc showing twist amount"""
        if not face or abs(angle) < 5:
            return
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Determine arc color and direction
        if angle > 0:
            color = (0.0, 1.0, 0.0, 0.7)  # Green for clockwise
            start_angle = 0
            end_angle = min(90, angle)
        else:
            color = (1.0, 0.0, 0.0, 0.7)  # Red for counter-clockwise
            start_angle = max(-90, angle)
            end_angle = 0
        
        glColor4fv(color)
        glLineWidth(4.0)
        
        # Draw arc around the face
        glPushMatrix()
        
        # Position arc
        face_positions = {
            'U': (0, self.cube_size/2 + 0.2, 0),
            'D': (0, -self.cube_size/2 - 0.2, 0),
            'L': (-self.cube_size/2 - 0.2, 0, 0),
            'R': (self.cube_size/2 + 0.2, 0, 0),
            'F': (0, 0, self.cube_size/2 + 0.2),
            'B': (0, 0, -self.cube_size/2 - 0.2)
        }
        
        if face in face_positions:
            pos = face_positions[face]
            glTranslatef(pos[0], pos[1], pos[2])
            
            # Orient arc
            if face in ['U', 'D']:
                glRotatef(90, 1, 0, 0)
            elif face in ['L', 'R']:
                glRotatef(90, 0, 1, 0)
            
            # Draw arc
            radius = self.cube_size/2 + 0.3
            glBegin(GL_LINE_STRIP)
            steps = 30
            for i in range(steps + 1):
                t = start_angle + (end_angle - start_angle) * (i / steps)
                rad = math.radians(t)
                x = radius * math.cos(rad)
                y = radius * math.sin(rad)
                glVertex3f(x, y, 0)
            glEnd()
        
        glPopMatrix()
        
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
    
    def render(self, rotation_x, rotation_y, grabbed_face=None, twist_angle=0, 
               mode_text="", move_count=0, timer=0):
        """Main render with physical feedback"""
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Update visual feedback
        current_time = time.time()
        if grabbed_face:
            self.grab_highlight = min(1.0, self.grab_highlight + 0.1)
        else:
            self.grab_highlight = max(0, self.grab_highlight - 0.05)
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.08, 0.08, 0.12, 1.0)  # Dark space-like background
        
        # Draw starfield background (optional)
        self.draw_starfield()
        
        # Set up view
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -10)
        
        # Apply smooth cube rotation
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)
        
        # Draw grabbed face highlight
        if grabbed_face:
            self.draw_grabbed_face_highlight(grabbed_face, twist_angle)
            self.draw_twist_indicator(grabbed_face, twist_angle)
        
        # Draw the cube (simplified - you'd integrate your actual cube drawing)
        self.draw_simplified_cube(grabbed_face)
        
        # Draw UI
        self.draw_ui(mode_text, move_count, timer, grabbed_face, twist_angle)
        
        pygame.display.flip()
        return True
    
    def draw_simplified_cube(self, grabbed_face):
        """Draw a simplified cube for demonstration"""
        # You would replace this with your actual cube drawing
        glPushMatrix()
        
        # Draw a simple colored cube
        size = self.cube_size / 2
        
        # Draw each face with a different color
        colors = [
            (0, 1, 0),   # Front - Green
            (0, 0, 1),   # Back - Blue
            (1, 1, 1),   # Top - White
            (1, 1, 0),   # Bottom - Yellow
            (1, 0.5, 0), # Left - Orange
            (1, 0, 0)    # Right - Red
        ]
        
        glBegin(GL_QUADS)
        
        # Front
        glColor3fv(colors[0])
        glVertex3f(-size, -size, size)
        glVertex3f(size, -size, size)
        glVertex3f(size, size, size)
        glVertex3f(-size, size, size)
        
        # Back
        glColor3fv(colors[1])
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(size, -size, -size)
        
        # Top
        glColor3fv(colors[2])
        glVertex3f(-size, size, -size)
        glVertex3f(-size, size, size)
        glVertex3f(size, size, size)
        glVertex3f(size, size, -size)
        
        # Bottom
        glColor3fv(colors[3])
        glVertex3f(-size, -size, -size)
        glVertex3f(size, -size, -size)
        glVertex3f(size, -size, size)
        glVertex3f(-size, -size, size)
        
        # Left
        glColor3fv(colors[4])
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, -size, size)
        glVertex3f(-size, size, size)
        glVertex3f(-size, size, -size)
        
        # Right
        glColor3fv(colors[5])
        glVertex3f(size, -size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(size, size, size)
        glVertex3f(size, -size, size)
        
        glEnd()
        
        glPopMatrix()
    
    def draw_starfield(self):
        """Draw a starfield background for immersion"""
        glDisable(GL_LIGHTING)
        glPointSize(2.0)
        glBegin(GL_POINTS)
        glColor3f(1, 1, 1)
        
        # Draw random stars
        import random
        for _ in range(100):
            x = random.uniform(-20, 20)
            y = random.uniform(-20, 20)
            z = random.uniform(-15, -5)
            glVertex3f(x, y, z)
        
        glEnd()
        glEnable(GL_LIGHTING)
    
    def draw_ui(self, mode_text, move_count, timer, grabbed_face, twist_angle):
        """Draw immersive UI"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Draw mode text
        mode_surface = self.font.render(mode_text, True, (255, 255, 0))
        mode_data = pygame.image.tostring(mode_surface, "RGBA", True)
        glRasterPos2f(20, 40)
        glDrawPixels(mode_surface.get_width(), mode_surface.get_height(),
                     GL_RGBA, GL_UNSIGNED_BYTE, mode_data)
        
        # Draw twist angle if grabbing
        if grabbed_face and abs(twist_angle) > 5:
            twist_text = f"Twist: {abs(twist_angle):.0f}° ({'CW' if twist_angle > 0 else 'CCW'})"
            twist_surface = self.font.render(twist_text, True, 
                (0, 255, 0) if twist_angle > 0 else (255, 0, 0))
            twist_data = pygame.image.tostring(twist_surface, "RGBA", True)
            glRasterPos2f(self.display[0] - 300, 40)
            glDrawPixels(twist_surface.get_width(), twist_surface.get_height(),
                         GL_RGBA, GL_UNSIGNED_BYTE, twist_data)
        
        # Draw stats
        stats_y = 80
        stats = [
            f"Moves: {move_count}",
            f"Time: {timer:.1f}s",
            "Press Q to quit",
            "Grab near a face and twist!"
        ]
        
        for i, text in enumerate(stats):
            surf = self.small_font.render(text, True, (200, 200, 255))
            data = pygame.image.tostring(surf, "RGBA", True)
            glRasterPos2f(20, stats_y + i * 30)
            glDrawPixels(surf.get_width(), surf.get_height(),
                         GL_RGBA, GL_UNSIGNED_BYTE, data)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return True
        return False
    
    def cleanup(self):
        pygame.quit()