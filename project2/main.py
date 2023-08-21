import math

from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os

g_cam_ang = 0.
g_cam_height = .1
triangle_x = 0.
triangle_y = 0.
triangle_z = 0.

current_mouse_xpos = 0
current_mouse_ypos = 0

clicked_mouse_xpos = 0
clicked_mouse_ypos = 0

moved_mouse_xpos = 0
moved_mouse_ypos = 0

moved_mouse_xpos_right = 0
moved_mouse_ypos_right = 0

saved_xpos = 0
saved_ypos = 0

saved_xpos_right = 0
saved_ypos_right = 0

mouse_clicked = False
mouse_clicked_right = False

distance = 0.1

mode = 1

solid_mode = 1

file_mode = -1
file_path = ""
obj_object = None
obj_vao = None

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_norm;

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(transpose(inverse(M))) * vin_norm);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 material_color;

void main()
{
    // light and material properties
    vec3 light_pos = vec3(6,4,8);
    vec3 light_color = vec3(1,1,1);
    
    vec3 light_pos2 = vec3(-6,-4,-8);
    vec3 light_color2 = vec3(1,1,1);
     
    // vec3 material_color = vec3(1,1,1);
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;
    
    vec3 light_ambient2 = 0.1*light_color2;
    vec3 light_diffuse2 = light_color2;
    vec3 light_specular2 = light_color2;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // or can be material_color
    
    vec3 material_specular2 = light_color2;
    
    // ambient
    vec3 ambient = light_ambient * material_ambient;
    
    vec3 ambient2 = light_ambient2 * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);
    vec3 light_dir2 = normalize(light_pos2 - surface_pos);
    
    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;
    
    float diff2 = max(dot(normal, light_dir2), 0);
    vec3 diffuse2 = diff * light_diffuse2 * material_diffuse;
    
    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;
    
    vec3 reflect_dir2 = reflect(-light_dir2, normal);
    float spec2 = pow( max(dot(view_dir, reflect_dir2), 0.0), material_shininess);
    vec3 specular2 = spec2 * light_specular2 * material_specular2;
    
    vec3 color = ambient + diffuse + specular + ambient2 + diffuse2 + specular2;
    FragColor = vec4(color, 1.);
}
'''


class OBJLoader:
    def __init__(self):
        self.vertex_coordinates = [] #벡터3 vertex의 좌표
        self.norm_coordinates = [] #벡터3 법선벡터

        self.vertex_index = []  # 벡터3 vertex의 좌표
        self.norm_index = []  # 벡터3 법선벡터

        self.result = [] #인덱스를 실제 값으로 바꿈
        self.tri_count = 0
        self.quad_count = 0
        self.multiple_count = 0
        self.total_triangle = 0

    def get_triangle_count(self):
        return (self.tri_count + self.quad_count*2)*3

    def load_model(self, file):

        for line in open(file, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                self.vertex_coordinates.append(values[1:4])
            if values[0] == 'vn':
                self.norm_coordinates.append(values[1:4])

        for line in open(file, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'f':
                if len(values) == 4:
                    self.tri_count += 1;
                    self.total_triangle += 1;
                    for v in values[1:4]:
                        w = v.split('/')
                        self.vertex_index.append(int(w[0])-1)
                        self.norm_index.append(int(w[2])-1)
                if len(values) == 5:
                    self.quad_count += 1;
                    for i in range(3,5):
                        self.total_triangle += 1;
                        first = values[1].split('/')
                        self.vertex_index.append(int(first[0]) - 1)
                        self.norm_index.append(int(first[2]) - 1)
                        second = values[i-1].split('/')
                        self.vertex_index.append(int(second[0]) - 1)
                        self.norm_index.append(int(second[2]) - 1)
                        third = values[i].split('/')
                        self.vertex_index.append(int(third[0]) - 1)
                        self.norm_index.append(int(third[2]) - 1)
                if len(values) > 5:
                    self.multiple_count += 1;
                    for i in range(3, len(values)):
                        self.total_triangle += 1;
                        first = values[1].split('/')
                        self.vertex_index.append(int(first[0]) - 1)
                        self.norm_index.append(int(first[2]) - 1)
                        second = values[i - 1].split('/')
                        self.vertex_index.append(int(second[0]) - 1)
                        self.norm_index.append(int(second[2]) - 1)
                        third = values[i].split('/')
                        self.vertex_index.append(int(third[0]) - 1)
                        self.norm_index.append(int(third[2]) - 1)



    def prepare_vao_obj(self):
        for i in range(0, 3*(self.total_triangle)):
            for j in self.vertex_coordinates[self.vertex_index[i]]:
                self.result.append(j)  # j번째의 좌표를 추가
            for j in self.norm_coordinates[self.norm_index[i]]:
                self.result.append(j) #j번째의 놂를 추가
        # result = [vx1 vy1 vz1 tx1 ty1 nx1 ny1 nz1, vx2 vy2 vz2 tx2 ty2 nx2 ny2 nz2, ... ]

        self.result = np.array(self.result, 'float32')
        if file_mode == 1:
            print("Total number of faces : " + str(self.total_triangle))
            print("Number of faces with 3 vertices : " + str(self.tri_count))
            print("Number of faces with 4 vertices : " + str(self.quad_count))
            print("Number of faces with more than 4 vertices : " + str(self.multiple_count))

        # create and activate VAO (vertex array object)
        VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
        glBindVertexArray(VAO)  # activate VAO

        # create and activate VBO (vertex buffer object)
        VBO = glGenBuffers(1)  # create a buffer object ID and store it to VBO variable
        glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

        # copy vertex data to VBO
        glBufferData(GL_ARRAY_BUFFER, self.result.itemsize * len(self.result) , self.result,
                     GL_STATIC_DRAW)  # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

        # configure vertex positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * self.result.itemsize, None)
        glEnableVertexAttribArray(0)


        # configure vertex colors
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * self.result.itemsize,
                              ctypes.c_void_p(3 * self.result.itemsize))
        glEnableVertexAttribArray(1)

        return VAO

class Node:
    def __init__(self, parent, shape_transform):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform

    def get_shape_transform(self):
        return self.shape_transform

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------

    # vertex shader
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)  # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source)  # provide shader source code
    glCompileShader(vertex_shader)  # compile the shader object

    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())

    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)  # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source)  # provide shader source code
    glCompileShader(fragment_shader)  # compile the shader object

    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()  # create an empty program object
    glAttachShader(shader_program, vertex_shader)  # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)  # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program  # return the shader program


def key_callback(window, key, scancode, action, mods):
    global mode, file_mode, solid_mode
    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action == GLFW_PRESS or action == GLFW_REPEAT:
            if key == GLFW_KEY_V:
                if mode == 1:
                    mode = 0
                else:
                    mode = 1
            if key == GLFW_KEY_H:
                file_mode = 0
            if key == GLFW_KEY_Z:
                if solid_mode == 1:
                    solid_mode = 0
                else:
                    solid_mode = 1

def file_callback(window, paths):
    global file_mode, file_path,obj_object,obj_vao
    file_mode = 1
    print(os.path.basename(paths[0]))
    obj_object = OBJLoader()
    obj_object.load_model(os.path.join(paths[0]))
    obj_vao = obj_object.prepare_vao_obj()



def cursor_callback(window, xpos, ypos):
    global current_mouse_xpos, current_mouse_ypos,moved_mouse_xpos, moved_mouse_ypos,moved_mouse_xpos_right, moved_mouse_ypos_right
    current_mouse_xpos = xpos
    current_mouse_ypos = ypos

    if mouse_clicked:
        moved_mouse_xpos = (current_mouse_xpos - clicked_mouse_xpos)*0.1
        moved_mouse_ypos = (current_mouse_ypos - clicked_mouse_ypos)*0.1

        if glm.cos(glm.radians(moved_mouse_ypos + saved_ypos)) < 0:
            moved_mouse_xpos = -moved_mouse_xpos

    if mouse_clicked_right:
        moved_mouse_xpos_right = (current_mouse_xpos - clicked_mouse_xpos)*0.01
        moved_mouse_ypos_right = (current_mouse_ypos - clicked_mouse_ypos)*0.01


# 좌표계가 위에서 아래로 내려오면서 y좌표값이 커짐 -> 마우스를 위에서 아래로 내림 = 현재마우스 - 클릭시작점 = 양수
# x도 마찬가지로 오른쪽으로 갈수록 숫자가 커짐 -> 마우스를 왼쪽에서 오른쪽으로 이동 = 현제 마우스 = 클릭 시작점 = 양수

def button_callback(window, button, action, mod):
    global mouse_clicked,mouse_clicked_right, clicked_mouse_ypos, clicked_mouse_xpos, saved_xpos, \
        saved_ypos,moved_mouse_xpos,moved_mouse_ypos, saved_xpos_right, saved_ypos_right, moved_mouse_xpos_right, moved_mouse_ypos_right
    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action == GLFW_PRESS:
            mouse_clicked = True
            clicked_mouse_xpos = glfwGetCursorPos(window)[0]
            clicked_mouse_ypos = glfwGetCursorPos(window)[1]
        elif action == GLFW_RELEASE:
            mouse_clicked = False
            saved_xpos += moved_mouse_xpos
            saved_ypos += moved_mouse_ypos

            moved_mouse_xpos = 0
            moved_mouse_ypos = 0

    if button == GLFW_MOUSE_BUTTON_RIGHT:
        if action == GLFW_PRESS:
            mouse_clicked_right = True
            clicked_mouse_xpos = glfwGetCursorPos(window)[0]
            clicked_mouse_ypos = glfwGetCursorPos(window)[1]
        elif action == GLFW_RELEASE:
            mouse_clicked_right = False
            saved_xpos_right += moved_mouse_xpos_right
            saved_ypos_right += moved_mouse_ypos_right
            moved_mouse_xpos_right = 0
            moved_mouse_ypos_right = 0


def scroll_callback(window, xoffset, yoffset):
    global distance
    if distance+yoffset*0.005>=0:
        distance += yoffset*0.005

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
                         # position        # color
                         1.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # x-axis start
                         -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, # x-axis end
                         1.0, 0.0, 0.2,  0.0, 1.0, 0.0,# y-axis start
                         -1.0, 0.0, 0.2,  0.0, 1.0, 0.0,# y-axis end
                         1.0, 0.0, 0.4,  0.0, 1.0, 0.0,# y-axis start
                         -1.0, 0.0, 0.4,  0.0, 1.0, 0.0,# y-axis end
                         1.0, 0.0, -0.2,  0.0, 1.0, 0.0,# y-axis start
                         -1.0, 0.0, -0.2, 0.0, 1.0, 0.0, # y-axis end
                         1.0, 0.0, -0.4,  0.0, 1.0, 0.0, # y-axis start
                         -1.0, 0.0, -0.4,   0.0, 1.0, 0.0,# y-axis end
                         0.0, 0.0, 1.0,0.0, 1.0, 0.0,
                         0.0, 0.0, -1.0,0.0, 1.0, 0.0,
                         0.2, 0.0, 1.0,0.0, 1.0, 0.0,
                         0.2, 0.0, -1.0,0.0, 1.0, 0.0,
                         0.4, 0.0, 1.0,0.0, 1.0, 0.0,
                         0.4, 0.0, -1.0,0.0, 1.0, 0.0,
                         -0.2, 0.0, 1.0,0.0, 1.0, 0.0,
                         -0.2, 0.0, -1.0,0.0, 1.0, 0.0,
                         -0.4, 0.0, 1.0,0.0, 1.0, 0.0,
                         -0.4, 0.0, -1.0,0.0, 1.0, 0.0,
                         )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)  # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)  # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr,
                 GL_STATIC_DRAW)  # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def draw_node(vao, node, VP, MVP_loc, M_loc, view_pos, view_pos_loc, temp_count, color_loc, color_vec):
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    # M = node.get_global_transform() * node.get_shape_transform()
    M = glm.mat4()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(view_pos_loc, view_pos.x, view_pos.y, view_pos.z)
    glUniform3f(color_loc, color_vec.x, color_vec.y, color_vec.z)
    glDrawArrays(GL_TRIANGLES, 0, temp_count*3)




def main():
    global saved_xpos,obj_object,obj_vao
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)  # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)  # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '2019038513', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetScrollCallback(window, scroll_callback)

    glfwSetDropCallback(window, file_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    M_loc = glGetUniformLocation(shader_program, 'M')
    view_pos_loc = glGetUniformLocation(shader_program, 'view_pos')
    color_loc = glGetUniformLocation(shader_program, 'material_color')

    # prepare vaos
    vao_frame = prepare_vao_frame()

    obj_object = OBJLoader()
    obj_object.load_model(os.path.join("./cube-tri.obj"))
    obj_vao = obj_object.prepare_vao_obj()

    obj_head = OBJLoader()
    obj_head.load_model(os.path.join("./head.obj"))
    head_vao = obj_head.prepare_vao_obj()

    obj_coffee = OBJLoader()
    obj_coffee.load_model(os.path.join("./coffee.obj"))
    coffee_vao = obj_coffee.prepare_vao_obj()

    obj_jack = OBJLoader()
    obj_jack.load_model(os.path.join("./jack.obj"))
    jack_vao = obj_jack.prepare_vao_obj()

    obj_pipe = OBJLoader()
    obj_pipe.load_model(os.path.join("./pipe.obj"))
    pipe_vao = obj_pipe.prepare_vao_obj()

    obj_cloud = OBJLoader()
    obj_cloud.load_model(os.path.join("./cloud.obj"))
    cloud_vao = obj_cloud.prepare_vao_obj()

    # paremt, transform, norm
    head = Node(None, glm.translate(glm.vec3(0, -1.6,0)))
    coffee = Node(head, glm.translate(glm.vec3(0, -0.8, 1.3))*glm.scale(glm.vec3(0.01,0.01,0.01)))
    pipe = Node(head, glm.translate(glm.vec3(0, 0, 0.18))*glm.scale(glm.vec3(0.02,0.02,0.02))*glm.rotate(glm.radians(90),glm.vec3(0,1,0)))
    jack = Node(head, glm.translate(glm.vec3(0, 0, 0.3))*glm.scale(glm.vec3(0.8,0.8,0.8))*glm.rotate(glm.radians(-90),glm.vec3(1,0,0)))
    cloud = Node(coffee, glm.translate(glm.vec3(0,0.1,0.2))*glm.scale(glm.vec3(0.02,0.02,0.02)))
    cloud2 = Node(coffee, glm.translate(glm.vec3(0, 0.1, 0.2)) * glm.scale(glm.vec3(0.02, 0.02, 0.02)))

    cloud3 = Node(pipe, glm.translate(glm.vec3(0, 0.3, 0.3)) * glm.scale(glm.vec3(0.02, 0.02, 0.02)))
    cloud4 = Node(pipe, glm.translate(glm.vec3(0, 0.3, 0.3)) * glm.scale(glm.vec3(0.02, 0.02, 0.02)))


    temp_position = glm.vec3(0.0,0.0,0.0)
    saved_position = glm.vec3(0.0,0.0,0.0)
    temp_target = glm.vec3(0.0,0.0,0.0)
    saved_target = glm.vec3(0.0,0.0,0.0)

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # view matrix
        # rotate camera position with g_cam_ang / move camera up & down with g_cam_heigh
        xpos = (saved_xpos + moved_mouse_xpos)%360
        ypos = (saved_ypos + moved_mouse_ypos)%360

        xpos_right = moved_mouse_xpos_right
        ypos_right = moved_mouse_ypos_right

        position_x = distance * np.cos(glm.radians(ypos)) * np.cos(glm.radians(xpos))
        position_y = distance * np.sin(glm.radians(ypos))
        position_z = distance * np.cos(glm.radians(ypos)) * np.sin(glm.radians(xpos))
        position = glm.vec3(position_x, position_y, position_z)

        target = glm.vec3(0,0,0)
        direction = glm.normalize(position - target)

        up = glm.vec3(0.0, 1.0, 0.0)
        cameraRight = glm.normalize(glm.cross(up, direction))
        cameraUp = glm.cross(direction,cameraRight)
        camera_move = cameraRight*(-xpos_right) + cameraUp*(ypos_right)

        # 동산 넘어가기
        if glm.cos(glm.radians(ypos)) < 0:
            cameraUp = -cameraUp
            camera_move = cameraRight * (-xpos_right) + cameraUp * (ypos_right)

        if mouse_clicked or mouse_clicked_right:
            temp_position = camera_move + saved_position
            temp_target = camera_move + saved_target
        else:
            saved_position = temp_position
            saved_target = temp_target

        if mode == 0:
            P = glm.ortho(-1, 1, -1, 1, -1, 1)
        else:
            P = glm.perspective(45, 1, 0.1, 10000)

        V = glm.lookAt(position + temp_position, temp_target, cameraUp)
        M = glm.mat4()
        view_pos = position + temp_position

        MVP = P * V * M

        # draw current frame
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(view_pos_loc, view_pos.x, view_pos.y, view_pos.z)
        glUniform3f(color_loc, 1, 1, 1)
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 20)

        if solid_mode == 1:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if file_mode == 1:
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
            glUniform3f(view_pos_loc, view_pos.x, view_pos.y, view_pos.z)
            glUniform3f(color_loc, 1, 1, 1)
            glBindVertexArray(obj_vao)
            glDrawArrays(GL_TRIANGLES, 0, obj_object.total_triangle * 3)
        elif file_mode == 0:
            t = glfwGetTime()
            head.set_transform(glm.rotate(t, glm.vec3(0,1,0)))
            pipe.set_transform(glm.rotate(glm.radians(40),glm.vec3(1,0,0))*glm.translate(glm.vec3(0,-0.15,(glm.sin(t+3)+1)/3)))
            coffee.set_transform(glm.rotate(glm.radians(-40),glm.vec3(1,0,0))*glm.translate(glm.vec3(0, -0.15, (glm.sin(t+1.5)+1)/3)))
            jack.set_transform(glm.translate(glm.vec3(0, 0, (glm.sin(t)+1)/3 )))
            cloud.set_transform(glm.translate(glm.vec3(0,(glm.sin(t)+1)/10,0)))
            cloud2.set_transform(glm.translate(glm.vec3(0, 0.2 - (glm.sin(t)+1)/10, 0)))
            cloud3.set_transform(glm.translate(glm.vec3(0, (glm.sin(t) + 1) / 10, 0)))
            cloud4.set_transform(glm.translate(glm.vec3(0, 0.2 - (glm.sin(t) + 1) / 10, 0)))
            head.update_tree_global_transform()


            draw_node(head_vao, head, P*V, MVP_loc, M_loc, view_pos, view_pos_loc, obj_head.total_triangle, color_loc, glm.vec3(251/255,206/255,177/255))
            draw_node(pipe_vao, pipe, P*V, MVP_loc, M_loc, view_pos, view_pos_loc, obj_pipe.total_triangle, color_loc, glm.vec3(111/255,79/255,40/255))
            draw_node(coffee_vao, coffee, P*V, MVP_loc, M_loc, view_pos, view_pos_loc, obj_coffee.total_triangle, color_loc, glm.vec3(111/255,79/255,40/255))
            draw_node(jack_vao, jack, P*V, MVP_loc, M_loc, view_pos, view_pos_loc, obj_jack.total_triangle, color_loc, glm.vec3(0/255,128/255,0/255))
            draw_node(cloud_vao, cloud, P * V, MVP_loc, M_loc, view_pos, view_pos_loc, obj_cloud.total_triangle, color_loc, glm.vec3(1,1,1))
            draw_node(cloud_vao, cloud2, P * V, MVP_loc, M_loc, view_pos, view_pos_loc, obj_cloud.total_triangle, color_loc, glm.vec3(1,1,1))
            draw_node(cloud_vao, cloud3, P * V, MVP_loc, M_loc, view_pos, view_pos_loc, obj_cloud.total_triangle, color_loc, glm.vec3(211/255,211/255,211/255))
            draw_node(cloud_vao, cloud4, P * V, MVP_loc, M_loc, view_pos, view_pos_loc, obj_cloud.total_triangle, color_loc, glm.vec3(211/255,211/255,211/255))


        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()


if __name__ == "__main__":
    main()
