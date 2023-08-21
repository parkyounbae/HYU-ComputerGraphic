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
line_mode = 0
moving_mode = 0
now_time = 0
start_time = 0
move_index = 0
obj_object = None

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

void main()
{
    // light and material properties
    vec3 light_pos = vec3(6,4,8);
    vec3 light_color = vec3(1,1,1);
    
    vec3 light_pos2 = vec3(-6,4,-8);

    vec3 material_color = vec3(1,1,1);
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // or can be material_color

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);
    
    vec3 light_dir2 = normalize(light_pos2 - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;
    
    float diff2 = max(dot(normal, light_dir2), 0);
    vec3 diffuse2 = diff2 * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;
    
    vec3 reflect_dir2 = reflect(-light_dir2, normal);
    float spec2 = pow( max(dot(view_dir, reflect_dir2), 0.0), material_shininess);
    vec3 specular2 = spec2 * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular+ diffuse2 + specular2;
    FragColor = vec4(color, 1.);
}
'''

class Node:
    def __init__(self,name, parent, link_transform_from_parent, shape_transform, channels, is_joint,base_scale):
        self.name = name
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.link_transform_from_parent = link_transform_from_parent
        self.joint_transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = glm.vec3(1,1,1)
        self.channels = channels
        self.is_joint = is_joint

        self.base_scale = base_scale


    def set_joint_transform(self, joint_transform):
        self.joint_transform = joint_transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.link_transform_from_parent * self.joint_transform
        else:
            self.global_transform = self.base_scale * self.link_transform_from_parent * self.joint_transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color

class BVHLoader:
    def __init__(self):
        self.node_list = []
        self.move_list = []
        self.frames = 0
        self.frame_time = 0.0
        self.joint_num = 0

    def print_result(self):
        print("Frames : " + str(self.frames))
        print("FPS : " + str(1/self.frame_time))
        print("Number of joints : " + str(self.joint_num))
        for i in self.node_list:
            if(i.is_joint == 1):
                print(i.name, end=" ")
        print(" ")



    def load_model(self, file):
        read_object = open(file, 'r')
        stack = []
        width = 0.04

        max_len = 0

        for line in read_object:
            values = line.lstrip().split()
            if values[0] == "ROOT":
                read_object.readline()
                offset = read_object.readline().split()
                channels = read_object.readline().split()
                channels_data = channels[2:]
                result_node = Node(values[1],None, glm.translate(glm.vec3(0,0,0)), glm.scale(glm.vec3(0,0,0)),channels_data,1, glm.mat4())
                self.joint_num += 1
                self.node_list.append(result_node)
                stack.append(result_node)
            elif values[0] == "JOINT":
                read_object.readline()
                offset = read_object.readline().split()
                offset_vec = glm.vec3(float(offset[1]), float(offset[2]), float(offset[3]))
                offset_scale = glm.vec3(width, glm.length(offset_vec)/2, width)
                offset_length = glm.length(offset_vec)

                if max_len < offset_length:
                    max_len = offset_length

                initial_vec = glm.normalize(glm.vec3(0,1,0))
                offset_normal_vec = glm.normalize(offset_vec)
                axis = glm.cross(initial_vec, offset_normal_vec)
                angle = glm.acos(glm.dot(initial_vec, offset_normal_vec))
                rotation_mat = glm.rotate(angle, axis)
                if glm.isnan(rotation_mat[0][0]):
                    rotation_mat = glm.mat4()
                    if float(offset[2])<0:
                        offset_length = -offset_length

                initial_vec2 = glm.normalize(glm.vec3(0, 1, 0))
                offset_normal_vec2 = glm.normalize(offset_vec)
                axis2 = glm.cross(offset_normal_vec2, initial_vec2)
                angle2 = glm.acos(glm.dot(offset_normal_vec2, initial_vec2))
                rotation_mat2 = glm.rotate(angle2, axis2)
                if glm.isnan(rotation_mat2[0][0]):
                    rotation_mat2 = glm.mat4()

                # 와이 좌표가 음수일때를 고려
                channels = read_object.readline().split()
                channels_data = channels[2:]
                result_object = Node(values[1], stack[len(stack)-1], rotation_mat*glm.mat4(),glm.translate(glm.vec3(0,offset_length/2,0))*glm.scale(offset_scale), channels_data, 0, glm.mat4())
                result_node = Node(values[1], result_object, glm.translate(glm.vec3(0,offset_length,0))*rotation_mat2, glm.scale(glm.vec3(0,0,0)), channels_data,1, glm.mat4())
                self.node_list.append(result_object)
                self.node_list.append(result_node)
                self.joint_num += 1
                stack.append(result_node)

            elif values[0] == "End":
                read_object.readline()
                offset = read_object.readline().split()
                offset_vec = glm.vec3(float(offset[1]), float(offset[2]), float(offset[3]))
                offset_scale = glm.vec3(width, glm.length(offset_vec)/2, width)
                offset_length = glm.length(offset_vec)

                if max_len < offset_length:
                    max_len = offset_length
                read_object.readline()

                initial_vec = glm.normalize(glm.vec3(0, 1, 0))
                offset_normal_vec = glm.normalize(offset_vec)
                axis = glm.cross(initial_vec, offset_normal_vec)
                angle = glm.acos(glm.dot(initial_vec, offset_normal_vec))
                rotation_mat = glm.rotate(angle, axis)
                if glm.isnan(rotation_mat[0][0]):
                    rotation_mat = glm.mat4()
                    if float(offset[2]) < 0:
                        offset_length = -offset_length

                result_object = Node(values[1], stack[len(stack)-1], rotation_mat*glm.mat4(),glm.translate(glm.vec3(0,offset_length/2,0))*glm.scale(offset_scale), None, 0, glm.mat4())
                self.node_list.append(result_object)

            elif values[0] == "}":
                if len(stack) != 0:
                    stack.pop()

            elif values[0] == "MOTION":
                temp = read_object.readline().split()
                self.frames = int(temp[1])
                temp = read_object.readline().split()
                self.frame_time = float(temp[2])
                self.move_list = read_object.readlines()

        digit = len(str(int(max_len)))
        temp = glm.mat4()
        if digit > 1:
            for i in range(digit):
                temp = temp * glm.scale(glm.vec3(0.4,0.4,0.4))
            self.node_list[0].base_scale = temp
            for i in self.node_list:
                if i.is_joint == 0:
                    for j in range(digit):
                        i.shape_transform = i.shape_transform * glm.scale(glm.vec3(5,1,5))






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
    global mode,line_mode,start_time, moving_mode,move_index
    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    else:
        if action == GLFW_PRESS or action == GLFW_REPEAT:
            if key == GLFW_KEY_V:
                if mode == 1:
                    mode = 0
                else:
                    mode = 1
            if key == GLFW_KEY_1:
                line_mode = 1
            if key == GLFW_KEY_2:
                line_mode = 0
            if key == GLFW_KEY_SPACE:
                if moving_mode == 1:
                    moving_mode = 0
                else:
                    moving_mode = 1
                for i in obj_object.node_list:
                    if i.is_joint == 1:
                        i.set_joint_transform(glm.mat4())
                start_time = now_time
                move_index = 0

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
    global mouse_clicked,mouse_clicked_right, clicked_mouse_ypos, clicked_mouse_xpos, saved_xpos,  \
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
    # print('mouse wheel scroll: %d, %d' % (xoffset, yoffset))
    global distance
    if distance+yoffset*0.005>=0:
        distance += yoffset*0.005

def file_callback(window, paths):
    global obj_object,obj_vao, moving_mode,move_index

    print(os.path.basename(paths[0]))
    obj_object = BVHLoader()
    obj_object.load_model(os.path.join(paths[0]))
    moving_mode = 0
    move_index = 0
    obj_object.print_result()



def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
                         # position        # color
                         1.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # x-axis start
                         -1.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # x-axis end
                         1.0, 0.0, 0.2, 1.0, 1.0, 0.0,  # y-axis start
                         -1.0, 0.0, 0.2, 1.0, 1.0, 0.0,  # y-axis end
                         1.0, 0.0, 0.4, 1.0, 1.0, 0.0,  # y-axis start
                         -1.0, 0.0, 0.4, 1.0, 1.0, 0.0,  # y-axis end
                         1.0, 0.0, -0.2, 1.0, 1.0, 0.0,  # y-axis start
                         -1.0, 0.0, -0.2, 1.0, 1.0, 0.0,  # y-axis end
                         1.0, 0.0, -0.4, 1.0, 1.0, 0.0,  # y-axis start
                         -1.0, 0.0, -0.4, 1.0, 1.0, 0.0,  # y-axis end
                         0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, -1.0, 1.0, 0.0, 0.0,
                         0.2, 0.0, 1.0, 1.0, 1.0, 1.0,
                         0.2, 0.0, -1.0, 1.0, 1.0, 1.0,
                         0.4, 0.0, 1.0, 1.0, 1.0, 1.0,
                         0.4, 0.0, -1.0, 1.0, 1.0, 1.0,
                         -0.2, 0.0, 1.0, 1.0, 1.0, 1.0,
                         -0.2, 0.0, -1.0, 1.0, 1.0, 1.0,
                         -0.4, 0.0, 1.0, 1.0, 1.0, 1.0,
                         -0.4, 0.0, -1.0, 1.0, 1.0, 1.0,
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

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32),
                          ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_frame2():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
                         # position        # color
                         10.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # x-axis start
                         -10.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # x-axis end
                         0.0, 0.0, 10.0, 1.0, 1.0, 1.0,
                         0.0, 0.0, -10.0, 1.0, 1.0, 1.0,
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

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32),
                          ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_line():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
                         # position        # color
                         0.0, 1.0, 0.0, 0.0, 1.0, 0.0,  # x-axis start
                         0.0, -1.0, 0.0, 0.0, 1.0, 0.0,  # x-axis end
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

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32),
                          ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_cube():
    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    vertices = glm.array(glm.float32,
                         # position      normal
                         -1, 1, 1, 0, 0, 1,  # v0
                         1, -1, 1, 0, 0, 1,  # v2
                         1, 1, 1, 0, 0, 1,  # v1

                         -1, 1, 1, 0, 0, 1,  # v0
                         -1, -1, 1, 0, 0, 1,  # v3
                         1, -1, 1, 0, 0, 1,  # v2

                         -1, 1, -1, 0, 0, -1,  # v4
                         1, 1, -1, 0, 0, -1,  # v5
                         1, -1, -1, 0, 0, -1,  # v6

                         -1, 1, -1, 0, 0, -1,  # v4
                         1, -1, -1, 0, 0, -1,  # v6
                         -1, -1, -1, 0, 0, -1,  # v7

                         -1, 1, 1, 0, 1, 0,  # v0
                         1, 1, 1, 0, 1, 0,  # v1
                         1, 1, -1, 0, 1, 0,  # v5

                         -1, 1, 1, 0, 1, 0,  # v0
                         1, 1, -1, 0, 1, 0,  # v5
                         -1, 1, -1, 0, 1, 0,  # v4

                         -1, -1, 1, 0, -1, 0,  # v3
                         1, -1, -1, 0, -1, 0,  # v6
                         1, -1, 1, 0, -1, 0,  # v2

                         -1, -1, 1, 0, -1, 0,  # v3
                         -1, -1, -1, 0, -1, 0,  # v7
                         1, -1, -1, 0, -1, 0,  # v6

                         1, 1, 1, 1, 0, 0,  # v1
                         1, -1, 1, 1, 0, 0,  # v2
                         1, -1, -1, 1, 0, 0,  # v6

                         1, 1, 1, 1, 0, 0,  # v1
                         1, -1, -1, 1, 0, 0,  # v6
                         1, 1, -1, 1, 0, 0,  # v5

                         -1, 1, 1, -1, 0, 0,  # v0
                         -1, -1, -1, -1, 0, 0,  # v7
                         -1, -1, 1, -1, 0, 0,  # v3

                         -1, 1, 1, -1, 0, 0,  # v0
                         -1, 1, -1, -1, 0, 0,  # v4
                         -1, -1, -1, -1, 0, 0,  # v7
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

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32),
                          ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def draw_frame_array(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    for i in range(-20, 21):
        for j in range(-20, 21):
            MVP_cube = MVP * glm.translate(glm.vec3(10 * i, 0, 10 * j))
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_cube))
            glDrawArrays(GL_LINES, 0, 4)


def draw_node(vao, node, VP, MVP_loc):
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_TRIANGLES, 0, 36)

def draw_line(vao, node, VP, MVP_loc):
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_LINES, 0, 2)

def main():
    global saved_xpos, obj_object, now_time, start_time,move_index
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


    # prepare vaos
    vao_frame = prepare_vao_frame()
    vao_cube = prepare_vao_cube()
    vao_line = prepare_vao_line()
    vao_frame2 = prepare_vao_frame2()

    temp_position = glm.vec3(0.0,0.0,0.0)
    saved_position = glm.vec3(0.0,0.0,0.0)
    temp_target = glm.vec3(0.0,0.0,0.0)
    saved_target = glm.vec3(0.0,0.0,0.0)
    obj_object = BVHLoader()
    move_index = 0

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
            P = glm.ortho(-30, 30 , -30 , 30 , -30, 30)
        else:
            P = glm.perspective(45, 1, 0.1, 1000)

        V = glm.lookAt(position + temp_position, temp_target, cameraUp)
        I = glm.mat4()

        MVP = P * V * I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame
        # glBindVertexArray(vao_frame)
        # glDrawArrays(GL_LINES, 0, 20)
        draw_frame_array(vao_frame2,MVP,MVP_loc)

        now_time = glfwGetTime()

        if moving_mode == 1:
            if (now_time-start_time)>obj_object.frame_time:

                if move_index == obj_object.frames:
                    move_index = 0
                start_time = now_time
                move_data = obj_object.move_list[move_index].split()
                node_index = 0
                move_index = move_index + 1

                for node in obj_object.node_list:
                    if(node.is_joint == 1):
                        result_move = glm.mat4()
                        for joint_move in node.channels:
                            joint_move = joint_move.upper()
                            if joint_move == "XROTATION":
                                result_move = result_move * glm.rotate(glm.radians(float(move_data[node_index])), glm.vec3(1,0,0))
                            elif joint_move == "YROTATION":
                                result_move = result_move * glm.rotate(glm.radians(float(move_data[node_index])),
                                                                               glm.vec3(0, 1, 0))
                            elif joint_move == "ZROTATION":
                                result_move = result_move * glm.rotate(glm.radians(float(move_data[node_index])),
                                                                               glm.vec3(0, 0, 1))
                            elif joint_move == "XPOSITION":
                                result_move = result_move * glm.translate(glm.vec3(float(move_data[node_index]),0,0))
                            elif joint_move == "YPOSITION":
                                result_move = result_move * glm.translate(glm.vec3(0,float(move_data[node_index]),0))
                            elif joint_move == "ZPOSITION":
                                result_move = result_move * glm.translate(glm.vec3(0,0,float(move_data[node_index])))
                            node_index += 1
                        # print(result_move)
                        node.set_joint_transform(result_move)

        for node in obj_object.node_list:
            if node == obj_object.node_list[0]:
                node.update_tree_global_transform()
            if line_mode == 1:
                draw_line(vao_line, node, P*V, MVP_loc)
            else:
                draw_node(vao_cube, node, P * V, MVP_loc)



        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()


if __name__ == "__main__":
    main()
