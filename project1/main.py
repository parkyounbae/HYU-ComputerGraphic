import math

from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

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

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''


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
    global mode
    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    else:
        if action == GLFW_PRESS or action == GLFW_REPEAT:
            if key == GLFW_KEY_V:
                if mode == 1:
                    mode = 0
                else:
                    mode = 1

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
    print('mouse wheel scroll: %d, %d' % (xoffset, yoffset))
    global distance
    if distance+yoffset*0.005>=0:
        distance += yoffset*0.005

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


def main():
    global saved_xpos
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)  # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)  # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '20190384513', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetScrollCallback(window, scroll_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')

    # prepare vaos
    vao_frame = prepare_vao_frame()

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
            P = glm.ortho(-1, 1 , -1 , 1 , -1, 1)
        else:
            P = glm.perspective(45, 1, 0.1, 10)

        V = glm.lookAt(position + temp_position, temp_target, cameraUp)
        I = glm.mat4()

        MVP = P * V * I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 20)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()


if __name__ == "__main__":
    main()
