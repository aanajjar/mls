import math
import random
import numpy as np
import matplotlib.pyplot as plt

class vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        return vec2(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return vec2(self.x - other.x, self.y - other.y)
    def __mul__(self, k):
        return vec2(self.x*k, self.y*k)
    def __truediv__(self, k):
        return vec2(self.x/k, self.y/k)
    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"
    def dot(self, other):
        return self.x*other.x + self.y*other.y
    def transpose_multiply(self, other):
        return mat2([self.x*other.x, self.x*other.y, self.y*other.x, self.y*other.y])
    def str_repr(self, ndigits):
        return "(" + str(round(self.x, ndigits)) + ", " + str(round(self.y, ndigits)) + ")"

class vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, k):
        return vec3(self.x*k, self.y*k, self.z*k)
    def __truediv__(self, k):
        return vec3(self.x/k, self.y/k, self.z/k)
    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"
    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z
    def transpose_multiply(self, other):
        value = [self.x*other.x, self.x*other.y, self.x*other.z,
        self.y*other.x, self.y*other.y, self.y*other.z,
        self.z*other.x, self.z*other.y, self.z*other.z]
        return mat3(value)
    def str_repr(self, ndigits):
        return "(" + str(round(self.x, ndigits)) + ", " + str(round(self.y, ndigits)) + ", " + str(round(self.z, ndigits)) + ")"

class mat2:
    def __init__(self, value = None):
        if (value != None):
            if (isinstance(value, float)):
                self.matrix = [value, 0.0, 0.0, value]
            elif (isinstance(value, int)):
                self.matrix = [value, 0, 0, value]
            elif (isinstance(value, list)):
                self.matrix = value
        else:
            self.matrix = [0.0, 0.0, 0.0, 0.0]
    def __getitem__(self, index):
        return self.matrix[index]
    def __add__(self, other):
        return mat2([self[0] + other[0], self[1] + other[1], self[2] + other[2], self[3] + other[3]])
    def __sub__(self, other):
        return mat2([self[0] - other[0], self[1] - other[1], self[2] - other[2], self[3] - other[3]])
    def __mul__(self, other):
        if (isinstance(other, float) or isinstance(other, int)):
            return mat2([self[0]*other, self[1]*other, self[2]*other, self[3]*other])
        if (isinstance(other, mat2)):
            return mat2([self[0]*other[0] + self[1]*other[2], self[0]*other[1] + self[1]*other[3], self[2]*other[0] + self[3]*other[2], self[2]*other[1] + self[3]*other[3]])
        if (isinstance(other, vec2)):
            return vec2(self[0]*other.x + self[1]*other.y, self[2]*other.x + self[3]*other.y)
    def __truediv__(self, k):
        return mat2([self[0]/k, self[1]/k, self[2]/k, self[3]/k])
    def __str__(self):
        return "[" + str(self[0]) + ", " + str(self[1]) + "; " + str(self[2]) + ", " + str(self[3]) + "]"
    def det(self):
        return self[0]*self[3] - self[1]*self[2]
    def inverse(self):
        det = self.det()
        return mat2([self[3]/det, -self[1]/det, -self[2]/det, self[0]/det])
    def transpose(self):
        return mat2([self[0], self[2], self[1], self[3]])

class mat3:
    def __init__(self, value = None):
        if (value != None):
            if (isinstance(value, float)):
                self.matrix = [value, 0.0, 0.0, 0.0, value, 0.0, 0.0, 0.0, value]
            elif (isinstance(value, int)):
                self.matrix = [value, 0, 0, 0, value, 0, 0, 0, value]
            elif (isinstance(value, list)):
                self.matrix = value
        else:
            self.matrix = [0.0 for i in range(9)]
    def __getitem__(self, index):
        return self.matrix[index]
    def __add__(self, other):
        value = [self[i] + other[i] for i in range(9)]
        return mat3(value)
    def __sub__(self, other):
        value = [self[i] - other[i] for i in range(9)]
        return mat3(value)
    def __mul__(self, other):
        if (isinstance(other, float) or isinstance(other, int)):
            value = [self[i]*other for i in range(9)]
            return mat3(value)
        if (isinstance(other, mat3)):
            value = [self[i]*other[j] + self[i + 1]*other[j + 3] + self[i + 2]*other[j + 6] for i in range(0, 9, 3) for j in range(3)]
            return mat3(value)
        if (isinstance(other, vec3)):
            value = [self[i]*other.x + self[i + 1]*other.y + self[i + 2]*other.z for i in range(0, 9, 3)]
            return vec3(*value)
    def __truediv__(self, k):
        value = [self[i]/k for i in range(9)]
        return mat3(value)
    def __str__(self):
        return "[" + str(self[0]) + ", " + str(self[1]) + ", " + str(self[2]) + "; " + \
        str(self[3]) + ", " + str(self[4]) + ", " + str(self[5]) + "; " + \
        str(self[6]) + ", " + str(self[7]) + ", " + str(self[8]) + "]"
    def det(self):
        return self[0]*(self[4]*self[8] - self[5]*self[7]) - self[1]*(self[3]*self[8] - self[5]*self[6]) + self[2]*(self[3]*self[7] - self[4]*self[6])
    def inverse(self):
        det = self.det()
        value = [(self[4]*self[8] - self[5]*self[7])/det, (self[2]*self[7] - self[1]*self[8])/det, (self[1]*self[5] - self[2]*self[4])/det,
        (self[5]*self[6] - self[3]*self[8])/det, (self[0]*self[8] - self[2]*self[6])/det, (self[2]*self[3] - self[0]*self[5])/det,
        (self[3]*self[7] - self[4]*self[6])/det, (self[1]*self[6] - self[0]*self[7])/det, (self[0]*self[4] - self[1]*self[3])/det]
        return mat3(value)
    def transpose(self):
        value = [self[0], self[3], self[6], self[1], self[4], self[7], self[2], self[5], self[8]]
        return mat3(value)

def affine_transform_2d(v, mapping, alpha = 1):
    p_wgt = vec2(0, 0)
    q_wgt = vec2(0, 0)
    w = len(mapping)*[None]
    w_sum = 0
    for i in range(len(mapping)):
        mp = mapping[i]
        x = mp[0].x - v.x
        y = mp[0].y - v.y
        if (x == 0 and y == 0): return mp[1]
        w[i] = 1/((x*x + y*y) ** alpha)
        p_wgt += mp[0]*w[i]
        q_wgt += mp[1]*w[i]
        w_sum += w[i]
    p_wgt /= w_sum
    q_wgt /= w_sum
    M1 = mat2(0)
    M2 = mat2(0)
    for i in range(len(mapping)):
        mp = mapping[i]
        p_adj = mp[0] - p_wgt
        q_adj = mp[1] - q_wgt
        M1 += p_adj.transpose_multiply(p_adj)*w[i]
        M2 += p_adj.transpose_multiply(q_adj)*w[i]
    M1 = M1.inverse()
    M = M1*M2
    M = M.transpose()
    v_out = M*(v - p_wgt) + q_wgt
    return v_out

def affine_transform_3d(v, mapping, alpha = 1):
    p_wgt = vec3(0, 0, 0)
    q_wgt = vec3(0, 0, 0)
    w = len(mapping)*[None]
    w_sum = 0
    for i in range(len(mapping)):
        mp = mapping[i]
        x = mp[0].x - v.x
        y = mp[0].y - v.y
        z = mp[0].z - v.z
        if (x == 0 and y == 0 and z == 0): return mp[1]
        w[i] = 1/((x*x + y*y + z*z) ** alpha)
        p_wgt += mp[0]*w[i]
        q_wgt += mp[1]*w[i]
        w_sum += w[i]
    p_wgt /= w_sum
    q_wgt /= w_sum
    M1 = mat3(0)
    M2 = mat3(0)
    for i in range(len(mapping)):
        mp = mapping[i]
        p_adj = mp[0] - p_wgt
        q_adj = mp[1] - q_wgt
        M1 += p_adj.transpose_multiply(p_adj)*w[i]
        M2 += p_adj.transpose_multiply(q_adj)*w[i]
    M1 = M1.inverse()
    M = M1*M2
    M = M.transpose()
    v_out = M*(v - p_wgt) + q_wgt
    return v_out

def similarity_transform_2d(v, mapping, alpha = 1):
    p_wgt = vec2(0, 0)
    q_wgt = vec2(0, 0)
    w = len(mapping)*[None]
    w_sum = 0
    for i in range(len(mapping)):
        mp = mapping[i]
        x = mp[0].x - v.x
        y = mp[0].y - v.y
        if (x == 0 and y == 0): return mp[1]
        w[i] = 1/((x*x + y*y) ** alpha)
        p_wgt += mp[0]*w[i]
        q_wgt += mp[1]*w[i]
        w_sum += w[i]
    p_wgt /= w_sum
    q_wgt /= w_sum
    mu = 0
    for i in range(len(mapping)):
        mp = mapping[i]
        p_adj = mp[0] - p_wgt
        mu += w[i]*(p_adj.dot(p_adj))
    A_fac = mat2([v.x - p_wgt.x, v.y - p_wgt.y, v.y - p_wgt.y, p_wgt.x - v.x])
    v_out = vec2(0, 0)
    for i in range(len(mapping)):
        mp = mapping[i]
        p_adj = mp[0] - p_wgt
        q_adj = mp[1] - q_wgt
        A = mat2([p_adj.x, p_adj.y, p_adj.y, -p_adj.x])*A_fac*w[i]
        A = A.transpose()
        v_out += A*q_adj/mu
    v_out += q_wgt
    return v_out

def rigid_transform_2d(v, mapping, alpha = 1):
    p_wgt = vec2(0, 0)
    q_wgt = vec2(0, 0)
    w = len(mapping)*[None]
    w_sum = 0
    for i in range(len(mapping)):
        mp = mapping[i]
        x = mp[0].x - v.x
        y = mp[0].y - v.y
        if (x == 0 and y == 0): return mp[1]
        w[i] = 1/((x*x + y*y) ** alpha)
        p_wgt += mp[0]*w[i]
        q_wgt += mp[1]*w[i]
        w_sum += w[i]
    p_wgt /= w_sum
    q_wgt /= w_sum
    A_fac = mat2([v.x - p_wgt.x, v.y - p_wgt.y, v.y - p_wgt.y, p_wgt.x - v.x])
    v_out = vec2(0, 0)
    for i in range(len(mapping)):
        mp = mapping[i]
        p_adj = mp[0] - p_wgt
        q_adj = mp[1] - q_wgt
        A = mat2([p_adj.x, p_adj.y, p_adj.y, -p_adj.x])*A_fac*w[i]
        A = A.transpose()
        v_out += A*q_adj
    r = math.sqrt(v_out.dot(v_out))
    v_out /= r
    v_sub = v - p_wgt
    r = math.sqrt(v_sub.dot(v_sub))
    v_out *= r
    v_out += q_wgt
    return v_out

def test_transform_2d(transform, alpha = 1):
    points = 20*[None]
    for i in range(20):
        x = random.randrange(-40, 41)
        y = random.randrange(-40, 41)
        points[i] = vec2(x, y)
    tr_x = random.randrange(-40, 41)
    tr_y = random.randrange(-40, 41)
    mapping = [(p, vec2(p.x + tr_x, p.y + tr_y)) for p in points]
    print("Translation")
    print("Input".ljust(20), "Translation".ljust(20), "Transformation".ljust(20))
    for i in range(20):
        x = random.randrange(-40, 41)
        y = random.randrange(-40, 41)
        v_in = vec2(x, y)
        v_translate = vec2(x + tr_x, y + tr_y)
        v_transform = transform(v_in, mapping, alpha)
        print(str(v_in).ljust(20), str(v_translate.str_repr(4)).ljust(20), str(v_transform.str_repr(4)).ljust(20))
    print()
    th = 2*math.pi*random.random()
    mapping = [(p, vec2(p.x*math.cos(th) - p.y*math.sin(th), p.x*math.sin(th) + p.y*math.cos(th))) for p in points]
    print("Rotation")
    print("Input".ljust(20), "Rotation".ljust(20), "Transformation".ljust(20))
    for i in range(20):
        x = random.randrange(-40, 41)
        y = random.randrange(-40, 41)
        v_in = vec2(x, y)
        v_rotate = vec2(x*math.cos(th) - y*math.sin(th), x*math.sin(th) + y*math.cos(th))
        v_transform = transform(v_in, mapping, alpha)
        print(str(v_in).ljust(20), str(v_rotate.str_repr(4)).ljust(20), str(v_transform.str_repr(4)).ljust(20))
    print()
    k = math.exp(2*random.random() - 1)
    mapping = [(p, vec2(k*p.x, k*p.y)) for p in points]
    print("Uniform scaling")
    print("Input".ljust(20), "Scaling".ljust(20), "Transformation".ljust(20))
    for i in range(20):
        x = random.randrange(-40, 41)
        y = random.randrange(-40, 41)
        v_in = vec2(x, y)
        v_scale = vec2(k*x, k*y)
        v_transform = transform(v_in, mapping, alpha)
        print(str(v_in).ljust(20), str(v_scale.str_repr(4)).ljust(20), str(v_transform.str_repr(4)).ljust(20))
    print()
    k_x = math.exp(2*random.random() - 1)
    k_y = 3*random.random() + 1
    if (k_x >= k_y + math.exp(-1)): k_y = k_x - k_y
    else: k_y = k_x + k_y
    mapping = [(p, vec2(k_x*p.x, k_y*p.y)) for p in points]
    print("Non-uniform scaling")
    print("Input".ljust(20), "Scaling".ljust(20), "Transformation".ljust(20))
    for i in range(20):
        x = random.randrange(-40, 41)
        y = random.randrange(-40, 41)
        v_in = vec2(x, y)
        v_scale = vec2(k_x*x, k_y*y)
        v_transform = transform(v_in, mapping, alpha)
        print(str(v_in).ljust(20), str(v_scale.str_repr(4)).ljust(20), str(v_transform.str_repr(4)).ljust(20))
    print()

def test_transform_3d(transform, alpha = 1):
    points = 20*[None]
    for i in range(20):
        x = random.randrange(-40, 41)
        y = random.randrange(-40, 41)
        z = random.randrange(-40, 41)
        points[i] = vec3(x, y, z)
    tr_x = random.randrange(-40, 41)
    tr_y = random.randrange(-40, 41)
    tr_z = random.randrange(-40, 41)
    mapping = [(p, vec3(p.x + tr_x, p.y + tr_y, p.z + tr_z)) for p in points]
    print("Translation")
    print("Input".ljust(30), "Translation".ljust(30), "Transformation".ljust(30))
    for i in range(20):
        x = random.randrange(-40, 41)
        y = random.randrange(-40, 41)
        z = random.randrange(-40, 41)
        v_in = vec3(x, y, z)
        v_translate = vec3(x + tr_x, y + tr_y, z + tr_z)
        v_transform = transform(v_in, mapping, alpha)
        print(str(v_in).ljust(30), str(v_translate.str_repr(4)).ljust(30), str(v_transform.str_repr(4)).ljust(30))
    print()
    th_x = 2*math.pi*random.random()
    points_rot = [vec3(p.x, p.y*math.cos(th_x) - p.z*math.sin(th_x), p.y*math.sin(th_x) + p.z*math.cos(th_x)) for p in points]
    th_y = 2*math.pi*random.random()
    points_rot = [vec3(p.z*math.sin(th_y) + p.x*math.cos(th_y), p.y, p.z*math.cos(th_y) - p.x*math.sin(th_y)) for p in points_rot]
    th_z = 2*math.pi*random.random()
    points_rot = [vec3(p.x*math.cos(th_z) - p.y*math.sin(th_z), p.x*math.sin(th_z) + p.y*math.cos(th_z), p.z) for p in points_rot]
    mapping = [(points[i], points_rot[i]) for i in range(len(points))]
    print("Rotation")
    print("Input".ljust(30), "Rotation".ljust(30), "Transformation".ljust(30))
    for i in range(20):
        x = random.randrange(-40, 41)
        y = random.randrange(-40, 41)
        z = random.randrange(-40, 41)
        v_in = vec3(x, y, z)
        v_rotate = vec3(v_in.x, v_in.y*math.cos(th_x) - v_in.z*math.sin(th_x), v_in.y*math.sin(th_x) + v_in.z*math.cos(th_x))
        v_rotate = vec3(v_rotate.z*math.sin(th_y) + v_rotate.x*math.cos(th_y), v_rotate.y, v_rotate.z*math.cos(th_y) - v_rotate.x*math.sin(th_y))
        v_rotate = vec3(v_rotate.x*math.cos(th_z) - v_rotate.y*math.sin(th_z), v_rotate.x*math.sin(th_z) + v_rotate.y*math.cos(th_z), v_rotate.z)
        v_transform = transform(v_in, mapping, alpha)
        print(str(v_in).ljust(30), str(v_rotate.str_repr(4)).ljust(30), str(v_transform.str_repr(4)).ljust(30))
    print()
    k = math.exp(2*random.random() - 1)
    mapping = [(p, vec3(k*p.x, k*p.y, k*p.z)) for p in points]
    print("Uniform scaling")
    print("Input".ljust(30), "Scaling".ljust(30), "Transformation".ljust(30))
    for i in range(20):
        x = random.randrange(-40, 41)
        y = random.randrange(-40, 41)
        z = random.randrange(-40, 41)
        v_in = vec3(x, y, z)
        v_scale = vec3(k*x, k*y, k*z)
        v_transform = transform(v_in, mapping, alpha)
        print(str(v_in).ljust(30), str(v_scale.str_repr(4)).ljust(30), str(v_transform.str_repr(4)).ljust(30))
    print()
    k_x = math.exp(2*random.random() - 1)
    k_y = 3*random.random() + 1
    k_z = 3*random.random() + 1
    if (k_x >= k_y + math.exp(-1)): k_y = k_x - k_y
    else: k_y = k_x + k_y
    if ((k_x + k_y)/2 >= k_z + math.exp(-1)): k_z = (k_x + k_y)/2 - k_z
    else: k_z = (k_x + k_y)/2 + k_z
    mapping = [(p, vec3(k_x*p.x, k_y*p.y, k_z*p.z)) for p in points]
    print("Non-uniform scaling")
    print("Input".ljust(30), "Scaling".ljust(30), "Transformation".ljust(30))
    for i in range(20):
        x = random.randrange(-40, 41)
        y = random.randrange(-40, 41)
        z = random.randrange(-40, 41)
        v_in = vec3(x, y, z)
        v_scale = vec3(k_x*x, k_y*y, k_z*z)
        v_transform = transform(v_in, mapping, alpha)
        print(str(v_in).ljust(30), str(v_scale.str_repr(4)).ljust(30), str(v_transform.str_repr(4)).ljust(30))
    print()