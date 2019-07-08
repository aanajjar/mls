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
    def __rmul__(self, k):
        return vec2(self.x*k, self.y*k)
    def __truediv__(self, k):
        return vec2(self.x/k, self.y/k)
    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"
    def scale(self, k_x, k_y = None):
        if (k_y is None):
            return vec2(k_x*self.x, k_x*self.y)
        else:
            return vec2(k_x*self.x, k_y*self.y)
    def translate(self, tr_x, tr_y):
        return vec2(self.x + tr_x, self.y + tr_y)
    def rotate(self, th):
        return vec2(self.x*math.cos(th) - self.y*math.sin(th), self.x*math.sin(th) + self.y*math.cos(th))
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
    def __rmul__(self, k):
        return vec3(self.x*k, self.y*k, self.z*k)
    def __truediv__(self, k):
        return vec3(self.x/k, self.y/k, self.z/k)
    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"
    def scale(self, k_x, k_y = None, k_z = None):
        if (k_y is None):
            return vec3(k_x*self.x, k_x*self.y, k_x*self.z)
        else:
            return vec3(k_x*self.x, k_y*self.y, k_z*self.z)
    def translate(self, tr_x, tr_y, tr_z):
        return vec3(self.x + tr_x, self.y + tr_y, self.z + tr_z)
    def rotate_x(self, th_x):
        return vec3(self.x, self.y*math.cos(th_x) - self.z*math.sin(th_x), self.y*math.sin(th_x) + self.z*math.cos(th_x))
    def rotate_y(self, th_y):
        return vec3(self.z*math.sin(th_y) + self.x*math.cos(th_y), self.y, self.z*math.cos(th_y) - self.x*math.sin(th_y))
    def rotate_z(self, th_z):
        return vec3(self.x*math.cos(th_z) - self.y*math.sin(th_z), self.x*math.sin(th_z) + self.y*math.cos(th_z), self.z)
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
    def __rmul__(self, k):
        return mat2([self[0]*k, self[1]*k, self[2]*k, self[3]*k])
    def __truediv__(self, k):
        return mat2([self[0]/k, self[1]/k, self[2]/k, self[3]/k])
    def __repr__(self):
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
    def __rmul__(self, k):
        value = [self[i]*k for i in range(9)]
        return mat3(value)
    def __truediv__(self, k):
        value = [self[i]/k for i in range(9)]
        return mat3(value)
    def __repr__(self):
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
    th_y = 2*math.pi*random.random()
    th_z = 2*math.pi*random.random()
    points_rot = [vec3(p.x, p.y*math.cos(th_x) - p.z*math.sin(th_x), p.y*math.sin(th_x) + p.z*math.cos(th_x)) for p in points]
    points_rot = [vec3(p.z*math.sin(th_y) + p.x*math.cos(th_y), p.y, p.z*math.cos(th_y) - p.x*math.sin(th_y)) for p in points_rot]
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

def triangle(p1, p2, p3, width, height):
    v1 = vec2(round(p1.x), round(p1.y))
    v2 = vec2(round(p2.x), round(p2.y))
    v3 = vec2(round(p3.x), round(p3.y))
    if (v1.y > v2.y):
        temp = v1
        v1 = v2
        v2 = temp
    if (v1.y > v3.y):
        temp = v1
        v1 = v3
        v3 = temp
    if (v2.y > v3.y):
        temp = v2
        v2 = v3
        v3 = temp
    if (v1.y != v2.y): k_12 = (v2.x - v1.x)/(v2.y - v1.y)
    if (v1.y != v3.y): k_13 = (v3.x - v1.x)/(v3.y - v1.y)
    if (v2.y != v3.y): k_23 = (v3.x - v2.x)/(v3.y - v2.y)
    if (v1.y == v2.y):
        if (v1.x < v2.x):
            xl, xu = v1.x, v2.x
            left = False
        else:
            xl, xu = v2.x, v1.x
            left = True
        if (v1.y >= 0 and v1.y < height):
            xl = max(xl, 0)
            xu = min(xu, width - 1)
            for x in range(xl, xu + 1):
                yield vec2(x, v1.y)
    else:
        left = v2.x < k_13*(v2.y - v1.y) + v1.x
        if (left):
            k1, k2 = k_12, k_13
        else:
            k1, k2 = k_13, k_12
        yl = max(v1.y, 0)
        yu = min(v2.y, height)
        for y in range(yl, yu):
            xl = max(math.floor(k1*(y - v1.y) + v1.x + 0.5), 0)
            xu = min(math.floor(k2*(y - v1.y) + v1.x + 0.5), width - 1)
            for x in range(xl, xu + 1):
                yield vec2(x, y)
    if (v2.y == v3.y):
        if (v2.x < v3.x):
            xl, xu = v2.x, v3.x
        else:
            xl, xu = v3.x, v2.x
        if (v2.y >= 0 and v2.y < height):
            xl = max(xl, 0)
            xu = min(xu, width - 1)
            for x in range(xl, xu + 1):
                yield vec2(x, v2.y)
    else:
        if (left):
            k1, k2 = k_23, k_13
            t1, t2 = v2, v1
        else:
            k1, k2 = k_13, k_23
            t1, t2 = v1, v2
        yl = max(v2.y, 0)
        yu = min(v3.y + 1, height)
        for y in range(yl, yu):
            xl = max(math.floor(k1*(y - t1.y) + t1.x + 0.5), 0)
            xu = min(math.floor(k2*(y - t2.y) + t2.x + 0.5), width - 1)
            for x in range(xl, xu + 1):
                yield vec2(x, y)

from PIL import Image, ImageTk

def transform_image(image, transform, mapping, alpha = 1, incr_x = 10, incr_y = 10):
    width, height = image.size
    image_in = np.array(image.convert("RGBA"))
    image_out = [[[255, 255, 255, 255] for j in range(width)] for i in range(height)]
    for i in range(0, height, incr_y):
        p_ur = transform(vec2(0, i), mapping, alpha)
        p_lr = transform(vec2(0, i + incr_y), mapping, alpha)
        for j in range(incr_x, width, incr_x):
            p_ul = p_ur
            p_ll = p_lr
            p_ur = transform(vec2(j, i), mapping, alpha)
            p_lr = transform(vec2(j, i + incr_y), mapping, alpha)
            a = p_ur - p_ul
            b = p_ll - p_ul
            det = a.x*b.y - a.y*b.x
            if (det != 0.0):
                for p in triangle(p_ul, p_ur, p_ll, width, height):
                    c = p - p_ul
                    rx = (b.y*c.x - b.x*c.y)/det
                    ry = (a.x*c.y - a.y*c.x)/det
                    image_out[p.y][p.x] = image_in[min(height - 1, max(0, round(i + ry*incr_y)))][min(width - 1, max(0, round(j + (rx - 1)*incr_x)))]
            a = p_lr - p_ll
            b = p_lr - p_ur
            det = a.x*b.y - a.y*b.x
            if (det != 0.0):
                p_ulr = p_ur + p_ll - p_lr
                for p in triangle(p_ur, p_ll, p_lr, width, height):
                    c = p - p_ulr
                    rx = (b.y*c.x - b.x*c.y)/det
                    ry = (a.x*c.y - a.y*c.x)/det
                    image_out[p.y][p.x] = image_in[min(height - 1, max(0, round(i + ry*incr_y)))][min(width - 1, max(0, round(j + (rx - 1)*incr_x)))]
    image_out = Image.fromarray(np.uint8(image_out))
    return image_out

import tkinter as tk
from tkinter import filedialog, messagebox

class ImageTransform:
    min_image_width = 400
    min_image_height = 250
    image = None
    fixed_label = None
    moving_label = None
    line = None
    r_point = 2
    def __init__(self, master):
        self.master = master
        self.master.title("Image Transformation")
        self.master.minsize(self.min_image_width + 20, self.min_image_height + 20)
        self.menubar = tk.Menu(self.master)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Load Image", command=self.load_image)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.master.destroy)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.master.config(menu=self.menubar)
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack()
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(padx=10, pady=10)
        self.canvas = tk.Canvas(self.canvas_frame, width=self.min_image_width, height=self.min_image_height, bg="#CCCCCC", bd=0, highlightthickness=0)
        self.canvas.bind("<Button-1>", self.place_point)
        self.canvas.bind("<Enter>", self.show_point)
        self.canvas.bind("<Leave>", self.hide_point)
        self.canvas.bind("<Motion>", self.render_point)
        self.canvas.pack()
        self.radio_frame = tk.Frame(self.main_frame)
        self.radio_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.hide = tk.IntVar(value=0)
        self.toggle_hide = tk.Checkbutton(self.radio_frame, text="Hide points", variable=self.hide, command=self.hide_mapping)
        self.toggle_hide.pack(side="right", anchor=tk.NE)
        self.transform_label = tk.Label(self.radio_frame, text="Transformation")
        self.transform_label.pack(side="top", anchor=tk.NW)
        self.transform_option = tk.IntVar(value=0)
        self.option_affine = tk.Radiobutton(self.radio_frame, text="Affine", variable=self.transform_option, value=1)
        self.option_affine.pack(side="top", anchor=tk.NW)
        self.option_similarity = tk.Radiobutton(self.radio_frame, text="Similarity", variable=self.transform_option, value=2)
        self.option_similarity.pack(side="top", anchor=tk.NW)
        self.option_rigid = tk.Radiobutton(self.radio_frame, text="Rigid", variable=self.transform_option, value=3)
        self.option_rigid.pack(side="top", anchor=tk.NW)
        self.alpha_frame = tk.Frame(self.radio_frame)
        self.alpha_frame.pack(side="top", anchor=tk.NW)
        self.alpha_label = tk.Label(self.alpha_frame, text="Alpha = ")
        self.alpha_label.pack(side="left")
        self.alpha_entry = tk.Entry(self.alpha_frame, width=20)
        self.alpha_entry.delete(0, tk.END)
        self.alpha_entry.insert(0, "1.0000")
        self.alpha_entry.pack(side="left")
        self.transform_frame = tk.Frame(self.main_frame)
        self.transform_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.transform = tk.Button(self.transform_frame, text="Transfrom Image", command=self.transform_input)
        self.transform.pack(side="left")
        self.view_original = tk.Button(self.transform_frame, text="View Original", command=self.show_original)
        self.view_original.pack(side="left")
        self.action_reset = tk.Button(self.transform_frame, text="Reset", command=self.reset)
        self.action_reset.pack(side="right")
        self.fixed = []
        self.moving = []
    def load_image(self):
        image_source = filedialog.askopenfilename(initialdir="/", title="Select File")
        if (image_source == ""): return
        try:
            self.image_original = Image.open(image_source)
        except Exception as err:
            messagebox.showerror("Error", "Failed to load image.\n" + str(err))
            return
        width, height = self.image_original.size
        self.max_image_width = int(0.8*self.master.winfo_screenwidth())
        self.max_image_height = int(0.8*self.master.winfo_screenheight())
        cwidth = min(max(width, self.min_image_width), self.max_image_width)
        cheight = min(max(height, self.min_image_height), self.max_image_height)
        self.k = max(width/cwidth, height/cheight)
        if (self.k > 1):
            width, height = int(width/self.k), int(height/self.k)
            self.image = ImageTk.PhotoImage(self.image_original.resize((width, height), Image.ANTIALIAS))
            cwidth = min(max(width, self.min_image_width), self.max_image_width)
            cheight = min(max(height, self.min_image_height), self.max_image_height)
        else:
            self.image = ImageTk.PhotoImage(self.image_original)
        self.fixed = []
        self.moving = []
        self.place_fixed_or_moving = 0
        self.canvas.delete(tk.ALL)
        self.canvas.config(width=cwidth, height=cheight, bg=self.master.cget("background"))
        self.canvas.create_image((cwidth/2, cheight/2), image=self.image)
    def show_original(self):
        if (self.image is None):
            messagebox.showerror("Error", "No image was selected.")
            return
        if (self.k > 1):
            width, height = self.image_transformed.size
            self.image = ImageTk.PhotoImage(self.image_original, Image.ANTIALIAS)
        else:
            self.image = ImageTk.PhotoImage(self.image_original)
        cwidth = self.canvas.winfo_width()
        cheight = self.canvas.winfo_height()
        self.canvas.tag_lower(self.canvas.create_image((cwidth/2, cheight/2), image=self.image))
    def transform_input(self):
        if (self.image is None):
            messagebox.showerror("Error", "No image was selected.")
            return
        option = self.transform_option.get()
        if (option == 1): transformation = affine_transform_2d
        elif (option == 2): transformation = similarity_transform_2d
        elif (option == 3): transformation = rigid_transform_2d
        else:
            messagebox.showerror("Error", "No transformation was selected.")
            return
        mapping = [(self.fixed[i], self.moving[i]) for i in range(len(self.moving))]
        try:
            alpha = float(self.alpha_entry.get())
            self.image_transformed = transform_image(self.image_original, transformation, mapping, alpha, 10, 10)
            if (self.k > 1):
                width, height = self.image_transformed.size
                self.image = ImageTk.PhotoImage(self.image_transformed.resize((int(width/self.k), int(height/self.k)), Image.ANTIALIAS))
            else:
                self.image = ImageTk.PhotoImage(self.image_transformed)
        except Exception as err:
            messagebox.showerror("Error", "Failed to transform image.\n" + str(err))
            return
        cwidth = self.canvas.winfo_width()
        cheight = self.canvas.winfo_height()
        self.canvas.tag_lower(self.canvas.create_image((cwidth/2, cheight/2), image=self.image))
    def hide_mapping(self):
        if (self.hide.get() == 1):
            self.canvas.itemconfig("fixed", state=tk.HIDDEN)
            self.canvas.itemconfig("moving", state=tk.HIDDEN)
            self.canvas.itemconfig("line", state=tk.HIDDEN)
        else:
            self.canvas.itemconfig("fixed", state=tk.NORMAL)
            self.canvas.itemconfig("moving", state=tk.NORMAL)
            self.canvas.itemconfig("line", state=tk.NORMAL)
    def reset(self):
        self.fixed = []
        self.moving = []
        self.place_fixed_or_moving = 0
        self.canvas.delete(tk.ALL)
        self.show_original()
    place_fixed_or_moving = 0
    def place_point(self, event):
        if (self.image is None or self.hide.get() == 1): return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        k = max(self.k, 1)
        img_x = k*(x - (self.canvas.winfo_width() - self.image.width())/2)
        img_y = k*(y - (self.canvas.winfo_height() - self.image.height())/2)
        if (self.place_fixed_or_moving == 0):
            self.fixed.append(vec2(img_x, img_y))
            self.fixed_x = x
            self.fixed_y = y
            self.canvas.create_oval(x - self.r_point, y - self.r_point, x + self.r_point, y + self.r_point, fill="red", width=0, tags="fixed")
            self.canvas.delete(self.fixed_label)
            self.moving_label = self.canvas.create_oval(x - self.r_point, y - self.r_point, x + self.r_point, y + self.r_point, fill="blue", width=0, tags="moving")
            self.line = self.canvas.create_line(self.fixed_x, self.fixed_y, x, y, fill="black", dash=(4, 4), tags="line")
            self.place_fixed_or_moving = 1
        elif (self.place_fixed_or_moving == 1):
            self.moving.append(vec2(img_x, img_y))
            self.canvas.create_oval(x - self.r_point, y - self.r_point, x + self.r_point, y + self.r_point, fill="blue", width=0, tags="moving")
            self.canvas.create_line(self.fixed_x, self.fixed_y, x, y, fill="black", dash=(4, 4), tags="line")
            self.canvas.delete(self.moving_label)
            self.canvas.delete(self.line)
            self.fixed_label = self.canvas.create_oval(x - self.r_point, y - self.r_point, x + self.r_point, y + self.r_point, fill="red", width=0, tags="fixed")
            self.place_fixed_or_moving = 0
    def show_point(self, event):
        if (self.image is None or self.hide.get() == 1): return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if (self.place_fixed_or_moving == 0):
            self.fixed_label = self.canvas.create_oval(x - self.r_point, y - self.r_point, x + self.r_point, y + self.r_point, fill="red", width=0, tags="fixed")
        elif (self.place_fixed_or_moving == 1):
            self.moving_label = self.canvas.create_oval(x - self.r_point, y - self.r_point, x + self.r_point, y + self.r_point, fill="blue", width=0, tags="moving")
            self.line = self.canvas.create_line(self.fixed_x, self.fixed_y, x, y, fill="black", dash=(4, 4))
    def hide_point(self, event):
        if (self.place_fixed_or_moving == 0):
            if (self.fixed_label is not None):
                self.canvas.delete(self.fixed_label)
        elif (self.place_fixed_or_moving == 1):
            if (self.moving_label is not None):
                self.canvas.delete(self.moving_label)
            if (self.line is not None):
                self.canvas.delete(self.line)
    def render_point(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if (self.place_fixed_or_moving == 0):
            if (self.fixed_label is not None):
                self.canvas.coords(self.fixed_label, x - self.r_point, y - self.r_point, x + self.r_point, y + self.r_point)
        elif (self.place_fixed_or_moving == 1):
            if (self.moving_label is not None and self.line is not None):
                self.canvas.coords(self.moving_label, x - self.r_point, y - self.r_point, x + self.r_point, y + self.r_point)
                self.canvas.coords(self.line, self.fixed_x, self.fixed_y, x, y)

def open_gui():
    root = tk.Tk()
    app = ImageTransform(root)
    app.master.mainloop()
