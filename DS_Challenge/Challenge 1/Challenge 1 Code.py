import numpy as np


class TreeNode(object):
    def __init__(self, key, seg):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1
        self.seg = seg

    def __repr__(self):
        return str(self.seg) + " " + str(self.key)


def cmp(node1, node2):
    valid = True
    for i in range(4):
        if node1[i] != node2[i]:
            valid = False
            break
    if valid:
        return False

    x_loc = max(node1[0], node2[0])
    y1 = ((node1[2] - x_loc) * node1[1] + (x_loc - node1[0]) * node1[3]) / \
         (node1[2] - node1[0])
    y2 = ((node2[2] - x_loc) * node2[1] + (x_loc - node2[0]) * node2[3]) / \
         (node2[2] - node2[0])

    if y1 != y2:
        return y1 > y2

    return ((node1[3] - node1[1]) / (node1[2] - node1[0])) > \
           ((node2[3] - node2[1]) / (node2[2] - node2[0]))


def cmp_point(seg, p):
    y_seg = (seg[1] * (seg[2] - p[0]) + seg[3] * (p[0] - seg[0])) / (seg[2] - seg[0])
    return p[1] > y_seg


class AVLTree(object):

    def insert_node(self, root, key, seg):
        if not root:
            return TreeNode(key, seg)
        elif cmp(root.key, key):
            root.left = self.insert_node(root.left, key, seg)
        else:
            root.right = self.insert_node(root.right, key, seg)

        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))

        balanceFactor = self.getBalance(root)
        if balanceFactor > 1:
            if cmp(root.left.key, key):
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)

        if balanceFactor < -1:
            if cmp(key, root.right.key):
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)

        return root

    def delete_node(self, root, key):
        if not root:
            return root
        elif cmp(root.key, key):
            root.left = self.delete_node(root.left, key)
        elif cmp(key, root.key):
            root.right = self.delete_node(root.right, key)
        else:
            if root.left is None:
                temp = root.right
                root = None
                return temp
            elif root.right is None:
                temp = root.left
                root = None
                return temp
            temp = self.getMinValueNode(root.right)
            root.key = temp.key
            root.seg = temp.seg
            root.right = self.delete_node(root.right,
                                          temp.key)
        if root is None:
            return root

        # Update the balance factor of nodes
        root.height = 1 + max(self.getHeight(root.left),
                              self.getHeight(root.right))

        balanceFactor = self.getBalance(root)

        # Balance the tree
        if balanceFactor > 1:
            if self.getBalance(root.left) >= 0:
                return self.rightRotate(root)
            else:
                root.left = self.leftRotate(root.left)
                return self.rightRotate(root)
        if balanceFactor < -1:
            if self.getBalance(root.right) <= 0:
                return self.leftRotate(root)
            else:
                root.right = self.rightRotate(root.right)
                return self.leftRotate(root)
        return root

    def leftRotate(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.getHeight(z.left),
                           self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                           self.getHeight(y.right))
        return y

    def rightRotate(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.getHeight(z.left),
                           self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                           self.getHeight(y.right))
        return y

    def getHeight(self, root):
        if not root:
            return 0
        return root.height

    def getBalance(self, root):
        if not root:
            return 0
        return self.getHeight(root.left) - self.getHeight(root.right)

    def getMinValueNode(self, root):
        if root is None or root.left is None:
            return root
        return self.getMinValueNode(root.left)

    def below_line(self, root, key):
        if not root:
            return
        dec = cmp(root.key, key)
        if dec:
            return self.below_line(root.left, key)
        ret = self.below_line(root.right, key)
        if not ret:
            return root
        return ret

    def below_point(self, root, point_):
        if not root:
            return
        dec = cmp_point(root.key, point_)
        if not dec:
            return self.below_point(root.left, point_)
        ret = self.below_point(root.right, point_)
        if not ret:
            return root
        return ret

    def preOrder(self, root):
        if not root:
            return
        print(root)
        print("Left :")
        self.preOrder(root.left)
        print("Right :")
        self.preOrder(root.right)


class Point:
    def __init__(self, coordinate, mode, seg=None):
        self.coords = coordinate
        self.mode = mode
        self.seg = seg
        self.reg = None

    def __lt__(self, other):
        if self.coords[0] != other.coords[0]:
            return self.coords[0] < other.coords[0]
        if self.mode != other.mode:
            return self.mode > other.mode
        if self.mode == 2:
            return True
        x = segment[self.seg]
        m1 = (x[3] - x[1]) / (x[2] - x[0])
        x = segment[other.seg]
        m2 = (x[3] - x[1]) / (x[2] - x[0])
        return m1 > m2 if self.mode == 0 else m1 < m2

    def __repr__(self):
        return str(self.coords) + " " + str(self.mode) + " " + str(self.seg)


class Node:
    def __init__(self, num):
        self.num = num
        self.parent = None
        self.height = 1
        self.data = 0

    def get_set(self):
        nodes = []
        node = self
        while node.parent is not None:
            nodes.append(node)
            node = node.parent
        for t in nodes:
            t.parent = node
        return node

    def __repr__(self):
        return str(self.num) + " " + str(self.parent.num if self.parent else None) + " " + str(self.data)


class Regions:
    def __init__(self):
        self.nodes = []

    def add_node(self):
        num = len(self.nodes)
        self.nodes.append(Node(num))
        return num

    def add_data(self, index):
        par1 = self.nodes[index[0]].get_set()
        par2 = self.nodes[index[1]].get_set()
        if par2 != par1:
            par2.data += 1
            par1.data += 1

    def merge(self, one, two):
        par1 = self.nodes[one].get_set()
        par2 = self.nodes[two].get_set()
        if par1 == par2:
            return
        if par1.height > par2.height:
            par1, par2 = par2, par1
        par1.parent = par2
        if par1.height == par2.height:
            par2.height += 1
        par2.data += par1.data


cos = np.cos(np.pi / 120)
sin = np.sin(np.pi / 120)


def rotate(point_):
    global sin, cos
    point_[0], point_[1] = point_[0] * cos - point_[1] * sin, point_[0] * sin + point_[1] * cos
    if len(point_) == 4:
        point_[2], point_[3] = point_[2] * cos - point_[3] * sin, point_[2] * sin + point_[3] * cos
        if point_[0] > point_[2]:
            point_[0], point_[2] = point_[2], point_[0]
            point_[1], point_[3] = point_[3], point_[1]
    return point_


n = int(input())
segment = []
points = []
for i in range(n):
    e = list(map(float, input().split()))
    segment.append(rotate(e))
    points.append(Point(segment[-1][0:2], 0, len(segment) - 1))
    points.append(Point(segment[-1][2:4], 1, len(segment) - 1))
n = int(input())
point = []
regions_up = [None] * len(segment)
regions_down = [None] * len(segment)

for i in range(n):
    e = list(map(float, input().split()))
    point.append(Point(rotate(e), 2))
    points.append(point[-1])
points.sort()
regions = Regions()
avl = AVLTree()
root = None
regions.add_node()
i = 0
while i < len(points):
    first = i
    i = i + 1
    while i < len(points) and points[i].coords[0] == points[first].coords[0]:
        i = i + 1
    end = i - 1
    if points[first].mode == 2:
        ret = avl.below_point(root, points[first].coords)
        for p in points[first:end + 1]:
            if not ret:
                p.reg = 0
            else:
                p.reg = regions_up[ret.seg]
    elif points[first].mode == 1 and points[end].mode == 1:
        regions.merge(regions_up[points[first].seg], regions_down[points[end].seg])
        for p in points[first:end + 1]:
            root = avl.delete_node(root, segment[p.seg])
    elif points[first].mode == 1 and points[end].mode == 0:
        down = 0
        prev = None
        up = regions_up[points[first].seg]
        for p in points[first:end + 1]:
            if p.mode == 1:
                root = avl.delete_node(root, segment[p.seg])
                down = regions_down[p.seg]
            else:
                root = avl.insert_node(root, segment[p.seg], p.seg)
                if not prev:
                    regions_up[p.seg] = up
                else:
                    r = regions.add_node()
                    regions_up[p.seg] = r
                    regions_down[prev.seg] = r
                prev = p
        regions_down[points[end].seg] = down
    else:
        ret = avl.below_line(root, segment[points[first].seg])
        glob = 0
        if ret:
            glob = regions_up[ret.seg]
        prev = None
        for p in points[first:end + 1]:
            root = avl.insert_node(root, segment[p.seg], p.seg)
            if not prev:
                regions_up[p.seg] = glob
            else:
                r = regions.add_node()
                regions_up[p.seg] = r
                regions_down[prev.seg] = r
            prev = p
        regions_down[points[end].seg] = glob

for i in range(len(segment)):
    regions.add_data([regions_up[i], regions_down[i]])
for p in point:
    print(regions.nodes[p.reg].get_set().data)
