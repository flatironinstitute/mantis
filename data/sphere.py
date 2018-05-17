import math
import numpy as np


def generate_sphere_grid(n_subdivisions=2, return_faces=False):
    verts, faces = icosahedron()
    for x in range(n_subdivisions):
        verts, faces = subdivide(verts, faces)

    verts = np.array(verts)

    if return_faces:
        return verts, faces
    else:
        return verts


def icosahedron():
    """Construct an icosahedron"""

    # create 12 vertices of a icosahedron
    t = (1.0 + math.sqrt(5.0)) / 2.0;

    verts = [np.array((-1, t, 0)),
             np.array((1, t, 0)),
             np.array((-1, -t, 0)),
             np.array((1, -t, 0)),
             np.array((0, -1, t)),
             np.array((0, 1, t)),
             np.array((0, -1, -t)),
             np.array((0, 1, -t)),
             np.array((t, 0, -1)),
             np.array((t, 0, 1)),
             np.array((-t, 0, -1)),
             np.array((-t, 0, 1))]

    verts = [v / np.linalg.norm(v) for v in verts]

    faces = [
        # 5 faces around point 0
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        # 5 adjacent faces
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        # 5 faces around point 3
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        # 5 adjacent faces
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1)]

    return verts, faces


def subdivide(verts, faces):
    """
    Subdivide each triangle into four triangles, pushing verts to
    the unit sphere
    """

    def add_and_normalize(v1, v2):
        v3 = v1 + v2
        v3 /= np.linalg.norm(v3)
        return v3

    triangles = len(faces)
    for face_index in range(triangles):
        # Create three new verts at the midpoints of each edge:
        face = faces[face_index]
        a, b, c = (verts[vert_index] for vert_index in face)
        verts.append(add_and_normalize(a, b))
        verts.append(add_and_normalize(b, c))
        verts.append(add_and_normalize(a, c))

        # Split the current triangle into four smaller triangles:
        i = len(verts) - 3
        j, k = i + 1, i + 2
        faces.append((i, j, k))
        faces.append((face[0], i, k))
        faces.append((i, face[1], j))
        faces[face_index] = (k, j, face[2])

    return verts, faces


def main():
    num_subdivisions = 2
    verts, faces = icosahedron()
    for x in range(num_subdivisions):
        verts, faces = subdivide(verts, faces)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    verts = np.array(verts)
    print(len(verts))

    import scipy.io

    scipy.io.savemat('sphere_{}.mat'.format(len(verts)),
                     {'X': verts, 'tri': faces})

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces)
    ax.axis('equal')
    plt.show()

    dists = []
    for fa in faces:
        for vi in fa:
            for vj in fa:
                if vi == vj:
                    continue
                d = np.linalg.norm(verts[vi] - verts[vj])
                dists.append(d)

    print(min(dists), max(dists))


if __name__ == '__main__':
    main()
