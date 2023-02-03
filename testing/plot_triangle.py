import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.stats import dirichlet


def triangle_area(xy, pair):
    return 0.5 * np.linalg.norm(np.cross(*(pair - xy)))


def convert_cartesian_to_barycentric(xy, pairs, tol=1e-4):
    barycentric = np.array([triangle_area(xy, pair) for pair in pairs]) / AREA
    return barycentric


def draw_pdf_contours(
        triangle,
        pairs,
        alpha,
        num_of_levels=200,
        subdivision=8,
        **kwargs
):
    triangle_refiner = tri.UniformTriRefiner(triangle)
    triangle_mesh = triangle_refiner.refine_triangulation(subdiv=subdivision)
    p_values = np.array(
        [
            dirichlet.pdf(convert_cartesian_to_barycentric(xy, pairs), alpha)
            for xy in zip(triangle_mesh.x, triangle_mesh.y)
        ])

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    ax.tricontourf(triangle_mesh, p_values, num_of_levels,
                   cmap='viridis', **kwargs)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.75 ** 0.5)
    ax.axis('off')

    # playing som with adding axis labels
    ax.axline((0, 0), (1, 0), color='black', lw=2)
    ax.axline((0, 0), (0.5, 0.75 ** 0.5), color='black', lw=2)
    ax.axline((1, 0), (0.5, 0.75 ** 0.5), color='black', lw=2)

    plt.tight_layout()


if __name__ == "__main__":
    # Set corners of triangle mesh
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75 ** 0.5]])
    # area of triangle
    AREA = 0.5 * 1.0 * 0.75 ** 0.5
    # construct triangle object
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    # construct refiner for triangle mesh
    triangle_refiner = tri.UniformTriRefiner(triangle)
    # create triangle mesh
    triangle_mesh = triangle_refiner.refine_triangulation(subdiv=4)
    print(dir(triangle_mesh))

    # Plot for test
    # fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(2 * 7, 7))
    # for i, mesh in enumerate((triangle, triangle_mesh)):
    #     ax[i].triplot(mesh)
    #     ax[i].axis('off')
    #     ax[i].set_aspect('equal')
    # plt.tight_layout()

    # extract (x,y) coordinates
    coordinate_pairs = np.array(
        [
            corners[np.roll(range(3), -i)[1:]]
            for i in range(3)
        ])

    # draw_pdf_contours(triangle, coordinate_pairs, [1, 1, 1])
    draw_pdf_contours(triangle, coordinate_pairs, [1.0, 1.0, 2.72])
    plt.show()
