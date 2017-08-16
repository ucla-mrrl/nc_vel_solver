import numpy as np

def gen_encoding(encode_type, null_encode=False, dvenc=2):

    if encode_type == '4pt-null':
        A = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    elif encode_type == '6pt-Zwart':
        psi_a = 0.5 * (1.0 + np.sqrt(5.0))

        A = np.array([[0,      1,     psi_a],
                      [1,      psi_a, 0],
                      [psi_a,  0,     1],
                      [0,      -1,    psi_a],
                      [-1,     psi_a, 0],
                      [-psi_a, 0,     1]])

        scale = np.sqrt(1 + psi_a * psi_a)
        A /= scale

        if null_encode:
            A = np.vstack((np.array([0,0,0]), A))
    elif encode_type == 'dualvenc':
        A = np.array([[1, 0, 0],
                      [dvenc, 0, 0],
                      [0, 1, 0],
                      [0, dvenc, 0],
                      [0, 0, 1],
                      [0, 0, dvenc]
                      ])

        if null_encode:
            A = np.vstack((np.array([0, 0, 0]), A))

    return A