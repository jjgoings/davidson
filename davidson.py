""" Davidson iterative diagonalization """
import numpy as np


def davidson(hamiltonian, num_roots=3, convergence=1e-6):
    """
    Return the lowest several eigenvalues and associated eigenvectors of a real (M x M) symmetric
    matrix using the Davidson iterative diagonalization method.
    Returns two objects, a 1-D array containing the eigenvalues of `hamiltonian`, and
    a 2-D square array of the corresponding eigenvectors (in columns).

    Parameters
    ----------
    hamiltonian : (M, M) ndarray
        Real symmetric matrix whose eigenvalues and eigenvectors are to be computed.

    num_roots : int
        Number of lowest eigenvalues and eigenvectors to be computed.

    convergence : float
        Calculation will end when the residual norm for each root falls below `convergence`

    Returns
    -------
    eigenvalues : (num_roots) ndarray
        The num_roots lowest eigenvalues in ascending order
    eigenvectors : (M, num_roots) ndarray
        The column ``eigenvectors[:, i]`` is the normalized eigenvector corresponding
        to the eigenvalue ``eigenvalue[i]``.
    """

    dim_hamiltonian = hamiltonian.shape[0]
    dim_subspace = min(3 * num_roots, dim_hamiltonian)
    guess_vectors = np.eye(dim_hamiltonian, dim_subspace)

    converged = False
    while not converged:
        # project hamiltonian onto subspace
        subspace_hamiltonian = np.dot(np.dot(guess_vectors.T, hamiltonian),
                                      guess_vectors)

        # diag subspace_hamiltonian
        eigenvalues, subspace_eigenvectors = np.linalg.eigh(subspace_hamiltonian)
        eigenvalues, subspace_eigenvectors = \
            eigenvalues[:num_roots], subspace_eigenvectors[:, :num_roots]

        # get current approx. eigenvectors
        eigenvectors = np.dot(guess_vectors, subspace_eigenvectors)

        # form residual vectors
        residual_vectors = np.zeros((dim_hamiltonian, num_roots))
        correction_vectors = np.zeros((dim_hamiltonian, num_roots))

        # orthonormalize and add correction vectors
        for j in range(num_roots):
            residual_vectors[:, j] = np.dot(
                (hamiltonian - eigenvalues[j] * np.eye(dim_hamiltonian)),
                eigenvectors[:, j])
            converged = True
            if np.linalg.norm(residual_vectors[:, j]) > convergence:
                converged = False
                # clip values to avoid potential divergence
                preconditioner = np.clip(
                    1 / (eigenvalues[j] - np.diag(hamiltonian)),
                    a_min=-1e5,
                    a_max=1e5)

                # normalize correction vectors
                correction_vectors[:, j] = preconditioner * residual_vectors[:, j]
                correction_vectors[:, j] /= np.linalg.norm(correction_vectors[:, j])

                # project corrections onto orthogonal complement
                new_vector = np.dot((np.eye(dim_hamiltonian) -
                                     np.dot(guess_vectors, guess_vectors.T)),
                                    correction_vectors[:, j])
                new_vector_norm = np.linalg.norm(new_vector)

                # only add new vectors with 'significant' norm -- ignore the others
                # usually ~ 1e-4 is sufficient, but feel free to play with this
                if new_vector_norm > 1e-4:
                    # if subspace too big, collapse to best set of eigenvectors
                    if dim_subspace + 1 > min(500,dim_hamiltonian):
                        dim_subspace = num_roots
                        guess_vectors = eigenvectors
                        # enforce orthogonality (cheap b/c dim guess_vectors is small)
                        guess_vectors, _ = np.linalg.qr(guess_vectors)
                        break

                    guess_vectors_copy = np.copy(guess_vectors)
                    dim_subspace += 1
                    guess_vectors = np.eye(dim_hamiltonian, dim_subspace)
                    guess_vectors[:, :(dim_subspace - 1)] = guess_vectors_copy
                    # add new orthonormal vector to guess vectors
                    guess_vectors[:, -1] = new_vector / new_vector_norm

    if converged:
        return eigenvalues, eigenvectors


if __name__ == '__main__':

    import time

    t0 = time.time()
    np.random.seed(0)
    NDIM = 1000
    A = np.diag(np.arange(NDIM, dtype=np.float64))
    A[1:3, 1:3] = 0
    M = np.random.randn(NDIM, NDIM)
    M += M.T
    A += 1e-2 * M

    t1 = time.time()
    print("Hamiltonian formation (ms): %.2f" % ((t1 - t0) * 1e3))

    NUM_ROOTS = 3
    E, C = davidson(A, NUM_ROOTS)
    t2 = time.time()
    print("Davidson (ms):         %.2f" % ((t2 - t1) * 1e3))
    print(C.shape)

    E_true, C_true = np.linalg.eigh(A)
    E_true, C_true = E_true[:NUM_ROOTS], C_true[:, :NUM_ROOTS]
    t3 = time.time()
    print("Numpy (ms):            %.2f" % ((t3 - t2) * 1e3))

    assert np.allclose(E, E_true)
    print("Eigs:                 ", E)
