""" Testing Davidson implementation vs NumPy builtin linear algebra """

import numpy as np
from davidson import davidson

def test_h6_1p00_chain():
    """ FCI Hamiltonian for H6 chain w/ interatomic spacing of 1.0 A in STO-6G basis """

    num_roots = 10
    hamiltonian = np.loadtxt('data/h6_1p00_chain.txt')

    dav_eigs, _ = davidson(hamiltonian,num_roots)

    np_eigs, _ = np.linalg.eigh(hamiltonian)
    np_eigs = np_eigs[:num_roots]

    assert np.allclose(dav_eigs,np_eigs)

def test_h6_1p50_chain():
    """ FCI Hamiltonian for H6 chain w/ interatomic spacing of 1.5 A in STO-6G basis """

    num_roots = 10
    hamiltonian = np.loadtxt('data/h6_1p50_chain.txt')

    dav_eigs, _ = davidson(hamiltonian,num_roots)

    np_eigs, _ = np.linalg.eigh(hamiltonian)
    np_eigs = np_eigs[:num_roots]

    assert np.allclose(dav_eigs,np_eigs)

def test_h6_2p00_chain():
    """ FCI Hamiltonian for H6 chain w/ interatomic spacing of 2.0 A in STO-6G basis """

    num_roots = 10
    hamiltonian = np.loadtxt('data/h6_2p00_chain.txt')

    dav_eigs, _ = davidson(hamiltonian,num_roots)

    np_eigs, _ = np.linalg.eigh(hamiltonian)
    np_eigs = np_eigs[:num_roots]

    assert np.allclose(dav_eigs,np_eigs)

def test_h6_1p00_ring():
    """ FCI Hamiltonian for H6 ring w/ interatomic spacing of 1.0 A in STO-6G basis """

    num_roots = 10
    hamiltonian = np.loadtxt('data/h6_1p00_ring.txt')

    dav_eigs, _ = davidson(hamiltonian,num_roots)

    np_eigs, _ = np.linalg.eigh(hamiltonian)
    np_eigs = np_eigs[:num_roots]

    assert np.allclose(dav_eigs,np_eigs)

def test_h6_1p50_ring():
    """ FCI Hamiltonian for H6 ring w/ interatomic spacing of 1.5 A in STO-6G basis """

    num_roots = 10
    hamiltonian = np.loadtxt('data/h6_1p50_ring.txt')

    dav_eigs, _ = davidson(hamiltonian,num_roots)

    np_eigs, _ = np.linalg.eigh(hamiltonian)
    np_eigs = np_eigs[:num_roots]

    assert np.allclose(dav_eigs,np_eigs)

def test_h6_2p00_ring():
    """ FCI Hamiltonian for H6 ring w/ interatomic spacing of 2.0 A in STO-6G basis """

    num_roots = 10
    hamiltonian = np.loadtxt('data/h6_2p00_ring.txt')

    dav_eigs, _ = davidson(hamiltonian,num_roots)

    np_eigs, _ = np.linalg.eigh(hamiltonian)
    np_eigs = np_eigs[:num_roots]

    assert np.allclose(dav_eigs,np_eigs)

