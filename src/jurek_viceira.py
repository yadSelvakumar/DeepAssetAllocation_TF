import numpy as np
import scipy as sp
import tensorflow as tf


def compute_JV_matrices(Phi0, Phi1, Sigma_v, settings):
    stateNames = [name[0] for name in settings['stateNames'][0][0][0]]#settings['stateNames'][0][0][0]
    gammas = settings['allocationSettings'][0][0][0][0][0][0][0]
    r = sum(['EXCESS_RETURN' in name and name not in settings['ExcludeAssets'][0][0] for name in stateNames]) #
    m = settings['allocationSettings'][0][0][0][0][3][0][0]#settings['allocationSettings']['m']
    K = settings['allocationSettings'][0][0][0][0][1][0][0]#settings['allocationSettings']['H']
    
    No_States = max(1 + r + m, Phi0.shape[0])
    
    L = np.zeros((1, No_States))
    
    L[0, fillID('INFLATION', stateNames)] = 1
    L[0, fillID('INFLATION', stateNames) + settings['nstates'][0][0][0][0]] = -1
    L[0, fillID('REAL_RISKFREE', stateNames) + settings['nstates'][0][0][0][0]] = -1
    
    AA = np.eye(No_States + 1)
    AA[0, 1:] = L
    iAA = np.linalg.inv(AA)
    
    Phi0 = np.dot(iAA, np.vstack((0, Phi0)))
    Phi1 = np.dot(iAA, sp.linalg.block_diag(0, Phi1))
    Sigma_v = np.dot(np.dot(iAA, sp.linalg.block_diag(0, Sigma_v)), iAA.T)
    
    No_States = No_States + 1
    
    stateNames = ['XP_REAL_RATE'] + stateNames
    
    A0 = np.nan * np.zeros((r, K))
    A1 = np.nan * np.zeros((r, No_States, K))
    
    B1 = np.nan * np.zeros((1, No_States, K))
    B2 = np.nan * np.zeros((No_States, No_States, K))
    
    H1 = np.zeros((1, No_States))
    H1[0, fillID('XP_REAL_RATE', stateNames)] = 1
    
    for i in range(len(stateNames)):
        if stateNames[i] in settings['ExcludeAssets']:
            stateNames[i] = 'OUT'
    
    idx_returns = np.where(['RETURN' in name for name in stateNames])[0]
    Hx = np.zeros((r, No_States))
    Hx[:,idx_returns] = np.eye(r)
    
    Sigma_xx = np.dot(np.dot(Hx, Sigma_v), Hx.T)
    Sigma_xx_inv = np.linalg.inv(Sigma_xx)
    sig2_x = np.expand_dims(np.diag(Sigma_xx),axis = 1)
    Sigma_1x = np.dot(np.dot(Hx, Sigma_v), H1.T)
    sig2_1 = np.dot(np.dot(H1, Sigma_v), H1.T)
    
    Phi0_x = np.dot(Hx, Phi0)
    Phi1_x = np.dot(Hx, Phi1)
    
    Phi0_1 = np.dot(H1, Phi0)
    Phi1_1 = np.dot(H1, Phi1)
    
    A0[:, [-1]] = (1 / gammas) * Sigma_xx_inv @ (Phi0_x + 0.5 * sig2_x + (1 - gammas) * Sigma_1x)
    A1[:, :, -1] = (1 / gammas) * Sigma_xx_inv @ Phi1_x
    
    B1[:, :, -1] = Phi1_1 + A0[:, [-1]].T @ (Phi1_x - gammas * Sigma_xx @ A1[:, :, -1]) + \
                   (Phi0_x + 0.5 * sig2_x + (1 - gammas) * Sigma_1x).T @ A1[:, :, -1]
    B2[:, :, -1] = A1[:, :, -1].T @ (Phi1_x - 0.5 * gammas * Sigma_xx @ A1[:, :, -1])
    
    for k in range(K - 2, -1, -1):

        A0[:, [k]] = (1 / gammas) * Sigma_xx_inv @(Phi0_x + 0.5 * sig2_x + (1 - gammas) * (
            Sigma_1x + Hx @ Sigma_v @ (B1[:, :, k + 1].T + (B2[:, :, k + 1].T + B2[:, :, k + 1])@ Phi0)))

        A1[:, :, k] = (1 / gammas) * Sigma_xx_inv @ (Phi1_x + (1 - gammas) * Hx@ Sigma_v @ (B2[:, :, k + 1].T + B2[:, :, k + 1])@ Phi1)

        LLambda = 0.5 * (B2[:, :, k + 1].T + B2[:, :, k + 1]) @ Sigma_v @ (B2[:, :, k + 1].T + B2[:, :, k + 1]).T * 0.5
        GGamma = (B2[:, :, k + 1].T + B2[:, :, k + 1])@ Sigma_v
        PSI = GGamma.T

        B1[:, :, k] = Phi1_1 + A0[:, [k]].T @ (Phi1_x - gammas * Sigma_xx@ A1[:, :, k]) + \
                      (Phi0_x + 0.5 * sig2_x + (1 - gammas) * Sigma_1x).T @ A1[:, :, k] + \
                      (B1[:, :, k + 1] + Phi0.T @ (B2[:, :, k + 1].T + B2[:, :, k + 1])) @ Phi1 + \
                      (1 - gammas) * (B1[:, :, k + 1] @ (Hx @ Sigma_v).T + Phi0.T @ (Hx @ PSI).T)@A1[:, :, k] + \
                        (1 - gammas) * (2 * Phi0.T @ (LLambda + LLambda.T) + H1 @ PSI + A0[:, [k]].T @ Hx @ PSI + B1[:, :, k + 1]@ PSI) @ Phi1

        B2[:, :, k] = A1[:, :, k].T @ (Phi1_x - 0.5 * gammas * Sigma_xx @ A1[:, :, k]) + \
                      Phi1.T @ (B2[:, :, k + 1] + 2 * (1 - gammas) * LLambda) @ Phi1 + \
                      (1 - gammas) * Phi1.T @ (Hx @ PSI).T @ (A1[:, :, k])

    ReorderVec = np.arange(K - 1, -1, -1)
    
    A0 = A0[:, ReorderVec]
    A1 = A1[:, 1:, ReorderVec]
    
    B1 = B1[:, :, ReorderVec]
    B2 = B2[:, :, ReorderVec]
    
    return tf.cast(A0,tf.float32), tf.cast(A1,tf.float32)

def fillID(nam1, Names1, nam2=None, Names2=None):
    if Names2 is None:
        Names2 = Names1

        if nam2 is None:
            nam2 = ''

    N = len(Names1)
    M = len(Names2)

    if nam2 == '':
        posIdx = np.where(np.array(Names1) == nam1)[0]

    else:
        p1 = np.where(np.array(Names1) == nam1)[0]
        p2 = np.where(np.array(Names2) == nam2)[0]

        if len(p1) == 0 or len(p2) == 0:
            posIdx = np.array([])

        else:
            posIdx = np.ravel_multi_index((p1, p2), (N, M))

    return posIdx