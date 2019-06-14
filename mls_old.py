import math

def affine_transform(v, mapping, alpha = 1):
    """affine_transform(v, mapping, alpha = 1)
    
    Maps a point on the undeformed image to a point on the deformed image given a set
    of points mapping points on the deformed image to points on the undeformed image using
    affine transformation.
    
    Parameters
    ----------
    v: tuple
        A point on the undeformed image represented by a tuple containing a pair of values.
    
    mapping: list of tuples of tuples
        A list of tuples containing a pair of tuples that represent points on the undeformed
        image mapping to points on the deformed image.
    
    alpha: float, optional
        Can affect where a point maps exactly on the deformed image and is set to 1 by
        default if it is not specified.
    """
    p_wgt = (0, 0)
    q_wgt = (0, 0)
    w_sum = 0
    for mp in mapping:
        x = mp[0][0] - v[0]
        y = mp[0][1] - v[1]
        if (x == 0 and y == 0): return mp[1]
        w = 1/((x*x + y*y) ** alpha)
        p_wgt = (p_wgt[0] + w*mp[0][0], p_wgt[1] + w*mp[0][1])
        q_wgt = (q_wgt[0] + w*mp[1][0], q_wgt[1] + w*mp[1][1])
        w_sum += w
    p_wgt = (p_wgt[0]/w_sum, p_wgt[1]/w_sum)
    q_wgt = (q_wgt[0]/w_sum, q_wgt[1]/w_sum)
    M1 = [0, 0, 0, 0]
    M2 = [0, 0, 0, 0]
    for mp in mapping:
        p_adj = (mp[0][0] - p_wgt[0], mp[0][1] - p_wgt[1])
        q_adj = (mp[1][0] - q_wgt[0], mp[1][1] - q_wgt[1])
        x = mp[0][0] - v[0]
        y = mp[0][1] - v[1]
        w = 1/((x*x + y*y) ** alpha)
        M1 = [M1[0] + w*p_adj[0]*p_adj[0], M1[1] + w*p_adj[0]*p_adj[1], M1[2] + w*p_adj[1]*p_adj[0], M1[3] + w*p_adj[1]*p_adj[1]]
        M2 = [M2[0] + w*p_adj[0]*q_adj[0], M2[1] + w*p_adj[0]*q_adj[1], M2[2] + w*p_adj[1]*q_adj[0], M2[3] + w*p_adj[1]*q_adj[1]]
    disc = M1[0]*M1[3] - M1[1]*M1[2]
    M1 = [M1[3]/disc, -M1[2]/disc, -M1[1]/disc, M1[0]/disc]
    M = [M1[0]*M2[0] + M1[1]*M2[2], M1[0]*M2[1] + M1[1]*M2[3], M1[2]*M2[0] + M1[3]*M2[2], M1[2]*M2[1] + M1[3]*M2[3]]
    v_out = (v[0] - p_wgt[0], v[1] - p_wgt[1])
    v_out = (v_out[0]*M[0] + v_out[1]*M[2] + q_wgt[0], v_out[0]*M[1] + v_out[1]*M[3] + q_wgt[1])
    return v_out

def similarity_transform(v, mapping, alpha = 1):
    """similarity_transform(v, mapping, alpha = 1)
    
    Maps a point on the undeformed image to a point on the deformed image given a set
    of points mapping points on the deformed image to points on the undeformed image using
    similarity transformation.
    
    Parameters
    ----------
    v: tuple
        A point on the undeformed image represented by a tuple containing a pair of values.
    
    mapping: list of tuples of tuples
        A list of tuples containing a pair of tuples that represent points on the undeformed
        image mapping to points on the deformed image.
    
    alpha: float, optional
        Can affect where a point maps exactly on the deformed image and is set to 1 by
        default if it is not specified.
    """
    p_wgt = (0, 0)
    q_wgt = (0, 0)
    w_sum = 0
    for mp in mapping:
        x = mp[0][0] - v[0]
        y = mp[0][1] - v[1]
        if (x == 0 and y == 0): return mp[1]
        w = 1/((x*x + y*y) ** alpha)
        p_wgt = (p_wgt[0] + w*mp[0][0], p_wgt[1] + w*mp[0][1])
        q_wgt = (q_wgt[0] + w*mp[1][0], q_wgt[1] + w*mp[1][1])
        w_sum += w
    p_wgt = (p_wgt[0]/w_sum, p_wgt[1]/w_sum)
    q_wgt = (q_wgt[0]/w_sum, q_wgt[1]/w_sum)
    mu = 0
    for mp in mapping:
        p_adj = (mp[0][0] - p_wgt[0], mp[0][1] - p_wgt[1])
        x = mp[0][0] - v[0]
        y = mp[0][1] - v[1]
        w = 1/((x*x + y*y) ** alpha)
        mu += w*(p_adj[0]*p_adj[0] + p_adj[1]*p_adj[1])
    A_fac = [v[0] - p_wgt[0], v[1] - p_wgt[1], v[1] - p_wgt[1], p_wgt[0] - v[0]]
    v_out = (0, 0)
    for mp in mapping:
        p_adj = (mp[0][0] - p_wgt[0], mp[0][1] - p_wgt[1])
        q_adj = (mp[1][0] - q_wgt[0], mp[1][1] - q_wgt[1])
        x = mp[0][0] - v[0]
        y = mp[0][1] - v[1]
        w = 1/((x*x + y*y) ** alpha)
        A = [w*(p_adj[0]*A_fac[0] + p_adj[1]*A_fac[2]), w*(p_adj[0]*A_fac[1] + p_adj[1]*A_fac[3]), w*(p_adj[1]*A_fac[0] - p_adj[0]*A_fac[2]), w*(p_adj[1]*A_fac[1] - p_adj[0]*A_fac[3])]
        v_out = (v_out[0] + (q_adj[0]*A[0] + q_adj[1]*A[2])/mu, v_out[1] + (q_adj[0]*A[1] + q_adj[1]*A[3])/mu)
    v_out = (v_out[0] + q_wgt[0], v_out[1] + q_wgt[1])
    return v_out

def rigid_transform(v, mapping, alpha = 1):
    """rigid_transform(v, mapping, alpha = 1)
    
    Maps a point on the undeformed image to a point on the deformed image given a set
    of points mapping points on the deformed image to points on the undeformed image using
    rigid transformation.
    
    Parameters
    ----------
    v: tuple
        A point on the undeformed image represented by a tuple containing a pair of values.
    
    mapping: list of tuples of tuples
        A list of tuples containing a pair of tuples that represent points on the undeformed
        image mapping to points on the deformed image.
    
    alpha: float, optional
        Can affect where a point maps exactly on the deformed image and is set to 1 by
        default if it is not specified.
    """
    p_wgt = (0, 0)
    q_wgt = (0, 0)
    w_sum = 0
    for mp in mapping:
        x = mp[0][0] - v[0]
        y = mp[0][1] - v[1]
        if (x == 0 and y == 0): return mp[1]
        w = 1/((x*x + y*y) ** alpha)
        p_wgt = (p_wgt[0] + w*mp[0][0], p_wgt[1] + w*mp[0][1])
        q_wgt = (q_wgt[0] + w*mp[1][0], q_wgt[1] + w*mp[1][1])
        w_sum += w
    p_wgt = (p_wgt[0]/w_sum, p_wgt[1]/w_sum)
    q_wgt = (q_wgt[0]/w_sum, q_wgt[1]/w_sum)
    A_fac = [v[0] - p_wgt[0], v[1] - p_wgt[1], v[1] - p_wgt[1], p_wgt[0] - v[0]]
    v_out = (0, 0)
    for mp in mapping:
        p_adj = (mp[0][0] - p_wgt[0], mp[0][1] - p_wgt[1])
        q_adj = (mp[1][0] - q_wgt[0], mp[1][1] - q_wgt[1])
        x = mp[0][0] - v[0]
        y = mp[0][1] - v[1]
        w = 1/((x*x + y*y) ** alpha)
        A = [w*(p_adj[0]*A_fac[0] + p_adj[1]*A_fac[2]), w*(p_adj[0]*A_fac[1] + p_adj[1]*A_fac[3]), w*(p_adj[1]*A_fac[0] - p_adj[0]*A_fac[2]), w*(p_adj[1]*A_fac[1] - p_adj[0]*A_fac[3])]
        v_out = (v_out[0] + q_adj[0]*A[0] + q_adj[1]*A[2], v_out[1] + q_adj[0]*A[1] + q_adj[1]*A[3])
    r = math.sqrt(v_out[0]*v_out[0] + v_out[1]*v_out[1])
    v_out = (v_out[0]/r, v_out[1]/r)
    v_sub = (v[0] - p_wgt[0], v[1] - p_wgt[1])
    r = math.sqrt(v_sub[0]*v_sub[0] + v_sub[1]*v_sub[1])
    v_out = (r*v_out[0] + q_wgt[0], r*v_out[1] + q_wgt[1])
    return v_out
