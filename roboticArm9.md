# Zorder
I’ve been thinking this morning and zorder is the one that saves the most time. It’s very easy to develop and demonstrates that the forward mode is superior to backpropagation when changing the operation graph for each frame, which is unfeasible from a practical standpoint because the others require “compilation” time.

From a mathematical point of view, you only have to find the intersection point between two segments. That intersection point is what exerts a force in the direction of the line of sight.

## Python code of intersection of a two segment giving two extremes  points in a 3D space.

Below is a self-contained Python example for finding the intersection of **two line segments in 3D**. The function can return:

- **No intersection** (`None`)  
- **A single intersection point** (as a 3D tuple)  
- **A line segment** (two 3D tuples) if they are collinear and overlapping  

Thus, in the collinear-overlap case, you get “two extreme points” of their intersection segment in 3D. In the single-point case, those two points will be the same.

> **Important**: This approach handles three main cases:
> 1. **Non-parallel** segments (may intersect in exactly one point, or not at all if skew).
> 2. **Parallel but not collinear** (no intersection).
> 3. **Collinear** segments (may overlap in a segment or not at all).

---

## Code

```python
import numpy as np

def intersect_segments_3d(P1, P2, Q1, Q2, tol=1e-9):
    """
    Find the intersection of two line segments in 3D.

    Parameters:
    -----------
    P1, P2 : array-like of shape (3,)
        Endpoints of the first segment.
    Q1, Q2 : array-like of shape (3,)
        Endpoints of the second segment.
    tol : float, optional
        A small tolerance for floating-point comparisons.

    Returns:
    --------
    None
        If there is no intersection.
    (x, y, z)
        If the segments intersect in exactly one point, returns that point as a tuple.
    ((x1, y1, z1), (x2, y2, z2))
        If the segments are collinear and overlap in a segment,
        returns the two extreme points of the overlapping segment as tuples.
    """
    # Convert inputs to numpy arrays (float)
    P1 = np.array(P1, dtype=float)
    P2 = np.array(P2, dtype=float)
    Q1 = np.array(Q1, dtype=float)
    Q2 = np.array(Q2, dtype=float)

    # Define vectors
    u = P2 - P1  # Direction of segment P1->P2
    v = Q2 - Q1  # Direction of segment Q1->Q2
    w = P1 - Q1

    # Squared lengths (for use in various checks)
    u_len_sq = np.dot(u, u)
    v_len_sq = np.dot(v, v)

    # Cross product to check parallelism
    cross_uv = np.cross(u, v)
    cross_uv_len_sq = np.dot(cross_uv, cross_uv)

    # ----------------------------------------------------------------
    # CASE 1: Non-parallel lines (cross_uv != 0)
    # ----------------------------------------------------------------
    if cross_uv_len_sq > tol:
        # Solve for t and s in:
        #   P1 + t*u = Q1 + s*v
        # using the analytic formula for 3D line-line intersection

        # Some helpers:
        a = np.dot(u, u)  # = u_len_sq
        b = np.dot(u, v)
        c = np.dot(v, v)  # = v_len_sq
        d = np.dot(u, w)
        e = np.dot(v, w)

        denom = a*c - b*b
        if abs(denom) < tol:
            # They are almost parallel but didn't get caught by cross_uv_len_sq > tol check —
            # might be a numerical edge case. Treat as parallel/collinear below.
            pass
        else:
            t = (b*e - c*d) / denom
            s = (a*e - b*d) / denom

            # Intersection points on infinite lines
            # R_P = P1 + t*u
            # R_Q = Q1 + s*v
            # For an actual intersection of segments, 0 <= t <= 1 and 0 <= s <= 1
            if 0.0 - tol <= t <= 1.0 + tol and 0.0 - tol <= s <= 1.0 + tol:
                # Compute the actual intersection point (single point)
                # Use either line’s param. We'll use R_P
                R = P1 + t*u
                # (Optional) You might want to clamp t and s to [0,1] if needed,
                # but checking the condition with tolerance is usually enough.
                
                # Double-check the distance between R_P and R_Q is near zero
                RQ = Q1 + s*v
                if np.linalg.norm(R - RQ) < tol:
                    # It's a valid intersection point
                    return tuple(R)
            
            # If we get here, no valid intersection
            return None

    # ----------------------------------------------------------------
    # CASE 2: Parallel or nearly parallel lines
    # ----------------------------------------------------------------
    # Check if they are collinear by seeing if (P1 - Q1) is also parallel to u
    # i.e., cross(u, (P1-Q1)) ~ 0
    cross_uw = np.cross(u, w)
    if np.dot(cross_uw, cross_uw) > tol:
        # They are parallel but NOT collinear => no intersection
        return None

    # ----------------------------------------------------------------
    # CASE 3: Collinear segments
    # ----------------------------------------------------------------
    # We can "project" each segment onto the same direction (say 'u' if it's non-zero)
    # to reduce the problem to a 1D overlap of intervals.

    # If u is too small (segment P1->P2 is basically a point), 
    # switch to v for projection to avoid division by zero.
    if u_len_sq < tol:
        # P1 and P2 are (nearly) the same point. Check if that point is on Q1->Q2
        # Project onto v
        if point_on_segment_3d(P1, Q1, Q2, tol):
            # Intersection is that single point
            return tuple(P1)
        else:
            return None

    # Likewise, if v is too small => Q1->Q2 is basically a point
    if v_len_sq < tol:
        # Q1 and Q2 are (nearly) the same point. Check if that point is on P1->P2
        if point_on_segment_3d(Q1, P1, P2, tol):
            # Intersection is that single point
            return tuple(Q1)
        else:
            return None

    # Now, both segments have non-zero length and are collinear.
    # Project onto vector u.
    # Parametrize: any point on the line can be written as P1 + alpha*u.
    # We'll find the range of alpha for:
    #   - Segment P1->P2 is alpha in [0, 1].
    #   - Segment Q1->Q2 is beta in [0, 1], but we convert it to alpha.
    
    # 1) alpha range for P-segment is obviously [0, 1].
    # 2) For Q1->Q2, we solve P1 + alpha*u = Q1 + beta*v => alpha*u - beta*v = Q1 - P1
    #    But simpler is to project Q1 and Q2 onto the direction of u, 
    #    with P1 as an origin (alpha=0).
    # 
    # Let:
    #   alphaQ1 = ( (Q1 - P1) . u ) / (u . u)
    #   alphaQ2 = ( (Q2 - P1) . u ) / (u . u)
    #
    # Then Q1, Q2 correspond to alpha in [alphaQ1, alphaQ2] (assuming alphaQ1 < alphaQ2).
    
    denom_u = np.dot(u, u)  # = u_len_sq
    alphaQ1 = np.dot((Q1 - P1), u) / denom_u
    alphaQ2 = np.dot((Q2 - P1), u) / denom_u

    # Sort them so alphaQ1 <= alphaQ2
    alpha_min = min(alphaQ1, alphaQ2)
    alpha_max = max(alphaQ1, alphaQ2)

    # Our P-segment is alpha in [0, 1].
    # Q-segment is alpha in [alpha_min, alpha_max].
    # Overlap in alpha is [max(0, alpha_min), min(1, alpha_max)].
    overlap_start = max(0.0, alpha_min)
    overlap_end   = min(1.0, alpha_max)

    if overlap_start > overlap_end + tol:
        # No overlap
        return None

    # Compute actual overlap points in 3D
    # clamp them carefully to [0,1] if they’re slightly out of bounds
    overlap_start_clamped = max(0.0, min(1.0, overlap_start))
    overlap_end_clamped   = max(0.0, min(1.0, overlap_end))

    # Endpoints in 3D
    pointA = P1 + overlap_start_clamped * u
    pointB = P1 + overlap_end_clamped * u

    # If the segment degenerates to a single point, return that
    if np.linalg.norm(pointA - pointB) < tol:
        return tuple(pointA)
    
    # Otherwise, return the two extreme points of overlap
    return (tuple(pointA), tuple(pointB))

def point_on_segment_3d(point, A, B, tol=1e-9):
    """
    Check if 'point' lies on the segment A->B (in 3D) within a tolerance.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    P = np.array(point, dtype=float)
    AB = B - A
    AP = P - A
    AB_len_sq = np.dot(AB, AB)
    if AB_len_sq < tol:
        # Segment A->B is effectively a point. Check distance
        return np.linalg.norm(A - P) < tol
    # Check collinearity:
    cross_AB_AP = np.cross(AB, AP)
    if np.dot(cross_AB_AP, cross_AB_AP) > tol:
        return False
    # Check if P is between A and B by projecting
    t = np.dot(AP, AB) / AB_len_sq
    return -tol <= t <= 1.0 + tol


# ------------------------------
# Example usage / tests
# ------------------------------
if __name__ == "__main__":
    # 1) Single point intersection (non-parallel)
    seg1_P1 = (0,0,0)
    seg1_P2 = (1,1,1)
    seg2_P1 = (1,0,0)
    seg2_P2 = (0,1,1)

    res = intersect_segments_3d(seg1_P1, seg1_P2, seg2_P1, seg2_P2)
    print("Example 1:", res)

    # 2) Collinear overlapping
    seg1_P1 = (0,0,0)
    seg1_P2 = (2,2,2)
    seg2_P1 = (1,1,1)
    seg2_P2 = (3,3,3)

    res = intersect_segments_3d(seg1_P1, seg1_P2, seg2_P1, seg2_P2)
    print("Example 2:", res)

    # 3) Parallel, non-collinear => no intersection
    seg1_P1 = (0,0,0)
    seg1_P2 = (1,0,0)
    seg2_P1 = (0,1,1)
    seg2_P2 = (1,1,1)

    res = intersect_segments_3d(seg1_P1, seg1_P2, seg2_P1, seg2_P2)
    print("Example 3:", res)
```

### How It Works

1. **Convert endpoints to NumPy arrays** for easy vector arithmetic.
2. **Compute vectors**:
   - \(u = P2 - P1\)  
   - \(v = Q2 - Q1\)  
   - \(w = P1 - Q1\)
3. **Check if the lines are parallel** via the cross product \(\text{cross}(u, v)\). 
   - If its norm is large enough, the lines are **not parallel**. We solve for parameters \(t\) and \(s\) such that:
$$
P1 + t\,u = Q1 + s\,v.
$$

     If \(0 \le t \le 1\) and \(0 \le s \le 1\), they intersect in that point within both segments.
   - If the lines are parallel, check if they’re **collinear**. If they aren’t collinear, there is **no intersection**.
   - If they’re **collinear**, reduce to a **1D overlap** check by projecting onto one of the direction vectors.
4. **Return**:
   - `None` if no intersection,
   - A single point if they intersect in exactly one point,
   - Two points \((pointA, pointB)\) if they overlap in a line segment (the “extreme endpoints”).

This is the standard approach for finding intersection or overlap of two **finite line segments** in 3D space.