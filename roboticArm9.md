# Zorder
I've been thinking this morning and zorder is the one that saves the most time. It's very easy to develop and demonstrates that the forward mode is superior to backpropagation when changing the operation graph for each frame, which is unfeasible from a practical standpoint because the others require "compilation" time.

From a mathematical point of view, you only have to find the intersection point between two segments. That intersection point is what exerts a force in the direction of the line of sight.

## Python code for intersection of two segments given two endpoints in a 2D space.

```python
import math

def intersect_segments_2d(P1, P2, Q1, Q2, tol=1e-9):
    """
    Find the intersection of two line segments in 2D.
    
    Parameters
    ----------
    P1, P2 : (float, float)
        Endpoints of the first segment.
    Q1, Q2 : (float, float)
        Endpoints of the second segment.
    tol : float
        Tolerance for floating-point comparisons.
    
    Returns
    -------
    None
        If there is no intersection.
    (x, y)
        If the segments intersect at exactly one point.
    ((x1, y1), (x2, y2))
        If the segments are collinear and overlap in a segment;
        returns the two extreme endpoints of that overlapping segment.
    """

    # Unpack points for convenience
    x1, y1 = P1
    x2, y2 = P2
    x3, y3 = Q1
    x4, y4 = Q2

    # Define some helper vectors
    # u = vector P1->P2
    # v = vector Q1->Q2
    # w = vector Q1->P1
    u = (x2 - x1, y2 - y1)
    v = (x4 - x3, y4 - y3)
    w = (x1 - x3, y1 - y3)

    # A small cross product function in 2D
    def cross_2d(a, b):
        return a[0]*b[1] - a[1]*b[0]
    
    # A small dot product function in 2D
    def dot_2d(a, b):
        return a[0]*b[0] + a[1]*b[1]

    cross_uv = cross_2d(u, v)
    
    # ------------------------------------------------------------
    # CASE 1: If u and v are NOT parallel (cross_uv != 0), 
    #         the lines may intersect at one point.
    # ------------------------------------------------------------
    if abs(cross_uv) > tol:
        # Solve the system:
        #   P1 + t*u = Q1 + s*v
        # using known formulas for line intersection in 2D.
        
        # We want to find 't' such that:
        #   cross(v, (Q1 - P1)) / cross(v, u) = t
        # or equivalently use param expansions.

        cross_v_w = cross_2d(v, w)
        cross_u_w = cross_2d(u, w)
        
        t = cross_v_w / cross_uv
        s = cross_u_w / cross_uv

        # For an intersection within both segments, we need t & s in [0,1].
        if -tol <= t <= 1 + tol and -tol <= s <= 1 + tol:
            # Intersection point (we can plug t into P1 + t*u)
            ix = x1 + t*u[0]
            iy = y1 + t*u[1]
            
            # (Optional) check the actual difference using s as well:
            # Q1 + s*v should be (ix, iy). If numerical error is small, it's valid.
            
            return (ix, iy)  # Single intersection point
        
        # If we get here, the infinite lines intersect, 
        # but not within the bounds of both segments
        return None

    # ------------------------------------------------------------
    # CASE 2: If u and v are parallel (cross_uv ~ 0)
    # ------------------------------------------------------------
    # Check if they are collinear by seeing if 
    # the vector (P1 - Q1) is also parallel to u (or v).
    # cross(u, w) = 0 => collinear
    if abs(cross_2d(u, w)) > tol:
        # Parallel but not collinear => no intersection
        return None

    # ------------------------------------------------------------
    # CASE 3: Collinear segments
    # ------------------------------------------------------------
    # We can project the points onto the 'u' direction (if u is not zero-length).
    # If u is near zero-length, we use v or just check the single point scenario.

    # Squared length of u (to check if P1~P2)
    u_len_sq = dot_2d(u, u)
    # Squared length of v
    v_len_sq = dot_2d(v, v)
    
    def point_on_segment_2d(pt, A, B):
        """Check if pt lies on the segment A->B (collinear case)."""
        (Ax, Ay), (Bx, By) = A, B
        (Px, Py) = pt
        # Check bounding box
        if (min(Ax, Bx) - tol <= Px <= max(Ax, Bx) + tol and
            min(Ay, By) - tol <= Py <= max(Ay, By) + tol):
            return True
        return False

    # If the first segment is effectively a point:
    if u_len_sq < tol:
        # Then P1==P2. Check if that point is on segment Q1->Q2
        if point_on_segment_2d(P1, Q1, Q2):
            return P1  # A single intersection point
        else:
            return None

    # If second segment is effectively a point:
    if v_len_sq < tol:
        # Then Q1==Q2. Check if that point is on segment P1->P2
        if point_on_segment_2d(Q1, P1, P2):
            return Q1
        else:
            return None

    # Both segments have non-zero length and are collinear.
    # Project Q1, Q2 onto the line of segment P by param alpha:
    #    Any point on P can be written as P1 + alpha*u, alpha in [0,1].
    # Then alpha for Q1 is ( (Q1 - P1) · u ) / (u · u ).
    # We'll do similarly for Q2, then find overlap in alpha-space.
    def param_on_P(pt, P1, u):
        # param alpha for 'pt' on line (P1 + alpha*u)
        # here: alpha = (pt - P1) · u / (u · u)
        px, py = pt
        p1x, p1y = P1
        return ((px - p1x)*u[0] + (py - p1y)*u[1]) / (dot_2d(u, u))

    alphaQ1 = param_on_P(Q1, P1, u)
    alphaQ2 = param_on_P(Q2, P1, u)

    alpha_min = min(alphaQ1, alphaQ2)
    alpha_max = max(alphaQ1, alphaQ2)

    # Overlap in alpha is [max(0, alpha_min), min(1, alpha_max)]
    overlap_start = max(0.0, alpha_min)
    overlap_end   = min(1.0, alpha_max)

    if overlap_start > overlap_end + tol:
        # No overlap
        return None

    # Compute the actual overlapping segment in (x,y)
    start_pt = (x1 + overlap_start*u[0], y1 + overlap_start*u[1])
    end_pt   = (x1 + overlap_end  *u[0], y1 + overlap_end  *u[1])

    # If overlap is effectively a single point
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    if dx*dx + dy*dy < tol*tol:
        # Single point
        return start_pt

    # Return the two extreme points
    return (start_pt, end_pt)


# ----------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------
if __name__ == "__main__":
    # 1) One-point intersection (non-parallel)
    s1_p1 = (0, 0)
    s1_p2 = (1, 1)
    s2_p1 = (1, 0)
    s2_p2 = (0, 1)
    res = intersect_segments_2d(s1_p1, s1_p2, s2_p1, s2_p2)
    print("Example 1:", res)  # Should be (0.5, 0.5) roughly

    # 2) Collinear overlapping
    s1_p1 = (0, 0)
    s1_p2 = (3, 3)
    s2_p1 = (2, 2)
    s2_p2 = (5, 5)
    res = intersect_segments_2d(s1_p1, s1_p2, s2_p1, s2_p2)
    print("Example 2:", res)  # Overlaps from (2,2) to (3,3)

    # 3) Parallel but non-collinear => No intersection
    s1_p1 = (0, 0)
    s1_p2 = (1, 0)
    s2_p1 = (0, 1)
    s2_p2 = (1, 1)
    res = intersect_segments_2d(s1_p1, s1_p2, s2_p1, s2_p2)
    print("Example 3:", res)  # None
```
One of the challenges in parallel computation is instructions like:

```python
if abs(cross_uv) > tol:
```

That is, since parallel processing is done for all individuals in the population, it is difficult to execute a conditional instruction.

These are for cases where one of the endpoints is on the segment. Do these cases occur? Can we dispense with this check?