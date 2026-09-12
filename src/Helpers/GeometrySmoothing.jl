"""
Geometry smoothing and simplification helpers for closed polygonal contours.

All methods here are deliberately low-order and local to keep behavior
mathematically tractable and avoid high-order spline oscillations.
"""

@inline function _point_segment_dist(x::Float64, y::Float64,
                                     x1::Float64, y1::Float64,
                                     x2::Float64, y2::Float64)
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-14
        return hypot(x - x1, y - y1)
    end
    t = clamp(((x - x1) * dx + (y - y1) * dy) / seg_len_sq, 0.0, 1.0)
    px = x1 + t * dx
    py = y1 + t * dy
    return hypot(x - px, y - py)
end

function _dp_open(pts::Vector{Tuple{Float64,Float64}}, epsilon::Float64)
    n = length(pts)
    n <= 2 && return pts

    x1, y1 = pts[1]
    x2, y2 = pts[end]
    dmax = -1.0
    imax = 1
    for i in 2:(n - 1)
        xi, yi = pts[i]
        d = _point_segment_dist(xi, yi, x1, y1, x2, y2)
        if d > dmax
            dmax = d
            imax = i
        end
    end

    if dmax > epsilon
        left = _dp_open(pts[1:imax], epsilon)
        right = _dp_open(pts[imax:end], epsilon)
        return vcat(left[1:end-1], right)
    else
        return [pts[1], pts[end]]
    end
end

"""
    simplify_polygon(pts, epsilon)

Douglas-Peucker simplification for CLOSED contours (cyclic-safe).

This implementation first converts the loop to an open representation,
splits the cycle into two open chains using the farthest pair heuristic,
applies standard DP on each chain, then re-closes the polygon.
"""
function simplify_polygon(pts::Vector{Tuple{Float64,Float64}}, epsilon::Float64)
    n = length(pts)
    n <= 4 && return pts

    # Work with open representation (drop duplicate closing point if present)
    open_pts = (pts[1] == pts[end]) ? pts[1:end-1] : copy(pts)
    m = length(open_pts)
    m < 3 && return pts

    # Pick farthest pair to split cycle into two open chains
    i0, j0 = 1, 2
    dmax_sq = -1.0
    for i in 1:m
        xi, yi = open_pts[i]
        for j in (i + 1):m
            xj, yj = open_pts[j]
            d_sq = (xi - xj)^2 + (yi - yj)^2
            if d_sq > dmax_sq
                dmax_sq = d_sq
                i0, j0 = i, j
            end
        end
    end

    chain1 = open_pts[i0:j0]
    chain2 = vcat(open_pts[j0:end], open_pts[1:i0])

    s1 = _dp_open(chain1, epsilon)
    s2 = _dp_open(chain2, epsilon)

    merged_open = vcat(s1[1:end-1], s2[1:end-1])
    if isempty(merged_open)
        merged_open = open_pts
    end
    return vcat(merged_open, merged_open[1])
end

"""
    smooth_polygon_chaikin(pts, iters, alpha)

Chaikin corner-cutting smoothing for closed polygons.

Each edge is replaced by two points:
Q = (1-a) P_i + a P_{i+1}
R = a P_i + (1-a) P_{i+1}

with a in (0, 0.5). This produces a smooth, local, non-oscillatory curve.
"""
function smooth_polygon_chaikin(pts::Vector{Tuple{Float64,Float64}},
                                iters::Int,
                                alpha::Float64)
    iters <= 0 && return pts
    alpha <= 0 && return pts
    alpha >= 0.5 && return pts

    cur = copy(pts)
    for _ in 1:iters
        open_pts = (cur[1] == cur[end]) ? cur[1:end-1] : cur
        m = length(open_pts)
        m < 3 && return cur

        nxt = Tuple{Float64,Float64}[]
        sizehint!(nxt, 2 * m + 1)
        for i in 1:m
            j = (i == m) ? 1 : (i + 1)
            x1, y1 = open_pts[i]
            x2, y2 = open_pts[j]
            q = ((1 - alpha) * x1 + alpha * x2,
                 (1 - alpha) * y1 + alpha * y2)
            r = (alpha * x1 + (1 - alpha) * x2,
                 alpha * y1 + (1 - alpha) * y2)
            push!(nxt, q)
            push!(nxt, r)
        end
        push!(nxt, nxt[1])
        cur = nxt
    end
    return cur
end

"""
    smooth_polygon_laplacian(pts, iters, lambda)

Uniform Laplacian smoothing on closed polygons:
P_i <- (1-lambda) P_i + lambda * 0.5 * (P_{i-1} + P_{i+1})

Local, linear, and easy to reason about. Use moderate lambda and few iters
to avoid over-shrinking.
"""
function smooth_polygon_laplacian(pts::Vector{Tuple{Float64,Float64}},
                                  iters::Int,
                                  λ::Float64)
    iters <= 0 && return pts
    λ <= 0 && return pts
    λ >= 1 && return pts

    cur = copy(pts)
    for _ in 1:iters
        open_pts = (cur[1] == cur[end]) ? cur[1:end-1] : cur
        m = length(open_pts)
        m < 4 && return cur

        nxt = copy(open_pts)
        @inbounds for i in 1:m
            ip = (i == 1) ? m : (i - 1)
            inx = (i == m) ? 1 : (i + 1)
            x, y = open_pts[i]
            xp, yp = open_pts[ip]
            xn, yn = open_pts[inx]
            x_avg = 0.5 * (xp + xn)
            y_avg = 0.5 * (yp + yn)
            nxt[i] = ((1 - λ) * x + λ * x_avg,
                      (1 - λ) * y + λ * y_avg)
        end
        cur = vcat(nxt, nxt[1])
    end
    return cur
end

"""
    cap_polygon_points(pts, max_points)

Uniformly subsample polygon points to keep runtime tractable.
"""
function cap_polygon_points(pts::Vector{Tuple{Float64,Float64}}, max_points::Int)
    max_points <= 0 && return pts
    n = length(pts)
    n <= max_points && return pts
    idx = round.(Int, range(1, n, length=max_points))
    capped = pts[idx]
    if capped[1] != capped[end]
        push!(capped, capped[1])
    end
    return capped
end
