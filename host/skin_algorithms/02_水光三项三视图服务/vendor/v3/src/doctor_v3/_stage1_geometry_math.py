"""Regional 2.5D sampling helpers. No dense map is fabricated from face landmarks."""
import cv2
import numpy as np
from scipy.spatial import Delaunay

VERSION = "v3-regional-mesh-curvature-2"
PARAMETERS = {"plane_min_samples": 6, "curvature_min_samples": 9,
              "minimum_triangle_shape": .05, "max_reference_condition": 10000.,
              "minimum_curvature_edges": 3, "minimum_valid_triangle_fraction": .50,
              "max_plane_condition": 100000.}


def scale_of(points):
    p = np.asarray(points, float)
    if len(p) < 400 or not np.isfinite(p).all():
        return None
    scale = float(np.linalg.norm(p[33, :2] - p[263, :2]))
    return scale if scale > 10 else None


def local_samples(arrays, domain):
    p = np.asarray(arrays.get("landmarks", []), float)
    z = arrays.get("relative_z", arrays.get("landmarks_relative_z"))
    scale = scale_of(p)
    if z is None or scale is None:
        return None, "missing_relative_z"
    z = np.asarray(z, float).reshape(-1)
    if len(z) != len(p):
        return None, "invalid_relative_z_shape"
    xy = np.rint(p[:, :2]).astype(int)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < domain.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < domain.shape[0])
    ids = np.flatnonzero(inside)
    ids = ids[domain[xy[ids, 1], xy[ids, 0]] & np.isfinite(z[ids])]
    if len(ids) < PARAMETERS["plane_min_samples"]:
        return None, "insufficient_regional_z_samples"
    coords = (p[ids, :2] - p[ids, :2].mean(axis=0)) / scale
    values = z[ids] / scale
    design = np.column_stack((np.ones(len(ids)), coords))
    if np.linalg.matrix_rank(design) < 3 or np.linalg.cond(design)>PARAMETERS["max_plane_condition"]:
        return None, "degenerate_regional_geometry"
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ coefficients
    return {"ids": ids, "xy": coords, "pixel_xy": p[ids, :2], "z": values,
            "relative_depth": residual, "scale_px": scale, "plane": coefficients,
            "domain": np.asarray(domain,dtype=bool)}, None


def sparse_relief(samples, domain, minimum_depth, support=None):
    """Area-weighted triangle quadrature on regional samples, never rasterized."""
    xy, depth = samples["pixel_xy"], samples["relative_depth"]
    try:
        triangles = Delaunay(xy).simplices
    except Exception:
        return None, "degenerate_regional_geometry"
    positions = xy[triangles]
    centers = positions.mean(axis=1)
    center_pixels = np.rint(centers).astype(int)
    inside = domain[center_pixels[:, 1], center_pixels[:, 0]]
    support_weights = None
    if support is not None:
        # Select triangles intersecting observed line evidence. A triangle's
        # centroid alone is not the measured groove and used to reject every
        # valid no-line region, or miss lines near a triangle edge.
        intersection = np.zeros(len(triangles),bool)
        support_weights = np.zeros(len(triangles),float)
        accounted = np.zeros_like(domain,dtype=bool)
        for i, vertices in enumerate(positions):
            x0,y0 = np.floor(vertices.min(axis=0)).astype(int)
            x1,y1 = np.ceil(vertices.max(axis=0)).astype(int)
            patch = np.zeros((y1-y0+1,x1-x0+1),np.uint8)
            cv2.fillConvexPoly(patch,np.rint(vertices-[x0,y0]).astype(np.int32),1)
            overlap = (patch>0)&support[y0:y1+1,x0:x1+1]&domain[y0:y1+1,x0:x1+1]&~accounted[y0:y1+1,x0:x1+1]
            support_weights[i] = int(overlap.sum())
            intersection[i] = support_weights[i]>0
            if inside[i]:
                accounted[y0:y1+1,x0:x1+1] |= overlap
        inside &= intersection
    edges = np.linalg.norm(positions - np.roll(positions, 1, axis=1), axis=2)
    # Do not span a large unsupported patch or a hole in the regional domain.
    med_edge = float(np.median(edges))
    inside &= edges.max(axis=1) <= max(med_edge * 2.5, 1)
    positions, triangles = positions[inside], triangles[inside]
    if not len(triangles):
        return None, "insufficient_regional_surface_coverage"
    triangle_area = np.abs(np.cross(positions[:, 1]-positions[:, 0], positions[:, 2]-positions[:, 0])) / 2
    area = support_weights[inside] if support_weights is not None else triangle_area
    depths = np.maximum(depth[triangles], 0).mean(axis=1)
    affected = depths > minimum_depth
    total = float(area.sum())
    if total < (5 if support is not None else 100):
        return None, "insufficient_regional_surface_coverage"
    positive = depths[affected]
    weights = area[affected]
    return {"surface_area_px": total, "affected_area_px": float(weights.sum()),
            "mean_depth": float(np.average(positive, weights=weights)) if len(positive) else None,
            "p90_depth": weighted_percentile(positive, weights, .9) if len(positive) else None,
            "volume": float(np.sum(weights * positive) / max(int(domain.sum()), 1)),
            "triangles": samples["ids"][triangles].tolist(), "triangle_area_px": triangle_area.tolist(),
            "triangle_support_area_px": area.tolist(),
            "triangle_depth": depths.tolist(), "relative_z_samples": depth.tolist()}, None


def weighted_percentile(values, weights, quantile):
    if not len(values):
        return None
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[min(int(np.searchsorted(cumulative, quantile*cumulative[-1])), len(values)-1)]])


def curvature(samples):
    """Coarse mesh curvature from adjacent measured triangles, not local Z pixels.

    A regional quadratic is a six-coefficient reference. At least nine original
    non-collinear samples provide residual degrees of freedom. Adjacent triangle
    slopes are evaluated from original vertex Z and compared to the same mesh
    evaluated on that reference, avoiding the old impossible 24-node demand.
    """
    xy, z = samples["xy"], samples["z"]
    if len(xy) < PARAMETERS["curvature_min_samples"]:
        return None, "insufficient_curvature_samples"
    spread = np.std(xy,axis=0)
    if np.any(spread<1e-5):
        return None, "degenerate_regional_geometry"
    uv = xy/spread
    design = np.column_stack((np.ones(len(xy)), uv[:,0], uv[:,1], uv[:,0]**2, uv[:,0]*uv[:,1], uv[:,1]**2))
    if np.linalg.matrix_rank(design)<6 or np.linalg.cond(design)>PARAMETERS["max_reference_condition"]:
        return None, "degenerate_regional_geometry"
    reference = np.linalg.lstsq(design,z,rcond=None)[0]
    reference_z = design@reference
    try:
        mesh = Delaunay(xy)
    except Exception:
        return None, "degenerate_regional_geometry"
    triangles = mesh.simplices
    local_gradients, reference_gradients, centers, accepted = {}, {}, {}, []
    domain = samples.get("domain")
    for i, tri in enumerate(triangles):
        q = xy[tri]
        edge = np.linalg.norm(q-np.roll(q,1,axis=0),axis=1)
        twice_area = abs(float(np.cross(q[1]-q[0],q[2]-q[0])))
        if twice_area/max(float(edge.max()**2),1e-12)<PARAMETERS["minimum_triangle_shape"]:
            continue
        if domain is not None:
            px = samples["pixel_xy"][tri]
            checks = np.rint(np.vstack((px.mean(axis=0),(px+np.roll(px,1,axis=0))/2))).astype(int)
            if not np.all(domain[checks[:,1],checks[:,0]]):
                continue
        plane = np.column_stack((np.ones(3),q))
        local_gradients[i] = np.linalg.solve(plane,z[tri])[1:]
        reference_gradients[i] = np.linalg.solve(plane,reference_z[tri])[1:]
        centers[i] = q.mean(axis=0)
        accepted.append(i)
    if len(accepted)/max(len(triangles),1)<PARAMETERS["minimum_valid_triangle_fraction"]:
        return None, "insufficient_valid_mesh_coverage"
    edges = []
    values = []
    for i in accepted:
        for j in mesh.neighbors[i]:
            j = int(j)
            if j<=i or j not in local_gradients:
                continue
            delta = centers[j]-centers[i]
            distance2 = float(delta@delta)
            if distance2<=1e-12:
                continue
            raw = float((local_gradients[j]-local_gradients[i])@delta/distance2)
            baseline = float((reference_gradients[j]-reference_gradients[i])@delta/distance2)
            values.append(raw-baseline)
            edges.append([i,j])
    if len(values)<PARAMETERS["minimum_curvature_edges"]:
        return None, "insufficient_valid_mesh_edges"
    values = np.asarray(values)
    # Turn burden compares adjacent curvature-edge samples sharing a triangle.
    transitions = []
    for i in range(len(edges)):
        for j in range(i+1,len(edges)):
            if set(edges[i]) & set(edges[j]) and values[i]*values[j]<0:
                transitions.append(abs(float(values[i]-values[j])))
    return {"p90_deviation":float(np.percentile(abs(values),90)),
            "turning":float(sum(transitions)/max(len(edges),1)),
            "convex":float(np.mean(np.maximum(values,0))),
            "curvature_deviation_samples":values.tolist(),
            "sample_ids":samples["ids"].tolist(),
            "reference_quadratic":reference.tolist(),"reference_coordinate_scale":spread.tolist(),
            "original_triangles":samples["ids"][triangles].tolist(),"accepted_triangle_indices":accepted,
            "triangle_gradient_samples":{str(k):v.tolist() for k,v in local_gradients.items()},
            "reference_gradient_samples":{str(k):v.tolist() for k,v in reference_gradients.items()},
            "edge_triangle_indices":edges,"edge_count":len(edges),"turning_edge_count":len(transitions),
            "input_sample_count":len(xy),"valid_triangle_fraction":len(accepted)/len(triangles),
            "geometry_sampling":"coarse_original_vertex_mesh",
            "method":"adjacent_triangle_slope_deviation_from_regional_quadratic"}, None


def dense_residual(arrays, meta, sigma_px=12):
    z = arrays.get("relative_z_map")
    if z is None:
        return None, "insufficient_geometry_resolution"
    if meta.get("geometry_sampling") != "dense_surface" or meta.get("interpolated_from_sparse", False):
        return None, "unverified_dense_geometry_provenance"
    z = np.asarray(z, float)
    valid = np.asarray(arrays["valid"]) > 0
    if z.shape != valid.shape:
        return None, "invalid_dense_geometry_shape"
    scale = scale_of(arrays["landmarks"])
    if scale is None:
        return None, "invalid_facial_scale"
    valid &= np.isfinite(z)
    weight = cv2.GaussianBlur(valid.astype(float), (0, 0), sigma_px)
    smooth = cv2.GaussianBlur(np.where(valid, z / scale, 0), (0, 0), sigma_px) / np.maximum(weight, 1e-8)
    # Reference support must surround the pixel, not simply exist on one side.
    interior = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    residual = np.where(interior & (weight >= .95), z / scale-smooth, np.nan)
    return residual, None
