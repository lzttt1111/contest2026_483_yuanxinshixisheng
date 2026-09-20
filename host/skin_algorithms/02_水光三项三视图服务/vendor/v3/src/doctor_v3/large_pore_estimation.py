"""Explicit quantile-based large-pore estimate from retained size summaries."""
def estimate_large_density(density,p50,p90,maximum,threshold):
    if density==0:
        return 0.0
    if any(v is None for v in (density,p50,p90,maximum,threshold)):
        return None
    if density<0 or not 0<p50<=p90<=maximum or threshold<=0:
        return None
    points=[(0.0,0.0),(p50,.5),(p90,.9),(maximum,1.0)]
    if threshold>=maximum:return 0.0
    for (lo,fl),(hi,fh) in zip(points,points[1:]):
        if lo<=threshold<hi and hi>lo:
            cdf=fl+(fh-fl)*(threshold-lo)/(hi-lo)
            return density*(1-cdf)
    return None

