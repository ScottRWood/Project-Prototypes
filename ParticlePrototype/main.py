from numpy import *
from numpy.random import *

def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = random(), 0

    for u in [(u0+i)/n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j-1)

    return indices


def particle_filter(sequence, pos, stepsize, n):
    seq = iter(sequence)
    x = ones((n, 1), int) * pos
    f0 = seq.__next__()[tuple(pos)] * ones(n)
    yield pos, x, ones(n)/n

    for im in seq:
        np.add(x, uniform(-stepsize, stepsize, x.shape), out=x, casting="unsafe")
        x = x.clip(zeros(2), array(im.shape)-1).astype(int)
        f = im[tuple(x.T)]
        w = 1./(1. + (f0-f)**2)

        w /= sum(w)
        yield sum(x.T*w, axis=1), x, w

        if 1./sum(w**2) < n/2. :
            x = x[resample(w),:]


if __name__ == "__main__":
    from pylab import *
    import time
    from IPython import display

    ion()
    seq = [ im for im in zeros((20,240,320), int)]
    x0 = array([120, 160])

    xs = vstack((arange(20)*3, arange(20)*2)).T + x0

    for t,x in enumerate(xs):
        xslice = slice(x[0]-8, x[0]+8)
        yslice = slice(x[1]-8, x[1]+8)
        seq[t][xslice, yslice] = 255

    for im, p in zip(seq, particle_filter(seq, x0, 8, 100)):
        pos, xs, ws = p
        position_overlay = zeros_like(im)
        position_overlay[np.array(pos).astype(int)] = 1
        particle_overlay = zeros_like(im)
        particle_overlay[tuple(xs.T)] = 1
        draw()
        time.sleep(0.3)
        clf()

        imshow(im, cmap=cm.gray)
        spy(position_overlay, marker='.', color='b')
        spy(particle_overlay, marker=',', color='r')
        display.clear_output(wait=True)
        display.display(show())
