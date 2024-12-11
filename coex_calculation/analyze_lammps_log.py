import argparse, gzip, math
import numpy as np

def read_logdata(path, nsectionskip=0, verbose=1):
    ready = False
    sections = 0
    data = []
    if path[-3:] == '.gz':
        fileopen, mode = gzip.open, 'rt'
    else:
        fileopen, mode = open, 'r'
    with fileopen(path, mode) as f:
        for line in f:
            if len(line.split()) > 0 and line.split()[0] == 'Step':
                fields = line.split()
                ready = True
                sections += 1
                sectioncount = 0
                if verbose > 1:
                    print(fields)
            elif line[:7] == 'WARNING' and verbose > 2:
                print(line.strip())
                continue
            elif ready:
                if len(line.split()) == len(fields):
                    sectioncount += 1
                    if sectioncount <= nsectionskip:
                        continue
                    try:
                        data.append([float(x) for x in line.split()])
                    except ValueError:
                        continue
                # else:
                #     ready = False
    if verbose:
        print("# Loaded %d sections" % sections)
    return fields, np.array(data)

def autocorr(X):
    dX = X - np.mean(X)
    result = np.correlate(dX, dX, mode='full')
    C = result[result.size//2:] / result[result.size//2]
    return C

def count_crossings(X, y):
    return sum(1 for i in range(1, len(X)) if X[i] >= y and X[i-1] < y)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to log file")
    parser.add_argument('--nsectionskip', type=int, default=0, \
                        help="number of lines to skip per section [0]")
    parser.add_argument('--neq', type=int, default=0, help="number of equilibration lines [0]")
    parser.add_argument('--natoms-mol', type=int, default=1, help="number of atoms per molecule [1]")
    parser.add_argument('--colvar', type=str, default=None, help="collective variable [None]")
    parser.add_argument('--colvar-dx', type=float, default=1., help="collective variable bin width [1]")
    parser.add_argument('--colvar-label', type=str, default=None, \
                        help="variable to averages for col var histogram label [None]")
    parser.add_argument('--colvar-percentile', type=float, default=None, \
                        help="calculate the specified percentile of the collective variable [None]")
    parser.add_argument('--colvar-flux', type=float, default=None, \
                        help="count flux of collective variable across a specified threshold [None]")
    clargs = parser.parse_args()

    fields, data = read_logdata(clargs.path, nsectionskip=clargs.nsectionskip)
    print("# neq =", clargs.neq)
    print("# nlines =", data.shape[0])

    print("# %18s %14s %14s %14s %14s %14s" % \
          ('', 'mean', 'stderr', 'stddev', 'min', 'max'))
    for i in range(len(fields)):
        stderr = np.std(data[clargs.neq:,i]) / math.sqrt(data.shape[0] - 1.)
        print("%20s %14g %14g %14g %14g %14g" % \
              (fields[i], np.mean(data[clargs.neq:,i]), stderr, np.std(data[clargs.neq:,i]), \
               np.min(data[clargs.neq:,i]), np.max(data[clargs.neq:,i])))

    if clargs.colvar is not None:
        field_index = fields.index(clargs.colvar)
        if clargs.colvar_label is not None:
            label_index = fields.index(clargs.colvar_label)
            colvar_label = np.mean(data[clargs.neq:,label_index])
        else:
            clargs.colvar_label = 'None'
            colvar_label = np.nan
        colvar_min = np.min(data[clargs.neq:,field_index])
        colvar_max = np.max(data[clargs.neq:,field_index])
        minbin = colvar_min - colvar_min % clargs.colvar_dx
        maxbin = colvar_max - colvar_max % clargs.colvar_dx + clargs.colvar_dx
        nbins = int((maxbin - minbin) / clargs.colvar_dx)
        hist, bins = np.histogram(data[clargs.neq:,field_index], bins=nbins, range=(minbin, maxbin))
        print("Writing colvar.dat")
        with open('colvar.dat', 'w') as f:
            f.write("# histogram of collective variable: %s\n\n" % clargs.colvar)
            f.write("# label=%s colvar normalized_frequency\n" % clargs.colvar_label)
            f.write("%g %g 0\n" % (colvar_label, bins[0]))
            for i in range(hist.shape[0]):
                f.write("%g %g %g\n%g %g %g\n" % \
                        (colvar_label, bins[i], hist[i] / hist.sum(),
                         colvar_label, bins[i+1], hist[i] / hist.sum()))
            f.write("%g %g 0\n\n" % (colvar_label, bins[-1]))

        if clargs.colvar_percentile is not None:
            print("colvar mean, %g-percentile =" % clargs.colvar_percentile, \
                  np.mean(data[clargs.neq:,field_index]), \
                  np.percentile(data[clargs.neq:,field_index], clargs.colvar_percentile))

        if clargs.colvar_flux is not None:
            ncrossings = count_crossings(data[clargs.neq:,field_index], clargs.colvar_flux)
            dt = data[1,fields.index('Step')] - data[0,fields.index('Step')]
            flux = ncrossings / (data.shape[0] - clargs.neq) / dt
            print("Flux (colvar > %g) = %g crossings / timestep" % (clargs.colvar_flux, flux))
